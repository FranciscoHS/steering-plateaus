#!/usr/bin/env python3
"""Build a word-searchable index over a whole dictionary of directions.

Companion to build_neuronpedia_dashboard.py.  Where that script renders a fixed
set of directions into cards, this one indexes ALL K directions so you can later
search for a word and get the directions that respond to it most strongly.

It runs the model once over a FineWeb corpus, caches the layer-L activations,
then sweeps the dictionary in chunks computing per-direction aggregates.  Output
is a single JSON (search_data.json) consumed by build_search_dashboard.py to
emit a static, offline, no-server searchable HTML page.

Two notions of "responds to a word" are indexed (both shown side by side):
  - input-side  : the direction fires on tokens that ARE that word in the corpus
                  (per-direction top activating token TYPES -> inverted index).
  - output-side : the direction PROMOTES that word in W_U . ln_f(d) (logit lens
                  top tokens -> inverted index).  No corpus needed, exact.

Per-token SCORE depends on --score_mode (same as the dashboard):
  - dot          : a . d_k
  - cos_centered : (a - abar).d_k / ||a - abar||   (matches the featureness rank)

Run on a pod (needs the model + the dict on the network volume):

    HF_HOME=/workspace/hf_cache python -u \
      scripts/superellipse/index_directions_for_search.py \
      --dict_path /workspace/.../dictionary.pt --model qwen3-1.7b --layer 14 \
      --score_mode cos_centered --n_seqs 8000 --seq_len 64 \
      --out /workspace/.../search_index

The expensive part is the indexing pass; regenerating the HTML from the JSON is
cheap and runs anywhere (build_search_dashboard.py).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, '.')

import numpy as np
import torch

from src.model import load_model, _get_lm_head, _get_final_norm
from src.data import load_fineweb_fixed_length


def cache_activations(model, tok_ids, layer, batch_size, device, dtype):
    """One forward pass over the corpus; return layer-L activations [N,T,d] on CPU.

    Kept in `dtype` (bf16 by default) to bound RAM — for 8000x64x2048 bf16 that
    is ~2GB, which we can hold while we sweep the dictionary in chunks.
    """
    N, T = tok_ids.shape
    H = None
    for i in range(0, N, batch_size):
        bt = tok_ids[i:i + batch_size].to(device)
        with torch.no_grad():
            h = model(bt, output_hidden_states=True).hidden_states[layer + 1]
        if H is None:
            H = torch.empty(N, T, h.shape[-1], dtype=dtype)
        H[i:i + bt.shape[0]] = h.to(dtype).cpu()
        if (i // batch_size) % 20 == 0:
            print(f'  cached {i + bt.shape[0]}/{N}', flush=True)
    return H


def score_chunk(H, D_chunk, abar, score_mode, seq_batch, device):
    """Per-token scores [N,T,c] for a chunk of c directions, from cached H.

    Streams over sequence-batches to bound GPU memory; the result is a CPU
    float32 tensor we aggregate over.
    """
    N, T, d = H.shape
    c = D_chunk.shape[0]
    out = torch.empty(N, T, c, dtype=torch.float32)
    Dt = D_chunk.T.contiguous()
    for i in range(0, N, seq_batch):
        hb = H[i:i + seq_batch].to(device).float()
        if score_mode == 'cos_centered':
            hb = hb - abar
            hb = hb / hb.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        out[i:i + hb.shape[0]] = (hb @ Dt).cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--score_mode', default='cos_centered',
                    choices=['dot', 'cos_centered'])
    ap.add_argument('--n_seqs', type=int, default=8000)
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--batch_size', type=int, default=64,
                    help='forward-pass batch for caching activations')
    ap.add_argument('--seq_batch', type=int, default=512,
                    help='sequence-batch when scoring a direction chunk')
    ap.add_argument('--dir_chunk', type=int, default=1000,
                    help='# directions scored at once (bounds CPU RAM: '
                         'N*T*dir_chunk float32)')
    ap.add_argument('--n_top_tokens', type=int, default=25,
                    help='# distinct top activating token TYPES per direction')
    ap.add_argument('--cand_tokens', type=int, default=300,
                    help='# raw top hits scanned before de-duping to token types')
    ap.add_argument('--n_snippets', type=int, default=5,
                    help='# activating example snippets stored per direction')
    ap.add_argument('--sample_top_frac', type=float, default=0.10,
                    help='sample the stored snippets from the top this-fraction '
                         'of sequences (ranked by peak score), once, at index '
                         'time. <=0 restores strict top-n.')
    ap.add_argument('--sample_seed', type=int, default=0,
                    help='RNG seed for --sample_top_frac snippet sampling')
    ap.add_argument('--ctx_left', type=int, default=18)
    ap.add_argument('--ctx_right', type=int, default=8)
    ap.add_argument('--topk_logits', type=int, default=12)
    ap.add_argument('--max_dirs_per_token', type=int, default=200,
                    help='cap directions stored per token in the inverted index')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    cache_dtype = torch.bfloat16  # for the H cache; model runs in args.dtype

    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])

    blob = torch.load(args.dict_path, map_location='cpu')
    D_full = blob['dictionary'].to(torch.float32)
    D_full = D_full / (D_full.norm(dim=-1, keepdim=True) + 1e-12)
    K, d_model = D_full.shape
    names_full = blob.get('names')
    print(f'Dictionary: {K} directions, d_model={d_model}', flush=True)

    # --- Corpus -----------------------------------------------------------
    print(f'Loading {args.n_seqs} FineWeb seqs (seq_len={args.seq_len})', flush=True)
    tok_list = load_fineweb_fixed_length(args.n_seqs, tokenizer,
                                         seq_len=args.seq_len, seed=args.seed)
    tok_ids = torch.stack(tok_list)
    N, T = tok_ids.shape

    print('Caching layer activations', flush=True)
    H = cache_activations(model, tok_ids, args.layer, args.batch_size,
                          device, cache_dtype)

    abar = None
    if args.score_mode == 'cos_centered':
        # corpus mean over all tokens, pos-0 skipped
        abar = H[:, 1:, :].float().reshape(-1, d_model).mean(0).to(device)
        print('Computed corpus mean abar', flush=True)

    # decode cache: token id -> string (vocab is bounded, avoid re-decoding)
    _dec = {}
    def decode_tok(tid):
        tid = int(tid)
        s = _dec.get(tid)
        if s is None:
            s = tokenizer.decode([tid])
            _dec[tid] = s
        return s

    def norm_key(s):
        return s.strip().lower()

    # --- Logit lens (chunked) --------------------------------------------
    print('Computing logit lens', flush=True)
    lm_head = _get_lm_head(model)
    ln_f = _get_final_norm(model)
    W_U = lm_head.weight.detach().float().to(device)   # [vocab, d]
    ln_dtype = next(ln_f.parameters()).dtype

    flat_tok = tok_ids[:, 1:].reshape(-1)   # corpus token ids, pos-0 skipped

    dirs = [None] * K            # per-direction summary dicts
    inv_input = {}               # norm token -> list[(dir, score)]
    inv_output = {}              # norm token -> list[(dir, logit)]

    t0 = time.time()
    for c0 in range(0, K, args.dir_chunk):
        c1 = min(K, c0 + args.dir_chunk)
        D_chunk = D_full[c0:c1].to(device)
        c = c1 - c0

        # logit lens for this chunk
        with torch.no_grad():
            ln_D = ln_f(D_chunk.to(ln_dtype)).float()
            lg = ln_D @ W_U.T                     # [c, vocab]
        top_pos = torch.topk(lg, args.topk_logits, dim=1)
        top_neg = torch.topk(-lg, args.topk_logits, dim=1)

        # per-token corpus scores for this chunk
        sc = score_chunk(H, D_chunk, abar, args.score_mode, args.seq_batch, device)
        pos_scores = sc[:, 1:, :]                 # [N, T-1, c], pos-0 skipped

        # histogram stats
        flatN = pos_scores.reshape(-1, c)         # [N*(T-1), c]
        col_max = flatN.max(0).values
        col_mean = flatN.mean(0)
        frac_pos = (flatN > 0).float().mean(0)
        # vmax for coloring: 99th pctile of |score|, on a strided sample
        # (torch.quantile caps at ~2^24 elems, so downsample + numpy).
        step = max(1, flatN.shape[0] // 20000)
        samp = flatN[::step].numpy()
        vmax_pos = np.percentile(np.clip(samp, 0, None), 99, axis=0)
        vmax_neg = np.percentile(np.clip(-samp, 0, None), 99, axis=0)

        # top activating token TYPES: top cand hits per dir, then de-dup ids
        cand = min(args.cand_tokens, flatN.shape[0])
        tk = torch.topk(flatN.T, cand, dim=1)     # [c, cand]
        tk_vals, tk_idx = tk.values, tk.indices

        # example sequences per dir: snippets are sampled (once, here) from the
        # top sample_top_frac of sequences ranked by peak score, rather than the
        # strict top-n, so the examples aren't all near-duplicate extremes.
        seq_max, seq_argmax = pos_scores.max(dim=1)   # [N, c] (argmax in 1..T-1)
        ranked = torch.argsort(seq_max.T, dim=1, descending=True)   # [c, N]
        if args.sample_top_frac > 0:
            pool_n = min(N, max(args.n_snippets,
                                int(round(args.sample_top_frac * N))))
        else:
            pool_n = min(N, args.n_snippets)

        for j in range(c):
            k = c0 + j

            # input-side top token types
            ids = tk_idx[j]
            seen, toptokens = set(), []
            for r in range(cand):
                tid = int(flat_tok[int(ids[r])])
                if tid in seen:
                    continue
                seen.add(tid)
                toptokens.append((decode_tok(tid), round(float(tk_vals[j, r]), 4)))
                if len(toptokens) >= args.n_top_tokens:
                    break

            # output-side logit lens tokens
            lpos = [(decode_tok(top_pos.indices[j, r]),
                     round(float(top_pos.values[j, r]), 4))
                    for r in range(args.topk_logits)]
            lneg = [(decode_tok(top_neg.indices[j, r]),
                     round(-float(top_neg.values[j, r]), 4))
                    for r in range(args.topk_logits)]

            vp = max(float(vmax_pos[j]), 1e-6)
            vn = max(float(vmax_neg[j]), 1e-6)

            # pick example sequences (sampled from the top pool, deterministic)
            ranked_j = ranked[j]
            take = min(args.n_snippets, pool_n)
            if args.sample_top_frac > 0 and pool_n > take:
                rng = np.random.default_rng(args.sample_seed + k)
                pick = np.sort(rng.choice(pool_n, size=take, replace=False))
            else:
                pick = np.arange(take)
            ex_idx = [int(ranked_j[p]) for p in pick]

            # example snippets (tokens + per-token scores around the peak)
            snippets = []
            for si in ex_idx:
                peak = int(seq_argmax[si, j]) + 1     # +1: pos-0 was skipped
                lo = max(1, peak - args.ctx_left)
                hi = min(T, peak + args.ctx_right + 1)
                toks = [decode_tok(tok_ids[si, p]) for p in range(lo, hi)]
                scs = [round(float(sc[si, p, j]), 3) for p in range(lo, hi)]
                snippets.append({'peak': peak - lo,
                                 'max': round(float(seq_max[si, j]), 3),
                                 'toks': toks, 'scs': scs})

            dirs[k] = {
                'id': k,
                'name': names_full[k] if names_full else None,
                'max': round(float(col_max[j]), 3),
                'mean': round(float(col_mean[j]), 4),
                'frac_pos': round(float(frac_pos[j]), 3),
                'vmax_pos': round(vp, 4), 'vmax_neg': round(vn, 4),
                'top_tokens': toptokens,
                'logit_pos': lpos, 'logit_neg': lneg,
                'snippets': snippets,
            }

            # inverted indices (keyed by normalized token text)
            for tokstr, val in toptokens:
                if val <= 0:
                    continue
                inv_input.setdefault(norm_key(tokstr), []).append((k, val))
            for tokstr, val in lpos:
                inv_output.setdefault(norm_key(tokstr), []).append((k, val))

        done = c1
        rate = done / max(1e-9, time.time() - t0)
        print(f'  indexed {done}/{K} dirs ({rate:.0f}/s)', flush=True)

    # sort + cap inverted indices
    def finalize_inv(inv):
        out = {}
        for key, lst in inv.items():
            if not key:
                continue
            lst.sort(key=lambda x: -x[1])
            out[key] = [[i, round(v, 4)] for i, v in lst[:args.max_dirs_per_token]]
        return out

    data = {
        'meta': {
            'model': args.model, 'layer': args.layer,
            'score_mode': args.score_mode, 'dict_path': args.dict_path,
            'K': K, 'n_seqs': N, 'seq_len': T,
            'n_top_tokens': args.n_top_tokens, 'n_snippets': args.n_snippets,
        },
        'dirs': dirs,
        'inv_input': finalize_inv(inv_input),
        'inv_output': finalize_inv(inv_output),
    }

    out_json = os.path.join(args.out, 'search_data.json')
    with open(out_json, 'w') as f:
        json.dump(data, f)
    sz = os.path.getsize(out_json) / 1e6
    print(f'\nSaved {out_json} ({sz:.1f} MB)', flush=True)
    print(f'  {len(data["inv_input"])} input tokens, '
          f'{len(data["inv_output"])} output tokens indexed', flush=True)


if __name__ == '__main__':
    main()
