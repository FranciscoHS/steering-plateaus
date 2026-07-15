#!/usr/bin/env python3
"""Causal ablation of directions -> per-token KL on a natural corpus.

For each direction d (unit norm), project it out of the residual stream at `layer`
during a forward pass over FineWeb:  h' = h - (d . h) d, at ALL positions.
Measure KL(p_baseline || p_ablated) per token, aggregate per direction.

Families compared:
  - found        : random sample of 40k-dictionary atoms
  - contrastive  : diff-of-means directions (lang + pairwise)
  - random       : random Gaussian unit vectors (matched norm = unit)

Efficiency: cache resid@layer once per batch (one full forward), then re-run only the
upper blocks (layer+1 .. end) per direction. Baseline log-probs come from the same
truncated path on the unablated resid, so KL>=0 and KL=0 at zero projection.
"""
import sys, os, argparse, pickle
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import (load_model, _get_blocks, _get_final_norm, _get_lm_head,
                        _is_rope_model, _get_position_embeddings, _run_block,
                        _is_cat_pos_emb_model, _get_cat_pos_emb)
from src.data import load_fineweb_fixed_length


def truncated_forward_allpos(model, hidden, start_layer, pos_emb):
    """Run blocks[start_layer+1:] + final norm + lm_head, returning ALL-position logits.

    Like src.model.forward_from_layer_allpos but keeps every position (that helper
    only returns the final token)."""
    h = hidden
    for block in _get_blocks(model)[start_layer + 1:]:
        h = _run_block(block, h, model, pos_emb)
    h = _get_final_norm(model)(h)
    return _get_lm_head(model)(h)  # [B, S, vocab]


def build_directions(args, d_model, device):
    """Return (dirs [N, d_model] unit-norm fp32 on CPU, meta list of (family, label))."""
    fams = set(args.families.split(','))
    dirs, meta = [], []

    def sampled_atoms(dict_path, fam, n, seed):
        blob = torch.load(dict_path, map_location='cpu', weights_only=False)
        D = blob['dictionary'].float()
        D = D / D.norm(dim=-1, keepdim=True)
        K = D.shape[0]
        print(f"[{fam}] dictionary has {K} atoms, dim {D.shape[1]}")
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(K, generator=g)[:min(n, K)]
        for i in idx.tolist():
            dirs.append(D[i]); meta.append((fam, f'atom{i}'))

    # found: 40k dictionary atoms (with alignment-to-activations term)
    if 'found' in fams:
        sampled_atoms(args.dict_path, 'found', args.n_found, args.found_seed)
    # found_noalign: same recipe dictionary built without the activation-alignment term
    if 'found_noalign' in fams:
        sampled_atoms(args.noalign_dict_path, 'found_noalign', args.n_found, args.found_seed)

    # contrastive: diff-of-means
    if 'contrastive' in fams:
        cd = pickle.load(open(args.contrastive_pkl, 'rb'))
        for name, v in cd['features'].items():
            v = v.float(); v = v / v.norm()
            dirs.append(v); meta.append(('contrastive', f'lang_{name}'))
        if args.include_pairwise and 'pairwise_means' in cd:
            for key, v in cd['pairwise_means'].items():
                v = v.float(); v = v / v.norm()
                label = 'pair_' + '_'.join(map(str, key)) if isinstance(key, tuple) else f'pair_{key}'
                dirs.append(v); meta.append(('contrastive', label))

    # sae: decoder directions of the Qwen-Scope SAE (unit-normalized, same as probe script)
    if 'sae' in fams and args.sae_ckpt:
        sd = torch.load(args.sae_ckpt, map_location='cpu', weights_only=False)
        W_enc = sd['W_enc'].float()
        W_dec = sd['W_dec'].float()
        if W_dec.shape[0] != W_enc.shape[0]:
            W_dec = W_dec.T  # -> [n_feat, d_model]
        W_dec = W_dec / (W_dec.norm(dim=-1, keepdim=True) + 1e-12)
        nf = W_dec.shape[0]
        print(f"[sae] decoder has {nf} latents, dim {W_dec.shape[1]}")
        gs = torch.Generator().manual_seed(args.sae_seed)
        n_sae = min(args.n_sae, nf)
        sidx = torch.randperm(nf, generator=gs)[:n_sae]
        for i in sidx.tolist():
            dirs.append(W_dec[i]); meta.append(('sae', f'lat{i}'))

    # random: Gaussian unit vectors (matched norm = unit)
    if 'random' in fams:
        gr = torch.Generator().manual_seed(args.random_seed)
        R = torch.randn(args.n_random, d_model, generator=gr)
        R = R / R.norm(dim=-1, keepdim=True)
        for j in range(args.n_random):
            dirs.append(R[j]); meta.append(('random', f'rand{j}'))

    dirs = torch.stack(dirs).float()  # [N, d_model], CPU
    present = [m[0] for m in meta]
    n_by_fam = {f: present.count(f) for f in dict.fromkeys(present)}
    print(f"[dirs] total {dirs.shape[0]}  by family: {n_by_fam}")
    return dirs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--dict_path', default=None)
    ap.add_argument('--noalign_dict_path', default=None,
                    help='dictionary built without the activation-alignment term '
                         '(family "found_noalign")')
    ap.add_argument('--contrastive_pkl', default=None)
    ap.add_argument('--families', default='found,contrastive,sae,random',
                    help='comma list: found,found_noalign,contrastive,sae,random')
    ap.add_argument('--include_pairwise', action='store_true')
    ap.add_argument('--cond_pcts', default='5,10,25',
                    help='percentiles for top-p%% activation-conditioned mean KL')
    ap.add_argument('--n_found', type=int, default=256)
    ap.add_argument('--n_random', type=int, default=256)
    ap.add_argument('--n_sae', type=int, default=256)
    ap.add_argument('--sae_ckpt', default=None,
                    help='Qwen-Scope SAE .pt (W_enc/W_dec/...); adds an "sae" family of '
                         'unit-normalized decoder directions')
    ap.add_argument('--found_seed', type=int, default=0)
    ap.add_argument('--random_seed', type=int, default=0)
    ap.add_argument('--sae_seed', type=int, default=0)
    ap.add_argument('--n_seqs', type=int, default=512)
    ap.add_argument('--load_n', type=int, default=None,
                    help='n to request from the fineweb cache (>= n_seqs); slice to n_seqs. '
                         'Set to an existing cache n to avoid re-streaming.')
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--corpus_seed', type=int, default=42)
    ap.add_argument('--batch_seqs', type=int, default=16)
    ap.add_argument('--fineweb_cache', default='/workspace/fineweb_cache')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer, device = load_model(args.model, device=device, dtype=torch.bfloat16)
    model.eval()
    d_model = getattr(model.config, 'hidden_size', None) or getattr(model.config, 'n_embd')
    print(f"[model] {args.model} device={device} d_model={d_model}")

    # corpus
    load_n = args.load_n or args.n_seqs
    tok_list = load_fineweb_fixed_length(load_n, tokenizer, seq_len=args.seq_len,
                                         seed=args.corpus_seed, cache_dir=args.fineweb_cache)
    tokens = torch.stack([t[:args.seq_len] for t in tok_list])[:args.n_seqs]  # [n_seqs, seq_len]
    n_seqs, S = tokens.shape
    n_tokens = n_seqs * S
    print(f"[corpus] {n_seqs} seqs x {S} tokens = {n_tokens} positions")

    dirs, meta = build_directions(args, d_model, device)
    N = dirs.shape[0]
    dirs_dev = dirs.to(device)  # [N, d_model] fp32

    kl_per_token = torch.zeros(N, n_tokens, dtype=torch.float32)   # CPU
    act_per_token = torch.zeros(N, n_tokens, dtype=torch.float32)  # |h.d| per token
    proj_sq_sum = torch.zeros(N, dtype=torch.float64)             # for captured-proj
    layer = args.layer

    pos = 0
    nb = (n_seqs + args.batch_seqs - 1) // args.batch_seqs
    for b in range(nb):
        sl = slice(b * args.batch_seqs, (b + 1) * args.batch_seqs)
        tok_b = tokens[sl].to(device)
        B = tok_b.shape[0]
        with torch.no_grad():
            out = model(tok_b, output_hidden_states=True)
            resid = out.hidden_states[layer + 1]  # [B, S, d_model], model dtype
        # position embeddings for the truncated forward (depend only on S)
        pos_emb = None
        if _is_rope_model(model):
            pos_emb = _get_position_embeddings(model, resid)
        elif _is_cat_pos_emb_model(model):
            pos_emb = _get_cat_pos_emb(model, resid.shape[1])

        with torch.no_grad():
            base_logits = truncated_forward_allpos(model, resid, layer, pos_emb)
            base_logp = F.log_softmax(base_logits.float(), dim=-1)  # [B,S,V]
            base_p = base_logp.exp()

        resid_f = resid.float()  # [B,S,d] for projection precision
        bt = B * S
        for di in range(N):
            d = dirs_dev[di]  # [d_model] fp32
            coef = (resid_f * d).sum(-1, keepdim=True)  # [B,S,1]
            csq = coef.squeeze(-1)  # [B,S]
            proj_sq_sum[di] += (csq ** 2).double().sum().cpu()
            act_per_token[di, pos:pos + bt] = csq.abs().reshape(-1).cpu()
            resid_abl = (resid_f - coef * d).to(resid.dtype)
            with torch.no_grad():
                abl_logits = truncated_forward_allpos(model, resid_abl, layer, pos_emb)
                abl_logp = F.log_softmax(abl_logits.float(), dim=-1)
                kl = (base_p * (base_logp - abl_logp)).sum(-1)  # [B,S]
            kl_per_token[di, pos:pos + bt] = kl.reshape(-1).cpu()
        pos += bt
        print(f"[batch {b+1}/{nb}] done ({pos}/{n_tokens} positions)", flush=True)

    # summaries
    proj_var = (proj_sq_sum / n_tokens).float()  # mean (h.d)^2
    mean_kl = kl_per_token.mean(1)
    median_kl = kl_per_token.median(1).values
    p90_kl = kl_per_token.quantile(0.90, dim=1)
    max_kl = kl_per_token.max(1).values

    # conditional-on-activation: for each direction, mean KL over its own top-p%
    # most-active tokens (ranked by |h.d|).
    pcts = [int(x) for x in args.cond_pcts.split(',')]
    cond_kl = {}
    for p in pcts:
        k = max(1, round(n_tokens * p / 100))
        top_idx = act_per_token.topk(k, dim=1).indices            # [N, k]
        cond_kl[p] = kl_per_token.gather(1, top_idx).mean(1)       # [N]

    summary = {
        'meta': meta, 'family': [m[0] for m in meta], 'label': [m[1] for m in meta],
        'mean_kl': mean_kl, 'median_kl': median_kl, 'p90_kl': p90_kl, 'max_kl': max_kl,
        'cond_kl': cond_kl, 'cond_pcts': pcts,
        'captured_proj_var': proj_var,
        'args': vars(args), 'n_tokens': n_tokens, 'n_seqs': n_seqs, 'seq_len': S,
    }
    with open(os.path.join(args.out_dir, 'summary.pkl'), 'wb') as f:
        pickle.dump(summary, f)
    torch.save(kl_per_token, os.path.join(args.out_dir, 'kl_per_token.pt'))
    torch.save(act_per_token, os.path.join(args.out_dir, 'act_per_token.pt'))

    # quick console report
    fams = np.array([m[0] for m in meta])
    for fam in ('found', 'found_noalign', 'contrastive', 'sae', 'random'):
        m = fams == fam
        if m.sum() == 0:
            continue
        mt = torch.from_numpy(m)
        v = mean_kl[mt]
        cond_str = '  '.join(f'top{p}%={cond_kl[p][mt].median():.4e}' for p in pcts)
        print(f"[{fam:13s}] n={m.sum():4d}  mean-KL median={v.median():.4e}  "
              f"| cond-KL median {cond_str}")

    # plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        eps = 1e-12
        lo = max(eps, float(mean_kl[mean_kl > 0].min()) if (mean_kl > 0).any() else eps)
        hi = float(mean_kl.max()) + eps
        bins = np.logspace(np.log10(lo), np.log10(hi), 50)
        for fam, c in (('random', 'gray'), ('sae', 'tab:green'),
                       ('contrastive', 'tab:orange'), ('found_noalign', 'tab:red'),
                       ('found', 'tab:blue')):
            m = fams == fam
            if m.sum() == 0:
                continue
            ax.hist(mean_kl[torch.from_numpy(m)].numpy().clip(lo, hi), bins=bins,
                    alpha=0.55, label=f'{fam} (n={m.sum()})', color=c, density=True)
        ax.set_xscale('log')
        ax.set_xlabel('per-direction mean KL(base || ablated)  [nats/token]')
        ax.set_ylabel('density')
        ax.set_title(f'{args.model} L{layer}: ablation KL by direction family')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, 'hist_meankl.png'), dpi=130)
        print('[plot] wrote hist_meankl.png')
    except Exception as e:
        print(f'[plot] skipped: {e}')

    print(f'[done] outputs in {args.out_dir}')


if __name__ == '__main__':
    main()
