#!/usr/bin/env python3
"""Normalized token-type diversity of each direction's firing tail.

For each direction, take the --n_pos top-|score| corpus positions (centered
cosine, as featureness) and look at the PEAK TOKENS. Normalize each token:
lowercase -> strip surrounding whitespace/punctuation -> map anything
digit-containing or a number-word to <NUM>; pure-punctuation tokens keep their
surface form. Then report the number of distinct normalized types and the
entropy over them.

Intuition: a token detector (even one firing on '1', ' 1', 'one', 'One')
collapses to ~1 normalized type; a semantic feature fires on many. Combine
with kurtosis: feature-like = high kurtosis AND high diversity.

One model pass over the corpus (activations cached), then pure matmuls per
atom chunk — cheap even for 20k atoms. Controls (labeled contrastive dicts)
are appended for calibration.

    HF_HOME=/workspace/hf_cache python -u \\
        scripts/superellipse/token_diversity.py \\
        --dict_path .../dictionary.pt --ranking_pkl .../featureness/ranking.pkl \\
        --control_dicts results/scaffold_qwen_L14/dictionary.pt \\
                        results/langcode_qwen_L14/dictionary.pt \\
        --out .../featureness/token_diversity
"""
import argparse
import os
import pickle
import string
import sys
sys.path.insert(0, '.')

from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model
from src.data import load_fineweb_fixed_length

PUNCT = string.punctuation + '“”‘’«»–—·…¡¿'
NUMWORDS = {
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
    'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty',
    'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred',
    'thousand', 'million', 'billion', 'trillion', 'first', 'second', 'third',
    'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
}


def normalize_token(tok):
    t = tok.strip().lower()
    core = t.strip(PUNCT)
    if core == '':
        return t if t else '<WS>'      # pure punctuation keeps surface form
    if any(c.isdigit() for c in core) or core in NUMWORDS:
        return '<NUM>'
    return core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--ranking_pkl', required=True,
                    help='featureness ranking.pkl (kurtosis per atom)')
    ap.add_argument('--control_dicts', nargs='*', default=[])
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--n_seqs', type=int, default=2000)
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--n_pos', type=int, default=50)
    ap.add_argument('--atom_chunk', type=int, default=2000)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--topn_report', type=int, default=60)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])

    blob = torch.load(args.dict_path, map_location='cpu')
    D_full = blob['dictionary'].float()
    D_full = D_full / (D_full.norm(dim=-1, keepdim=True) + 1e-12)
    K = D_full.shape[0]
    rk = pickle.load(open(args.ranking_pkl, 'rb'))
    kurt = np.asarray(rk['kurtosis'])

    dirs = [D_full]
    meta = [('dict', f'atom {a}', float(kurt[a])) for a in range(K)]
    for path in args.control_dicts:
        cb = torch.load(path, map_location='cpu')
        C = cb['dictionary'].float()
        C = C / (C.norm(dim=-1, keepdim=True) + 1e-12)
        tag = os.path.basename(os.path.dirname(path))
        dirs.append(C)
        meta += [('control', f'{n} [{tag}]', None) for n in cb['names']]
    D_all = torch.cat(dirs, 0)
    Ndir = D_all.shape[0]
    print(f'{Ndir} directions ({K} dict + {Ndir - K} controls)', flush=True)

    print(f'Loading {args.n_seqs} FineWeb seqs (seq_len={args.seq_len})',
          flush=True)
    tok_list = load_fineweb_fixed_length(args.n_seqs, tokenizer,
                                         seq_len=args.seq_len, seed=args.seed)
    tok_ids = torch.stack(tok_list)
    N, T = tok_ids.shape

    # One model pass: cache centered+normalized activations (pos-0 skipped).
    print('Caching corpus activations', flush=True)
    acts = torch.empty(N * (T - 1), D_all.shape[1], dtype=torch.float16,
                       device=device)
    for i in range(0, N, args.batch_size):
        bt = tok_ids[i:i + args.batch_size].to(device)
        with torch.no_grad():
            h = model(bt, output_hidden_states=True).hidden_states[
                args.layer + 1][:, 1:, :].float()
        acts[i * (T - 1):(i + bt.shape[0]) * (T - 1)] = \
            h.reshape(-1, h.shape[-1]).to(torch.float16)
    abar = acts.float().mean(0)
    hc = acts.float() - abar
    del acts
    hc = (hc / hc.norm(dim=-1, keepdim=True).clamp_min(1e-9)).to(torch.float16)
    flat_tok = tok_ids[:, 1:].reshape(-1)

    print('Scoring + collecting peak tokens', flush=True)
    results = []
    for c0 in range(0, Ndir, args.atom_chunk):
        c1 = min(Ndir, c0 + args.atom_chunk)
        Dc = D_all[c0:c1].to(device=device, dtype=torch.float16)
        sc = (hc @ Dc.T).float()                            # [M, nc]
        top = sc.abs().topk(args.n_pos, dim=0).indices      # [n_pos, nc]
        top = top.cpu()
        for j in range(c1 - c0):
            tids = flat_tok[top[:, j]].tolist()
            norm = [normalize_token(tokenizer.decode([t])) for t in tids]
            cnt = Counter(norm)
            p = np.array(list(cnt.values()), dtype=np.float64)
            p = p / p.sum()
            ent = float(-(p * np.log(p)).sum())
            grp, lab, kv = meta[c0 + j]
            results.append({'group': grp, 'label': lab, 'kurt': kv,
                            'n_types': len(cnt), 'entropy': ent,
                            'top_types': cnt.most_common(8)})
        print(f'  {c1}/{Ndir}', flush=True)
    del hc

    with open(os.path.join(args.out, 'token_diversity.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'results': results}, f)

    # --- Report ------------------------------------------------------------
    dict_rs = [r for r in results if r['group'] == 'dict']
    ents = np.array([r['entropy'] for r in dict_rs])
    lines = [f'# Token-type diversity (top {args.n_pos} peaks, normalized)',
             '',
             f'- dict atoms: n={len(dict_rs)}  entropy quantiles: '
             f'p10={np.quantile(ents, .1):.2f}  p50={np.median(ents):.2f}  '
             f'p90={np.quantile(ents, .9):.2f}  max={ents.max():.2f}',
             f'- max possible entropy: log({args.n_pos})='
             f'{np.log(args.n_pos):.2f}', '',
             '## Controls']
    ctrl = sorted([r for r in results if r['group'] == 'control'],
                  key=lambda r: -r['entropy'])
    for r in ctrl:
        tt = ' '.join(f'{t}:{c}' for t, c in r['top_types'][:5])
        lines.append(f'- {r["label"]:30s} H={r["entropy"]:.2f} '
                     f'types={r["n_types"]:3d}  | {tt}')

    kthr = float(np.quantile(kurt, 0.99))
    cand = [r for r in dict_rs if r['kurt'] is not None and r['kurt'] >= kthr]
    cand.sort(key=lambda r: -r['entropy'])
    lines += ['', f'## Dict atoms with kurtosis >= p99 ({kthr:.2f}), '
                  f'sorted by entropy (n={len(cand)})']
    for r in cand[:args.topn_report]:
        tt = ' '.join(f'{t}:{c}' for t, c in r['top_types'][:5])
        lines.append(f'- {r["label"]:12s} H={r["entropy"]:.2f} '
                     f'types={r["n_types"]:3d} kurt={r["kurt"]:.2f} | {tt}')
    report = os.path.join(args.out, 'report.md')
    with open(report, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved {report}', flush=True)

    # --- Scatter: kurtosis vs entropy ---------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = [r['kurt'] for r in dict_rs]
    y = [r['entropy'] for r in dict_rs]
    ax.scatter(x, y, s=4, alpha=0.15, c='C0', label='dict atoms')
    ax.axvline(kthr, color='k', ls=':', lw=1, alpha=0.6,
               label=f'kurt p99={kthr:.2f}')
    ax.set_xlabel('excess kurtosis (centered cosine)')
    ax.set_ylabel(f'normalized token-type entropy (top {args.n_pos} peaks)')
    ax.set_title(f'Feature-ness vs token diversity ({args.model} '
                 f'L{args.layer})')
    ax.legend(); ax.grid(True, alpha=0.3)
    plot = os.path.join(args.out, 'kurt_vs_entropy.png')
    plt.savefig(plot, dpi=150, bbox_inches='tight')
    print(f'Saved {plot}', flush=True)


if __name__ == '__main__':
    main()
