#!/usr/bin/env python3
"""Cosine alignment of dictionary directions with an arbitrary reference set.

Like dict_sae_alignment.py, but the reference directions come from one or more
dictionary.pt files (key 'dictionary', optional 'names') instead of an SAE
decoder. For each dict atom, find its best-matching reference direction (max
cosine) and compare against a random-direction null over the same references.

Pure linear algebra. Usage:
    python scripts/superellipse/dict_ref_alignment.py \\
        --dict_path .../dictionary.pt \\
        --ref_paths .../scaffold.pt .../langcode.pt .../mwe.pt \\
        --ref_label contrastive --n_random 2000 --out .../contrastive_align
"""
import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--ref_paths', nargs='+', required=True,
                    help="dictionary.pt files providing reference directions")
    ap.add_argument('--ref_label', default='reference')
    ap.add_argument('--n_random', type=int, default=2000)
    ap.add_argument('--topm', type=int, default=5)
    ap.add_argument('--rand_seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # reference directions (concat, unit)
    parts, names = [], []
    for p in args.ref_paths:
        b = torch.load(p, map_location='cpu', weights_only=False)
        V = b['dictionary'].to(torch.float32)
        parts.append(V)
        nm = b.get('names', [f'{os.path.basename(os.path.dirname(p))}_{i}'
                              for i in range(V.shape[0])])
        names += list(nm)
    R = torch.cat(parts, 0)
    R = (R / (R.norm(dim=-1, keepdim=True) + 1e-12)).to(device)
    M, d_model = R.shape
    print(f'Reference ({args.ref_label}): {M} directions', flush=True)

    blob = torch.load(args.dict_path, map_location='cpu', weights_only=False)
    D = blob['dictionary'].to(device=device, dtype=torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    K = D.shape[0]
    print(f'Dictionary: {K} atoms', flush=True)

    cos = D @ R.T                                   # [K, M]
    found = cos.max(dim=1)
    found_max_cos = found.values.cpu().numpy()
    found_max_idx = found.indices.cpu().numpy()

    g = torch.Generator(device=device).manual_seed(args.rand_seed)
    Rnd = torch.randn(args.n_random, d_model, generator=g, device=device)
    Rnd = Rnd / Rnd.norm(dim=-1, keepdim=True)
    rand_max = (Rnd @ R.T).max(dim=1).values.cpu().numpy()

    r95 = np.quantile(rand_max, 0.95)
    print(f'\n=== found vs random null (max cosine over {M} {args.ref_label}) ===')
    print(f'  found  : mean={found_max_cos.mean():.3f}  '
          f'median={np.median(found_max_cos):.3f}  '
          f'min={found_max_cos.min():.3f}  max={found_max_cos.max():.3f}')
    print(f'  random : mean={rand_max.mean():.3f}  median={np.median(rand_max):.3f}'
          f'  95pct={r95:.3f}  max={rand_max.max():.3f}  (n={args.n_random})')
    n_above = int((found_max_cos > r95).sum())
    print(f'  found atoms above random 95th pct: {n_above}/{K}')

    # top close hits with their reference name
    order = np.argsort(-found_max_cos)[:20]
    print('\n=== top 20 close hits (atom -> reference : cos) ===')
    for a in order:
        print(f'  atom {int(a):5d} -> {names[int(found_max_idx[a])]:24s} '
              f'cos={found_max_cos[a]:.3f}', flush=True)

    with open(os.path.join(args.out, 'ref_alignment.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'found_max_cos': found_max_cos,
                     'found_max_idx': found_max_idx, 'rand_max': rand_max,
                     'names': names, 'M': M}, f)

    lo = min(found_max_cos.min(), rand_max.min())
    hi = max(found_max_cos.max(), rand_max.max())
    bins = np.linspace(lo, hi, 50)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rand_max, bins=bins, density=True, color='gray', alpha=0.5,
            label=f'random null (n={args.n_random})')
    ax.hist(found_max_cos, bins=bins, density=True, color='C2', alpha=0.6,
            label=f'dict atoms (n={K})')
    ax.axvline(r95, color='k', ls='--', lw=1, label=f'random 95th pct = {r95:.3f}')
    ax.set_xlabel(f'max cosine with any {args.ref_label} direction (best match)')
    ax.set_ylabel('density')
    ax.set_title(f'Dictionary alignment with {args.ref_label} dirs '
                 f'(M={M})\n{K} atoms vs {args.n_random} random  |  '
                 f'{n_above}/{K} above random 95th pct')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png = os.path.join(args.out, 'ref_alignment_hist.png')
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved {png}\nSaved {os.path.join(args.out, "ref_alignment.pkl")}',
          flush=True)


if __name__ == '__main__':
    main()
