#!/usr/bin/env python3
"""Cosine alignment of found dictionary directions with SAE decoder latents.

For each found atom d_k, find its best-matching SAE latent (max cosine with the
normalized decoder directions = columns of W_dec) and report the top matches.
Compared against a random-direction null: a random unit direction's max cosine
over the same 32k latents (the relevant baseline, since with 32k latents even a
random direction has an inflated best-match by chance).

No model forward needed — pure linear algebra on W_dec + the dictionary.

Usage:
    python scripts/superellipse/dict_sae_alignment.py \\
        --dict_path .../dictionary.pt --sae_ckpt .../layer14.sae.pt \\
        --n_random 500 --topm 5 --out .../sae_alignment
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
    ap.add_argument('--sae_ckpt', required=True)
    ap.add_argument('--n_random', type=int, default=500,
                    help='random unit dirs for the null max-cosine distribution')
    ap.add_argument('--topm', type=int, default=5,
                    help='# of top SAE latents to list per found atom')
    ap.add_argument('--rand_seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sae = torch.load(args.sae_ckpt, map_location='cpu')
    W_dec = sae['W_dec']                              # [d_model, d_sae]
    if W_dec.shape[0] > W_dec.shape[1]:              # ensure [d_model, d_sae]
        W_dec = W_dec.T
    d_model, d_sae = W_dec.shape
    L = W_dec.T.to(device=device, dtype=torch.float32)   # [d_sae, d_model]
    L = L / (L.norm(dim=-1, keepdim=True) + 1e-12)
    print(f'SAE decoder latents: {d_sae}  d_model={d_model}', flush=True)

    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(device=device, dtype=torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    K = D.shape[0]
    print(f'Dictionary: {K} atoms', flush=True)

    # Found atoms vs all latents: cosine [K, d_sae].
    cos_found = D @ L.T                               # [K, d_sae]
    found_max = cos_found.max(dim=1)
    found_max_cos = found_max.values.cpu().numpy()
    found_max_idx = found_max.indices.cpu().numpy()

    # Random null: max cosine of random unit dirs over the same latents.
    g = torch.Generator(device=device).manual_seed(args.rand_seed)
    Rnd = torch.randn(args.n_random, d_model, generator=g, device=device,
                      dtype=torch.float32)
    Rnd = Rnd / Rnd.norm(dim=-1, keepdim=True)
    rand_max = (Rnd @ L.T).max(dim=1).values.cpu().numpy()

    print('\n=== per-atom best SAE-latent match (top {} each) ==='.format(
        args.topm))
    per_atom = []
    for k in range(K):
        vals, idx = torch.topk(cos_found[k], args.topm)
        vals = vals.cpu().numpy(); idx = idx.cpu().numpy()
        per_atom.append({'atom': k, 'top_latent': idx.tolist(),
                         'top_cos': vals.tolist()})
        pairs = '  '.join(f'#{int(i)}:{v:.3f}' for v, i in zip(vals, idx))
        print(f'  atom {k}:  max_cos={vals[0]:.3f}   {pairs}', flush=True)

    print('\n=== found vs random null (max cosine over 32k latents) ===')
    print(f'  found  : mean={found_max_cos.mean():.3f}  '
          f'median={np.median(found_max_cos):.3f}  '
          f'min={found_max_cos.min():.3f}  max={found_max_cos.max():.3f}')
    print(f'  random : mean={rand_max.mean():.3f}  '
          f'median={np.median(rand_max):.3f}  '
          f'95pct={np.quantile(rand_max,0.95):.3f}  '
          f'max={rand_max.max():.3f}  (n={args.n_random})')
    n_above = int((found_max_cos > np.quantile(rand_max, 0.95)).sum())
    print(f'  found atoms above random 95th pct: {n_above}/{K}')

    with open(os.path.join(args.out, 'sae_alignment.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'found_max_cos': found_max_cos,
                     'found_max_idx': found_max_idx, 'rand_max': rand_max,
                     'per_atom': per_atom, 'd_sae': d_sae}, f)
    print(f"\nSaved {os.path.join(args.out, 'sae_alignment.pkl')}", flush=True)

    # Histogram: found vs random null max-cosine (density).
    lo = min(found_max_cos.min(), rand_max.min())
    hi = max(found_max_cos.max(), rand_max.max())
    bins = np.linspace(lo, hi, 60)
    r95 = np.quantile(rand_max, 0.95)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rand_max, bins=bins, density=True, color='gray', alpha=0.5,
            label=f'random null (n={args.n_random})')
    ax.hist(found_max_cos, bins=bins, density=True, color='C0', alpha=0.6,
            label=f'found atoms (n={K})')
    ax.axvline(r95, color='k', ls='--', lw=1,
               label=f'random 95th pct = {r95:.3f}')
    ax.set_xlabel('max cosine with any SAE latent (best match)')
    ax.set_ylabel('density')
    ax.set_title(f'Found-direction alignment with SAE latents (W32K, L0=50)\n'
                 f'{K} atoms vs {args.n_random} random  |  '
                 f'{int((found_max_cos>r95).sum())}/{K} above random 95th pct')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'sae_alignment_hist.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {plot_path}', flush=True)


if __name__ == '__main__':
    main()
