#!/usr/bin/env python3
"""Projection-distribution diagnostics for found dictionary directions.

For each found atom d_k, project a corpus of layer-L activations (all token
positions, pos-0 skipped) onto d_k and characterize the DISTRIBUTION of
projections, not just the max. A feature-like direction should be sparse /
heavy-tailed: ~0 for most tokens, large on a small subset. We compare the
found atoms to random unit directions on:

  - excess kurtosis (Gaussian = 0; heavy-tailed / sparse > 0),
  - skew,
  - top-1% L1-mass fraction (concentration: fraction of total |centered proj|
    carried by the top 1% of tokens).

Massive-activation tokens (a few huge-norm positions) are extreme outliers in
raw projection for ANY direction and inflate these stats uniformly, so every
metric is reported twice: on all tokens, and after excluding the top
--outlier_pct by activation norm.

Usage:
    python scripts/superellipse/dict_projection_dist.py \\
        --dict_path .../dictionary.pt --model qwen3-1.7b --layer 14 \\
        --n_seqs 2000 --seq_len 64 --n_random 20 --out .../proj_dist
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

from src.model import load_model
from src.data import load_fineweb_fixed_length


def dist_stats(proj, keep_mask):
    """Excess kurtosis, skew, top-1% L1-mass fraction on proj[keep_mask]."""
    x = proj[keep_mask].astype(np.float64)
    mu, sd = x.mean(), x.std()
    z = (x - mu) / (sd + 1e-12)
    kurt = float((z ** 4).mean() - 3.0)
    skew = float((z ** 3).mean())
    a = np.abs(x - mu)
    n_top = max(1, int(0.01 * a.size))
    top_mass = float(np.sort(a)[-n_top:].sum() / (a.sum() + 1e-12))
    return kurt, skew, top_mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--n_seqs', type=int, default=2000)
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--n_random', type=int, default=20)
    ap.add_argument('--outlier_pct', type=float, default=0.5,
                    help='exclude top this %% of tokens by activation norm')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--rand_seed', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=64)
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
    D = blob['dictionary'].to(torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    K, d_model = D.shape
    D_dev = D.to(device)
    print(f'Dictionary: {K} atoms, d_model={d_model}', flush=True)

    g = torch.Generator(device=device).manual_seed(args.rand_seed)
    R = torch.randn(args.n_random, d_model, generator=g, device=device,
                    dtype=torch.float32)
    R = R / R.norm(dim=-1, keepdim=True)

    print(f'Loading {args.n_seqs} FineWeb seqs (seq_len={args.seq_len})',
          flush=True)
    tok_list = load_fineweb_fixed_length(args.n_seqs, tokenizer,
                                         seq_len=args.seq_len, seed=args.seed)
    tok_ids = torch.stack(tok_list)
    N, T = tok_ids.shape

    print('Capturing activations', flush=True)
    acts_chunks = []
    for i in range(0, N, args.batch_size):
        bt = tok_ids[i:i + args.batch_size].to(device)
        with torch.no_grad():
            h = model(bt, output_hidden_states=True).hidden_states[args.layer + 1]
        acts_chunks.append(h[:, 1:, :].float().reshape(-1, d_model).cpu())
    acts = torch.cat(acts_chunks, 0).to(device)       # [M, d]
    M = acts.shape[0]
    norms = acts.norm(dim=-1)
    print(f'  {M} token activations  mean||a||={norms.mean():.1f}  '
          f'max||a||={norms.max():.1f}', flush=True)

    # Outlier mask: drop the top outlier_pct by activation norm.
    thr = torch.quantile(norms, 1.0 - args.outlier_pct / 100.0)
    keep = (norms <= thr).cpu().numpy()
    n_drop = int((~keep).sum())
    print(f'Outlier exclusion: drop {n_drop} tokens with ||a|| > {thr:.1f} '
          f'(top {args.outlier_pct}%)', flush=True)

    # Projections (raw dot) and cosine (norm-normalized, all tokens kept).
    found_proj = (acts @ D_dev.T).cpu().numpy()       # [M, K]
    rand_proj = (acts @ R.T).cpu().numpy()            # [M, n_random]
    inv_norm = (1.0 / norms).cpu().numpy()[:, None]
    found_cos = found_proj * inv_norm                 # [M, K]
    rand_cos = rand_proj * inv_norm                   # [M, n_random]
    all_mask = np.ones(M, dtype=bool)

    # --- Cosine distribution stats (all tokens; no norm exclusion) ---
    print('\n=== cosine-projection distribution (all tokens) ===')
    print(f'{"dir":>7}  {"kurt":>8}  {"skew":>7}  {"top1%mass":>9}')
    f_ck, f_cs, f_ctm = [], [], []
    for k in range(K):
        kk, sk, tm = dist_stats(found_cos[:, k], all_mask)
        f_ck.append(kk); f_cs.append(sk); f_ctm.append(tm)
        print(f'  atom{k}  {kk:8.2f}  {sk:7.2f}  {tm:9.3f}', flush=True)
    r_ck, r_ctm = [], []
    for r in range(args.n_random):
        kk, sk, tm = dist_stats(rand_cos[:, r], all_mask)
        r_ck.append(kk); r_ctm.append(tm)
    f_ck = np.array(f_ck); r_ck = np.array(r_ck)
    f_ctm = np.array(f_ctm); r_ctm = np.array(r_ctm)
    print(f'\ncosine kurt    found: median={np.median(f_ck):.2f} '
          f'[{f_ck.min():.2f},{f_ck.max():.2f}]   '
          f'random: median={np.median(r_ck):.2f} '
          f'[{r_ck.min():.2f},{r_ck.max():.2f}]')
    print(f'cosine top1%   found: median={np.median(f_ctm):.3f}   '
          f'random: median={np.median(r_ctm):.3f}')

    # Cosine histogram grid.
    rand_cos_pool = rand_cos.reshape(-1)
    ncol = 5
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False)
    lo = min(found_cos.min(), np.quantile(rand_cos_pool, 0.0005))
    hi = max(found_cos.max(), np.quantile(rand_cos_pool, 0.9995))
    bins = np.linspace(lo, hi, 140)
    for k in range(K):
        ax = axes[k // ncol][k % ncol]
        ax.hist(rand_cos_pool, bins=bins, density=True, color='gray',
                alpha=0.4, label='random (pooled)')
        ax.hist(found_cos[:, k], bins=bins, density=True, color='C1',
                alpha=0.6, label=f'atom {k}')
        ax.set_yscale('log')
        ax.set_title(f'atom {k}  kurt={f_ck[k]:.1f}', fontsize=9)
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(K, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle(f'Cosine-projection distributions (all tokens, log-density)  '
                 f'{args.model} L{args.layer}')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'cos_hist.png'), dpi=130,
                bbox_inches='tight')
    plt.close()
    print(f'Saved {os.path.join(args.out, "cos_hist.png")}')

    rows = []   # (label, kind, kurt_all, kurt_keep, skew_keep, topmass_keep)
    for k in range(K):
        ka, _, _ = dist_stats(found_proj[:, k], all_mask)
        kk, sk, tm = dist_stats(found_proj[:, k], keep)
        rows.append((f'atom{k}', 'found', ka, kk, sk, tm))
    rand_keep_kurt = []
    for r in range(args.n_random):
        ka, _, _ = dist_stats(rand_proj[:, r], all_mask)
        kk, sk, tm = dist_stats(rand_proj[:, r], keep)
        rows.append((f'rand{r}', 'random', ka, kk, sk, tm))
        rand_keep_kurt.append(kk)

    # Console table.
    print('\n=== projection distribution (outliers excluded) ===')
    print(f'{"dir":>7}  {"kurt(all)":>10}  {"kurt(keep)":>10}  '
          f'{"skew":>7}  {"top1%mass":>9}')
    for lbl, kind, ka, kk, sk, tm in rows:
        if kind == 'found':
            print(f'{lbl:>7}  {ka:10.1f}  {kk:10.2f}  {sk:7.2f}  {tm:9.3f}',
                  flush=True)
    f_kk = np.array([r[3] for r in rows if r[1] == 'found'])
    r_kk = np.array(rand_keep_kurt)
    f_tm = np.array([r[5] for r in rows if r[1] == 'found'])
    r_tm = np.array([r[5] for r in rows if r[1] == 'random'])
    print(f'\nkurt(keep)   found: median={np.median(f_kk):.2f} '
          f'[{f_kk.min():.2f},{f_kk.max():.2f}]   '
          f'random: median={np.median(r_kk):.2f} '
          f'[{r_kk.min():.2f},{r_kk.max():.2f}]')
    print(f'top1%mass    found: median={np.median(f_tm):.3f}   '
          f'random: median={np.median(r_tm):.3f}')

    with open(os.path.join(args.out, 'proj_dist.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'rows': rows,
                     'found_proj_keep_sample': found_proj[keep][:50000],
                     'keep_thr': float(thr), 'n_drop': n_drop}, f)

    # Histogram grid: each found atom, outliers excluded, vs pooled random.
    rand_keep = rand_proj[keep].reshape(-1)
    ncol = 5
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False)
    for k in range(K):
        ax = axes[k // ncol][k % ncol]
        fp = found_proj[keep, k]
        lo = min(fp.min(), np.quantile(rand_keep, 0.001))
        hi = max(fp.max(), np.quantile(rand_keep, 0.999))
        bins = np.linspace(lo, hi, 120)
        ax.hist(rand_keep, bins=bins, density=True, color='gray', alpha=0.4,
                label='random (pooled)')
        ax.hist(fp, bins=bins, density=True, color='C0', alpha=0.6,
                label=f'atom {k}')
        ax.set_yscale('log')
        ax.set_title(f'atom {k}  kurt={rows[k][3]:.1f}', fontsize=9)
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(K, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle(f'Projection distributions (outliers >{args.outlier_pct}%-norm '
                 f'excluded, log-density)  {args.model} L{args.layer}')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'proj_hist.png'), dpi=130,
                bbox_inches='tight')
    plt.close()

    print(f'\nSaved {os.path.join(args.out, "proj_dist.pkl")}')
    print(f'Saved {os.path.join(args.out, "proj_hist.png")}')


if __name__ == '__main__':
    main()
