#!/usr/bin/env python3
"""Diagnostics that the dictionary build is doing what we want, vs build order.

Two figures from one snapshot of the dictionary (atoms are stored in the order
they were optimized):

  (1) pooled superellipse-p vs build-order bucket.
      Split the N atoms into n_buckets contiguous build-order buckets; per
      bucket sample n_pairs WITHIN-bucket pairs and fit a pooled-p (bootstrap
      CI over pairs), reusing scripts/superellipse/p_vs_recon.estimate_pooled_p.
      Question: does p (which the `ratio` proxy pushes high) degrade in later
      buckets as high-p directions get used up?

  (2) mean / max |cos| vs corpus size (cumulative in build order).
      As atoms accumulate, track the pairwise alignment among the first n.
      max|cos| must rise with n; mean|cos| ideally stays low. Random-unit
      directions (same N, d) overlaid as the isotropic baseline.

Usage:
    python -u scripts/superellipse/p_and_overlap_vs_order.py \
        --dict_path .../snap_dict_first20k.pt \
        --model qwen3-1.7b --layer 14 \
        --n_buckets 6 --n_pairs 200 --n_anchors 20 \
        --out .../p_overlap_vs_order
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

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)
from scripts.superellipse.p_vs_recon import (
    estimate_pooled_p, sample_uniform_pairs,
)


def overlap_running_stats(D, chunk=512):
    """For unit rows of D in build order, return per-index (m, s):
      m[i] = max_{j<i} |cos(d_i, d_j)|
      s[i] = sum_{j<i} |cos(d_i, d_j)|
    (m[0]=s[0]=0). Vectorized per row-chunk with a strict lower-triangular mask.
    """
    N = D.shape[0]
    m = torch.zeros(N, device=D.device)
    s = torch.zeros(N, device=D.device)
    for b0 in range(0, N, chunk):
        b1 = min(N, b0 + chunk)
        block = (D[b0:b1] @ D[:b1].T).abs()           # [bc, b1]
        rows = torch.arange(b0, b1, device=D.device).view(-1, 1)
        cols = torch.arange(b1, device=D.device).view(1, -1)
        mask = cols < rows                            # strict: only earlier
        m[b0:b1] = block.masked_fill(~mask, float('-inf')).max(dim=1).values
        s[b0:b1] = (block * mask).sum(dim=1)
    m[0] = 0.0
    return m.cpu(), s.cpu()


def cumulative_overlap_curves(D, grid):
    """Return (mean|cos|(n), max|cos|(n)) evaluated at corpus sizes in `grid`."""
    m, s = overlap_running_stats(D)
    cmax = torch.cummax(m, dim=0).values              # max over first n
    csum = torch.cumsum(s, dim=0)                     # total |cos| sum over <n pairs
    mean_at, max_at = [], []
    for n in grid:
        n_pairs = n * (n - 1) / 2
        mean_at.append(float(csum[n - 1] / n_pairs) if n_pairs > 0 else float('nan'))
        max_at.append(float(cmax[n - 1]))
    return np.array(mean_at), np.array(max_at)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_atoms', type=int, default=20000,
                    help='use the first n_atoms of the snapshot (build order)')
    ap.add_argument('--n_buckets', type=int, default=6)
    ap.add_argument('--n_pairs', type=int, default=200)
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--pair_seed', type=int, default=0)
    ap.add_argument('--rand_seed', type=int, default=777)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--n_steps_grid', type=int, default=31)
    ap.add_argument('--n_pool_bootstrap', type=int, default=2000)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.smoke:
        args.n_buckets = 3
        args.n_pairs = 4
        args.n_pool_bootstrap = 100
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----- load dict (build order preserved) -----
    Draw = torch.load(args.dict_path, map_location='cpu')['dictionary']
    N = min(args.n_atoms, Draw.shape[0])
    D = Draw[:N].to(device=device, dtype=torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    d_model = D.shape[1]
    print(f'dict: {N} atoms (of {Draw.shape[0]}), d={d_model}', flush=True)

    # =================================================================== #
    # Figure 2: alignment vs corpus size (no model needed)                #
    # =================================================================== #
    grid = sorted(set(list(range(max(2, N // 40), N + 1, max(1, N // 40))) + [N]))
    grid = [g for g in grid if 2 <= g <= N]
    print(f'overlap grid: {len(grid)} points up to {grid[-1]}', flush=True)
    mean_dict, max_dict = cumulative_overlap_curves(D, grid)
    g = torch.Generator(device=device).manual_seed(args.rand_seed)
    Drand = torch.randn(N, d_model, generator=g, device=device,
                        dtype=torch.float32)
    Drand = Drand / Drand.norm(dim=-1, keepdim=True)
    mean_rand, max_rand = cumulative_overlap_curves(Drand, grid)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(grid, max_dict, '-', color='C0', label='dict  max |cos|')
    ax.plot(grid, mean_dict, '-', color='C0', alpha=0.6, lw=2,
            label='dict  mean |cos|')
    ax.plot(grid, max_rand, '--', color='C1', label='random  max |cos|')
    ax.plot(grid, mean_rand, '--', color='C1', alpha=0.6, lw=2,
            label='random  mean |cos|')
    ax.set_xlabel('corpus size n (build order)')
    ax.set_ylabel('|cos| among first n directions')
    ax.set_title(f'{args.model} L{args.layer}: dictionary alignment vs build order')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    png2 = os.path.join(args.out, 'overlap_vs_order.png')
    plt.savefig(png2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {png2}', flush=True)

    # =================================================================== #
    # Figure 1: pooled-p vs build-order bucket (needs model)              #
    # =================================================================== #
    print(f'\nLoading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, _ = load_model(args.model, dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    print(f'L{args.layer}->{measure_layer}', flush=True)

    tok = load_fineweb_fixed_length(args.n_anchors, tokenizer,
                                    seq_len=args.seq_len, seed=args.held_seed)
    ctx, act = extract_context_and_activation(model, tok, args.layer, device)
    contexts = [ctx[i:i + 1].float().cpu() for i in range(ctx.shape[0])]
    activations = act.float().cpu()

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps_grid)
    pair_rng = np.random.default_rng(args.pair_seed)
    boot_rng = np.random.default_rng(args.pair_seed + 1)

    edges = np.linspace(0, N, args.n_buckets + 1).astype(int)
    buckets = []
    for bi in range(args.n_buckets):
        b0, b1 = int(edges[bi]), int(edges[bi + 1])
        pairs = sample_uniform_pairs(list(range(b0, b1)), args.n_pairs, pair_rng)
        pair_dirs = [(D[i].cpu(), D[j].cpu()) for (i, j) in pairs]
        pest = estimate_pooled_p(
            pair_dirs, model, contexts, activations, args.layer, measure_layer,
            device, angles_deg, args.n_pool_bootstrap, boot_rng,
            label=f'bucket{bi}[{b0}:{b1}]')
        center = (b0 + b1) / 2.0
        print(f'  bucket {bi} [{b0}:{b1}) center={center:.0f}: '
              f'p_pool={pest["p_pool"]:.3f} CI={pest["p_pool_ci95"]} '
              f'p_med={pest["p_median"]:.3f}', flush=True)
        buckets.append({'bucket': bi, 'lo': b0, 'hi': b1, 'center': center,
                        **pest})

    # ----- save everything -----
    with open(os.path.join(args.out, 'p_overlap_vs_order.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'measure_layer': measure_layer,
                     'angles_deg': angles_deg, 'buckets': buckets,
                     'overlap_grid': grid,
                     'mean_dict': mean_dict, 'max_dict': max_dict,
                     'mean_rand': mean_rand, 'max_rand': max_rand}, f)

    centers = [b['center'] for b in buckets]
    pmed = [b['p_median'] for b in buckets]
    yerr = np.array([[b['p_median'] - b['p_iqr'][0] for b in buckets],
                     [b['p_iqr'][1] - b['p_median'] for b in buckets]])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    # faint scatter of the individual per-pair fits behind the median+IQR.
    for b in buckets:
        pf = np.asarray(b['p_fits'], dtype=float)
        pf = pf[np.isfinite(pf)]
        width = (b['hi'] - b['lo']) * 0.3
        jit = (np.random.default_rng(b['bucket']).random(pf.size) - 0.5) * width
        ax.scatter(b['center'] + jit, pf, s=6, color='C0', alpha=0.15, zorder=1)
    ax.errorbar(centers, pmed, yerr=yerr, fmt='o-', color='C0', capsize=4,
                lw=2, zorder=3, label='per-pair median p (IQR)')
    ax.axhline(2.0, color='gray', ls=':', lw=1, label='p=2 (ellipse)')
    # Keep y readable: cap at 1.5x the largest IQR top (per-pair fits can rail).
    ax.set_ylim(top=1.5 * max(b['p_iqr'][1] for b in buckets))
    ax.set_xlabel('build order (bucket center, atom index)')
    ax.set_ylabel('per-pair superellipse-p (median, IQR; dots = pairs)')
    ax.set_title(f'{args.model} L{args.layer}->{measure_layer}: '
                 f'p vs build order  ({args.n_buckets} buckets, '
                 f'{args.n_pairs} pairs each)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    png1 = os.path.join(args.out, 'p_vs_order.png')
    plt.savefig(png1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {png1}')
    print(f'Saved {os.path.join(args.out, "p_overlap_vs_order.pkl")}')


if __name__ == '__main__':
    main()
