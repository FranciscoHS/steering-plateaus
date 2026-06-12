#!/usr/bin/env python3
"""Superellipse-p across build 'time': split the dictionary into contiguous
build-order buckets and fit p from within-bucket random pairs in each.

Check: does the pairwise p of the atoms drift as the greedy build proceeds
(e.g. early atoms high-p, later atoms degrading toward p≈2, or vice versa)?

Loads the model + held-out anchors ONCE and reuses them for every bucket, so
the only thing varying across buckets is which build-order band the two
directions are drawn from. Per bucket we report per-pair p_fit (median + IQR)
and the pooled-points p with a bootstrap CI over pairs, exactly as
fit_pairs_dictionary.py does. Writes a checkpoint after each bucket.

Run on a GPU pod.

Usage:
    python scripts/superellipse/fit_p_vs_buildorder.py \\
        --dict_path /tmp/dict_25k_snapshot.pt --model qwen3-1.7b --layer 14 \\
        --n_buckets 6 --n_pairs 100 --n_anchors 20 \\
        --out .../p_vs_buildorder_n25k
"""
import argparse
import os
import pickle
import sys
import time
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
from scripts.superellipse.canonical.sweep_core import run_2d_per_anchors
from scripts.superellipse.canonical.superellipse import (
    robust_p_fit, fit_superellipse,
)
from scripts.superellipse.plot_boundary_pair import boundary_for_set
from scripts.superellipse.fit_pairs_dictionary import (
    summarize_fit, normalized_points,
)


def sample_pairs(lo, hi, n_pairs, rng):
    """n_pairs unordered i!=j absolute dict indices in [lo, hi)."""
    K = hi - lo
    seen = set()
    target = min(n_pairs, K * (K - 1) // 2)
    while len(seen) < target:
        i, j = int(rng.integers(K)), int(rng.integers(K))
        if i == j:
            continue
        if i > j:
            i, j = j, i
        seen.add((i, j))
    return [(lo + i, lo + j) for i, j in sorted(seen)]


def fit_bucket(model, contexts, activations, D, pairs, layer, measure_layer,
               device, angles_deg, angles_rad, n_bootstrap, n_pool_bootstrap,
               pair_seed):
    """Pooled + per-pair p for one bucket's pairs. Returns a stats dict."""
    p_fits, p_los, p_his, coss = [], [], [], []
    pool_xs, pool_ys = [], []
    for (i, j) in pairs:
        d1 = D[i].cpu(); d2 = D[j].cpu()
        r = run_2d_per_anchors(model, contexts, activations, d1, d2, layer,
                               measure_layer, device, angles_rad,
                               mode='geodesic', single_batch_ref=True)
        l2_grid = r['l2']
        median_grid = np.median(l2_grid, axis=0)
        fit = robust_p_fit(angles_deg, l2_grid, n_bootstrap=n_bootstrap,
                           exact_geodesic=True)
        p_lo, p_hi = summarize_fit(fit)
        b = boundary_for_set(angles_deg, median_grid, exact=True)
        p_fits.append(b['fit']['p']); p_los.append(p_lo); p_his.append(p_hi)
        coss.append(float((d1.to(device) * d2.to(device)).sum().item()))
        xs, ys = normalized_points(angles_deg, median_grid)
        pool_xs.append(xs); pool_ys.append(ys)
    p_fits = np.array(p_fits, float)
    fin = np.isfinite(p_fits)
    p_med = float(np.median(p_fits[fin])) if fin.any() else float('nan')
    p_iqr = ((float(np.quantile(p_fits[fin], 0.25)),
              float(np.quantile(p_fits[fin], 0.75))) if fin.any()
             else (float('nan'), float('nan')))

    valid = [k for k in range(len(pool_xs)) if pool_xs[k].size > 0]
    xs_all = (np.concatenate([pool_xs[k] for k in valid]) if valid
              else np.zeros(0))
    ys_all = (np.concatenate([pool_ys[k] for k in valid]) if valid
              else np.zeros(0))
    p_pool = float(fit_superellipse(xs_all, ys_all)['p']) if xs_all.size \
        else float('nan')
    bs = np.random.default_rng(pair_seed + 7)
    boot = []
    for _ in range(n_pool_bootstrap if valid else 0):
        idx = bs.integers(0, len(valid), len(valid))
        xb = np.concatenate([pool_xs[valid[k]] for k in idx])
        yb = np.concatenate([pool_ys[valid[k]] for k in idx])
        if xb.size:
            boot.append(fit_superellipse(xb, yb)['p'])
    boot = np.array(boot, float)
    ci = ((float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
          if boot.size else (float('nan'), float('nan')))
    return {'p_fits': p_fits, 'p_los': np.array(p_los), 'p_his': np.array(p_his),
            'cos': np.array(coss), 'p_median': p_med, 'p_iqr': p_iqr,
            'p_pool': p_pool, 'p_pool_ci_95': ci, 'n_pairs': len(pairs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_buckets', type=int, default=6)
    ap.add_argument('--n_pairs', type=int, default=100)
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--pair_seed', type=int, default=0)
    ap.add_argument('--n_steps_grid', type=int, default=31)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--n_bootstrap', type=int, default=100)
    ap.add_argument('--n_pool_bootstrap', type=int, default=2000)
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
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    print(f'Device: {device}  L{args.layer}->{measure_layer}', flush=True)

    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(device=device, dtype=torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    N = D.shape[0]
    print(f'Dictionary: {N} atoms  -> {args.n_buckets} build-order buckets',
          flush=True)

    print(f'Extracting {args.n_anchors} anchors (seed={args.held_seed})',
          flush=True)
    tok = load_fineweb_fixed_length(args.n_anchors, tokenizer,
                                    seq_len=args.seq_len, seed=args.held_seed)
    ctx, act = extract_context_and_activation(model, tok, args.layer, device)
    contexts = [ctx[i:i + 1].float().cpu() for i in range(ctx.shape[0])]
    activations = act.float().cpu()

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps_grid)
    angles_rad = np.deg2rad(angles_deg)

    edges = np.linspace(0, N, args.n_buckets + 1).astype(int)
    buckets = []
    t0 = time.time()
    for bi in range(args.n_buckets):
        lo, hi = int(edges[bi]), int(edges[bi + 1])
        rng = np.random.default_rng(args.pair_seed + bi)
        pairs = sample_pairs(lo, hi, args.n_pairs, rng)
        print(f'\n[bucket {bi}] idx [{lo},{hi})  {len(pairs)} pairs', flush=True)
        s = fit_bucket(model, contexts, activations, D, pairs, args.layer,
                       measure_layer, device, angles_deg, angles_rad,
                       args.n_bootstrap, args.n_pool_bootstrap,
                       args.pair_seed + bi)
        s.update({'bucket': bi, 'lo': lo, 'hi': hi,
                  'mid_idx': (lo + hi) / 2.0})
        buckets.append(s)
        print(f'  p_pool={s["p_pool"]:.3f} CI[{s["p_pool_ci_95"][0]:.3f},'
              f'{s["p_pool_ci_95"][1]:.3f}]  p_med={s["p_median"]:.3f} '
              f'IQR({s["p_iqr"][0]:.3f},{s["p_iqr"][1]:.3f})  '
              f'elapsed={time.time()-t0:.0f}s', flush=True)
        # checkpoint after each bucket
        with open(os.path.join(args.out, 'p_vs_buildorder.pkl'), 'wb') as f:
            pickle.dump({'args': vars(args), 'N': N, 'edges': edges,
                         'measure_layer': measure_layer, 'buckets': buckets}, f)

    # --- Plot p vs build-order bucket ---
    mids = [b['mid_idx'] for b in buckets]
    p_pool = [b['p_pool'] for b in buckets]
    ci_lo = [b['p_pool_ci_95'][0] for b in buckets]
    ci_hi = [b['p_pool_ci_95'][1] for b in buckets]
    p_med = [b['p_median'] for b in buckets]
    iqr_lo = [b['p_iqr'][0] for b in buckets]
    iqr_hi = [b['p_iqr'][1] for b in buckets]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(mids, p_pool, 'o-', color='C2', lw=2, label='pooled-points p')
    ax.fill_between(mids, ci_lo, ci_hi, color='C2', alpha=0.2,
                    label='pooled 95% CI')
    ax.plot(mids, p_med, 's--', color='C1', lw=1.5, label='per-pair median p')
    ax.fill_between(mids, iqr_lo, iqr_hi, color='C1', alpha=0.15, label='IQR')
    ax.axhline(2.0, color='gray', lw=1, ls=':', label='p=2 (ellipse)')
    for e in edges:
        ax.axvline(e, color='k', lw=0.4, alpha=0.15)
    ax.set_xlabel('build-order index (atom discovery order)')
    ax.set_ylabel('superellipse p')
    ax.set_title(f'p across build "time"  ({args.model} L{args.layer}->'
                 f'{measure_layer}, {args.n_buckets} buckets, '
                 f'{args.n_pairs} pairs/bucket, n_anchors={args.n_anchors})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'p_vs_buildorder.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print('\n=== p across build order ===')
    for b in buckets:
        print(f"  bucket {b['bucket']} [{b['lo']},{b['hi']}): "
              f"p_pool={b['p_pool']:.3f} "
              f"CI[{b['p_pool_ci_95'][0]:.3f},{b['p_pool_ci_95'][1]:.3f}]  "
              f"p_med={b['p_median']:.3f}")
    print(f'\nSaved {plot_path}')


if __name__ == '__main__':
    main()
