#!/usr/bin/env python3
"""
Single estimate of the superellipse exponent p ACROSS a set of direction pairs.

Pools the normalized iso-L2 boundary points from every pair in a fit pkl
(e.g. fit_onevsrest.pkl: all 45 one-vs-rest class pairs) into one point cloud
and fits a single p in |x|^p+|y|^p=1. A confidence interval is obtained by
bootstrapping over the pairs (resample pairs with replacement, re-pool, refit).

Per-pair normalization matches plot_boundary_pair / boundary_for_set: threshold
= 0.5 * min(max axis response), exact-geodesic normalization, keep points in
(0.05, 1.5). Works on any *_pairs pkl with 'per_pair' (each having 'l2_grid')
and 'angles_deg'.

Usage:
    python pool_p_contrastive_mlp.py \
        --fit_pkl results/grok_mlp/against_contrastive/ckpt_s1000000_final_a30_nc10/fit_onevsrest_classes/fit_onevsrest.pkl
"""
import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.superellipse.canonical.superellipse import (
    extract_contour, axis_intercept, fit_superellipse, superellipse_curve,
)


def normalized_points(angles, median_grid):
    """Exact-geodesic normalized boundary points (xn, yn) at thresh=0.5*base,
    masked to (0.05, 1.5) — identical to boundary_for_set."""
    base = min(median_grid[:, 0].max(), median_grid[0, :].max())
    thresh = 0.5 * base
    t1 = axis_intercept(angles, median_grid[:, 0], thresh)
    t2 = axis_intercept(angles, median_grid[0, :], thresh)
    raw = extract_contour(angles, median_grid, thresh).astype(float)
    if raw.size == 0 or not np.isfinite(t1) or not np.isfinite(t2) or t1 <= 0 or t2 <= 0:
        return np.zeros(0), np.zeros(0)
    r_deg = np.hypot(raw[:, 0], raw[:, 1])
    r_safe = np.where(r_deg > 1e-12, r_deg, 1.0)
    r_rad = np.deg2rad(r_deg)
    xn = (np.sin(r_rad) / np.sin(np.deg2rad(t1))) * (raw[:, 0] / r_safe)
    yn = (np.sin(r_rad) / np.sin(np.deg2rad(t2))) * (raw[:, 1] / r_safe)
    m = (xn > 0.05) & (yn > 0.05) & (xn < 1.5) & (yn < 1.5)
    return xn[m], yn[m]


def fit_p_pool(xs, ys):
    return fit_superellipse(np.asarray(xs), np.asarray(ys))['p']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit_pkl', type=str, required=True)
    ap.add_argument('--n_bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    with open(args.fit_pkl, 'rb') as f:
        d = pickle.load(f)
    angles = d['angles_deg']
    per_pair = d['per_pair']

    # Per-pair normalized points + per-pair point p (from pooled-style fit).
    pair_points, per_pair_p, pair_labels = [], [], []
    for pp in per_pair:
        median_grid = np.median(pp['l2_grid'], axis=0)
        xn, yn = normalized_points(angles, median_grid)
        pair_points.append((xn, yn))
        per_pair_p.append(fit_p_pool(xn, yn) if len(xn) >= 3 else np.nan)
        pair_labels.append(tuple(pp['pair']))
    per_pair_p = np.array(per_pair_p)

    # Pooled point estimate over the whole set.
    all_x = np.concatenate([xy[0] for xy in pair_points])
    all_y = np.concatenate([xy[1] for xy in pair_points])
    p_pooled = fit_p_pool(all_x, all_y)

    # Bootstrap over pairs.
    rng = np.random.RandomState(args.seed)
    n = len(pair_points)
    boot = []
    for _ in range(args.n_bootstrap):
        idx = rng.randint(0, n, size=n)
        bx = np.concatenate([pair_points[k][0] for k in idx])
        by = np.concatenate([pair_points[k][1] for k in idx])
        if len(bx) >= 3:
            boot.append(fit_p_pool(bx, by))
    boot = np.array(boot)
    p_med = float(np.median(boot)); p_lo = float(np.percentile(boot, 5))
    p_hi = float(np.percentile(boot, 95))

    finite = per_pair_p[np.isfinite(per_pair_p)]
    print(f'Pairs pooled: {n}  (total boundary points: {len(all_x)})', flush=True)
    print(f'POOLED p across the set: point={p_pooled:.2f}, '
          f'bootstrap median={p_med:.2f}  90% CI [{p_lo:.2f}, {p_hi:.2f}]', flush=True)
    print(f'Per-pair p: median={np.median(finite):.2f}  '
          f'IQR [{np.percentile(finite,25):.2f}, {np.percentile(finite,75):.2f}]  '
          f'min={finite.min():.2f} max={finite.max():.2f}  (n={len(finite)})', flush=True)

    out = args.out or os.path.join(os.path.dirname(args.fit_pkl), 'pooled_p.png')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    # pooled scatter + fitted superellipse
    ax = axes[0]
    ax.scatter(all_x, all_y, s=6, alpha=0.25, color='#3a3a3a', label='all pairs (normalized)')
    xc, yc = superellipse_curve(2.0); ax.plot(xc, yc, '-', color='#1F77B4', lw=2, label='p=2')
    xf, yf = superellipse_curve(p_pooled)
    ax.plot(xf, yf, '-', color='#3a7d3a', lw=2.5, label=f'pooled fit p={p_pooled:.2f}')
    ax.set_xlim(0, 1.5); ax.set_ylim(0, 1.5); ax.set_aspect('equal')
    ax.set_xlabel('normalized x'); ax.set_ylabel('normalized y')
    ax.set_title('Pooled boundary points + fit'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    # histogram of per-pair p
    ax = axes[1]
    ax.hist(finite, bins=15, color='C0', alpha=0.8)
    ax.axvline(p_pooled, color='#3a7d3a', ls='-', lw=2, label=f'pooled p={p_pooled:.2f}')
    ax.axvspan(p_lo, p_hi, color='#3a7d3a', alpha=0.15, label=f'90% CI [{p_lo:.2f},{p_hi:.2f}]')
    ax.axvline(2.0, color='gray', ls=':', label='p=2')
    ax.set_xlabel('per-pair fitted p'); ax.set_ylabel('count')
    ax.set_title(f'Per-pair p (median={np.median(finite):.2f})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f'p across the set ({n} pairs): pooled {p_pooled:.2f}, '
                 f'CI [{p_lo:.2f}, {p_hi:.2f}]', fontsize=13)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()

    with open(out.replace('.png', '.pkl'), 'wb') as f:
        pickle.dump({'p_pooled': p_pooled, 'p_boot_median': p_med,
                     'p_ci': (p_lo, p_hi), 'per_pair_p': per_pair_p,
                     'pair_labels': pair_labels, 'boot': boot}, f)
    print(f'Saved: {out}\nSaved: {out.replace(".png", ".pkl")}', flush=True)


if __name__ == '__main__':
    main()
