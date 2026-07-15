#!/usr/bin/env python3
"""Shape-collapse test on the L2 (perturb_layer=2) exp_map 1d sweeps.

Same analysis as plot_curve_shape_collapse.py, but the data lives across many
per-concept files (exp_map/data/sweeps_1d/sweep1d_<model>_L2_*_holdout_75deg.pkl),
each holding:
    angles_deg : [n_angles]
    l2         : [n_anchors, n_angles]              (one contrastive direction)
    l2_random  : [n_random, n_anchors, n_angles]    (random baselines)

We aggregate across anchors (mean) -> one contrastive curve per concept file, and
per-random-direction curves pooled across files. Then run the 4-panel collapse.

Usage:
    python scripts/perturbation/plot_curve_shape_collapse_L2.py --model gemma
"""
import argparse
import glob
import os

import sys
sys.path.insert(0, '.')

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_curve_shape_collapse import (
    crossing, affine_normalize, group_mean_on_grid, C_CONTRAST, C_RANDOM,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gemma')
    ap.add_argument('--glob', default=None,
                    help='override file glob; default = gemma L2 holdout concept sweeps')
    ap.add_argument('--pool_random', action='store_true',
                    help='one random curve per file (mean over the 10 dirs) instead of all 10')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    patt = args.glob or f'results/exp_map/data/sweeps_1d/sweep1d_{args.model}_L2_*_prompts_holdout_75deg.pkl'
    files = sorted(glob.glob(patt))
    if not files:
        raise SystemExit(f'no files match {patt}')

    theta = None
    contrast = {}        # concept -> r(theta)
    random_curves = []   # list of r(theta)
    pl = ml = None
    for f in files:
        d = pickle.load(open(f, 'rb'))
        t = np.array(d['angles_deg'], dtype=float)
        theta = t if theta is None else theta
        pl, ml = d['perturb_layer'], d['measure_layer']
        contrast[d['direction_name']] = np.array(d['l2'], dtype=float).mean(axis=0)   # mean over anchors
        rnd = np.array(d['l2_random'], dtype=float).mean(axis=1)                      # [n_random, n_angles]
        if args.pool_random:
            random_curves.append(rnd.mean(axis=0))
        else:
            random_curves.extend(list(rnd))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axA, axB, axC, axD = axes.ravel()

    # ---- (A) raw curves ----
    for r in random_curves:
        axA.plot(theta, r, color=C_RANDOM, alpha=0.2, lw=0.8)
    axA.plot([], [], color=C_RANDOM, alpha=0.6, lw=1.5, label=f'random (n={len(random_curves)})')
    for name, r in contrast.items():
        axA.plot(theta, r, color=C_CONTRAST, alpha=0.7, lw=1.5)
    axA.plot([], [], color=C_CONTRAST, alpha=0.8, lw=2, label=f'contrastive (n={len(contrast)})')
    axA.set_xlabel('perturbation angle (deg)')
    axA.set_ylabel('L2 response')
    axA.set_title('(A) raw response curves')
    axA.legend(fontsize=8)

    # ---- (B) 2-point affine collapse ----
    xgrid = np.linspace(0, 1, 50)
    norm_contrast, norm_random = [], []
    for name, r in contrast.items():
        res = affine_normalize(theta, r)
        if res:
            x, y, *_ = res
            axB.plot(x, y, color=C_CONTRAST, alpha=0.4, lw=0.8)
            norm_contrast.append((x, y))
    for r in random_curves:
        res = affine_normalize(theta, r)
        if res:
            x, y, *_ = res
            axB.plot(x, y, color=C_RANDOM, alpha=0.12, lw=0.8)
            norm_random.append((x, y))
    mc, sc = group_mean_on_grid(norm_contrast, xgrid)
    mr, sr = group_mean_on_grid(norm_random, xgrid)
    axB.plot(xgrid, mc, color=C_CONTRAST, lw=3, label='contrastive mean')
    axB.fill_between(xgrid, mc - sc, mc + sc, color=C_CONTRAST, alpha=0.15)
    axB.plot(xgrid, mr, color=C_RANDOM, lw=3, label='random mean')
    axB.fill_between(xgrid, mr - sr, mr + sr, color=C_RANDOM, alpha=0.15)
    axB.scatter([0, 1], [0, 1], color='k', zorder=5, s=20)
    axB.set_xlabel('normalized angle  (theta_lo->0, theta_hi->1)')
    axB.set_ylabel('normalized response')
    axB.set_title('(B) 2-point affine collapse — onset & scale removed')
    axB.legend(fontsize=8)

    # ---- (C) x-only (onset) normalization ----
    for name, r in contrast.items():
        res = affine_normalize(theta, r)
        if res:
            _, _, tlo, _ = res
            axC.plot(theta / tlo, r / r.max(), color=C_CONTRAST, alpha=0.5, lw=1)
    for r in random_curves:
        res = affine_normalize(theta, r)
        if res:
            _, _, tlo, _ = res
            axC.plot(theta / tlo, r / r.max(), color=C_RANDOM, alpha=0.15, lw=0.8)
    axC.set_xlabel('angle / theta_break  (break = 10% of own max)')
    axC.set_ylabel('response / own max')
    axC.set_title('(C) onset-only normalization — tests "steeper than onset predicts"')
    axC.set_xlim(0, 8)

    # ---- (D) shape scalar: normalized y at x_norm = 0.5 ----
    def y_at_half(curve):
        res = affine_normalize(theta, curve)
        if not res:
            return None
        x, y, *_ = res
        return float(np.interp(0.5, x, y))
    yc = [v for v in (y_at_half(r) for r in contrast.values()) if v is not None]
    yr = [v for v in (y_at_half(r) for r in random_curves) if v is not None]
    rng = np.random.RandomState(0)
    axD.scatter(rng.uniform(-.05, .05, len(yr)), yr, color=C_RANDOM, alpha=0.5,
                label=f'random (med={np.median(yr):.2f})')
    axD.scatter(1 + rng.uniform(-.05, .05, len(yc)), yc, color=C_CONTRAST, alpha=0.8,
                label=f'contrastive (med={np.median(yc):.2f})')
    axD.axhline(0.5, color='gray', ls=':', lw=1)
    axD.set_xticks([0, 1]); axD.set_xticklabels(['random', 'contrastive'])
    axD.set_ylabel('normalized response at midpoint angle')
    axD.set_title('(D) shape scalar at x_norm=0.5  (0.5 = linear)')
    axD.legend(fontsize=8)

    fig.suptitle(f'Response-curve shape collapse — {args.model} perturb L{pl} -> measure L{ml} (L2)',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = args.out or f'results/feature_vs_random_sweep/curve_shape_collapse_{args.model}_L2.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f'saved: {out}')
    print(f'midpoint y  contrastive (n={len(yc)}): med={np.median(yc):.3f} range=[{min(yc):.3f},{max(yc):.3f}]')
    print(f'midpoint y  random      (n={len(yr)}): med={np.median(yr):.3f} range=[{min(yr):.3f},{max(yr):.3f}]')


if __name__ == '__main__':
    main()
