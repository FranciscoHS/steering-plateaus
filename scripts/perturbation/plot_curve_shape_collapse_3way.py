#!/usr/bin/env python3
"""Three-way shape-collapse: high-p dictionary atoms vs contrastive vs random.

Loads two dict_vs_random_sweep pkls (produced by run_dict_vs_random_sweep.py with the
same model/layer/seed, so activations and random dirs match):
  --highp_pkl       : its 'atom*' curves are the high-p (soft_p) dictionary atoms
  --contrastive_pkl : its 'atom*' curves are the contrastive (diff-of-means) directions
Random baseline is taken from the high-p pkl (identical across both by construction).

Same 4 panels as the 2-way version: raw, 2-point affine collapse, onset-only
normalization, midpoint shape scalar.

Usage:
    python scripts/perturbation/plot_curve_shape_collapse_3way.py \
        --highp_pkl results/dict_vs_random_sweep/qwen3-1.7b_L14_softp/dict_vs_random_sweep.pkl \
        --contrastive_pkl results/dict_vs_random_sweep/qwen3-1.7b_L14_contrastive/dict_vs_random_sweep.pkl
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

from scripts.perturbation.plot_curve_shape_collapse import (
    affine_normalize, group_mean_on_grid,
)

C_HIGHP = '#16a085'      # teal
C_CONTRAST = '#c0392b'   # red
C_RANDOM = '#2c3e50'     # dark slate


def atom_curves(pkl, metric):
    """Mean-over-activations curve for each 'atom*' direction in a sweep pkl."""
    M = pkl[metric]
    return [np.array(M[k], dtype=float).mean(axis=0) for k in M if k.startswith('atom')]


def random_curves(pkl, metric):
    arr = np.array(pkl[metric]['random'], dtype=float)   # [n_act, n_rand, n_ang]
    return [arr[:, j, :].mean(axis=0) for j in range(arr.shape[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--highp_pkl', required=True)
    ap.add_argument('--contrastive_pkl', required=True)
    ap.add_argument('--metric', default='l2', choices=['l2', 'kl'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    hp = pickle.load(open(args.highp_pkl, 'rb'))
    ct = pickle.load(open(args.contrastive_pkl, 'rb'))
    theta = np.array(hp['angles_deg'], dtype=float)

    groups = [
        ('high-p', C_HIGHP, atom_curves(hp, args.metric)),
        ('contrastive', C_CONTRAST, atom_curves(ct, args.metric)),
        ('random', C_RANDOM, random_curves(hp, args.metric)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axA, axB, axC, axD = axes.ravel()
    xgrid = np.linspace(0, 1, 50)

    for label, color, curves in groups:
        # (A) raw
        for r in curves:
            axA.plot(theta, r, color=color, alpha=0.25, lw=0.9)
        axA.plot([], [], color=color, lw=2, label=f'{label} (n={len(curves)})')
        # (B) affine collapse + (C) onset-only
        norm = []
        for r in curves:
            res = affine_normalize(theta, r)
            if not res:
                continue
            x, y, tlo, _ = res
            axB.plot(x, y, color=color, alpha=0.18, lw=0.8)
            axC.plot(theta / tlo, r / r.max(), color=color, alpha=0.2, lw=0.8)
            norm.append((x, y))
        m, s = group_mean_on_grid(norm, xgrid)
        axB.plot(xgrid, m, color=color, lw=3, label=f'{label} mean')
        axB.fill_between(xgrid, m - s, m + s, color=color, alpha=0.12)

    axA.set_xlabel('perturbation angle (deg)')
    axA.set_ylabel(f"{ {'l2': 'L2-distance', 'kl': 'KL'}[args.metric] } response")
    axA.set_title('(A) raw response curves'); axA.legend(fontsize=8)
    axB.scatter([0, 1], [0, 1], color='k', zorder=5, s=20)
    axB.set_xlabel('normalized angle  (theta_lo->0, theta_hi->1)')
    axB.set_ylabel('normalized response')
    axB.set_title('(B) 2-point affine collapse — onset & scale removed'); axB.legend(fontsize=8)
    axC.set_xlabel('angle / theta_break  (break = 10% of own max)')
    axC.set_ylabel('response / own max')
    axC.set_title('(C) onset-only normalization'); axC.set_xlim(0, 8)

    # (D) midpoint scalar
    rng = np.random.RandomState(0)
    for i, (label, color, curves) in enumerate(groups):
        ys = []
        for r in curves:
            res = affine_normalize(theta, r)
            if res:
                x, y, *_ = res
                ys.append(float(np.interp(0.5, x, y)))
        axD.scatter(i + rng.uniform(-.07, .07, len(ys)), ys, color=color, alpha=0.7,
                    label=f'{label} (med={np.median(ys):.2f})')
    axD.axhline(0.5, color='gray', ls=':', lw=1)
    axD.set_xticks(range(len(groups))); axD.set_xticklabels([g[0] for g in groups])
    axD.set_ylabel('normalized response at midpoint angle')
    axD.set_title('(D) shape scalar at x_norm=0.5  (0.5 = linear)'); axD.legend(fontsize=8)

    metric_label = {'l2': 'L2-distance', 'kl': 'KL'}[args.metric]
    fig.suptitle(f'Response-curve shape collapse (3-way) — qwen3-1.7b, perturb layer 14 '
                 f'(response metric: {metric_label})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or os.path.join(os.path.dirname(args.highp_pkl), f'collapse_3way_{args.metric}.png')
    fig.savefig(out, dpi=140)
    print(f'saved: {out}')


if __name__ == '__main__':
    main()
