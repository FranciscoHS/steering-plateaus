#!/usr/bin/env python3
"""Shape-collapse test for perturbation response curves (Direction 1).

Question: are contrastive (feature) and random response curves the same SHAPE up to
rescaling, or does the contrastive curve differ in shape (e.g. steeper) beyond just
breaking the plateau earlier?

Input: a feature_vs_random_sweep pkl with
    angles_deg : [n_angles]
    l2['random'] : [n_act, n_rand_dirs, n_angles]
    l2[<name>]   : [n_act, n_angles]   for each contrastive direction

We aggregate across activations (mean over act) -> one response curve r(theta) per
direction. Then:

  (A) raw curves                         : r vs theta, contrastive vs random
  (B) 2-point affine collapse            : anchor each curve at theta_lo (10% of its
        own max) and theta_hi (90% of its own max); map x->[0,1], y->[0,1]. Onset AND
        scale divided out -> only shape remains. If curves coincide, no shape diff.
  (C) x-only (onset) normalization       : theta / theta_break (theta_break = theta_lo),
        y = r / max. Tests whether steepness is just proportional to onset, or whether
        contrastive is steeper than its earlier onset predicts.
  (D) one shape scalar per curve         : normalized y at x_norm = 0.5 (concavity /
        steepness after onset+scale removed). Strip plot, contrastive vs random.

Usage:
    python scripts/perturbation/plot_curve_shape_collapse.py \
        --pkl results/feature_vs_random_sweep/qwen3-4b/feature_vs_random_sweep.pkl \
        --metric l2
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_CONTRAST = '#c0392b'
C_RANDOM = '#2c3e50'


def crossing(theta, r, level):
    """First angle (linear-interpolated) at which r rises through `level`.

    Uses the first upward crossing so noisy tails don't pull the anchor."""
    above = r >= level
    if not above.any():
        return theta[-1]
    i = int(np.argmax(above))  # first True
    if i == 0:
        return theta[0]
    t0, t1 = theta[i - 1], theta[i]
    r0, r1 = r[i - 1], r[i]
    if r1 == r0:
        return t1
    return t0 + (level - r0) * (t1 - t0) / (r1 - r0)


def affine_normalize(theta, r, lo_frac=0.1, hi_frac=0.9):
    """2-point affine collapse. Returns (x_norm, y_norm, theta_lo, theta_hi).

    y_norm = (r/max - lo_frac)/(hi_frac - lo_frac); x mapped so theta_lo->0, theta_hi->1.
    """
    rmax = r.max()
    if rmax <= 0:
        return None
    theta_lo = crossing(theta, r, lo_frac * rmax)
    theta_hi = crossing(theta, r, hi_frac * rmax)
    if theta_hi <= theta_lo:
        return None
    x = (theta - theta_lo) / (theta_hi - theta_lo)
    y = (r / rmax - lo_frac) / (hi_frac - lo_frac)
    return x, y, theta_lo, theta_hi


def group_mean_on_grid(curves, xgrid):
    """Interpolate each (x,y) curve onto xgrid and return (mean, std) across curves."""
    stacked = np.array([np.interp(xgrid, x, y, left=np.nan, right=np.nan)
                        for x, y in curves])
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
    return mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='results/feature_vs_random_sweep/qwen3-4b/feature_vs_random_sweep.pkl')
    ap.add_argument('--metric', default='l2', choices=['l2', 'kl'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    d = pickle.load(open(args.pkl, 'rb'))
    theta = np.array(d['angles_deg'], dtype=float)
    M = d[args.metric]

    # One curve per direction, aggregated across activations (mean over act).
    contrast = {}   # name -> r(theta)
    random_curves = []  # list of r(theta), one per random direction
    for name, arr in M.items():
        arr = np.array(arr, dtype=float)
        if name == 'random':
            # [n_act, n_rand, n_angles] -> per-direction mean over activations
            for j in range(arr.shape[1]):
                random_curves.append(arr[:, j, :].mean(axis=0))
        else:
            contrast[name] = arr.mean(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axA, axB, axC, axD = axes.ravel()

    # ---- (A) raw curves ----
    for r in random_curves:
        axA.plot(theta, r, color=C_RANDOM, alpha=0.35, lw=1)
    axA.plot([], [], color=C_RANDOM, alpha=0.6, lw=1.5, label=f'random (n={len(random_curves)})')
    for name, r in contrast.items():
        axA.plot(theta, r, color=C_CONTRAST, lw=2, label=f'contrastive: {name}')
    axA.set_xlabel('perturbation angle (deg)')
    axA.set_ylabel(f'{args.metric.upper()} response')
    axA.set_title('(A) raw response curves')
    axA.legend(fontsize=8)

    # ---- (B) 2-point affine collapse ----
    xgrid = np.linspace(0, 1, 50)
    norm_contrast, norm_random = [], []
    for name, r in contrast.items():
        res = affine_normalize(theta, r)
        if res:
            x, y, *_ = res
            axB.plot(x, y, color=C_CONTRAST, alpha=0.7, lw=1)
            norm_contrast.append((x, y))
    for r in random_curves:
        res = affine_normalize(theta, r)
        if res:
            x, y, *_ = res
            axB.plot(x, y, color=C_RANDOM, alpha=0.25, lw=1)
            norm_random.append((x, y))
    mc, sc = group_mean_on_grid(norm_contrast, xgrid)
    mr, sr = group_mean_on_grid(norm_random, xgrid)
    axB.plot(xgrid, mc, color=C_CONTRAST, lw=3, label='contrastive mean')
    axB.fill_between(xgrid, mc - sc, mc + sc, color=C_CONTRAST, alpha=0.15)
    axB.plot(xgrid, mr, color=C_RANDOM, lw=3, label='random mean')
    axB.fill_between(xgrid, mr - sr, mr + sr, color=C_RANDOM, alpha=0.15)
    axB.scatter([0, 1], [0, 1], color='k', zorder=5, s=20)  # the two anchor points
    axB.set_xlabel('normalized angle  (theta_lo->0, theta_hi->1)')
    axB.set_ylabel('normalized response')
    axB.set_title('(B) 2-point affine collapse — onset & scale removed')
    axB.legend(fontsize=8)

    # ---- (C) x-only (onset) normalization ----
    for name, r in contrast.items():
        res = affine_normalize(theta, r)
        if res:
            _, _, tlo, _ = res
            axC.plot(theta / tlo, r / r.max(), color=C_CONTRAST, lw=2)
    for r in random_curves:
        res = affine_normalize(theta, r)
        if res:
            _, _, tlo, _ = res
            axC.plot(theta / tlo, r / r.max(), color=C_RANDOM, alpha=0.3, lw=1)
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
    axD.scatter(np.zeros(len(yr)) + np.random.uniform(-.05, .05, len(yr)), yr,
                color=C_RANDOM, alpha=0.6, label=f'random (med={np.median(yr):.2f})')
    axD.scatter(np.ones(len(yc)) + np.random.uniform(-.05, .05, len(yc)), yc,
                color=C_CONTRAST, alpha=0.8, label=f'contrastive (med={np.median(yc):.2f})')
    axD.axhline(0.5, color='gray', ls=':', lw=1)  # 0.5 = straight line between anchors
    axD.set_xticks([0, 1]); axD.set_xticklabels(['random', 'contrastive'])
    axD.set_ylabel('normalized response at midpoint angle')
    axD.set_title('(D) shape scalar: concavity at x_norm=0.5  (0.5 = linear)')
    axD.legend(fontsize=8)

    fig.suptitle(f'Response-curve shape collapse — {d["args"]["model"]} L{d["args"]["layer"]} ({args.metric.upper()})',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = args.out or os.path.join(os.path.dirname(args.pkl), f'curve_shape_collapse_{args.metric}.png')
    fig.savefig(out, dpi=140)
    print(f'saved: {out}')
    # also dump the midpoint scalars for inspection
    print(f'midpoint y  contrastive: {np.round(yc, 3)}')
    print(f'midpoint y  random med={np.median(yr):.3f}  range=[{min(yr):.3f},{max(yr):.3f}]')


if __name__ == '__main__':
    main()
