#!/usr/bin/env python3
"""Threshold sensitivity for Fig 3 (PT x Gender at L2).

Reads sweep_isoplateau_L2.pkl, sweeps the L2-distance threshold, refits the
superellipse exponent p each time, plots p +/- bootstrap CI vs threshold.
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse)

PKL = 'results/paper_L2_gemma2_9b/sweep_isoplateau_L2.pkl'
OUT = 'writeup/figures/fig3_threshold_sensitivity.png'

THRESHOLDS = np.array([10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250])


def find_threshold(curve, angles, thresh):
    idx = np.searchsorted(curve, thresh)
    if idx == 0 or idx >= len(angles):
        return angles[min(idx, len(angles) - 1)]
    x0, x1 = angles[idx - 1], angles[idx]
    y0, y1 = curve[idx - 1], curve[idx]
    if y1 == y0: return x0
    return x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)


def main():
    with open(PKL, 'rb') as f:
        res = pickle.load(f)
    angles = res['angles_deg']
    median_grid = np.median(res['l2'], axis=0)
    curve1 = median_grid[:, 0]
    curve2 = median_grid[0, :]
    noise_floor = median_grid[0, 0]
    print(f'angle grid max: {angles[-1]:.1f}°')
    print(f'curve1 max: {curve1.max():.1f}  curve2 max: {curve2.max():.1f}')
    print(f'noise floor at (0,0): {noise_floor:.3f}')

    results = []
    for thr in THRESHOLDS:
        theta1 = find_threshold(curve1, angles, thr)
        theta2 = find_threshold(curve2, angles, thr)
        if theta1 >= angles[-1] * 0.98 or theta2 >= angles[-1] * 0.98:
            print(f'  thr={thr}: hits grid ceiling, skipped'); continue
        if theta1 <= angles[1] or theta2 <= angles[1]:
            print(f'  thr={thr}: inside first grid step, skipped'); continue

        pts = extract_contour_points(angles, median_grid, thr)
        if len(pts) == 0:
            print(f'  thr={thr}: empty contour'); continue
        cn = pts.astype(float).copy()
        cn[:, 0] /= theta1
        cn[:, 1] /= theta2
        avg_theta = (theta1 + theta2) / 2
        p_fit, ci, info = fit_superellipse(cn, avg_theta)
        if p_fit is None:
            print(f'  thr={thr}: fit failed'); continue
        results.append({
            'thr': thr, 'theta1': theta1, 'theta2': theta2,
            'p': p_fit, 'ci': ci, 'err': info['mean_radial_frac'],
            'err_deg': info['mean_radial_deg'],
        })
        print(f'  thr={thr:4g}  θ1={theta1:5.2f}  θ2={theta2:5.2f}  '
              f'p={p_fit:.2f}  CI=[{ci[0]:.2f}, {ci[1]:.2f}]  '
              f'err={info["mean_radial_frac"]:.3f} ({info["mean_radial_deg"]:.2f}°)')

    if not results:
        print('No valid fits'); return

    ts = np.array([r['thr'] for r in results])
    ps = np.array([r['p'] for r in results])
    lo = np.array([r['ci'][0] for r in results])
    hi = np.array([r['ci'][1] for r in results])
    errs = np.array([r['err'] for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.fill_between(ts, lo, hi, color='#1f77b4', alpha=0.2, label='95% bootstrap CI')
    ax1.plot(ts, ps, 'o-', color='#1f77b4', lw=1.8, markersize=7, label='fit $p$')
    ax1.axhline(2.0, ls='--', color='#d62728', alpha=0.6, label=r'$p=2$ (Euclidean)')
    ax1.set_xlabel(r'$L^2$ threshold')
    ax1.set_ylabel(r'fitted exponent $p$')
    ax1.set_xscale('log')
    ax1.set_title('Portuguese × Gender @ L2 — p vs threshold')
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9)

    ax2.plot(ts, errs, 'o-', color='#2ca02c', lw=1.6, markersize=7)
    ax2.set_xlabel(r'$L^2$ threshold')
    ax2.set_ylabel('mean radial fit error (normalized)')
    ax2.set_xscale('log')
    ax2.set_title('Fit quality')
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
