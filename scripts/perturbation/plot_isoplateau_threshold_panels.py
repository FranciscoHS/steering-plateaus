#!/usr/bin/env python3
"""Three-panel iso-plateau plot at low/typical/high L2 thresholds.

Reads the xwide PT x Gender sweep (90° max angle, non-uniform grid). One panel
per threshold, each with normalized data points + p=2 circle and p=inf square
references + fitted-p annotation. Goal: see at a glance how the contour shape
evolves from low (linear / sub-Euclidean) -> typical (peaked above Euclidean)
-> high (back near Euclidean) thresholds.
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse, superellipse_curve)

PKL = 'results/paper_L2_gemma2_9b/sweep_isoplateau_L2_xwide.pkl'
OUT = 'writeup/figures/fig3_threshold_panels.png'
LABELS = ('Portuguese', 'Gender')

# (threshold, descriptive label)
PANELS = [
    (10,  'low'),
    (50,  'typical'),
    (380, 'high'),
]


def find_threshold(curve, angles, thresh):
    idx = np.searchsorted(curve, thresh)
    if idx == 0 or idx >= len(angles):
        return angles[min(idx, len(angles) - 1)]
    x0, x1 = angles[idx - 1], angles[idx]
    y0, y1 = curve[idx - 1], curve[idx]
    if y1 == y0:
        return x0
    return x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)


def main():
    with open(PKL, 'rb') as f:
        res = pickle.load(f)
    angles = res['angles_deg']
    median_grid = np.median(res['l2'], axis=0)
    curve1 = median_grid[:, 0]
    curve2 = median_grid[0, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    # references — same in every panel
    xc, yc = superellipse_curve(2)
    xs, ys = superellipse_curve(20)

    for ax, (thr, label) in zip(axes, PANELS):
        theta1 = find_threshold(curve1, angles, thr)
        theta2 = find_threshold(curve2, angles, thr)

        pts = extract_contour_points(angles, median_grid, thr)
        cn = pts.astype(float).copy()
        cn[:, 0] /= theta1
        cn[:, 1] /= theta2
        # First quadrant only
        mask = (cn[:, 0] >= -0.02) & (cn[:, 1] >= -0.02)
        cn = cn[mask]

        avg_theta = (theta1 + theta2) / 2
        p_fit, ci, info = fit_superellipse(cn, avg_theta)

        ax.plot(xc, yc, '--', color='#1f77b4', lw=1.8, alpha=0.85,
                label='Euclidean ($p=2$)')
        ax.plot(xs, ys, '--', color='#d62728', lw=1.8, alpha=0.85,
                label=r'per-feature ($p\to\infty$)')
        ax.scatter(cn[:, 0], cn[:, 1], s=14, color='#2ca02c',
                   alpha=0.85, edgecolors='none', label='data')
        ax.plot(1, 0, 'kv', markersize=8, zorder=5)
        ax.plot(0, 1, 'k>', markersize=8, zorder=5)

        ax.set_xlim(-0.03, 1.20)
        ax.set_ylim(-0.03, 1.20)
        ax.set_aspect('equal')
        ax.set_xlabel(rf'$\theta(\varphi)\cos\varphi / \theta_1$  ({LABELS[0]})',
                      fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel(rf'$\theta(\varphi)\sin\varphi / \theta_2$  ({LABELS[1]})',
                          fontsize=11)

        p_str = f'{p_fit:.2f}' if p_fit is not None else '—'
        ci_str = (f'  CI [{ci[0]:.2f}, {ci[1]:.2f}]'
                  if ci is not None else '')
        title = (f'{label}: $L^2 = {thr}$\n'
                 rf'$\theta_1 = {theta1:.2f}°,\ \theta_2 = {theta2:.2f}°$;'
                 rf'  fit $p = {p_str}${ci_str}')
        ax.set_title(title, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc='lower left', frameon=True,
                      framealpha=0.95)

        print(f'  thr={thr:4g}  θ1={theta1:5.2f}  θ2={theta2:5.2f}  '
              f'p={p_str}')

    fig.suptitle(f'{LABELS[0]} × {LABELS[1]} @ Gemma-2-9B L2 — iso-plateau '
                 'shape at low / typical / high $L^2$ threshold', fontsize=12)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
