#!/usr/bin/env python3
"""All-threshold overlay of iso-plateau contours, L2 only.

Reads sweep_isoplateau_L2.pkl (PT x Gender, Gemma-2-9B, perturb L2, measure
L40), then on a single normalized plot overlays the contour at every threshold
in a sweep. Each contour is normalized by its own (theta1, theta2) axis
intercepts and color-coded by threshold value. p=2 circle and p=inf square
references are drawn for context.

Goal: see the regime where the modelling 'makes sense' (contours stack onto
the p~2 circle) vs where it breaks down (low thr -> blob inside; high thr ->
hits grid ceiling, fitted p drifts).
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse, superellipse_curve)

PKL = 'results/paper_L2_gemma2_9b/sweep_isoplateau_L2_xwide.pkl'
OUT = 'writeup/figures/fig3_threshold_overlay_xwide.png'
LABELS = ('Portuguese', 'Gender')

THRESHOLDS = np.array(
    [7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500])


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
    noise_floor = median_grid[0, 0]
    grid_max = max(curve1.max(), curve2.max())
    print(f'angles up to {angles[-1]:.1f}°, noise floor {noise_floor:.2f}, '
          f'grid max along axes ~{grid_max:.1f}')

    cmap = cm.viridis
    norm = LogNorm(vmin=THRESHOLDS.min(), vmax=THRESHOLDS.max())

    fig, ax = plt.subplots(figsize=(7, 7))

    summary_rows = []
    for thr in THRESHOLDS:
        theta1 = find_threshold(curve1, angles, thr)
        theta2 = find_threshold(curve2, angles, thr)

        flag = ''
        # Below noise floor: not a real contour
        if thr <= noise_floor:
            flag = 'below noise floor'
        # Hits grid ceiling on either axis
        elif theta1 >= angles[-1] * 0.98 or theta2 >= angles[-1] * 0.98:
            flag = 'grid ceiling'

        pts = extract_contour_points(angles, median_grid, thr)
        if len(pts) == 0:
            print(f'  thr={thr:6g}  empty contour ({flag})')
            continue
        cn = pts.astype(float).copy()
        cn[:, 0] /= theta1
        cn[:, 1] /= theta2

        avg_theta = (theta1 + theta2) / 2
        p_fit, ci, info = fit_superellipse(cn, avg_theta)

        color = cmap(norm(thr))
        # Restrict to first quadrant for plotting
        mask = (cn[:, 0] >= -0.02) & (cn[:, 1] >= -0.02)
        pts = cn[mask]
        size = 12 if not flag else 8
        alpha = 0.85 if not flag else 0.35
        ax.scatter(pts[:, 0], pts[:, 1], s=size, color=color,
                   alpha=alpha, edgecolors='none')

        summary_rows.append({
            'thr': thr, 'theta1': theta1, 'theta2': theta2,
            'p_fit': p_fit if p_fit is not None else float('nan'),
            'flag': flag,
        })
        p_str = f'{p_fit:.2f}' if p_fit is not None else '  -- '
        print(f'  thr={thr:6g}  θ1={theta1:5.2f}  θ2={theta2:5.2f}  '
              f'p={p_str}  {flag}')

    # Axis intercepts marker
    ax.plot(1, 0, 'kv', markersize=8, zorder=5)
    ax.plot(0, 1, 'k>', markersize=8, zorder=5)

    ax.set_xlim(-0.03, 1.25)
    ax.set_ylim(-0.03, 1.25)
    ax.set_aspect('equal')
    ax.set_xlabel(rf'$\theta(\varphi)\cos\varphi / \theta_1$  ({LABELS[0]})',
                  fontsize=12)
    ax.set_ylabel(rf'$\theta(\varphi)\sin\varphi / \theta_2$  ({LABELS[1]})',
                  fontsize=12)
    ax.set_title(f'{LABELS[0]} × {LABELS[1]} @ Gemma-2-9B L2 — '
                 'all $L^2$ thresholds, normalized', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Colorbar for thresholds
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$L^2$ threshold')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
    pdf_out = OUT.replace('.png', '.pdf')
    plt.savefig(pdf_out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved {OUT}')
    print(f'Saved {pdf_out}')

    # Save summary table alongside
    summary_pkl = OUT.replace('.png', '_summary.pkl')
    with open(summary_pkl, 'wb') as f:
        pickle.dump({'rows': summary_rows, 'noise_floor': noise_floor,
                     'grid_max_along_axes': grid_max,
                     'angles_deg': angles}, f)
    print(f'Saved {summary_pkl}')


if __name__ == '__main__':
    main()
