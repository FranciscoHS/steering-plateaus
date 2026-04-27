#!/usr/bin/env python3
"""Fig 3: Portuguese x Gender iso-plateau boundary at L2, normalized.

Plots data contour points + p=2 (Euclidean) reference + p=inf (per-feature
threshold) reference + data-fitted superellipse. L2 metric only.
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse, superellipse_curve)

PKL = 'results/paper_L2_gemma2_9b/sweep_isoplateau_L2.pkl'
OUT_PNG = 'writeup/figures/fig3_boundary_pt_gender_L2.png'
OUT_PDF = 'writeup/figures/fig3_boundary_pt_gender_L2.pdf'
THRESH = 50.0
LABELS = ('Portuguese', 'Gender')


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
    data = res['l2']                    # [N_ACT, N, N]
    median_grid = np.median(data, axis=0)

    pt_curve = median_grid[:, 0]        # PT axis (α2=0)
    gen_curve = median_grid[0, :]       # Gender axis (α1=0)
    theta1 = find_threshold(pt_curve, angles, THRESH)
    theta2 = find_threshold(gen_curve, angles, THRESH)
    print(f'theta_PT = {theta1:.2f}°, theta_Gender = {theta2:.2f}°')

    contour_pts = extract_contour_points(angles, median_grid, THRESH)
    if len(contour_pts) == 0:
        print('No contour points extracted'); return
    cn = contour_pts.astype(float).copy()
    cn[:, 0] /= theta1
    cn[:, 1] /= theta2

    avg_theta = (theta1 + theta2) / 2
    p_fit, ci, info = fit_superellipse(cn, avg_theta)
    print(f'Fitted p = {p_fit:.3f}  (95% CI [{ci[0]:.2f}, {ci[1]:.2f}])')
    print(f'  mean radial err = {info["mean_radial_frac"]:.3f} '
          f'({info["mean_radial_deg"]:.3f}° at avg θ = {avg_theta:.2f}°)')
    print(f'  max  radial err = {info["max_radial_frac"]:.3f} '
          f'({info["max_radial_deg"]:.3f}°)')

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # References
    xi, yi = superellipse_curve(20)   # p = 20 ≈ square (per-feature threshold)
    ax.plot(xi, yi, '--', color='#d62728', lw=2.0,
            label=r'per-feature threshold ($p\to\infty$)')
    xc, yc = superellipse_curve(2)
    ax.plot(xc, yc, '--', color='#1f77b4', lw=2.0,
            label=r'Euclidean combination ($p=2$)')

    # Data-fitted superellipse
    xf, yf = superellipse_curve(p_fit)
    ax.plot(xf, yf, '-', color='#333', lw=1.8, alpha=0.85,
            label=rf'fit $p={p_fit:.2f}$')

    # Data dots
    cont_angle = np.arctan2(cn[:, 1], cn[:, 0])
    order = np.argsort(cont_angle)
    ax.plot(cn[order, 0], cn[order, 1], 'o', color='#2ca02c',
            markersize=5, alpha=0.85, label='data', zorder=5)

    # Anchors at axis intercepts
    ax.plot(1, 0, 'kv', markersize=8, zorder=6)
    ax.plot(0, 1, 'k>', markersize=8, zorder=6)

    ax.set_xlim(-0.03, 1.15); ax.set_ylim(-0.03, 1.15)
    ax.set_aspect('equal')
    ax.set_xlabel(rf'$\theta(\varphi)\cos\varphi / \theta_1$  ({LABELS[0]})',
                  fontsize=12)
    ax.set_ylabel(rf'$\theta(\varphi)\sin\varphi / \theta_2$  ({LABELS[1]})',
                  fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, loc='lower left', frameon=True, framealpha=0.95)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=220, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    print(f'Saved {OUT_PNG}')
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
