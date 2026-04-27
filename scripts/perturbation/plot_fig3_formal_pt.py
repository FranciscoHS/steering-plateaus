#!/usr/bin/env python3
"""Fig 3 variant: Formal × Portuguese iso-plateau boundary at L2, normalized."""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse, superellipse_curve)

PKL = 'results/pair_exploration_L2_v2/sweep_2d_Formal_Portuguese.pkl'
OUT_PNG = 'writeup/figures/fig3_boundary_formal_pt_L2.png'
OUT_PDF = 'writeup/figures/fig3_boundary_formal_pt_L2.pdf'
THRESH = 50.0


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
    data = res['l2']
    median_grid = np.median(data, axis=0)
    labels = res['direction_labels']          # ('Formal', 'Portuguese')

    pt_curve = median_grid[:, 0]
    gen_curve = median_grid[0, :]
    theta1 = find_threshold(pt_curve, angles, THRESH)
    theta2 = find_threshold(gen_curve, angles, THRESH)
    print(f'theta_{labels[0]} = {theta1:.2f}°, '
          f'theta_{labels[1]} = {theta2:.2f}°')

    contour_pts = extract_contour_points(angles, median_grid, THRESH)
    cn = contour_pts.astype(float).copy()
    cn[:, 0] /= theta1; cn[:, 1] /= theta2
    avg_theta = (theta1 + theta2) / 2
    p_fit, ci, info = fit_superellipse(cn, avg_theta)
    print(f'Fitted p = {p_fit:.3f}  (95% CI [{ci[0]:.2f}, {ci[1]:.2f}])')
    print(f'  mean radial err = {info["mean_radial_frac"]:.3f} '
          f'({info["mean_radial_deg"]:.3f}° at avg θ = {avg_theta:.2f}°)')
    print(f'  max  radial err = {info["max_radial_frac"]:.3f} '
          f'({info["max_radial_deg"]:.3f}°)')

    fig, ax = plt.subplots(figsize=(5.8, 5.8))

    # Light grid at 0.2 intervals
    ax.set_xticks(np.arange(0, 1.21, 0.2))
    ax.set_yticks(np.arange(0, 1.21, 0.2))
    ax.grid(True, which='major', color='#bbbbbb', lw=0.5, alpha=0.45)

    # Reference curves
    xi, yi = superellipse_curve(20)
    ax.plot(xi, yi, ls=(0, (1.8, 2.5)), color='#D9822B', lw=1.6,
            alpha=0.95,
            label=r'per-feature threshold ($p\to\infty$)')
    xc, yc = superellipse_curve(2)
    ax.plot(xc, yc, '-', color='#1F77B4', lw=2.6, alpha=0.95,
            label=r'Euclidean combination ($p=2$)')

    # Data as small filled neutral-grey circles
    cont_angle = np.arctan2(cn[:, 1], cn[:, 0])
    order = np.argsort(cont_angle)
    ax.plot(cn[order, 0], cn[order, 1], 'o',
            color='#3a3a3a', markersize=4.0, alpha=0.85,
            markeredgecolor='none',
            label=rf'data  ($p_{{\mathrm{{fit}}}} = {p_fit:.2f}$)',
            zorder=5)

    ax.set_xlim(-0.03, 1.15); ax.set_ylim(-0.03, 1.15)
    ax.set_aspect('equal')
    ax.set_xlabel(rf'$\theta(\varphi)\cos\varphi / \theta_1$  ({labels[0]})',
                  fontsize=12)
    ax.set_ylabel(rf'$\theta(\varphi)\sin\varphi / \theta_2$  ({labels[1]})',
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
