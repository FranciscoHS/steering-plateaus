#!/usr/bin/env python3
"""Fig 1 intro: simplified Formal × Portuguese boundary for the headline."""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse, superellipse_curve)

PKL = 'results/pair_exploration_L2_v2/sweep_2d_Formal_Portuguese.pkl'
OUT_PNG = 'writeup/figures/fig1_intro_boundary.png'
OUT_PDF = 'writeup/figures/fig1_intro_boundary.pdf'
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
    plt.rcParams['font.family'] = 'Inter'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'cm'

    with open(PKL, 'rb') as f:
        res = pickle.load(f)
    angles = res['angles_deg']
    median_grid = np.median(res['l2'], axis=0)
    labels = res['direction_labels']

    pt_curve = median_grid[:, 0]; gen_curve = median_grid[0, :]
    theta1 = find_threshold(pt_curve, angles, THRESH)
    theta2 = find_threshold(gen_curve, angles, THRESH)
    contour_pts = extract_contour_points(angles, median_grid, THRESH)
    cn = contour_pts.astype(float).copy()
    cn[:, 0] /= theta1; cn[:, 1] /= theta2

    fig, ax = plt.subplots(figsize=(5.4, 5.4))

    # Reference curves — drawn longer than legend
    xi, yi = superellipse_curve(20)
    ax.plot(xi, yi, ls=(0, (2.5, 3.0)), color='#D9822B', lw=2.0, alpha=0.95)
    xc, yc = superellipse_curve(2)
    ax.plot(xc, yc, '-', color='#1F77B4', lw=3.0, alpha=0.95)

    # Data — slightly larger filled grey dots
    cont_angle = np.arctan2(cn[:, 1], cn[:, 0])
    order = np.argsort(cont_angle)
    ax.plot(cn[order, 0], cn[order, 1], 'o',
            color='#3a3a3a', markersize=5.5, alpha=0.9,
            markeredgecolor='none')

    # In-figure curve labels
    ax.annotate('per-feature threshold',
                xy=(0.83, 0.99), xytext=(0.55, 1.10),
                fontsize=12, color='#a05a18', ha='center',
                arrowprops=dict(arrowstyle='-', color='#a05a18', lw=0.8))
    ax.annotate('Euclidean combination',
                xy=(0.55, 0.85), xytext=(0.20, 0.55),
                fontsize=12, color='#1F77B4', ha='center',
                arrowprops=dict(arrowstyle='-', color='#1F77B4', lw=0.8))
    ax.annotate('data',
                xy=(0.85, 0.55), xytext=(0.95, 0.78),
                fontsize=12, color='#3a3a3a', ha='center',
                arrowprops=dict(arrowstyle='-', color='#3a3a3a', lw=0.8))

    ax.set_xlim(-0.05, 1.20); ax.set_ylim(-0.05, 1.20)
    ax.set_aspect('equal')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel(labels[0], fontsize=14)
    ax.set_ylabel(labels[1], fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=220, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    print(f'Saved {OUT_PNG}')


if __name__ == '__main__':
    main()
