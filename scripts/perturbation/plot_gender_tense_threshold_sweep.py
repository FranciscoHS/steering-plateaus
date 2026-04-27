#!/usr/bin/env python3
"""Gender x Tense iso-response boundary on Gemma-2-9B (L2 -> L40), one plot per
L2 threshold. Generates many candidate versions for Francisco to pick from.

Reads results/fig4_robustness/v2_T1_gemma_Gender_Tense.pkl. Each plot shows
normalised data points + p=2 (Euclidean) + p=infty (per-feature) references,
with the fitted superellipse exponent p annotated. Outputs one PNG per
threshold, plus a single overview grid.
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse, superellipse_curve)

PKL = 'results/fig4_robustness/v2_T1_gemma_Gender_Tense.pkl'
OUT_DIR = 'results/gender_tense_threshold_sweep'

# Median L2 range in this grid is ~8 to ~183 across the median, with axis
# intercepts going up to ~85-115 at 30 deg. Sweep a wide spread.
THRESHOLDS = [15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 130, 150]


def find_threshold(curve, angles, thresh):
    idx = np.searchsorted(curve, thresh)
    if idx == 0 or idx >= len(angles):
        return angles[min(idx, len(angles) - 1)]
    x0, x1 = angles[idx - 1], angles[idx]
    y0, y1 = curve[idx - 1], curve[idx]
    if y1 == y0:
        return x0
    return x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)


def make_panel(ax, angles, median_grid, thr, labels, show_legend=False):
    curve1 = median_grid[:, 0]
    curve2 = median_grid[0, :]
    theta1 = find_threshold(curve1, angles, thr)
    theta2 = find_threshold(curve2, angles, thr)

    pts = extract_contour_points(angles, median_grid, thr)
    if len(pts) == 0:
        ax.text(0.5, 0.5, f'no contour at L2={thr}', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        return None

    cn = pts.astype(float).copy()
    if theta1 > 0 and theta2 > 0:
        cn[:, 0] /= theta1
        cn[:, 1] /= theta2
    mask = (cn[:, 0] >= -0.02) & (cn[:, 1] >= -0.02)
    cn = cn[mask]

    avg_theta = (theta1 + theta2) / 2
    try:
        p_fit, ci, info = fit_superellipse(cn, avg_theta)
    except Exception:
        p_fit, ci = float('nan'), (float('nan'), float('nan'))

    xc, yc = superellipse_curve(2)
    xs, ys = superellipse_curve(20)
    ax.plot(xc, yc, '--', color='#1f77b4', lw=1.8, alpha=0.85,
            label='Euclidean ($p=2$)')
    ax.plot(xs, ys, '--', color='#d62728', lw=1.8, alpha=0.85,
            label=r'per-feature ($p\to\infty$)')
    ax.scatter(cn[:, 0], cn[:, 1], s=14, color='#2ca02c',
               alpha=0.85, edgecolors='none', label='data')
    ax.plot(1, 0, 'kv', markersize=8, zorder=5)
    ax.plot(0, 1, 'k>', markersize=8, zorder=5)

    ax.set_xlim(-0.05, 1.55)
    ax.set_ylim(-0.05, 1.55)
    ax.set_aspect('equal')
    ax.set_xlabel(f'$\\theta_1$ / $\\theta_1^*$  ({labels[0]})')
    ax.set_ylabel(f'$\\theta_2$ / $\\theta_2^*$  ({labels[1]})')
    ax.set_title(f'L2 threshold = {thr}\n'
                 f'$\\theta_1^*={theta1:.1f}^\\circ$, '
                 f'$\\theta_2^*={theta2:.1f}^\\circ$, '
                 f'fitted $p={p_fit:.2f}$', fontsize=11)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc='upper right', fontsize=9)
    return p_fit


def main():
    with open(PKL, 'rb') as f:
        res = pickle.load(f)
    angles = res['angles_deg']
    median_grid = np.median(res['l2'], axis=0)
    labels = res['direction_labels']
    print(f'Loaded {PKL}: {labels}, layer={res["layer"]}->{res["measure_layer"]}')

    os.makedirs(OUT_DIR, exist_ok=True)

    p_fits = {}
    for thr in THRESHOLDS:
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        p_fit = make_panel(ax, angles, median_grid, thr, labels,
                           show_legend=True)
        p_fits[thr] = p_fit
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'boundary_l2_{thr:03d}.png')
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'  thr={thr}: p={p_fit}')
        print(f'  saved {out}')

    # Overview grid: all thresholds in one figure.
    n = len(THRESHOLDS)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.2, nrow * 4.2))
    axes_flat = axes.flatten() if n > 1 else [axes]
    for ax, thr in zip(axes_flat, THRESHOLDS):
        make_panel(ax, angles, median_grid, thr, labels, show_legend=False)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(f'Gemma-2-9B {labels[0]} x {labels[1]} (L{res["layer"]} '
                 f'-> L{res["measure_layer"]}): boundary at varying L2 thresholds',
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(OUT_DIR, 'overview_grid.png')
    fig.savefig(out, dpi=120)
    print(f'  saved {out}')


if __name__ == '__main__':
    main()
