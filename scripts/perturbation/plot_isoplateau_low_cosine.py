#!/usr/bin/env python3
"""Plot superellipse shape fits at low cosine thresholds.

Loads existing 2D sweep data for all three feature pairs and generates
shape-fit panels at cosine thresholds approaching the noise floor.
No GPU needed.
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse, superellipse_curve)

RESULTS_DIR = 'results/mandarin_haskell/gemma-2-9b'

EXPERIMENTS = [
    ('Portuguese x Gender', os.path.join(RESULTS_DIR, 'sweep_isoplateau.pkl'), 'pt_gender'),
    ('Mandarin x Haskell', os.path.join(RESULTS_DIR, 'sweep_isoplateau_mandarin_haskell.pkl'), 'mand_hask'),
    ('French x Rust', os.path.join(RESULTS_DIR, 'sweep_isoplateau_french_rust.pkl'), 'fr_rust'),
]

COSINE_THRESHOLDS = [0.0002, 0.0005, 0.001, 0.002, 0.005]


def find_threshold(curve, angles, thresh):
    idx = np.searchsorted(curve, thresh)
    if idx == 0 or idx >= len(angles):
        return angles[idx] if idx < len(angles) else angles[-1]
    x0, x1 = angles[idx-1], angles[idx]
    y0, y1 = curve[idx-1], curve[idx]
    return x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)


def plot_experiment(name, pkl_path, prefix):
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    angles = results['angles_deg']
    median_grid = np.median(results['cosine'], axis=0)
    curve_1 = median_grid[:, 0]
    curve_2 = median_grid[0, :]
    noise = median_grid[0, 0]

    n_thresh = len(COSINE_THRESHOLDS)
    fig, axes = plt.subplots(1, n_thresh, figsize=(5 * n_thresh, 5))

    for ax_idx, thresh in enumerate(COSINE_THRESHOLDS):
        ax = axes[ax_idx]

        if thresh <= noise * 2:
            ax.text(0.5, 0.5, f'cos={thresh}\nbelow noise floor',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_aspect('equal')
            ax.set_xlim(-0.05, 1.5)
            ax.set_ylim(-0.05, 1.5)
            continue

        theta_1 = find_threshold(curve_1, angles, thresh)
        theta_2 = find_threshold(curve_2, angles, thresh)

        if theta_1 <= angles[1] or theta_2 <= angles[1]:
            ax.text(0.5, 0.5, f'cos={thresh}\nbelow grid res',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_aspect('equal')
            ax.set_xlim(-0.05, 1.5)
            ax.set_ylim(-0.05, 1.5)
            continue

        contour_pts = extract_contour_points(angles, median_grid, thresh)
        if len(contour_pts) == 0:
            ax.text(0.5, 0.5, f'cos={thresh}\nno contour',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_aspect('equal')
            ax.set_xlim(-0.05, 1.5)
            ax.set_ylim(-0.05, 1.5)
            continue

        contour_norm = contour_pts.copy()
        contour_norm[:, 0] /= theta_1
        contour_norm[:, 1] /= theta_2

        avg_theta = (theta_1 + theta_2) / 2
        p_fit, ci, fit_info = fit_superellipse(contour_norm, avg_theta)

        # Reference shapes
        x1 = 1 - np.linspace(0, 1, 200)
        y1 = np.linspace(0, 1, 200)
        ax.plot(x1, y1, '--', color='#aaaaaa', lw=1, label='p=1')
        xc, yc = superellipse_curve(2)
        ax.plot(xc, yc, '--', color='#888888', lw=1, label='p=2')
        xs, ys = superellipse_curve(20)
        ax.plot(xs, ys, '--', color='#bbbbbb', lw=1, label='p=inf')

        if p_fit is not None:
            xf, yf = superellipse_curve(p_fit)
            ax.plot(xf, yf, '-', color='#e41a1c', lw=2.5)
            ax.text(0.03, 0.97,
                    f'p={p_fit:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]\n'
                    f'err={fit_info["mean_radial_deg"]:.2f} mean, '
                    f'{fit_info["max_radial_deg"]:.2f} max\n'
                    f'({fit_info["n_pts"]} pts)',
                    transform=ax.transAxes, fontsize=8, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        contour_angle = np.arctan2(contour_norm[:, 1], contour_norm[:, 0])
        order = np.argsort(contour_angle)
        ax.plot(contour_norm[order, 0], contour_norm[order, 1], 'o',
                color='#377eb8', markersize=3, alpha=0.7)

        ax.set_title(f'cos = {thresh}\n'
                     f'theta1={theta_1:.1f}, theta2={theta_2:.1f}',
                     fontsize=11)
        ax.set_xlim(-0.05, 1.5)
        ax.set_ylim(-0.05, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc='lower left')

    fig.suptitle(f'Low Cosine Threshold Fits: {name}\n'
                 f'Gemma-2-9B, layer 20',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR,
                             f'isoplateau_low_cosine_shapes_{prefix}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f'Saved: {save_path}')


def main():
    for name, pkl_path, prefix in EXPERIMENTS:
        plot_experiment(name, pkl_path, prefix)


if __name__ == '__main__':
    main()
