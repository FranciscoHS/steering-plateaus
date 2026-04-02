#!/usr/bin/env python3
"""Plot best-fit superellipse exponent p vs cosine threshold for all feature pairs.

Single plot with one curve per pair, including bootstrap CIs.
No GPU needed.
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse)

RESULTS_DIR = 'results/mandarin_haskell/gemma-2-9b'

EXPERIMENTS = [
    ('Portuguese × Gender', os.path.join(RESULTS_DIR, 'sweep_isoplateau.pkl')),
    ('Mandarin × Haskell', os.path.join(RESULTS_DIR, 'sweep_isoplateau_mandarin_haskell.pkl')),
    ('French × Rust', os.path.join(RESULTS_DIR, 'sweep_isoplateau_french_rust.pkl')),
    ('French × C++', os.path.join(RESULTS_DIR, 'sweep_isoplateau_french_cpp.pkl')),
    ('Random × Random', os.path.join(RESULTS_DIR, 'sweep_isoplateau_random.pkl')),
]

COSINE_THRESHOLDS = np.array([
    0.0002, 0.0005, 0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05])

COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']


def find_threshold(curve, angles, thresh):
    idx = np.searchsorted(curve, thresh)
    if idx == 0 or idx >= len(angles):
        return angles[idx] if idx < len(angles) else angles[-1]
    x0, x1 = angles[idx-1], angles[idx]
    y0, y1 = curve[idx-1], curve[idx]
    return x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)


def main():
    fig, ax = plt.subplots(figsize=(8, 5))

    for (name, pkl_path), color in zip(EXPERIMENTS, COLORS):
        if not os.path.exists(pkl_path):
            print(f"Skipping {name}: {pkl_path} not found")
            continue

        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        angles = results['angles_deg']
        median_grid = np.median(results['cosine'], axis=0)
        curve_1 = median_grid[:, 0]
        curve_2 = median_grid[0, :]
        noise = median_grid[0, 0]

        ts, ps, ci_lo, ci_hi = [], [], [], []

        for thresh in COSINE_THRESHOLDS:
            if thresh <= noise * 2:
                continue

            theta_1 = find_threshold(curve_1, angles, thresh)
            theta_2 = find_threshold(curve_2, angles, thresh)

            if theta_1 <= angles[1] or theta_2 <= angles[1]:
                continue

            # Skip if either theta hits the grid max (ceiling artifact)
            if theta_1 >= angles[-1] * 0.98 or theta_2 >= angles[-1] * 0.98:
                continue

            contour_pts = extract_contour_points(angles, median_grid, thresh)
            if len(contour_pts) == 0:
                continue

            contour_norm = contour_pts.copy()
            contour_norm[:, 0] /= theta_1
            contour_norm[:, 1] /= theta_2

            avg_theta = (theta_1 + theta_2) / 2
            p_fit, ci, fit_info = fit_superellipse(contour_norm, avg_theta)
            if p_fit is None:
                continue

            ts.append(thresh)
            ps.append(p_fit)
            ci_lo.append(ci[0])
            ci_hi.append(ci[1])

        if ts:
            ax.fill_between(ts, ci_lo, ci_hi, color=color, alpha=0.15)
            ax.plot(ts, ps, 'o-', color=color, lw=2, markersize=5, label=name)

    ax.axhline(y=1, color='gray', ls=':', lw=1, alpha=0.5)
    ax.axhline(y=2, color='gray', ls=':', lw=1, alpha=0.5)
    ax.text(COSINE_THRESHOLDS[-1] * 1.02, 1, 'p=1', fontsize=8, color='gray', va='center')
    ax.text(COSINE_THRESHOLDS[-1] * 1.02, 2, 'p=2', fontsize=8, color='gray', va='center')

    ax.set_xscale('log')
    ax.set_xlabel('Cosine distance threshold', fontsize=12)
    ax.set_ylabel('Superellipse exponent p', fontsize=12)
    ax.set_title('Superellipse exponent vs cosine threshold\n'
                 'Gemma-2-9B · layer 20 · median over 30 prompts',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'isoplateau_p_vs_cosine_threshold.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f'Saved: {save_path}')


if __name__ == '__main__':
    main()
