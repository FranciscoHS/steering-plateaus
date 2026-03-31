#!/usr/bin/env python3
"""Generate superellipse shape plots at the threshold with lowest mean radial error."""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse, superellipse_curve)

RESULTS_DIR = 'results/mandarin_haskell/gemma-2-9b'

THRESHOLDS = {
    'l2': np.array([20, 30, 40, 50, 60, 70, 80, 100]),
    'cosine': np.array([0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]),
}

EXPERIMENTS = [
    {
        'name': 'PT × Gender',
        'pkl': os.path.join(RESULTS_DIR, 'sweep_isoplateau.pkl'),
        'label_1': 'Portuguese', 'label_2': 'Gender',
        'prefix': 'isoplateau_shape_best',
    },
    {
        'name': 'Mandarin × Haskell',
        'pkl': os.path.join(RESULTS_DIR, 'sweep_isoplateau_mandarin_haskell.pkl'),
        'label_1': 'Mandarin', 'label_2': 'Haskell',
        'prefix': 'isoplateau_shape_mandarin_haskell_best',
    },
    {
        'name': 'French × Rust',
        'pkl': os.path.join(RESULTS_DIR, 'sweep_isoplateau_french_rust.pkl'),
        'label_1': 'French', 'label_2': 'Rust',
        'prefix': 'isoplateau_french_rust_shape_best',
    },
]


def find_threshold(curve, angles, thresh):
    idx = np.searchsorted(curve, thresh)
    if idx == 0 or idx >= len(angles):
        return angles[idx] if idx < len(angles) else angles[-1]
    x0, x1 = angles[idx-1], angles[idx]
    y0, y1 = curve[idx-1], curve[idx]
    return x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)


def find_best_threshold(angles, median_grid, thresholds_arr):
    """Find the threshold with lowest mean radial error."""
    best_err, best_thresh = 999, None
    best_result = None
    for thresh in thresholds_arr:
        t1 = find_threshold(median_grid[:, 0], angles, thresh)
        t2 = find_threshold(median_grid[0, :], angles, thresh)
        pts = extract_contour_points(angles, median_grid, thresh)
        if len(pts) == 0:
            continue
        cn = pts.copy()
        cn[:, 0] /= t1
        cn[:, 1] /= t2
        avg_theta = (t1 + t2) / 2
        p, ci, info = fit_superellipse(cn, avg_theta)
        if p is not None and info['mean_radial_deg'] < best_err:
            best_err = info['mean_radial_deg']
            best_thresh = thresh
            best_result = (t1, t2, cn, p, ci, info)
    return best_thresh, best_result


def plot_shape(label_1, label_2, metric, threshold, theta_1, theta_2,
               contour_norm, p_fit, ci, fit_info, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Reference shapes
    x1 = 1 - np.linspace(0, 1, 200)
    y1 = np.linspace(0, 1, 200)
    ax.plot(x1, y1, '--', color='#aaaaaa', lw=1.5, label='p=1 (diamond)')
    xc, yc = superellipse_curve(2)
    ax.plot(xc, yc, '--', color='#888888', lw=1.5, label='p=2 (circle)')
    xs, ys = superellipse_curve(20)
    ax.plot(xs, ys, '--', color='#bbbbbb', lw=1.5, label='p→∞ (square)')

    # Fitted curve
    if p_fit is not None:
        xf, yf = superellipse_curve(p_fit)
        ax.plot(xf, yf, '-', color='#e41a1c', lw=2.5,
                label=f'fit: p={p_fit:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]')
        ax.text(0.03, 0.97,
                f'radial error: {fit_info["mean_radial_deg"]:.2f}° mean, '
                f'{fit_info["max_radial_deg"]:.2f}° max\n'
                f'({fit_info["n_pts"]} contour pts)',
                transform=ax.transAxes, fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    # Data points
    contour_angle = np.arctan2(contour_norm[:, 1], contour_norm[:, 0])
    order = np.argsort(contour_angle)
    ax.plot(contour_norm[order, 0], contour_norm[order, 1], 'o',
            color='#377eb8', markersize=3, alpha=0.7, label='data')

    units = f'L2={threshold}' if metric == 'l2' else f'cos={threshold}'
    ax.set_xlabel(f'α₁ / θ₁*  ({label_1}, normalized)', fontsize=12)
    ax.set_ylabel(f'α₂ / θ₂*  ({label_2}, normalized)', fontsize=12)
    ax.set_title(f'Iso-Plateau Shape ({units})\n'
                 f'θ_{label_1}*={theta_1:.1f}°, θ_{label_2}*={theta_2:.1f}°',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-0.05, 1.5)
    ax.set_ylim(-0.05, 1.5)
    ax.set_aspect('equal')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.plot(1, 0, 'kv', markersize=8, zorder=5)
    ax.plot(0, 1, 'k>', markersize=8, zorder=5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white')
    plt.close()


def main():
    for exp in EXPERIMENTS:
        with open(exp['pkl'], 'rb') as f:
            results = pickle.load(f)
        angles = results['angles_deg']

        print(f"\n{exp['name']}:")
        for metric in ['l2', 'cosine']:
            median_grid = np.median(results[metric], axis=0)
            best_thresh, best = find_best_threshold(
                angles, median_grid, THRESHOLDS[metric])
            if best is None:
                print(f"  {metric}: no valid fit found")
                continue
            t1, t2, cn, p, ci, info = best
            print(f"  {metric}: threshold={best_thresh}, p={p:.2f} "
                  f"[{ci[0]:.2f}, {ci[1]:.2f}], "
                  f"mean_err={info['mean_radial_deg']:.3f}°")

            save_path = os.path.join(
                RESULTS_DIR, f'{exp["prefix"]}_{metric}.png')
            plot_shape(exp['label_1'], exp['label_2'], metric, best_thresh,
                       t1, t2, cn, p, ci, info, save_path)


if __name__ == '__main__':
    main()
