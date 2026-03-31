#!/usr/bin/env python3
"""Sensitivity of superellipse fit to threshold choice.

Sweeps over a range of thresholds for both L2 and cosine metrics,
fits the superellipse at each, and plots p and mean radial error vs threshold.
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_isoplateau import extract_contour_points, fit_superellipse

RESULTS_DIR = 'results/mandarin_haskell/gemma-2-9b'
INPUT_PKL = os.path.join(RESULTS_DIR, 'sweep_isoplateau.pkl')

THRESHOLDS = {
    'l2': np.array([20, 30, 40, 50, 60, 70, 80, 100]),
    'cosine': np.array([0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]),
}

# Default thresholds used in main plots (for vertical line marker)
DEFAULT_THRESH = {'l2': 50.0, 'cosine': 0.01}


def find_threshold(curve, angles, thresh):
    """Interpolate the angle where curve crosses thresh."""
    idx = np.searchsorted(curve, thresh)
    if idx == 0 or idx >= len(angles):
        return angles[idx] if idx < len(angles) else angles[-1]
    x0, x1 = angles[idx-1], angles[idx]
    y0, y1 = curve[idx-1], curve[idx]
    return x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)


def sweep_thresholds(angles, median_grid, thresholds):
    """Fit superellipse at each threshold, return results dict."""
    pt_curve = median_grid[:, 0]
    gen_curve = median_grid[0, :]

    rows = []
    for thresh in thresholds:
        theta_pt = find_threshold(pt_curve, angles, thresh)
        theta_gen = find_threshold(gen_curve, angles, thresh)

        contour_pts = extract_contour_points(angles, median_grid, thresh)
        if len(contour_pts) == 0:
            continue

        contour_norm = contour_pts.copy()
        contour_norm[:, 0] /= theta_pt
        contour_norm[:, 1] /= theta_gen

        avg_theta = (theta_pt + theta_gen) / 2
        p_fit, ci, fit_info = fit_superellipse(contour_norm, avg_theta)
        if p_fit is None:
            continue

        rows.append({
            'threshold': thresh,
            'theta_pt': theta_pt,
            'theta_gen': theta_gen,
            'p_fit': p_fit,
            'ci_lo': ci[0],
            'ci_hi': ci[1],
            'mean_err_deg': fit_info['mean_radial_deg'],
            'max_err_deg': fit_info['max_radial_deg'],
            'n_pts': fit_info['n_pts'],
        })

    return rows


def plot_sensitivity(results_by_metric):
    """Plot p_fit and mean radial error vs threshold for both metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    metric_labels = {'l2': 'L2 distance', 'cosine': 'Cosine distance'}
    colors = {'l2': '#2166ac', 'cosine': '#c51b7d'}

    for col, metric in enumerate(['l2', 'cosine']):
        rows = results_by_metric[metric]
        if not rows:
            continue

        threshs = [r['threshold'] for r in rows]
        p_fits = [r['p_fit'] for r in rows]
        ci_lo = [r['ci_lo'] for r in rows]
        ci_hi = [r['ci_hi'] for r in rows]
        mean_errs = [r['mean_err_deg'] for r in rows]
        color = colors[metric]
        default_t = DEFAULT_THRESH[metric]

        # Top row: p_fit vs threshold
        ax = axes[0, col]
        ax.fill_between(threshs, ci_lo, ci_hi, color=color, alpha=0.2)
        ax.plot(threshs, p_fits, 'o-', color=color, lw=2, markersize=5)
        ax.set_ylabel('Superellipse exponent p', fontsize=11)
        ax.set_title(f'{metric_labels[metric]}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Bottom row: mean radial error vs threshold
        ax = axes[1, col]
        ax.plot(threshs, mean_errs, 's-', color=color, lw=2, markersize=5)
        ax.set_xlabel('Threshold', fontsize=11)
        ax.set_ylabel('Mean radial error (degrees)', fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Sensitivity of Superellipse Fit to Threshold Choice\n'
                 'Gemma-2-9B · layer 20 · median over 30 prompts',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, 'isoplateau_sensitivity.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white')
    plt.close()


def main():
    with open(INPUT_PKL, 'rb') as f:
        results = pickle.load(f)

    angles = results['angles_deg']
    results_by_metric = {}

    for metric in ['l2', 'cosine']:
        median_grid = np.median(results[metric], axis=0)
        thresholds = THRESHOLDS[metric]
        rows = sweep_thresholds(angles, median_grid, thresholds)
        results_by_metric[metric] = rows

        print(f"\n{metric.upper()} metric:")
        print(f"  {'threshold':>10s}  {'theta_pt':>8s}  {'theta_gen':>8s}  "
              f"{'p_fit':>6s}  {'CI':>14s}  {'mean_err':>8s}  {'n_pts':>5s}")
        for r in rows:
            print(f"  {r['threshold']:>10g}  {r['theta_pt']:>7.1f}°  "
                  f"{r['theta_gen']:>7.1f}°  {r['p_fit']:>6.2f}  "
                  f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]  "
                  f"{r['mean_err_deg']:>7.2f}°  {r['n_pts']:>5d}")

    plot_sensitivity(results_by_metric)


if __name__ == '__main__':
    main()
