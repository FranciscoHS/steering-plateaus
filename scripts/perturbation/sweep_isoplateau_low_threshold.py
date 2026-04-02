#!/usr/bin/env python3
"""Low-threshold iso-plateau analysis.

Re-analyzes existing 2D sweep data at L2 thresholds approaching zero
(10, 5, 3, 1) to see how the superellipse exponent p converges.

Runs on all three feature pairs: Portuguese×Gender, Mandarin×Haskell, French×Rust.
No GPU needed — just loads existing pkl files.
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
    {
        'name': 'Portuguese × Gender',
        'pkl': os.path.join(RESULTS_DIR, 'sweep_isoplateau.pkl'),
        'prefix': 'isoplateau',
    },
    {
        'name': 'Mandarin × Haskell',
        'pkl': os.path.join(RESULTS_DIR, 'sweep_isoplateau_mandarin_haskell.pkl'),
        'prefix': 'isoplateau_mandarin_haskell',
    },
    {
        'name': 'French × Rust',
        'pkl': os.path.join(RESULTS_DIR, 'sweep_isoplateau_french_rust.pkl'),
        'prefix': 'isoplateau_french_rust',
    },
]

# Extended thresholds: original range + low values
L2_THRESHOLDS = np.array([1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100])
COSINE_THRESHOLDS = np.array([0.0005, 0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05])


def find_threshold(curve, angles, thresh):
    idx = np.searchsorted(curve, thresh)
    if idx == 0 or idx >= len(angles):
        return angles[idx] if idx < len(angles) else angles[-1]
    x0, x1 = angles[idx-1], angles[idx]
    y0, y1 = curve[idx-1], curve[idx]
    return x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)


def analyze_experiment(exp):
    """Run sensitivity analysis for one experiment at extended thresholds."""
    print(f"\n{'='*60}")
    print(f"  {exp['name']}")
    print(f"{'='*60}")

    with open(exp['pkl'], 'rb') as f:
        results = pickle.load(f)
    angles = results['angles_deg']

    thresholds = {'l2': L2_THRESHOLDS, 'cosine': COSINE_THRESHOLDS}
    results_by_metric = {}

    for metric in ['l2', 'cosine']:
        median_grid = np.median(results[metric], axis=0)
        curve_1 = median_grid[:, 0]
        curve_2 = median_grid[0, :]

        rows = []
        print(f"\n{metric.upper()} metric:")
        print(f"  {'threshold':>10s}  {'theta_1':>8s}  {'theta_2':>8s}  "
              f"{'p_fit':>6s}  {'CI':>14s}  {'mean_err':>8s}  {'n_pts':>5s}")

        for thresh in thresholds[metric]:
            theta_1 = find_threshold(curve_1, angles, thresh)
            theta_2 = find_threshold(curve_2, angles, thresh)

            # Skip if threshold is below minimum grid value
            if theta_1 <= angles[1] or theta_2 <= angles[1]:
                print(f"  {thresh:>10g}  θ too small (< grid resolution), skipping")
                continue

            contour_pts = extract_contour_points(angles, median_grid, thresh)
            if len(contour_pts) == 0:
                print(f"  {thresh:>10g}  no contour found, skipping")
                continue

            contour_norm = contour_pts.copy()
            contour_norm[:, 0] /= theta_1
            contour_norm[:, 1] /= theta_2

            avg_theta = (theta_1 + theta_2) / 2
            p_fit, ci, fit_info = fit_superellipse(contour_norm, avg_theta)
            if p_fit is None:
                print(f"  {thresh:>10g}  fit failed (too few points)")
                continue

            rows.append({
                'threshold': thresh, 'p_fit': p_fit,
                'ci_lo': ci[0], 'ci_hi': ci[1],
                'mean_err_deg': fit_info['mean_radial_deg'],
                'theta_1': theta_1, 'theta_2': theta_2,
                'n_pts': fit_info['n_pts'],
            })
            print(f"  {thresh:>10g}  {theta_1:>7.1f}°  {theta_2:>7.1f}°  "
                  f"{p_fit:>6.2f}  [{ci[0]:.2f}, {ci[1]:.2f}]  "
                  f"{fit_info['mean_radial_deg']:>7.2f}°  {fit_info['n_pts']:>5d}")

        results_by_metric[metric] = rows

    return results_by_metric


def plot_combined(all_results):
    """Plot sensitivity across all experiments, highlighting the low-threshold region."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {
        'Portuguese × Gender': '#e41a1c',
        'Mandarin × Haskell': '#377eb8',
        'French × Rust': '#4daf4a',
    }

    for metric_idx, metric in enumerate(['l2', 'cosine']):
        for name, results_by_metric in all_results.items():
            rows = results_by_metric[metric]
            if not rows:
                continue
            ts = [r['threshold'] for r in rows]
            ps = [r['p_fit'] for r in rows]
            ci_lo = [r['ci_lo'] for r in rows]
            ci_hi = [r['ci_hi'] for r in rows]
            errs = [r['mean_err_deg'] for r in rows]
            color = colors[name]

            ax = axes[0, metric_idx]
            ax.fill_between(ts, ci_lo, ci_hi, color=color, alpha=0.15)
            ax.plot(ts, ps, 'o-', color=color, lw=2, markersize=5, label=name)

            ax = axes[1, metric_idx]
            ax.plot(ts, errs, 's-', color=color, lw=2, markersize=5, label=name)

    metric_labels = {'l2': 'L2 threshold', 'cosine': 'Cosine threshold'}
    for col, metric in enumerate(['l2', 'cosine']):
        axes[0, col].set_ylabel('Superellipse exponent p', fontsize=11)
        axes[0, col].set_title(metric_labels[metric], fontsize=12, fontweight='bold')
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].legend(fontsize=8)
        # Reference lines
        axes[0, col].axhline(y=1, color='gray', ls=':', lw=1, alpha=0.5)
        axes[0, col].axhline(y=2, color='gray', ls=':', lw=1, alpha=0.5)

        axes[1, col].set_xlabel(metric_labels[metric], fontsize=11)
        axes[1, col].set_ylabel('Mean radial error (degrees)', fontsize=11)
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].legend(fontsize=8)

    fig.suptitle('Sensitivity to threshold (extended to low values)\n'
                 'Gemma-2-9B · layer 20 · median over 30 prompts',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, 'isoplateau_low_threshold_sensitivity.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {save_path}")
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white')
    plt.close()


def plot_low_threshold_shapes(exp):
    """Plot superellipse shapes at low L2 thresholds for one experiment."""
    with open(exp['pkl'], 'rb') as f:
        results = pickle.load(f)
    angles = results['angles_deg']
    median_grid = np.median(results['l2'], axis=0)
    curve_1 = median_grid[:, 0]
    curve_2 = median_grid[0, :]

    low_thresholds = [1, 3, 5, 10]
    fig, axes = plt.subplots(1, len(low_thresholds), figsize=(5*len(low_thresholds), 5))

    for ax_idx, thresh in enumerate(low_thresholds):
        ax = axes[ax_idx]
        theta_1 = find_threshold(curve_1, angles, thresh)
        theta_2 = find_threshold(curve_2, angles, thresh)

        if theta_1 <= angles[1] or theta_2 <= angles[1]:
            ax.text(0.5, 0.5, f'L2={thresh}\nθ below grid\nresolution',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_aspect('equal')
            continue

        contour_pts = extract_contour_points(angles, median_grid, thresh)
        if len(contour_pts) == 0:
            ax.text(0.5, 0.5, f'L2={thresh}\nno contour', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12)
            ax.set_aspect('equal')
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
        ax.plot(xs, ys, '--', color='#bbbbbb', lw=1, label='p→∞')

        if p_fit is not None:
            xf, yf = superellipse_curve(p_fit)
            ax.plot(xf, yf, '-', color='#e41a1c', lw=2.5,
                    label=f'p={p_fit:.2f}')

        contour_angle = np.arctan2(contour_norm[:, 1], contour_norm[:, 0])
        order = np.argsort(contour_angle)
        ax.plot(contour_norm[order, 0], contour_norm[order, 1], 'o',
                color='#377eb8', markersize=2, alpha=0.7)

        ax.set_title(f'L2 = {thresh}\nθ₁={theta_1:.1f}°, θ₂={theta_2:.1f}°',
                     fontsize=11)
        ax.set_xlim(-0.05, 1.5)
        ax.set_ylim(-0.05, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f'Low-Threshold Shapes: {exp["name"]}\nGemma-2-9B · layer 20',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'{exp["prefix"]}_low_threshold_shapes.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white')
    plt.close()


def main():
    all_results = {}
    for exp in EXPERIMENTS:
        results_by_metric = analyze_experiment(exp)
        all_results[exp['name']] = results_by_metric

    plot_combined(all_results)

    # Also plot shape panels at low thresholds for each experiment
    for exp in EXPERIMENTS:
        plot_low_threshold_shapes(exp)


if __name__ == '__main__':
    main()
