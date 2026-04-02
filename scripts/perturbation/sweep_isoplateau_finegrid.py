#!/usr/bin/env python3
"""Fine-grid iso-plateau sweep for Portuguese × Gender at small angles.

Same as sweep_isoplateau.py but with angles 0.0 to 1.0° in steps of 0.1°,
to resolve the iso-plateau shape at very low L2 thresholds (1, 3, 5, 10).
"""
import os, sys, pickle, argparse
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks
from scripts.perturbation.generate_lesswrong_figures import (
    extract_activations_holdout,
)
from scripts.perturbation.sweep_isoplateau import sweep_2d
from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse, superellipse_curve)

# --- Config ---
N_ACT = 30
SEED = 123
LAYER = 20
MAX_ANGLE = 0.1    # degrees
N_STEPS = 21       # 0.0, 0.005, 0.01, ..., 0.1

RESULTS_DIR = 'results/mandarin_haskell/gemma-2-9b'
DIRECTIONS_PT = os.path.join(RESULTS_DIR, 'directions_portuguese_L20.pkl')
DIRECTIONS_GENDER = os.path.join(RESULTS_DIR, 'directions_gender_L20.pkl')
OUTPUT_PKL = os.path.join(RESULTS_DIR, 'sweep_isoplateau_finegrid2.pkl')
PREFIX = 'isoplateau_finegrid2'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()

    if not args.plot_only:
        run_sweep()
    plot_results()
    plot_shape()
    plot_sensitivity()


def run_sweep():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading directions...")
    with open(DIRECTIONS_PT, 'rb') as f:
        pt_data = pickle.load(f)
    with open(DIRECTIONS_GENDER, 'rb') as f:
        gen_data = pickle.load(f)

    d_portuguese = pt_data['d_portuguese']
    d_to_female = gen_data['d_to_female']

    print("Loading Gemma-2-9B...")
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    print(f"  n_layers={n_layers}, perturb@L{LAYER}, measure@L{measure_layer}")

    from data.male_prompts_holdout import MALE_PROMPTS_HOLDOUT
    holdout = MALE_PROMPTS_HOLDOUT[:N_ACT]
    print(f"Extracting activations from {len(holdout)} male holdout prompts...")
    contexts, activations = extract_activations_holdout(
        model, tokenizer, device, LAYER, holdout)

    angles_deg = np.linspace(0, MAX_ANGLE, N_STEPS)
    angles_rad = np.deg2rad(angles_deg)
    print(f"Grid: {N_STEPS}x{N_STEPS} = {N_STEPS**2} points per activation, "
          f"0-{MAX_ANGLE}° in {angles_deg[1]-angles_deg[0]:.2f}° steps")

    all_l2, all_cos = [], []
    for i in range(N_ACT):
        l2_grid, cos_grid = sweep_2d(
            model, contexts[i], activations[i],
            d_portuguese, d_to_female,
            LAYER, measure_layer, device, angles_rad, angles_rad)
        all_l2.append(l2_grid)
        all_cos.append(cos_grid)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{N_ACT}")

    results = {
        'angles_deg': angles_deg,
        'l2': np.stack(all_l2),
        'cosine': np.stack(all_cos),
        'layer': LAYER,
        'measure_layer': measure_layer,
        'direction_labels': ('Portuguese', 'Gender'),
    }
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {OUTPUT_PKL}")


def find_threshold(curve, angles, thresh):
    idx = np.searchsorted(curve, thresh)
    if idx == 0 or idx >= len(angles):
        return angles[idx] if idx < len(angles) else angles[-1]
    x0, x1 = angles[idx-1], angles[idx]
    y0, y1 = curve[idx-1], curve[idx]
    return x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)


def plot_results():
    with open(OUTPUT_PKL, 'rb') as f:
        results = pickle.load(f)
    angles = results['angles_deg']

    for metric, threshold, label, units in [
        ('l2', 0.5, 'L2 distance', 'L2=0.5'),
        ('l2', 1.0, 'L2 distance', 'L2=1'),
        ('l2', 3.0, 'L2 distance', 'L2=3'),
        ('l2', 5.0, 'L2 distance', 'L2=5'),
    ]:
        data = results[metric]
        median_grid = np.median(data, axis=0)

        fig, ax = plt.subplots(figsize=(7, 6))
        cf = ax.contourf(angles, angles, median_grid.T, levels=50, cmap='viridis')
        plt.colorbar(cf, ax=ax, label=label)
        cs = ax.contour(angles, angles, median_grid.T, levels=[threshold],
                        colors='red', linewidths=2.5)
        ax.clabel(cs, fmt=f'{threshold}', fontsize=10, colors='red')

        ax.set_xlabel('Portuguese angle α₁ (degrees)', fontsize=12)
        ax.set_ylabel('Gender angle α₂ (degrees)', fontsize=12)
        ax.set_title(f'Iso-Plateau Line ({units}) — Fine Grid\n'
                     f'Gemma-2-9B · layer 20 · median over {data.shape[0]} prompts',
                     fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        plt.tight_layout()
        png_path = os.path.join(RESULTS_DIR,
                                f'{PREFIX}_{metric}_{int(threshold)}.png')
        plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {png_path}")
        plt.savefig(png_path.replace('.png', '.pdf'), bbox_inches='tight',
                    facecolor='white')
        plt.close()


def plot_shape():
    with open(OUTPUT_PKL, 'rb') as f:
        results = pickle.load(f)
    angles = results['angles_deg']

    for metric, threshold, label, units in [
        ('l2', 0.5, 'L2 distance', 'L2=0.5'),
        ('l2', 1.0, 'L2 distance', 'L2=1'),
        ('l2', 3.0, 'L2 distance', 'L2=3'),
        ('l2', 5.0, 'L2 distance', 'L2=5'),
    ]:
        median_grid = np.median(results[metric], axis=0)
        theta_1 = find_threshold(median_grid[:, 0], angles, threshold)
        theta_2 = find_threshold(median_grid[0, :], angles, threshold)

        if theta_1 <= angles[1] or theta_2 <= angles[1]:
            print(f"  L2={threshold}: θ below grid resolution, skipping shape")
            continue

        contour_pts = extract_contour_points(angles, median_grid, threshold)
        if len(contour_pts) == 0:
            print(f"  No contour found for {metric} at {threshold}")
            continue

        contour_norm = contour_pts.copy()
        contour_norm[:, 0] /= theta_1
        contour_norm[:, 1] /= theta_2

        avg_theta = (theta_1 + theta_2) / 2
        p_fit, ci, fit_info = fit_superellipse(contour_norm, avg_theta)

        fig, ax = plt.subplots(figsize=(6, 6))

        x1 = 1 - np.linspace(0, 1, 200)
        y1 = np.linspace(0, 1, 200)
        ax.plot(x1, y1, '--', color='#aaaaaa', lw=1.5, label='p=1 (diamond)')
        xc, yc = superellipse_curve(2)
        ax.plot(xc, yc, '--', color='#888888', lw=1.5, label='p=2 (circle)')
        xs, ys = superellipse_curve(20)
        ax.plot(xs, ys, '--', color='#bbbbbb', lw=1.5, label='p→∞ (square)')

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

        contour_angle = np.arctan2(contour_norm[:, 1], contour_norm[:, 0])
        order = np.argsort(contour_angle)
        ax.plot(contour_norm[order, 0], contour_norm[order, 1], 'o',
                color='#377eb8', markersize=3, alpha=0.7, label='data')

        ax.set_xlabel('α₁ / θ₁*  (Portuguese, normalized)', fontsize=12)
        ax.set_ylabel('α₂ / θ₂*  (Gender, normalized)', fontsize=12)
        ax.set_title(f'Iso-Plateau Shape ({units}) — Fine Grid\n'
                     f'θ_PT*={theta_1:.2f}°, θ_gen*={theta_2:.2f}°',
                     fontsize=13, fontweight='bold')
        ax.set_xlim(-0.05, 1.5)
        ax.set_ylim(-0.05, 1.5)
        ax.set_aspect('equal')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.plot(1, 0, 'kv', markersize=8, zorder=5)
        ax.plot(0, 1, 'k>', markersize=8, zorder=5)

        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR,
                                 f'{PREFIX}_shape_{metric}_{int(threshold)}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight',
                    facecolor='white')
        plt.close()


def plot_sensitivity():
    """Print sensitivity table for low L2 thresholds."""
    with open(OUTPUT_PKL, 'rb') as f:
        results = pickle.load(f)
    angles = results['angles_deg']

    l2_thresholds = np.array([0.1, 0.2, 0.5, 1, 2, 3, 5])

    median_grid = np.median(results['l2'], axis=0)
    curve_1 = median_grid[:, 0]
    curve_2 = median_grid[0, :]

    print(f"\nFine-grid L2 sensitivity (Portuguese × Gender):")
    print(f"  {'threshold':>10s}  {'theta_1':>8s}  {'theta_2':>8s}  "
          f"{'p_fit':>6s}  {'CI':>14s}  {'mean_err':>8s}  {'n_pts':>5s}")

    for thresh in l2_thresholds:
        theta_1 = find_threshold(curve_1, angles, thresh)
        theta_2 = find_threshold(curve_2, angles, thresh)

        if theta_1 <= angles[1] or theta_2 <= angles[1]:
            print(f"  {thresh:>10g}  θ too small, skipping")
            continue

        contour_pts = extract_contour_points(angles, median_grid, thresh)
        if len(contour_pts) == 0:
            print(f"  {thresh:>10g}  no contour")
            continue

        contour_norm = contour_pts.copy()
        contour_norm[:, 0] /= theta_1
        contour_norm[:, 1] /= theta_2

        avg_theta = (theta_1 + theta_2) / 2
        p_fit, ci, fit_info = fit_superellipse(contour_norm, avg_theta)
        if p_fit is None:
            print(f"  {thresh:>10g}  fit failed")
            continue

        print(f"  {thresh:>10g}  {theta_1:>7.2f}°  {theta_2:>7.2f}°  "
              f"{p_fit:>6.2f}  [{ci[0]:.2f}, {ci[1]:.2f}]  "
              f"{fit_info['mean_radial_deg']:>7.3f}°  {fit_info['n_pts']:>5d}")


if __name__ == '__main__':
    main()
