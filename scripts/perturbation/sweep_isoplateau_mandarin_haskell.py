#!/usr/bin/env python3
"""Iso-plateau sweep for Mandarin × Haskell composition on Gemma-2-9B.

Extracts contrastive directions (EN→ZH Mandarin and Python→Haskell) at layer 20,
then runs a 2D iso-plateau sweep, fits superellipse, and runs sensitivity analysis.

Uses language-agnostic coding prompts as the holdout set (English text about
programming — sits in both domains).
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
    compute_contrastive_direction,
    extract_activations_holdout,
)
from scripts.perturbation.sweep_isoplateau import sweep_2d

# --- Config ---
N_ACT = 30
SEED = 123
LAYER = 20
MAX_ANGLE = 20.0
N_STEPS = 41

RESULTS_DIR = 'results/mandarin_haskell/gemma-2-9b'
DIRECTIONS_PKL = os.path.join(RESULTS_DIR, 'directions_mandarin_haskell_L20.pkl')
OUTPUT_PKL = os.path.join(RESULTS_DIR, 'sweep_isoplateau_mandarin_haskell.pkl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true',
                        help='Only plot from existing pkl data')
    parser.add_argument('--directions-only', action='store_true',
                        help='Only extract directions, skip sweep')
    args = parser.parse_args()

    if not args.plot_only:
        run_sweep(args.directions_only)
    plot_results()
    plot_shape()
    plot_sensitivity()


def extract_directions(model, tokenizer, device):
    """Extract Mandarin and Haskell directions at LAYER via diff-of-means."""
    from data.bilingual_prompts import ENGLISH_PROMPTS, CHINESE_PROMPTS
    from data.programming_prompts import PROMPT_SETS, LANGUAGES

    print("Extracting Mandarin direction (EN→ZH)...")
    d_mandarin = compute_contrastive_direction(
        model, tokenizer, device, LAYER,
        ENGLISH_PROMPTS, CHINESE_PROMPTS)

    print("Extracting Haskell direction (Python→Haskell)...")
    python_prompts = [ps['python'] for ps in PROMPT_SETS]
    haskell_prompts = [ps['haskell'] for ps in PROMPT_SETS]
    d_haskell = compute_contrastive_direction(
        model, tokenizer, device, LAYER,
        python_prompts, haskell_prompts)

    cos_sim = (d_mandarin @ d_haskell).item()
    print(f"  cos(Mandarin, Haskell) = {cos_sim:.4f}")

    dir_data = {
        'd_mandarin': d_mandarin,
        'd_haskell': d_haskell,
        'cosine_similarity': cos_sim,
        'layer': LAYER,
    }
    with open(DIRECTIONS_PKL, 'wb') as f:
        pickle.dump(dir_data, f)
    print(f"Saved: {DIRECTIONS_PKL}")

    return d_mandarin, d_haskell


def run_sweep(directions_only=False):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading Gemma-2-9B...")
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    print(f"  n_layers={n_layers}, perturb@L{LAYER}, measure@L{measure_layer}")

    # Extract or load directions
    if os.path.exists(DIRECTIONS_PKL):
        print(f"Loading existing directions from {DIRECTIONS_PKL}")
        with open(DIRECTIONS_PKL, 'rb') as f:
            dir_data = pickle.load(f)
        d_mandarin = dir_data['d_mandarin']
        d_haskell = dir_data['d_haskell']
        print(f"  cos(Mandarin, Haskell) = {dir_data['cosine_similarity']:.4f}")
    else:
        d_mandarin, d_haskell = extract_directions(model, tokenizer, device)

    if directions_only:
        return

    # Holdout activations: language-agnostic coding prompts
    from data.coding_prompts_agnostic import CODING_PROMPTS
    holdout = CODING_PROMPTS[:N_ACT]
    print(f"\nExtracting activations from {len(holdout)} coding-agnostic holdout prompts...")
    contexts, activations = extract_activations_holdout(
        model, tokenizer, device, LAYER, holdout)

    # Angle grid
    angles_deg = np.linspace(0, MAX_ANGLE, N_STEPS)
    angles_rad = np.deg2rad(angles_deg)
    print(f"Grid: {N_STEPS}x{N_STEPS} = {N_STEPS**2} points per activation, "
          f"0-{MAX_ANGLE}° in {angles_deg[1]-angles_deg[0]:.2f}° steps")

    # Sweep all activations
    all_l2 = []
    all_cos = []
    for i in range(N_ACT):
        l2_grid, cos_grid = sweep_2d(
            model, contexts[i], activations[i],
            d_mandarin, d_haskell,
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
        'direction_labels': ('Mandarin', 'Haskell'),
    }

    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {OUTPUT_PKL}")


def plot_results():
    """Plot filled contour iso-plateau maps."""
    from scripts.perturbation.plot_isoplateau import extract_contour_points

    with open(OUTPUT_PKL, 'rb') as f:
        results = pickle.load(f)

    angles = results['angles_deg']

    for metric, threshold, label, units in [
        ('l2', 50.0, 'L2 distance', 'L2=50'),
        ('cosine', 0.01, 'Cosine distance', 'cos=0.01'),
    ]:
        data = results[metric]
        median_grid = np.median(data, axis=0)

        fig, ax = plt.subplots(figsize=(7, 6))
        cf = ax.contourf(angles, angles, median_grid.T, levels=50, cmap='viridis')
        plt.colorbar(cf, ax=ax, label=label)

        cs = ax.contour(angles, angles, median_grid.T, levels=[threshold],
                        colors='red', linewidths=2.5)
        ax.clabel(cs, fmt=f'{threshold}', fontsize=10, colors='red')

        ax.set_xlabel('Mandarin angle α₁ (degrees)', fontsize=12)
        ax.set_ylabel('Haskell angle α₂ (degrees)', fontsize=12)
        ax.set_title(f'Iso-Plateau Line ({units})\n'
                     f'Gemma-2-9B · layer 20 · median over {data.shape[0]} prompts',
                     fontsize=13, fontweight='bold')
        ax.set_aspect('equal')

        plt.tight_layout()
        png_path = os.path.join(RESULTS_DIR, f'isoplateau_mandarin_haskell_{metric}.png')
        plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {png_path}")
        plt.savefig(png_path.replace('.png', '.pdf'), bbox_inches='tight',
                    facecolor='white')
        plt.close()


def plot_shape():
    """Plot normalized iso-plateau contour with superellipse fit."""
    from scripts.perturbation.plot_isoplateau import (
        extract_contour_points, fit_superellipse, superellipse_curve)

    with open(OUTPUT_PKL, 'rb') as f:
        results = pickle.load(f)

    angles = results['angles_deg']

    for metric, threshold, label, units in [
        ('l2', 50.0, 'L2 distance', 'L2=50'),
        ('cosine', 0.01, 'Cosine distance', 'cos=0.01'),
    ]:
        median_grid = np.median(results[metric], axis=0)
        pt_curve = median_grid[:, 0]
        gen_curve = median_grid[0, :]

        def find_threshold(curve, angs, thresh):
            idx = np.searchsorted(curve, thresh)
            if idx == 0 or idx >= len(angs):
                return angs[idx] if idx < len(angs) else angs[-1]
            x0, x1 = angs[idx-1], angs[idx]
            y0, y1 = curve[idx-1], curve[idx]
            return x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)

        theta_1 = find_threshold(pt_curve, angles, threshold)
        theta_2 = find_threshold(gen_curve, angles, threshold)

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

        # Reference shapes
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

        ax.set_xlabel('α₁ / θ₁*  (Mandarin, normalized)', fontsize=12)
        ax.set_ylabel('α₂ / θ₂*  (Haskell, normalized)', fontsize=12)
        ax.set_title(f'Iso-Plateau Shape ({units})\n'
                     f'θ_Mandarin*={theta_1:.1f}°, θ_Haskell*={theta_2:.1f}°',
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
                                 f'isoplateau_shape_mandarin_haskell_{metric}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight',
                    facecolor='white')
        plt.close()


def plot_sensitivity():
    """Plot sensitivity of superellipse fit to threshold choice."""
    from scripts.perturbation.plot_isoplateau import extract_contour_points, fit_superellipse

    with open(OUTPUT_PKL, 'rb') as f:
        results = pickle.load(f)

    angles = results['angles_deg']

    thresholds = {
        'l2': np.array([20, 30, 40, 50, 60, 70, 80, 100]),
        'cosine': np.array([0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]),
    }

    def find_threshold(curve, angs, thresh):
        idx = np.searchsorted(curve, thresh)
        if idx == 0 or idx >= len(angs):
            return angs[idx] if idx < len(angs) else angs[-1]
        x0, x1 = angs[idx-1], angs[idx]
        y0, y1 = curve[idx-1], curve[idx]
        return x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)

    results_by_metric = {}

    for metric in ['l2', 'cosine']:
        median_grid = np.median(results[metric], axis=0)
        pt_curve = median_grid[:, 0]
        gen_curve = median_grid[0, :]

        rows = []
        print(f"\n{metric.upper()} metric:")
        print(f"  {'threshold':>10s}  {'theta_1':>8s}  {'theta_2':>8s}  "
              f"{'p_fit':>6s}  {'CI':>14s}  {'mean_err':>8s}  {'n_pts':>5s}")

        for thresh in thresholds[metric]:
            theta_1 = find_threshold(pt_curve, angles, thresh)
            theta_2 = find_threshold(gen_curve, angles, thresh)

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

            rows.append({
                'threshold': thresh,
                'p_fit': p_fit,
                'ci_lo': ci[0], 'ci_hi': ci[1],
                'mean_err_deg': fit_info['mean_radial_deg'],
            })
            print(f"  {thresh:>10g}  {theta_1:>7.1f}°  {theta_2:>7.1f}°  "
                  f"{p_fit:>6.2f}  [{ci[0]:.2f}, {ci[1]:.2f}]  "
                  f"{fit_info['mean_radial_deg']:>7.2f}°  {fit_info['n_pts']:>5d}")

        results_by_metric[metric] = rows

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    metric_labels = {'l2': 'L2 distance', 'cosine': 'Cosine distance'}
    colors = {'l2': '#2166ac', 'cosine': '#c51b7d'}

    for col, metric in enumerate(['l2', 'cosine']):
        rows = results_by_metric[metric]
        if not rows:
            continue

        ts = [r['threshold'] for r in rows]
        ps = [r['p_fit'] for r in rows]
        ci_lo = [r['ci_lo'] for r in rows]
        ci_hi = [r['ci_hi'] for r in rows]
        errs = [r['mean_err_deg'] for r in rows]
        color = colors[metric]

        ax = axes[0, col]
        ax.fill_between(ts, ci_lo, ci_hi, color=color, alpha=0.2)
        ax.plot(ts, ps, 'o-', color=color, lw=2, markersize=5)
        ax.set_ylabel('Superellipse exponent p', fontsize=11)
        ax.set_title(f'{metric_labels[metric]}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.plot(ts, errs, 's-', color=color, lw=2, markersize=5)
        ax.set_xlabel('Threshold', fontsize=11)
        ax.set_ylabel('Mean radial error (degrees)', fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Sensitivity: Mandarin × Haskell\n'
                 'Gemma-2-9B · layer 20 · median over 30 prompts',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, 'isoplateau_sensitivity_mandarin_haskell.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white')
    plt.close()


if __name__ == '__main__':
    main()
