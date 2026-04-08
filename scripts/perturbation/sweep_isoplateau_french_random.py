#!/usr/bin/env python3
"""2D iso-plateau sweep: French direction × random direction.

Uses English holdout prompts as source activations.
Random direction is the improved baseline (diff of two random activations).
Runs multiple random directions and takes the median across them.
"""
import os, sys, pickle, argparse
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks, forward_from_layer_to_layer
from scripts.perturbation.generate_lesswrong_figures import (
    extract_activations_holdout,
)

# --- Config ---
N_ACT = 30
N_RANDOM_DIRS = 10   # number of random directions to average over
SEED = 123
LAYER = 20
MAX_ANGLE = 20.0
N_STEPS = 41

RESULTS_DIR = 'results/mandarin_haskell/gemma-2-9b'
DIRECTIONS_FR_RUST = os.path.join(RESULTS_DIR, 'directions_french_rust_L20.pkl')
OUTPUT_PKL = os.path.join(RESULTS_DIR, 'sweep_isoplateau_french_random.pkl')


def sweep_2d(model, ctx, act, d1, d2, layer, measure_layer, device,
             angles1_rad, angles2_rad, batch_size=64):
    """Sweep a 2D grid of combined perturbations on the sphere."""
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=torch.float32)
    R = a_d.norm()
    a_unit = a_d / R

    d1_d = d1.to(device=device, dtype=torch.float32)
    d1_orth = d1_d - (d1_d @ a_unit) * a_unit
    d1_orth = F.normalize(d1_orth, dim=0)

    d2_d = d2.to(device=device, dtype=torch.float32)
    d2_orth = d2_d - (d2_d @ a_unit) * a_unit - (d2_d @ d1_orth) * d1_orth
    d2_orth = F.normalize(d2_orth, dim=0)

    n1, n2 = len(angles1_rad), len(angles2_rad)
    s1 = torch.tensor(np.sin(angles1_rad), dtype=torch.float32, device=device)
    s2 = torch.tensor(np.sin(angles2_rad), dtype=torch.float32, device=device)

    S1, S2 = torch.meshgrid(s1, s2, indexing='ij')
    C = torch.sqrt(torch.clamp(1.0 - S1**2 - S2**2, min=0.0))

    all_points = (R * (C.reshape(-1, 1) * a_unit.unsqueeze(0)
                       + S1.reshape(-1, 1) * d1_orth.unsqueeze(0)
                       + S2.reshape(-1, 1) * d2_orth.unsqueeze(0))).to(model_dtype)

    n_points = n1 * n2

    with torch.no_grad():
        ref_act = forward_from_layer_to_layer(
            model, ctx_d, a_d[None, None].to(model_dtype), layer, measure_layer).float()

    l2_flat = np.zeros(n_points)
    cos_flat = np.zeros(n_points)

    with torch.no_grad():
        for start in range(0, n_points, batch_size):
            end = min(start + batch_size, n_points)
            batch = all_points[start:end].unsqueeze(1)
            pf = forward_from_layer_to_layer(
                model, ctx_d, batch, layer, measure_layer).float()
            l2_flat[start:end] = (pf - ref_act).norm(dim=-1).cpu().numpy()
            cos_flat[start:end] = (1.0 - F.cosine_similarity(
                ref_act, pf, dim=-1)).cpu().numpy()

    return l2_flat.reshape(n1, n2), cos_flat.reshape(n1, n2)


def generate_random_directions(activations, n_dirs, seed=42):
    """Improved random baseline: normalized difference of two random activations."""
    rng = np.random.RandomState(seed)
    n = len(activations)
    dirs = []
    for _ in range(n_dirs):
        i, j = rng.choice(n, size=2, replace=False)
        diff = activations[i] - activations[j]
        dirs.append(F.normalize(diff, dim=0))
    return dirs


def run_sweep():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load French direction
    print("Loading French direction...")
    with open(DIRECTIONS_FR_RUST, 'rb') as f:
        fr_data = pickle.load(f)
    d_french = fr_data['d_french']

    # Load model
    print("Loading Gemma-2-9B...")
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    print(f"  n_layers={n_layers}, perturb@L{LAYER}, measure@L{measure_layer}")

    # English holdout prompts
    from data.english_prompts_holdout import ENGLISH_PROMPTS_HOLDOUT
    holdout = ENGLISH_PROMPTS_HOLDOUT[:N_ACT]
    print(f"Extracting activations from {len(holdout)} English holdout prompts...")
    contexts, activations = extract_activations_holdout(
        model, tokenizer, device, LAYER, holdout)

    # Generate random directions from the holdout activations
    print(f"Generating {N_RANDOM_DIRS} random directions...")
    random_dirs = generate_random_directions(activations, N_RANDOM_DIRS)

    # Angle grid
    angles_deg = np.linspace(0, MAX_ANGLE, N_STEPS)
    angles_rad = np.deg2rad(angles_deg)
    print(f"Grid: {N_STEPS}x{N_STEPS} = {N_STEPS**2} points per activation, "
          f"0-{MAX_ANGLE}° in {angles_deg[1]-angles_deg[0]:.2f}° steps")

    # Sweep: for each random direction, sweep all activations
    all_results = []
    for rd_idx, d_random in enumerate(random_dirs):
        print(f"\n  Random direction {rd_idx+1}/{N_RANDOM_DIRS}:")
        cos_fr = (d_french @ d_random).item()
        print(f"    cos(French, random) = {cos_fr:.4f}")

        rd_l2, rd_cos = [], []
        for i in range(N_ACT):
            l2_grid, cos_grid = sweep_2d(
                model, contexts[i], activations[i],
                d_french, d_random,
                LAYER, measure_layer, device, angles_rad, angles_rad)
            rd_l2.append(l2_grid)
            rd_cos.append(cos_grid)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{N_ACT}")

        all_results.append({
            'l2': np.stack(rd_l2),      # [N_ACT, N_STEPS, N_STEPS]
            'cosine': np.stack(rd_cos),
            'cos_french': cos_fr,
        })

    results = {
        'angles_deg': angles_deg,
        'random_dirs': all_results,
        'n_random_dirs': N_RANDOM_DIRS,
        'layer': LAYER,
        'measure_layer': measure_layer,
        'holdout': 'english_prompts_holdout',
    }

    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {OUTPUT_PKL}")


def _get_median_grid(results, metric):
    """Compute median grid across random dirs and activations."""
    per_rd_medians = []
    for rd in results['random_dirs']:
        per_rd_medians.append(np.median(rd[metric], axis=0))
    stacked = np.stack(per_rd_medians)
    return np.median(stacked, axis=0)


def plot_results():
    from scripts.perturbation.plot_isoplateau import (
        extract_contour_points, fit_superellipse, superellipse_curve,
    )

    with open(OUTPUT_PKL, 'rb') as f:
        results = pickle.load(f)

    angles = results['angles_deg']
    n_rd = results['n_random_dirs']

    def find_threshold(curve, angles, thresh):
        idx = np.searchsorted(curve, thresh)
        if idx == 0 or idx >= len(angles):
            return angles[idx] if idx < len(angles) else angles[-1]
        x0, x1 = angles[idx-1], angles[idx]
        y0, y1 = curve[idx-1], curve[idx]
        return x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)

    for metric, threshold, units in [
        ('l2', 50.0, 'L2=50'),
        ('cosine', 0.01, 'cos=0.01'),
    ]:
        median_grid = _get_median_grid(results, metric)

        # --- Shape plot (normalized contour + superellipse fit) ---
        theta_fr = find_threshold(median_grid[:, 0], angles, threshold)
        theta_rd = find_threshold(median_grid[0, :], angles, threshold)

        contour_pts = extract_contour_points(angles, median_grid, threshold)
        if len(contour_pts) == 0:
            print(f"  No contour found for {metric} at {threshold}")
            continue

        contour_norm = contour_pts.copy()
        contour_norm[:, 0] /= theta_fr
        contour_norm[:, 1] /= theta_rd

        avg_theta = (theta_fr + theta_rd) / 2
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

        ax.set_xlabel('α₁ / θ₁*  (French, normalized)', fontsize=12)
        ax.set_ylabel('α₂ / θ₂*  (Random, normalized)', fontsize=12)
        ax.set_title(f'Iso-Plateau Shape: French × Random ({units})\n'
                     f'θ_French*={theta_fr:.1f}°, θ_Random*={theta_rd:.1f}° · '
                     f'median {N_ACT}×{n_rd}',
                     fontsize=11, fontweight='bold')
        ax.set_xlim(-0.05, 1.5)
        ax.set_ylim(-0.05, 1.5)
        ax.set_aspect('equal')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.plot(1, 0, 'kv', markersize=8, zorder=5)
        ax.plot(0, 1, 'k>', markersize=8, zorder=5)

        plt.tight_layout()
        png = os.path.join(RESULTS_DIR, f'isoplateau_french_random_shape_{metric}.png')
        plt.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
        plt.savefig(png.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {png}")
        plt.close()


def plot_1d_curves():
    """Plot 1D perturbation curves along French, Random, and French+Random."""
    from scripts.perturbation.sweep_isoplateau_french_random import _get_median_grid

    with open(OUTPUT_PKL, 'rb') as f:
        results = pickle.load(f)

    angles = results['angles_deg']
    n_rd = results['n_random_dirs']

    for metric, ylabel in [('l2', 'L2 distance'), ('cosine', 'Cosine distance')]:
        median_grid = _get_median_grid(results, metric)

        # French axis (α₂=0)
        french_curve = median_grid[:, 0]
        # Random axis (α₁=0)
        random_curve = median_grid[0, :]
        # Diagonal (α₁=α₂=α) — combined perturbation
        n = len(angles)
        diag_curve = np.array([median_grid[i, i] if i < n else np.nan for i in range(n)])

        # Also compute total angle for diagonal
        angles_rad = np.deg2rad(angles)
        total_angle_diag = np.rad2deg(np.arccos(np.sqrt(
            np.clip(1.0 - 2 * np.sin(angles_rad)**2, 0, 1))))

        # --- Per-direction angle plot ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(angles, french_curve, 'C0-', lw=2, label='French only')
        ax.plot(angles, random_curve, 'C1-', lw=2, label='Random only')
        ax.plot(angles, diag_curve, 'C2-', lw=2, label='French+Random (k=2)')
        ax.set_xlabel('Per-direction angle α (degrees)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Perturbation sweep: French vs Random vs Combined\n'
                     f'Gemma-2-9B · L20→L40 · English holdout · '
                     f'median {N_ACT}×{n_rd}',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        png = os.path.join(RESULTS_DIR, f'french_random_1d_per_dir_{metric}.png')
        plt.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
        plt.savefig(png.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {png}")
        plt.close()

        # --- Total angle plot ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(angles, french_curve, 'C0-', lw=2, label='French only (k=1)')
        ax.plot(angles, random_curve, 'C1-', lw=2, label='Random only (k=1)')
        ax.plot(total_angle_diag, diag_curve, 'C2-', lw=2,
                label='French+Random (k=2)')
        ax.set_xlabel('Total angle from activation (degrees)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Perturbation sweep: French vs Random vs Combined (total angle)\n'
                     f'Gemma-2-9B · L20→L40 · English holdout · '
                     f'median {N_ACT}×{n_rd}',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        png = os.path.join(RESULTS_DIR, f'french_random_1d_total_{metric}.png')
        plt.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
        plt.savefig(png.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {png}")
        plt.close()


def plot_raw_contour():
    """Plot unnormalized iso-plateau contours in raw angle space."""
    from scripts.perturbation.plot_isoplateau import extract_contour_points

    with open(OUTPUT_PKL, 'rb') as f:
        results = pickle.load(f)

    angles = results['angles_deg']
    n_rd = results['n_random_dirs']

    for metric, thresholds, units in [
        ('l2', [30, 50, 80, 120], 'L2'),
        ('cosine', [0.005, 0.01, 0.02, 0.05], 'Cosine'),
    ]:
        median_grid = _get_median_grid(results, metric)

        fig, ax = plt.subplots(figsize=(6, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(thresholds)))

        for thresh, color in zip(thresholds, colors):
            contour_pts = extract_contour_points(angles, median_grid, thresh)
            if len(contour_pts) == 0:
                continue
            contour_angle = np.arctan2(contour_pts[:, 1], contour_pts[:, 0])
            order = np.argsort(contour_angle)
            ax.plot(contour_pts[order, 0], contour_pts[order, 1], 'o-',
                    color=color, markersize=2, lw=1.5, label=f'{units}={thresh}')

        ax.set_xlabel('French angle α₁ (degrees)', fontsize=12)
        ax.set_ylabel('Random angle α₂ (degrees)', fontsize=12)
        ax.set_title(f'Iso-Plateau Contours: French × Random (raw)\n'
                     f'Gemma-2-9B · L20 · English holdout · '
                     f'median {N_ACT}×{n_rd}',
                     fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        png = os.path.join(RESULTS_DIR, f'isoplateau_french_random_raw_{metric}.png')
        plt.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
        plt.savefig(png.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {png}")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()

    if not args.plot_only:
        run_sweep()
    plot_results()
    plot_1d_curves()
    plot_raw_contour()


if __name__ == '__main__':
    main()
