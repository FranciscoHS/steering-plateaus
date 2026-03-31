#!/usr/bin/env python3
"""Iso-plateau line measurement for compositional steering.

Perturbs along Portuguese and Gender directions simultaneously on the sphere,
scanning a 2D grid of (α₁, α₂) angles. Measures downstream cosine distance
and L2 distance at each grid point, then plots contours at the plateau-breaking
threshold.

The combined perturbation is:
    p = R * (c * a_unit + sin(α₁) * d1_orth + sin(α₂) * d2_orth)
where c = sqrt(1 - sin²α₁ - sin²α₂), keeping ||p|| = R.
d1_orth and d2_orth are Gram-Schmidt orthogonalized against a_unit (and each other).
"""
import os, sys, pickle, argparse
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks, forward_from_layer_to_layer, forward_from_layer
from scripts.perturbation.generate_lesswrong_figures import (
    extract_activations_holdout,
)

# --- Config ---
N_ACT = 30
SEED = 123
LAYER = 20
MAX_ANGLE = 20.0   # degrees, for each axis
N_STEPS = 41       # grid resolution per axis (0, 0.5, 1.0, ..., 20.0)

RESULTS_DIR = 'results/mandarin_haskell/gemma-2-9b'
DIRECTIONS_PT = os.path.join(RESULTS_DIR, 'directions_portuguese_L20.pkl')
DIRECTIONS_GENDER = os.path.join(RESULTS_DIR, 'directions_gender_L20.pkl')
OUTPUT_PKL = os.path.join(RESULTS_DIR, 'sweep_isoplateau.pkl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()

    if not args.plot_only:
        run_sweep()
    plot_results()


def sweep_2d(model, ctx, act, d1, d2, layer, measure_layer, device,
             angles1_rad, angles2_rad, batch_size=64):
    """Sweep a 2D grid of combined perturbations on the sphere.

    Returns:
        l2_grid: [n1, n2] array of L2 distances
        cos_grid: [n1, n2] array of cosine distances
    """
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=torch.float32)
    R = a_d.norm()
    a_unit = a_d / R

    # Gram-Schmidt: orthogonalize d1 and d2 against a_unit and each other
    d1_d = d1.to(device=device, dtype=torch.float32)
    d1_orth = d1_d - (d1_d @ a_unit) * a_unit
    d1_orth = F.normalize(d1_orth, dim=0)

    d2_d = d2.to(device=device, dtype=torch.float32)
    d2_orth = d2_d - (d2_d @ a_unit) * a_unit - (d2_d @ d1_orth) * d1_orth
    d2_orth = F.normalize(d2_orth, dim=0)

    # Build 2D grid of perturbed points
    n1, n2 = len(angles1_rad), len(angles2_rad)
    s1 = torch.tensor(np.sin(angles1_rad), dtype=torch.float32, device=device)
    s2 = torch.tensor(np.sin(angles2_rad), dtype=torch.float32, device=device)

    # [n1, n2] grids
    S1, S2 = torch.meshgrid(s1, s2, indexing='ij')
    C = torch.sqrt(torch.clamp(1.0 - S1**2 - S2**2, min=0.0))

    # Flatten to [n1*n2, d_model]
    all_points = (R * (C.reshape(-1, 1) * a_unit.unsqueeze(0)
                       + S1.reshape(-1, 1) * d1_orth.unsqueeze(0)
                       + S2.reshape(-1, 1) * d2_orth.unsqueeze(0))).to(model_dtype)

    n_points = n1 * n2

    # Compute reference (unperturbed)
    with torch.no_grad():
        ref_act = forward_from_layer_to_layer(
            model, ctx_d, a_d[None, None].to(model_dtype), layer, measure_layer).float()
        ref_logits = forward_from_layer(
            model, ctx_d, a_d[None, None].to(model_dtype), layer).float()
        ref_logp = F.log_softmax(ref_logits, dim=-1)

    l2_flat = np.zeros(n_points)
    cos_flat = np.zeros(n_points)

    with torch.no_grad():
        for start in range(0, n_points, batch_size):
            end = min(start + batch_size, n_points)
            batch = all_points[start:end].unsqueeze(1)  # [B, 1, d_model]

            # L2 and cosine at penultimate layer
            pf = forward_from_layer_to_layer(
                model, ctx_d, batch, layer, measure_layer).float()
            l2_flat[start:end] = (pf - ref_act).norm(dim=-1).cpu().numpy()
            cos_flat[start:end] = (1.0 - F.cosine_similarity(
                ref_act, pf, dim=-1)).cpu().numpy()

    return l2_flat.reshape(n1, n2), cos_flat.reshape(n1, n2)


def run_sweep():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load directions
    print("Loading directions...")
    with open(DIRECTIONS_PT, 'rb') as f:
        pt_data = pickle.load(f)
    with open(DIRECTIONS_GENDER, 'rb') as f:
        gen_data = pickle.load(f)

    d_portuguese = pt_data['d_portuguese']
    d_to_female = gen_data['d_to_female']

    # Load model
    print("Loading Gemma-2-9B...")
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    print(f"  n_layers={n_layers}, perturb@L{LAYER}, measure@L{measure_layer}")

    # Holdout activations
    from data.male_prompts_holdout import MALE_PROMPTS_HOLDOUT
    holdout = MALE_PROMPTS_HOLDOUT[:N_ACT]
    print(f"Extracting activations from {len(holdout)} male holdout prompts...")
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
            d_portuguese, d_to_female,
            LAYER, measure_layer, device, angles_rad, angles_rad)
        all_l2.append(l2_grid)
        all_cos.append(cos_grid)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{N_ACT}")

    results = {
        'angles_deg': angles_deg,
        'l2': np.stack(all_l2),      # [N_ACT, N_STEPS, N_STEPS]
        'cosine': np.stack(all_cos),  # [N_ACT, N_STEPS, N_STEPS]
        'layer': LAYER,
        'measure_layer': measure_layer,
    }

    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {OUTPUT_PKL}")


def plot_results():
    with open(OUTPUT_PKL, 'rb') as f:
        results = pickle.load(f)

    angles = results['angles_deg']

    for metric, threshold, label, units in [
        ('l2', 50.0, 'L2 distance', 'L2=50'),
        ('cosine', 0.01, 'Cosine distance', 'cos=0.01'),
    ]:
        data = results[metric]  # [N_ACT, N_STEPS, N_STEPS]
        median_grid = np.median(data, axis=0)

        fig, ax = plt.subplots(figsize=(7, 6))

        # Filled contour of the metric
        cf = ax.contourf(angles, angles, median_grid.T, levels=50, cmap='viridis')
        plt.colorbar(cf, ax=ax, label=label)

        # Iso-plateau contour
        cs = ax.contour(angles, angles, median_grid.T, levels=[threshold],
                        colors='red', linewidths=2.5)
        ax.clabel(cs, fmt=f'{threshold}', fontsize=10, colors='red')

        # Mark individual 1D thresholds on axes
        # Portuguese threshold (α₂=0 row)
        row0 = median_grid[:, 0]
        idx_pt = np.searchsorted(row0, threshold)
        if idx_pt < len(angles):
            ax.plot(angles[idx_pt], 0, 'rv', markersize=10, zorder=5)

        # Gender threshold (α₁=0 column)
        col0 = median_grid[0, :]
        idx_gen = np.searchsorted(col0, threshold)
        if idx_gen < len(angles):
            ax.plot(0, angles[idx_gen], 'r>', markersize=10, zorder=5)

        ax.set_xlabel('Portuguese angle α₁ (degrees)', fontsize=12)
        ax.set_ylabel('Gender angle α₂ (degrees)', fontsize=12)
        ax.set_title(f'Iso-Plateau Line ({units})\n'
                     f'Gemma-2-9B · layer 20 · median over {data.shape[0]} prompts',
                     fontsize=13, fontweight='bold')
        ax.set_aspect('equal')

        plt.tight_layout()
        png_path = os.path.join(RESULTS_DIR, f'isoplateau_{metric}.png')
        plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {png_path}")
        plt.savefig(png_path.replace('.png', '.pdf'), bbox_inches='tight',
                    facecolor='white')
        plt.close()


if __name__ == '__main__':
    main()
