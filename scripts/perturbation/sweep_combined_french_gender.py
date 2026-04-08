#!/usr/bin/env python3
"""Perturbation sweep along combinations of French + Gender (to-female).

Uses male holdout prompts (matching original isoplateau setup).
"""
import os, sys, pickle, argparse, itertools
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks, forward_from_layer_to_layer
from scripts.perturbation.generate_lesswrong_figures import extract_activations_holdout

# --- Config ---
N_ACT = 30
SEED = 123
LAYER = 20
MAX_ANGLE = 20.0
N_STEPS = 81

RESULTS_DIR = 'results/mandarin_haskell/gemma-2-9b'
DIRECTIONS_FR_RUST = os.path.join(RESULTS_DIR, 'directions_french_rust_L20.pkl')
DIRECTIONS_GENDER = os.path.join(RESULTS_DIR, 'directions_gender_L20.pkl')
OUTPUT_PKL = os.path.join(RESULTS_DIR, 'sweep_combined_french_gender.pkl')


def sweep_combined(model, ctx, act, directions, layer, measure_layer,
                   device, angles_rad, batch_size=128):
    """Sweep along k combined directions with equal per-direction angle α."""
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=torch.float32)
    R = a_d.norm()
    a_unit = a_d / R

    d_orths = []
    for d in directions:
        d_d = d.to(device=device, dtype=torch.float32)
        d_orth = d_d - (d_d @ a_unit) * a_unit
        for prev in d_orths:
            d_orth = d_orth - (d_orth @ prev) * prev
        d_orth = F.normalize(d_orth, dim=0)
        d_orths.append(d_orth)

    k = len(d_orths)
    n = len(angles_rad)
    s = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)

    valid = k * s**2 <= 1.0
    c = torch.sqrt(torch.clamp(1.0 - k * s**2, min=0.0))

    d_sum = torch.stack(d_orths, dim=0).sum(dim=0)
    all_points = (R * (c.unsqueeze(1) * a_unit.unsqueeze(0)
                       + s.unsqueeze(1) * d_sum.unsqueeze(0))).to(model_dtype)

    with torch.no_grad():
        ref_act = forward_from_layer_to_layer(
            model, ctx_d, a_d[None, None].to(model_dtype), layer, measure_layer).float()

    l2_curve = np.full(n, np.nan)
    cos_curve = np.full(n, np.nan)

    valid_idx = valid.nonzero(as_tuple=True)[0]
    valid_points = all_points[valid_idx]

    with torch.no_grad():
        for start in range(0, len(valid_points), batch_size):
            end = min(start + batch_size, len(valid_points))
            batch = valid_points[start:end].unsqueeze(1)
            pf = forward_from_layer_to_layer(
                model, ctx_d, batch, layer, measure_layer).float()
            idx = valid_idx[start:end].cpu().numpy()
            l2_curve[idx] = (pf - ref_act).norm(dim=-1).cpu().numpy()
            cos_curve[idx] = (1.0 - F.cosine_similarity(
                ref_act, pf, dim=-1)).cpu().numpy()

    return l2_curve, cos_curve


def run_sweep():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load directions
    print("Loading directions...")
    with open(DIRECTIONS_FR_RUST, 'rb') as f:
        fr_data = pickle.load(f)
    with open(DIRECTIONS_GENDER, 'rb') as f:
        gen_data = pickle.load(f)

    d_french = fr_data['d_french']
    d_gender = gen_data['d_to_female']

    cos_fg = (d_french @ d_gender).item()
    print(f"  cos(French, Gender) = {cos_fg:.4f}")

    # Load model
    print("Loading Gemma-2-9B...")
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    print(f"  n_layers={n_layers}, perturb@L{LAYER}, measure@L{measure_layer}")

    # Male holdout prompts (matching original isoplateau setup)
    from data.male_prompts_holdout import MALE_PROMPTS_HOLDOUT
    holdout = MALE_PROMPTS_HOLDOUT[:N_ACT]
    print(f"Extracting activations from {len(holdout)} male holdout prompts...")
    contexts, activations = extract_activations_holdout(
        model, tokenizer, device, LAYER, holdout)

    # Angle grid
    angles_deg = np.linspace(0, MAX_ANGLE, N_STEPS)
    angles_rad = np.deg2rad(angles_deg)

    dir_labels = ['French', 'Gender']
    all_dirs = [d_french, d_gender]

    # k=1 (2 singles), k=2 (1 pair)
    combos = {}
    for k in [1, 2]:
        for combo_idx in itertools.combinations(range(2), k):
            label = '+'.join(dir_labels[i] for i in combo_idx)
            combos[label] = {'k': k, 'directions': [all_dirs[i] for i in combo_idx],
                             'indices': combo_idx}

    print(f"\nCombinations: {list(combos.keys())}")

    results_data = {}
    for combo_label, combo_info in combos.items():
        k = combo_info['k']
        dirs_subset = combo_info['directions']
        print(f"\n  [{combo_label}] (k={k}):")

        all_l2, all_cos = [], []
        for i in range(N_ACT):
            l2_curve, cos_curve = sweep_combined(
                model, contexts[i], activations[i],
                dirs_subset, LAYER, measure_layer, device, angles_rad)
            all_l2.append(l2_curve)
            all_cos.append(cos_curve)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{N_ACT}")

        results_data[combo_label] = {
            'k': k, 'indices': combo_info['indices'],
            'l2': np.stack(all_l2), 'cosine': np.stack(all_cos),
        }

    results = {
        'angles_deg': angles_deg,
        'combos': results_data,
        'layer': LAYER,
        'measure_layer': measure_layer,
        'direction_labels': dir_labels,
        'holdout': 'male_prompts_holdout',
    }

    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {OUTPUT_PKL}")


def plot_results():
    with open(OUTPUT_PKL, 'rb') as f:
        results = pickle.load(f)

    angles_deg = results['angles_deg']
    angles_rad = np.deg2rad(angles_deg)
    combos = results['combos']

    k_colors = {1: 'C0', 2: 'C1'}

    for metric, ylabel in [('l2', 'L2 distance'), ('cosine', 'Cosine distance')]:
        # --- Plot 1: vs per-direction angle ---
        fig, ax = plt.subplots(figsize=(8, 5))
        k_labels_seen = set()

        for combo_label, cdata in combos.items():
            k = cdata['k']
            data = cdata[metric]
            median = np.nanmedian(data, axis=0)
            q25 = np.nanpercentile(data, 25, axis=0)
            q75 = np.nanpercentile(data, 75, axis=0)
            color = k_colors[k]

            is_first = k not in k_labels_seen
            if is_first:
                k_labels_seen.add(k)
                legend_label = f'k={k}'
            else:
                legend_label = None

            ax.plot(angles_deg, median, color=color, alpha=0.7,
                    linewidth=2 if is_first else 1.2, label=legend_label)
            ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.08)

            last_valid = np.where(~np.isnan(median))[0]
            if len(last_valid) > 0:
                xi = last_valid[-1]
                ax.annotate(combo_label, (angles_deg[xi], median[xi]),
                            fontsize=8, alpha=0.7, ha='right',
                            xytext=(-5, 3), textcoords='offset points')

        ax.set_xlabel('Per-direction angle α (degrees)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Plateau-breaking: French + Gender (male holdout prompts)\n'
                     f'Gemma-2-9B · L{results["layer"]}→L{results["measure_layer"]} · '
                     f'median over {list(combos.values())[0][metric].shape[0]} prompts',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        png = os.path.join(RESULTS_DIR, f'combined_french_gender_per_dir_{metric}.png')
        plt.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
        plt.savefig(png.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {png}")
        plt.close()

        # --- Plot 2: vs total angle ---
        fig, ax = plt.subplots(figsize=(8, 5))
        k_labels_seen2 = set()

        for combo_label, cdata in combos.items():
            k = cdata['k']
            data = cdata[metric]
            median = np.nanmedian(data, axis=0)
            q25 = np.nanpercentile(data, 25, axis=0)
            q75 = np.nanpercentile(data, 75, axis=0)
            color = k_colors[k]

            total_angle_rad = np.arccos(np.sqrt(np.clip(
                1.0 - k * np.sin(angles_rad)**2, 0, 1)))
            total_angle_deg = np.rad2deg(total_angle_rad)

            is_first = k not in k_labels_seen2
            if is_first:
                k_labels_seen2.add(k)
                legend_label = f'k={k}'
            else:
                legend_label = None

            ax.plot(total_angle_deg, median, color=color, alpha=0.7,
                    linewidth=2 if is_first else 1.2, label=legend_label)
            ax.fill_between(total_angle_deg, q25, q75, color=color, alpha=0.08)

            last_valid = np.where(~np.isnan(median))[0]
            if len(last_valid) > 0:
                xi = last_valid[-1]
                ax.annotate(combo_label, (total_angle_deg[xi], median[xi]),
                            fontsize=8, alpha=0.7, ha='right',
                            xytext=(-5, 3), textcoords='offset points')

        ax.set_xlabel('Total angle from activation (degrees)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Plateau-breaking: French + Gender (male holdout, total angle)\n'
                     f'Gemma-2-9B · L{results["layer"]}→L{results["measure_layer"]} · '
                     f'median over {list(combos.values())[0][metric].shape[0]} prompts',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        png = os.path.join(RESULTS_DIR, f'combined_french_gender_total_{metric}.png')
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


if __name__ == '__main__':
    main()
