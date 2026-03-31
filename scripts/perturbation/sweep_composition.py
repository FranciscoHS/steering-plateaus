#!/usr/bin/env python3
"""Perturbation sweeps for compositional steering directions.

Sweeps along:
  (i)   Portuguese direction
  (ii)  Gender (to-female) direction
  (iii) Sum of Portuguese + Gender (normalized)
  (iv)  Random directions (improved baseline)

Uses pre-computed directions from the composition experiment (layer 20).
Measures L2 distance, cosine distance, and KL divergence.
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
    sweep_one_spherical,
    compute_improved_random_directions,
)

# --- Config ---
N_ACT = 30
N_RANDOM_DIRS = 10      # random directions per activation
MAX_ANGLE = 60.0
N_ANGLES = 60
SEED = 123
LAYER = 20
METRICS = ['l2', 'cosine', 'kl']

RESULTS_DIR = 'results/mandarin_haskell/gemma-2-9b'
DIRECTIONS_PT = os.path.join(RESULTS_DIR, 'directions_portuguese_L20.pkl')
DIRECTIONS_GENDER = os.path.join(RESULTS_DIR, 'directions_gender_L20.pkl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true',
                        help='Only plot from existing pkl data')
    parser.add_argument('--metrics', nargs='+', default=METRICS,
                        help='Which metrics to sweep (default: all)')
    args = parser.parse_args()

    if not args.plot_only:
        run_sweep(args.metrics)
    for metric in args.metrics:
        plot_results(metric)


def run_sweep(metrics):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load pre-computed directions
    print("Loading directions...")
    with open(DIRECTIONS_PT, 'rb') as f:
        pt_data = pickle.load(f)
    with open(DIRECTIONS_GENDER, 'rb') as f:
        gen_data = pickle.load(f)

    d_portuguese = pt_data['d_portuguese']
    d_to_female = gen_data['d_to_female']
    assert pt_data['layer'] == LAYER
    assert gen_data['layer'] == LAYER

    # Sum direction (normalized)
    d_sum = F.normalize(d_portuguese + d_to_female, dim=0)
    cos_pt_gen = (d_portuguese @ d_to_female).item()
    print(f"  cos(PT, female) = {cos_pt_gen:.4f}")

    # Load model
    print("\nLoading Gemma-2-9B...")
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    print(f"  n_layers={n_layers}, perturb@L{LAYER}, measure@L{measure_layer}")

    # Extract holdout activations (male-gendered, matching the contrastive source)
    from data.male_prompts_holdout import MALE_PROMPTS_HOLDOUT
    holdout = MALE_PROMPTS_HOLDOUT[:N_ACT]
    print(f"\nExtracting activations from {len(holdout)} male holdout prompts...")
    contexts, activations = extract_activations_holdout(
        model, tokenizer, device, LAYER, holdout)

    # Random directions (improved baseline)
    print(f"Computing {N_RANDOM_DIRS} random directions per activation...")
    rand_dirs = compute_improved_random_directions(activations, N_RANDOM_DIRS, SEED)

    # Angle grid
    angles_deg = np.linspace(0, MAX_ANGLE, N_ANGLES + 1)
    angles_rad = np.deg2rad(angles_deg)

    directions_to_sweep = {
        'portuguese': d_portuguese,
        'gender': d_to_female,
        'sum': d_sum,
    }

    for metric in metrics:
        print(f"\n{'='*60}")
        print(f"METRIC: {metric}")
        print(f"{'='*60}")

        results = {'angles_deg': angles_deg, 'metric': metric}

        for name, direction in directions_to_sweep.items():
            print(f"\nSweeping {name} direction ({N_ACT} activations)...")
            curves = []
            for i in range(N_ACT):
                curve = sweep_one_spherical(
                    model, contexts[i], activations[i], direction,
                    LAYER, measure_layer, device, angles_rad, metric=metric)
                curves.append(curve)
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{N_ACT}")
            results[name] = np.stack(curves)

        print(f"\nSweeping random directions ({N_ACT} x {N_RANDOM_DIRS} curves)...")
        rand_curves = []
        for i in range(N_ACT):
            for j in range(N_RANDOM_DIRS):
                curve = sweep_one_spherical(
                    model, contexts[i], activations[i], rand_dirs[j],
                    LAYER, measure_layer, device, angles_rad, metric=metric)
                rand_curves.append(curve)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{N_ACT}")
        results['random'] = np.stack(rand_curves)

        pkl_path = os.path.join(RESULTS_DIR, f'sweep_composition_{metric}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nSaved: {pkl_path}")


YLABEL = {
    'l2': 'L2 distance (penultimate layer)',
    'cosine': 'Cosine distance (penultimate layer)',
    'kl': 'KL divergence (logits)',
}


def plot_results(metric):
    """Plot median + IQR shading for all 4 direction types."""
    pkl_path = os.path.join(RESULTS_DIR, f'sweep_composition_{metric}.pkl')
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    angles = results['angles_deg']

    groups = [
        ('random',     'Random',              '#888888'),
        ('portuguese', 'Portuguese (DoM)',     '#2166ac'),
        ('gender',     'Gender/female (DoM)',  '#c51b7d'),
        ('sum',        'PT + Gender (sum)',    '#7b3294'),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for key, label, color in groups:
        data = results[key]
        med = np.median(data, axis=0)
        q25, q75 = np.percentile(data, [25, 75], axis=0)
        ax.fill_between(angles, q25, q75, color=color, alpha=0.15)
        ax.plot(angles, med, color=color, lw=2.5, label=label)

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel(YLABEL[metric], fontsize=12)
    ax.set_title(f'Perturbation Sweeps: Compositional Directions ({metric})\n'
                 f'Gemma-2-9B · layer 20 · male holdout prompts',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, MAX_ANGLE)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, f'sweep_composition_{metric}.png')
    plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png_path}")
    plt.savefig(png_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white')
    print(f"Saved PDF too")
    plt.close()


if __name__ == '__main__':
    main()
