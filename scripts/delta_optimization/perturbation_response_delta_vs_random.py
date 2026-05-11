#!/usr/bin/env python3
"""
Compare perturbation response along the optimized delta direction vs random directions.

For each real activation, spherically perturb along:
- v_perps[0]: the optimized delta direction (projected orthogonal to activation)
- v_perps[1:]: N random orthogonal directions

All batched together in one call to run_spherical_sweep.

Measure L2 distance at penultimate layer and KL divergence in logit space.

Usage:
    python scripts/delta_optimization/perturbation_response_delta_vs_random.py
    python scripts/delta_optimization/perturbation_response_delta_vs_random.py --which v2
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_to_final, forward_from_layer_to_layer,
    _get_final_norm, _get_lm_head, _get_blocks,
)
from src.data import load_fineweb_fixed_length
from scripts.perturbation.run_perturbation_sweep_spherical import (
    run_spherical_sweep, extract_context_and_activation,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=100)
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--which', type=str, default='both', choices=['v1', 'v2', 'v3b', 'v4', 'v5', 'both', 'all'])
    parser.add_argument('--v1_results', type=str, default='results/delta_opt/results_r66.pkl')
    parser.add_argument('--v2_results', type=str, default='results/delta_opt_v2/results_r66.pkl')
    parser.add_argument('--v3b_results', type=str, default='results/delta_opt_v3b/results_r66.pkl')
    parser.add_argument('--v4_results', type=str, default='results/delta_opt_v4/results_50.0deg.pkl')
    parser.add_argument('--v5_results', type=str, default='results/delta_opt_v5/results_50.0deg.pkl')
    parser.add_argument('--results_dir', type=str, default='results/perturbation_delta_vs_random')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    n_dirs = 1 + args.n_random_dirs  # delta + random

    angles_deg = np.linspace(0, 180, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Load deltas
    deltas = {}
    all_versions = [('v1', args.v1_results), ('v2', args.v2_results),
                    ('v3b', args.v3b_results), ('v4', args.v4_results),
                    ('v5', args.v5_results)]
    for label, path in all_versions:
        if args.which in (label, 'both', 'all') and os.path.exists(path):
            with open(path, 'rb') as f:
                results = pickle.load(f)
            # v4 stores 'direction' (unit vec), others store 'delta'
            if 'direction' in results:
                delta_unit = results['direction']
                delta_unit = delta_unit / delta_unit.norm()
                print(f"Loaded {label}: unit direction (v4 spherical)")
            else:
                delta = results['delta']
                delta_unit = delta / delta.norm()
                print(f"Loaded {label}: original norm={delta.norm().item():.1f}")
            deltas[label] = delta_unit

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2  # penultimate
    print(f"n_layers={n_layers}, measure_layer={measure_layer}")

    # Extract activations
    print(f"\nLoading {args.n_act} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print("Extracting layer-6 activations...")
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device
    )
    print(f"Activations shape: {activations.shape}, mean norm: {activations.norm(dim=-1).mean():.1f}")

    all_results = {}

    for label, delta_unit in deltas.items():
        print(f"\n{'='*60}")
        print(f"Direction: {label}")
        print(f"{'='*60}")

        # Build v_perps: [n_act, n_dirs, d_model]
        # First direction = delta, rest = random, all projected orthogonal to activation
        d_model = activations.shape[1]
        v_perps = torch.zeros(args.n_act, n_dirs, d_model)

        for i in range(args.n_act):
            a = activations[i]
            a_unit = a / a.norm()

            # Delta direction projected orthogonal
            v_delta = delta_unit.cpu() - (delta_unit.cpu() @ a_unit) * a_unit
            v_delta = v_delta / v_delta.norm()
            v_perps[i, 0] = v_delta

            # Random directions projected orthogonal
            v_rand = torch.randn(args.n_random_dirs, d_model)
            v_rand = v_rand - (v_rand @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
            v_rand = F.normalize(v_rand, dim=-1)
            v_perps[i, 1:] = v_rand

        # Run batched sweep
        all_l2 = np.zeros((args.n_act, len(angles_deg), n_dirs))
        all_kl = np.zeros((args.n_act, len(angles_deg), n_dirs))

        t0 = time.time()
        for i in range(args.n_act):
            all_l2[i], all_kl[i] = run_spherical_sweep(
                model, contexts[i], activations[i], v_perps[i],
                angles_rad, args.layer, device, n_dirs,
                measure_layer=measure_layer
            )
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")

        print(f"Done in {time.time()-t0:.0f}s")

        # Split: index 0 = delta, indices 1: = random
        all_results[label] = {
            'delta_l2': all_l2[:, :, 0],           # [n_act, n_angles]
            'delta_kl': all_kl[:, :, 0],
            'random_l2': all_l2[:, :, 1:],          # [n_act, n_angles, n_random_dirs]
            'random_kl': all_kl[:, :, 1:],
        }

    # Save
    save_data = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'results': all_results,
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(save_data, f)

    # Plot
    for label, res in all_results.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for metric_idx, (metric, metric_name) in enumerate([
            ('l2', f'L2 distance (layer {measure_layer})'),
            ('kl', 'KL divergence'),
        ]):
            ax = axes[metric_idx]

            delta_data = res[f'delta_{metric}']  # [n_act, n_angles]
            random_data = res[f'random_{metric}']  # [n_act, n_angles, n_random_dirs]

            # Delta: median across activations
            delta_median = np.median(delta_data, axis=0)
            delta_q25 = np.percentile(delta_data, 25, axis=0)
            delta_q75 = np.percentile(delta_data, 75, axis=0)

            # Random: median across dirs per activation, then across activations
            random_per_act = np.median(random_data, axis=2)
            random_median = np.median(random_per_act, axis=0)
            random_q25 = np.percentile(random_per_act, 25, axis=0)
            random_q75 = np.percentile(random_per_act, 75, axis=0)

            ax.plot(angles_deg, delta_median, color='red', linewidth=2, label=f'{label} direction')
            ax.fill_between(angles_deg, delta_q25, delta_q75, color='red', alpha=0.15)
            ax.plot(angles_deg, random_median, color='blue', linewidth=2, label='Random directions')
            ax.fill_between(angles_deg, random_q25, random_q75, color='blue', alpha=0.15)

            ax.set_xlabel('Perturbation angle (degrees)')
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 180)

        fig.suptitle(f'Perturbation response: {label} direction vs random (layer 6, GPT-2 Large)', fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(args.results_dir, f'perturbation_{label}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
