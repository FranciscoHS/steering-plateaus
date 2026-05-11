#!/usr/bin/env python3
"""
Perturbation sweep for response decomposition directions.

Loads directions from response_decomp results, sweeps perturbation angle
from 0° to max_angle, and compares cosine distance response against
random directions.

Usage:
    python scripts/delta_optimization/response_sweep.py
    python scripts/delta_optimization/response_sweep.py --results_pkl results/response_decomp/results.pkl
"""

import argparse
import math
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from src.data import load_fineweb_fixed_length


def spherical_perturb_batch(activations, d_hat, theta):
    """Rotate each activation toward d by angle theta on the hypersphere."""
    norms = activations.norm(dim=-1, keepdim=True)
    a_hat = activations / (norms + 1e-8)
    proj = (a_hat @ d_hat).unsqueeze(-1)
    d_perp = d_hat.unsqueeze(0) - proj * a_hat
    d_perp_hat = d_perp / (d_perp.norm(dim=-1, keepdim=True) + 1e-8)
    return norms * (math.cos(theta) * a_hat + math.sin(theta) * d_perp_hat)


def sweep_direction(model, stacked_contexts, stacked_activations, baselines,
                    direction, layer, end_layer, angles_rad):
    """Sweep perturbation angle for a single direction, return mean cosine distance per angle."""
    responses = []
    with torch.no_grad():
        for theta in angles_rad:
            perturbed = spherical_perturb_batch(stacked_activations, direction, theta)
            f_pert = forward_from_layer_to_layer(
                model, stacked_contexts, perturbed.unsqueeze(1), layer, end_layer)
            cos_sim = torch.nn.functional.cosine_similarity(f_pert, baselines, dim=-1)
            responses.append((1 - cos_sim).mean().item())
    return np.array(responses)


def main():
    parser = argparse.ArgumentParser(description='Perturbation sweep for response directions')
    parser.add_argument('--results_pkl', default='results/response_decomp/results.pkl')
    parser.add_argument('--max_angle', type=float, default=60.0, help='Max perturbation angle (degrees)')
    parser.add_argument('--n_angles', type=int, default=25, help='Number of angles to sweep')
    parser.add_argument('--n_random', type=int, default=20, help='Number of random directions')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load decomposition results
    with open(args.results_pkl, 'rb') as f:
        res = pickle.load(f)

    decomp_args = res['args']
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results_pkl)

    print(f"Loaded {len(res['directions'])} directions from {args.results_pkl}")
    print(f"  model={decomp_args['model']}, layer={decomp_args['layer']}")
    print(f"  Sweep: 0° to {args.max_angle}°, {args.n_angles} steps, {args.n_random} random dirs")

    # Load model
    model, tokenizer, device = load_model(decomp_args['model'])
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    end_layer = n_layers - 2
    layer = decomp_args['layer']

    # Load same data
    n_total = decomp_args['n_response_acts'] + 1
    token_ids_list = load_fineweb_fixed_length(
        n_total, tokenizer, seq_len=decomp_args['seq_len'], seed=decomp_args['seed']
    )

    # Extract activations
    response_indices = [i for i in range(n_total) if i != decomp_args['sample_idx']]
    contexts = []
    activations = []
    for token_ids in token_ids_list:
        tokens = token_ids.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        contexts.append(hidden[:, :-1, :].detach())
        activations.append(hidden[0, -1, :].detach().float())

    response_contexts = [contexts[i] for i in response_indices]
    response_activations = [activations[i] for i in response_indices]

    stacked_contexts = torch.cat(response_contexts, dim=0)
    stacked_activations = torch.stack(response_activations)

    # Precompute baselines
    with torch.no_grad():
        baselines = forward_from_layer_to_layer(
            model, stacked_contexts, stacked_activations.unsqueeze(1), layer, end_layer)

    # Angle grid
    angles_deg = np.linspace(0, args.max_angle, args.n_angles)
    angles_rad = [math.radians(a) for a in angles_deg]

    # Sweep found directions
    found_directions = [d.to(device) for d in res['directions']]
    n_found = len(found_directions)

    print(f"\nSweeping {n_found} found directions...")
    found_curves = []
    for i, d in enumerate(found_directions):
        curve = sweep_direction(model, stacked_contexts, stacked_activations,
                                baselines, d, layer, end_layer, angles_rad)
        found_curves.append(curve)
        print(f"  d_{i+1}: max response = {curve.max():.4f}")

    # Sweep random directions
    print(f"Sweeping {args.n_random} random directions...")
    random_curves = []
    for i in range(args.n_random):
        d_rand = torch.randn(d_model, device=device)
        d_rand = d_rand / d_rand.norm()
        curve = sweep_direction(model, stacked_contexts, stacked_activations,
                                baselines, d_rand, layer, end_layer, angles_rad)
        random_curves.append(curve)
    random_curves = np.array(random_curves)
    random_mean = random_curves.mean(axis=0)
    random_std = random_curves.std(axis=0)
    print(f"  Random mean max response = {random_mean.max():.4f}")

    # Save
    sweep_results = {
        'angles_deg': angles_deg,
        'found_curves': found_curves,
        'magnitudes': res['magnitudes'],
        'random_curves': random_curves,
        'random_mean': random_mean,
        'random_std': random_std,
        'decomp_args': decomp_args,
        'sweep_args': vars(args),
    }
    sweep_path = os.path.join(args.output_dir, 'sweep_results.pkl')
    with open(sweep_path, 'wb') as f:
        pickle.dump(sweep_results, f)
    print(f"\nSweep results saved to {sweep_path}")

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Random band
    ax.fill_between(angles_deg, random_mean - random_std, random_mean + random_std,
                    alpha=0.2, color='gray', label='Random ±1σ')
    ax.plot(angles_deg, random_mean, '--', color='gray', alpha=0.7, label='Random mean')

    # Found directions
    for i, curve in enumerate(found_curves):
        ax.plot(angles_deg, curve, '-', linewidth=2, alpha=0.8,
                label=f'd_{i+1} (α={res["magnitudes"][i]:.1f})')

    ax.axvline(10, color='red', linestyle=':', alpha=0.5, label='Training angle (10°)')
    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('Mean cosine distance (response)')
    ax.set_title(f'Response sweep: {decomp_args["model"]} layer {decomp_args["layer"]}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'sweep.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    main()
