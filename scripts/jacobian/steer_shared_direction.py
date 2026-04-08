#!/usr/bin/env python3
"""
Steer with shared directions found by reproducibility analysis, and run
perturbation sweeps to check if they're plateau-breaking.

Usage:
    python scripts/jacobian/steer_shared_direction.py \
        --shared_pt results/reproducibility_svd/eiffel/shared_directions.pt \
        --results_dir results/reproducibility_svd/eiffel/steering
"""

import argparse
import math
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from scripts.steering.spherical_steering_generate import generate_with_spherical_steering
from scripts.jacobian.case_study_decomp import spherical_perturb_batch


def perturbation_sweep(model, tokenizer, prompt, direction, layer, end_layer,
                       device, angles_deg):
    """Measure cosine distance at penultimate layer for a range of perturbation angles."""
    tokens = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    hidden = outputs.hidden_states[layer + 1]
    context = hidden[:, :-1, :].detach()
    activation = hidden[0, -1, :].detach().float()

    # Baseline
    with torch.no_grad():
        baseline = forward_from_layer_to_layer(
            model, context, activation.unsqueeze(0).unsqueeze(0), layer, end_layer)

    results = []
    for angle_deg in angles_deg:
        theta = math.radians(angle_deg)
        with torch.no_grad():
            perturbed = spherical_perturb_batch(activation.unsqueeze(0), direction, theta)
            f_pert = forward_from_layer_to_layer(
                model, context, perturbed.unsqueeze(1), layer, end_layer)
            cos_sim = torch.nn.functional.cosine_similarity(f_pert, baseline, dim=-1)
            cos_dist = (1 - cos_sim).item()
        results.append({'angle': angle_deg, 'cos_dist': cos_dist})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared_pt', required=True,
                        help='Path to shared_directions.pt')
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=7)
    parser.add_argument('--n_dirs', type=int, default=3,
                        help='Number of shared directions to test')
    parser.add_argument('--prompts', nargs='+',
                        default=['The Eiffel tower is in',
                                 'München liegt in',
                                 'def fibonacci(n)'])
    parser.add_argument('--steer_angles', type=str, default='0,5,10,15,20,30,45,60')
    parser.add_argument('--sweep_angles', type=str,
                        default='0,1,2,3,5,7,10,15,20,25,30,40,50,60')
    parser.add_argument('--n_tokens', type=int, default=30)
    parser.add_argument('--results_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    steer_angles = [float(a) for a in args.steer_angles.split(',')]
    sweep_angles = [float(a) for a in args.sweep_angles.split(',')]

    # Load shared directions
    shared = torch.load(args.shared_pt, map_location='cpu', weights_only=True)
    print(f"Loaded {len(shared['directions'])} shared directions")
    for i, (d, lam) in enumerate(zip(shared['directions'], shared['eigenvalues'])):
        print(f"  Shared dir {i+1}: λ={lam:.2f}, ||d||={d.norm():.4f}")

    # Load model
    model, tokenizer, device = load_model(args.model)
    n_layers = len(_get_blocks(model))
    end_layer = n_layers - 2
    print(f"Model: {args.model}, layer={args.layer}, end_layer={end_layer}, device={device}")

    output_lines = []
    all_sweeps = {}

    for d_idx in range(min(args.n_dirs, len(shared['directions']))):
        direction = shared['directions'][d_idx].to(device).float()
        direction = direction / direction.norm()
        lam = shared['eigenvalues'][d_idx]

        output_lines.append(f"\n{'='*70}")
        output_lines.append(f"SHARED DIRECTION {d_idx+1} (λ={lam:.2f})")
        output_lines.append(f"{'='*70}")

        for prompt in args.prompts:
            # ── Perturbation sweep ──
            sweep = perturbation_sweep(
                model, tokenizer, prompt, direction, args.layer, end_layer,
                device, sweep_angles)
            all_sweeps[(d_idx, prompt)] = sweep

            output_lines.append(f"\n  Prompt: '{prompt}'")
            output_lines.append(f"  Perturbation sweep (cosine distance at penultimate layer):")
            for s in sweep:
                output_lines.append(f"    {s['angle']:5.1f}°: cos_dist={s['cos_dist']:.6f}")

            # ── Steering ──
            output_lines.append(f"  Steering rollouts:")
            for angle in steer_angles:
                text, ids = generate_with_spherical_steering(
                    model, tokenizer, prompt, direction, angle,
                    args.layer, device, args.n_tokens, steer_mode='rolling'
                )
                output_lines.append(f"    {angle:5.1f}°: {text}")

    # Also sweep a random direction for comparison
    output_lines.append(f"\n{'='*70}")
    output_lines.append("RANDOM DIRECTION (baseline)")
    output_lines.append(f"{'='*70}")
    torch.manual_seed(12345)
    d_rand = torch.randn(direction.shape[0], device=device)
    d_rand = d_rand / d_rand.norm()

    for prompt in args.prompts:
        sweep = perturbation_sweep(
            model, tokenizer, prompt, d_rand, args.layer, end_layer,
            device, sweep_angles)
        all_sweeps[('random', prompt)] = sweep

        output_lines.append(f"\n  Prompt: '{prompt}'")
        output_lines.append(f"  Perturbation sweep:")
        for s in sweep:
            output_lines.append(f"    {s['angle']:5.1f}°: cos_dist={s['cos_dist']:.6f}")

    # Save text results
    output_text = '\n'.join(output_lines)
    txt_path = os.path.join(args.results_dir, 'steering_results.txt')
    with open(txt_path, 'w') as f:
        f.write(output_text)
    print(output_text)
    print(f"\nResults saved to {txt_path}")

    # Save sweep data
    sweep_path = os.path.join(args.results_dir, 'sweep_data.pkl')
    with open(sweep_path, 'wb') as f:
        pickle.dump(all_sweeps, f)

    # ── Plot perturbation sweeps ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(args.prompts), figsize=(6*len(args.prompts), 5))
    if len(args.prompts) == 1:
        axes = [axes]

    for p_idx, prompt in enumerate(args.prompts):
        ax = axes[p_idx]
        for d_idx in range(min(args.n_dirs, len(shared['directions']))):
            sweep = all_sweeps[(d_idx, prompt)]
            angles = [s['angle'] for s in sweep]
            dists = [s['cos_dist'] for s in sweep]
            lam = shared['eigenvalues'][d_idx]
            ax.plot(angles, dists, 'o-', label=f'Shared {d_idx+1} (λ={lam:.1f})',
                    markersize=4)

        # Random baseline
        sweep = all_sweeps[('random', prompt)]
        angles = [s['angle'] for s in sweep]
        dists = [s['cos_dist'] for s in sweep]
        ax.plot(angles, dists, 's--', color='gray', alpha=0.7, label='Random',
                markersize=4)

        ax.set_xlabel('Perturbation angle (°)')
        ax.set_ylabel('Cosine distance')
        short = prompt[:30] + '...' if len(prompt) > 30 else prompt
        ax.set_title(f"'{short}'")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.suptitle('Perturbation sweep: shared directions vs random', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'perturbation_sweeps.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    main()
