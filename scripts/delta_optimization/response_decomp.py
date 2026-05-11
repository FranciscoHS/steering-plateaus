#!/usr/bin/env python3
"""
Response-based feature decomposition of activations.

Decomposes activation a into k features {(d_i, α_i)} by greedily solving:
  max_{d}: α * Σ_j (1 - cos(f(perturb(a_j, d, θ)), f(a_j)))
           - λ_orth * Σ_{l<i} (d · d_l)²
  s.t. ||d|| = 1, α = d · residual

where perturb(a_j, d, θ) rotates a_j toward d by angle θ on the hypersphere,
and f maps layer-L activations to the second-to-last layer residual stream.

Response is cosine distance at 10° spherical perturbation, measured on 10
other activations from FineWeb.

Usage:
    python scripts/delta_optimization/response_decomp.py --k 5
    python scripts/delta_optimization/response_decomp.py --k 10 --layer 7 --model qwen3-1.7b
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


def extract_activations(model, token_ids_list, layer, device):
    """Extract contexts and last-token activations for multiple sequences."""
    contexts = []
    activations = []
    for token_ids in token_ids_list:
        tokens = token_ids.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        contexts.append(hidden[:, :-1, :].detach())
        activations.append(hidden[0, -1, :].detach().float())
    return contexts, activations


def spherical_perturb_batch(activations, d_hat, theta):
    """
    Rotate each activation toward d by angle theta on the hypersphere.

    Args:
        activations: [N, d_model] — activations to perturb
        d_hat: [d_model] — unit direction
        theta: float — rotation angle in radians

    Returns:
        perturbed: [N, d_model] — same norms, rotated by theta toward d
    """
    norms = activations.norm(dim=-1, keepdim=True)
    a_hat = activations / (norms + 1e-8)
    proj = (a_hat @ d_hat).unsqueeze(-1)  # [N, 1]
    d_perp = d_hat.unsqueeze(0) - proj * a_hat
    d_perp_hat = d_perp / (d_perp.norm(dim=-1, keepdim=True) + 1e-8)
    return norms * (math.cos(theta) * a_hat + math.sin(theta) * d_perp_hat)


def decompose_one_direction(model, stacked_contexts, stacked_activations, baselines,
                            residual, layer, end_layer, prev_directions,
                            theta, lambda_orth, n_steps, lr, device):
    """
    Find direction d maximizing α * Σ_j response_j - λ_orth * Σ (d·d_prev)².

    Response = cosine distance at penultimate layer after spherical perturbation.
    """
    d_model = residual.shape[0]

    d_raw = torch.randn(d_model, device=device, dtype=torch.float32)
    d_raw.requires_grad_(True)

    optimizer = torch.optim.Adam([d_raw], lr=lr)

    best = {'loss': float('inf'), 'd': None, 'alpha': None, 'response': None}
    log = {'loss': [], 'alpha': [], 'response': [], 'orth': []}

    r = residual.detach()

    for step in range(n_steps):
        optimizer.zero_grad()

        d_hat = d_raw / (d_raw.norm() + 1e-8)
        alpha = d_hat @ r

        # Spherical perturbation of all response activations
        perturbed = spherical_perturb_batch(stacked_activations, d_hat, theta)
        points = perturbed.unsqueeze(1)  # [N, 1, d]

        # Batched forward to penultimate layer
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            f_pert = forward_from_layer_to_layer(
                model, stacked_contexts, points, layer, end_layer)

        # Cosine distance from baselines
        cos_sim = torch.nn.functional.cosine_similarity(f_pert, baselines, dim=-1)
        total_response = (1 - cos_sim).sum()

        # Orthogonality penalty
        orth = torch.tensor(0.0, device=device)
        for d_prev in prev_directions:
            orth = orth + (d_hat @ d_prev).pow(2)

        loss = -(alpha * total_response) + lambda_orth * orth
        loss.backward()
        torch.nn.utils.clip_grad_norm_([d_raw], 1.0)
        optimizer.step()

        lv = loss.item()
        log['loss'].append(lv)
        log['alpha'].append(alpha.item())
        log['response'].append(total_response.item())
        log['orth'].append(orth.item())

        if lv < best['loss']:
            best['loss'] = lv
            best['d'] = d_hat.detach().clone()
            best['alpha'] = alpha.item()
            best['response'] = total_response.item()

    return best['d'], best['alpha'], best['response'], best['loss'], log


def main():
    parser = argparse.ArgumentParser(description='Response-based feature decomposition')
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=7)
    parser.add_argument('--k', type=int, default=10, help='Number of features to extract')
    parser.add_argument('--min_residual_frac', type=float, default=0.05,
                        help='Stop when ||residual|| / ||activation|| < this')
    parser.add_argument('--theta_deg', type=float, default=10.0,
                        help='Spherical perturbation angle in degrees')
    parser.add_argument('--n_response_acts', type=int, default=10,
                        help='Number of other activations for response testing')
    parser.add_argument('--response_self', action='store_true',
                        help='Use the target activation itself for response testing')
    parser.add_argument('--lambda_orth', type=float, default=None)
    parser.add_argument('--n_steps', type=int, default=200,
                        help='Optimization steps per restart')
    parser.add_argument('--n_restarts', type=int, default=5,
                        help='Random restarts per direction')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Which FineWeb sample to decompose (others used for response)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for data loading (which activations)')
    parser.add_argument('--opt_seed', type=int, default=None,
                        help='Seed for optimization randomness (default: same as --seed)')
    parser.add_argument('--results_dir', default='results/response_decomp')
    args = parser.parse_args()

    if args.opt_seed is None:
        args.opt_seed = args.seed

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    theta = math.radians(args.theta_deg)
    n_total = args.n_response_acts + 1

    print(f"Config: model={args.model}, layer={args.layer}, k={args.k}")
    print(f"  theta={args.theta_deg}°, n_response_acts={args.n_response_acts}")
    print(f"  n_steps={args.n_steps}, n_restarts={args.n_restarts}, lr={args.lr}")

    # Load model
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    end_layer = n_layers - 2
    print(f"  d_model={d_model}, n_layers={n_layers}, end_layer={end_layer}, device={device}")

    # Load data: 1 target + n_response_acts others
    token_ids_list = load_fineweb_fixed_length(
        n_total, tokenizer, seq_len=args.seq_len, seed=args.seed
    )

    # Extract activations at the specified layer
    contexts, activations = extract_activations(model, token_ids_list, args.layer, device)

    # Split target vs response
    target_activation = activations[args.sample_idx]
    act_norm = target_activation.norm().item()

    if args.response_self:
        response_contexts = [contexts[args.sample_idx]]
        response_activations = [activations[args.sample_idx]]
        print("  Using target activation for response testing (--response_self)")
    else:
        response_indices = [i for i in range(n_total) if i != args.sample_idx]
        response_contexts = [contexts[i] for i in response_indices]
        response_activations = [activations[i] for i in response_indices]

    print(f"  ||target activation|| = {act_norm:.2f}")

    # Stack response data for batched forward passes
    stacked_contexts = torch.cat(response_contexts, dim=0)      # [N, seq-1, d]
    stacked_activations = torch.stack(response_activations)      # [N, d]

    # Precompute baselines: f(a_j) at penultimate layer
    print("Precomputing baselines...")
    with torch.no_grad():
        baselines = forward_from_layer_to_layer(
            model, stacked_contexts, stacked_activations.unsqueeze(1),
            args.layer, end_layer)
    print(f"  baselines shape: {baselines.shape}")

    # Calibrate: measure response for a random direction
    print("\nCalibrating...")
    with torch.no_grad():
        d_rand = torch.randn(d_model, device=device)
        d_rand = d_rand / d_rand.norm()
        perturbed = spherical_perturb_batch(stacked_activations, d_rand, theta)
        f_pert = forward_from_layer_to_layer(
            model, stacked_contexts, perturbed.unsqueeze(1), args.layer, end_layer)
        cos_sim = torch.nn.functional.cosine_similarity(f_pert, baselines, dim=-1)
        random_response = (1 - cos_sim).sum().item()
    print(f"  Random direction response: {random_response:.4f}")
    print(f"  Expected α * response scale: ~{act_norm * random_response:.2f}")

    if args.lambda_orth is None:
        # Scale orthogonality penalty to match α * response magnitude
        args.lambda_orth = act_norm * random_response * 10
        print(f"  Auto lambda_orth = {args.lambda_orth:.2f}")

    # Greedy decomposition
    print(f"\n{'='*60}")
    print(f"Decomposing: k={args.k}, theta={args.theta_deg}°, "
          f"lambda_orth={args.lambda_orth:.1f}")
    print(f"{'='*60}")

    residual = target_activation.clone()
    directions = []
    magnitudes = []
    responses = []
    all_logs = []

    for i in range(args.k):
        residual_frac = residual.norm().item() / act_norm
        if residual_frac < args.min_residual_frac:
            print(f"\n--- Stopping: ||residual||/||a|| = {residual_frac:.4f} "
                  f"< {args.min_residual_frac} ---")
            break
        print(f"\n--- Direction {i+1}/{args.k} | ||residual|| = {residual.norm():.2f} "
              f"({residual_frac:.1%} of original) ---")

        best_across = {'loss': float('inf'), 'd': None, 'alpha': None,
                       'response': None, 'log': None}

        t0 = time.time()
        for r in range(args.n_restarts):
            torch.manual_seed(args.opt_seed + i * 1000 + r)
            d, alpha, response, loss, log = decompose_one_direction(
                model, stacked_contexts, stacked_activations, baselines,
                residual, args.layer, end_layer, directions,
                theta, args.lambda_orth, args.n_steps, args.lr, device
            )
            if loss < best_across['loss']:
                best_across = {'loss': loss, 'd': d, 'alpha': alpha,
                               'response': response, 'log': log}

            print(f"  restart {r+1}/{args.n_restarts}: "
                  f"α={alpha:.3f}, response={response:.4f}, loss={loss:.3f}, "
                  f"orth={log['orth'][-1]:.5f}")

        elapsed = time.time() - t0
        d_best = best_across['d']
        alpha_best = best_across['alpha']
        response_best = best_across['response']

        directions.append(d_best)
        magnitudes.append(alpha_best)
        responses.append(response_best)
        all_logs.append(best_across['log'])

        residual = residual - alpha_best * d_best

        print(f"  -> best α={alpha_best:.3f}, response={response_best:.4f} [{elapsed:.0f}s]")

        # Checkpoint after each direction
        checkpoint = {
            'args': vars(args),
            'directions': [d.cpu() for d in directions],
            'magnitudes': magnitudes,
            'responses': responses,
            'residual_norm': residual.norm().item(),
            'target_activation': target_activation.cpu(),
            'response_activations': stacked_activations.cpu(),
            'baselines': baselines.cpu(),
            'opt_logs': all_logs,
            'random_response': random_response,
            'n_directions_done': len(directions),
        }
        ckpt_path = os.path.join(args.results_dir, 'checkpoint.pkl')
        with open(ckpt_path, 'wb') as f:
            pickle.dump(checkpoint, f)

    # Cosine similarity matrix
    n = len(directions)
    cosines = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            cosines[a, b] = (directions[a] @ directions[b]).item()

    # Reconstruction quality
    recon = sum(m * d for m, d in zip(magnitudes, directions))
    recon_error = (target_activation - recon).norm().item() / act_norm

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for i in range(n):
        print(f"  d_{i+1}: α={magnitudes[i]:.3f}, response={responses[i]:.4f}")
    print(f"Reconstruction error (relative): {recon_error:.4f}")
    print(f"Final ||residual|| = {residual.norm():.2f} "
          f"(started at {act_norm:.2f})")
    print(f"\nCosine similarities:\n{cosines.round(3)}")

    # Save
    results = {
        'args': vars(args),
        'directions': [d.cpu() for d in directions],
        'magnitudes': magnitudes,
        'responses': responses,
        'cosines': cosines,
        'recon_error': recon_error,
        'residual_norm': residual.norm().item(),
        'target_activation': target_activation.cpu(),
        'response_activations': stacked_activations.cpu(),
        'baselines': baselines.cpu(),
        'opt_logs': all_logs,
        'random_response': random_response,
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Magnitudes
    ax = axes[0, 0]
    ax.bar(range(1, n+1), magnitudes)
    ax.set_xlabel('Direction')
    ax.set_ylabel('α')
    ax.set_title('Feature magnitudes')
    ax.grid(True, alpha=0.3)

    # Response per direction
    ax = axes[0, 1]
    ax.bar(range(1, n+1), responses)
    ax.axhline(random_response, color='red', linestyle='--', alpha=0.7, label='Random dir')
    ax.set_xlabel('Direction')
    ax.set_ylabel('Σ cosine distance')
    ax.set_title(f'Response ({args.theta_deg}° perturbation, {args.n_response_acts} activations)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cosine similarities
    ax = axes[1, 0]
    im = ax.imshow(cosines, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Direction cosine similarities')
    for a in range(n):
        for b in range(n):
            ax.text(b, a, f'{cosines[a,b]:.2f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax)

    # Optimization curves (response)
    ax = axes[1, 1]
    for i, log_data in enumerate(all_logs):
        ax.plot(log_data['response'], label=f'd_{i+1}', alpha=0.7)
    ax.axhline(random_response, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Step')
    ax.set_ylabel('Σ cosine distance')
    ax.set_title('Response during optimization (best restart)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Response decomposition: {args.model} layer {args.layer}, '
                 f'k={n}, θ={args.theta_deg}°', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    main()
