#!/usr/bin/env python3
"""
Test reproducibility of MELBO-like decomposition via SVD analysis.

Run the response-based decomposition N times on the same activation with
different optimization seeds. Stack all found directions into a matrix and
compute SVD. If singular values drop steeply after k, the method is finding
rotated versions of a fixed k-dimensional subspace. If they decay gradually,
the story is more nuanced.

Usage:
    python scripts/jacobian/reproducibility_svd.py \
        --target_prompt "The Eiffel tower is in" \
        --results_dir results/reproducibility_svd/eiffel

    # Faster test with fewer runs/restarts:
    python scripts/jacobian/reproducibility_svd.py \
        --target_prompt "The Eiffel tower is in" \
        --n_runs 5 --n_restarts 3 --n_steps 150 \
        --results_dir results/reproducibility_svd/eiffel_quick
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
from scripts.jacobian.case_study_decomp import (
    spherical_perturb_batch, decompose_one_direction
)


def run_single_decomposition(model, stacked_contexts, stacked_activations,
                             baselines, target_activation, act_norm,
                             layer, end_layer, args, opt_seed, device):
    """Run one full greedy decomposition with a given opt_seed."""
    theta = math.radians(args.theta_deg)
    d_model = target_activation.shape[0]

    residual = target_activation.clone()
    directions = []
    magnitudes = []
    responses = []

    for i in range(args.k):
        residual_frac = residual.norm().item() / act_norm
        if residual_frac < args.min_residual_frac:
            break

        best_across = {'loss': float('inf'), 'd': None, 'alpha': None, 'response': None}

        for r in range(args.n_restarts):
            torch.manual_seed(opt_seed + i * 1000 + r)
            d, alpha, response, loss = decompose_one_direction(
                model, stacked_contexts, stacked_activations, baselines,
                residual, layer, end_layer, directions,
                theta, args.lambda_orth, args.n_steps, args.lr, device
            )
            if loss < best_across['loss']:
                best_across = {'loss': loss, 'd': d, 'alpha': alpha, 'response': response}

        d_best = best_across['d']
        alpha_best = best_across['alpha']
        response_best = best_across['response']

        directions.append(d_best)
        magnitudes.append(alpha_best)
        responses.append(response_best)
        residual = residual - alpha_best * d_best

    return directions, magnitudes, responses


def main():
    parser = argparse.ArgumentParser(description='Reproducibility test via SVD')
    parser.add_argument('--target_prompt', required=True)
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=7)
    # Decomposition params
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--min_residual_frac', type=float, default=0.05)
    parser.add_argument('--theta_deg', type=float, default=10.0)
    parser.add_argument('--lambda_orth', type=float, default=None)
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    # Reproducibility params
    parser.add_argument('--n_runs', type=int, default=10,
                        help='Number of decomposition runs with different seeds')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='Seed for data loading (fixed across runs)')
    parser.add_argument('--results_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    theta = math.radians(args.theta_deg)

    print(f"=== Reproducibility SVD test ===")
    print(f"Prompt: '{args.target_prompt}'")
    print(f"Config: model={args.model}, layer={args.layer}, k={args.k}")
    print(f"  n_runs={args.n_runs}, n_restarts={args.n_restarts}, n_steps={args.n_steps}")

    # ── Load model ──
    torch.manual_seed(args.base_seed)
    np.random.seed(args.base_seed)
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    end_layer = n_layers - 2
    print(f"  d_model={d_model}, n_layers={n_layers}, device={device}")

    # ── Extract target activation ──
    target_tokens = tokenizer(args.target_prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**target_tokens, output_hidden_states=True)
    target_hidden = outputs.hidden_states[args.layer + 1]
    target_context = target_hidden[:, :-1, :].detach()
    target_activation = target_hidden[0, -1, :].detach().float()
    act_norm = target_activation.norm().item()
    print(f"  ||target activation|| = {act_norm:.2f}")

    # ── Response setup (self mode) ──
    stacked_contexts = target_context
    stacked_activations = target_activation.unsqueeze(0)

    print("Precomputing baselines...")
    with torch.no_grad():
        baselines = forward_from_layer_to_layer(
            model, stacked_contexts, stacked_activations.unsqueeze(1),
            args.layer, end_layer)

    # ── Calibrate lambda_orth ──
    with torch.no_grad():
        d_rand = torch.randn(d_model, device=device)
        d_rand = d_rand / d_rand.norm()
        perturbed = spherical_perturb_batch(stacked_activations, d_rand, theta)
        f_pert = forward_from_layer_to_layer(
            model, stacked_contexts, perturbed.unsqueeze(1), args.layer, end_layer)
        cos_sim = torch.nn.functional.cosine_similarity(f_pert, baselines, dim=-1)
        random_response = (1 - cos_sim).sum().item()
    print(f"  Random direction response: {random_response:.4f}")

    if args.lambda_orth is None:
        args.lambda_orth = act_norm * random_response * 10
        print(f"  Auto lambda_orth = {args.lambda_orth:.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # RUN DECOMPOSITION N TIMES
    # ══════════════════════════════════════════════════════════════════════
    all_directions = []  # flat list of all directions across all runs
    all_run_results = []

    for run_idx in range(args.n_runs):
        opt_seed = args.base_seed + (run_idx + 1) * 10000  # well-separated seeds
        t0 = time.time()
        print(f"\n--- Run {run_idx+1}/{args.n_runs} (opt_seed={opt_seed}) ---")

        dirs, mags, resps = run_single_decomposition(
            model, stacked_contexts, stacked_activations, baselines,
            target_activation, act_norm, args.layer, end_layer,
            args, opt_seed, device
        )

        elapsed = time.time() - t0
        n_found = len(dirs)
        print(f"  Found {n_found} directions in {elapsed:.0f}s")
        for i, (m, r) in enumerate(zip(mags, resps)):
            print(f"    d_{i+1}: α={m:.3f}, response={r:.4f}")

        all_directions.extend(dirs)
        all_run_results.append({
            'opt_seed': opt_seed,
            'directions': [d.cpu() for d in dirs],
            'magnitudes': mags,
            'responses': resps,
        })

    # ══════════════════════════════════════════════════════════════════════
    # SVD ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SVD ANALYSIS")
    print(f"{'='*60}")

    # Stack all directions: [total_dirs, d_model]
    D = torch.stack(all_directions).cpu().float()
    total_dirs = D.shape[0]
    print(f"Direction matrix: {D.shape[0]} directions × {D.shape[1]} dimensions")

    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    S = S.numpy()

    print(f"\nSingular values (top 20):")
    for i, s in enumerate(S[:20]):
        cumvar = (S[:i+1]**2).sum() / (S**2).sum()
        print(f"  σ_{i+1} = {s:.4f}  (cumulative variance: {cumvar:.1%})")

    # Find effective dimensionality at various thresholds
    total_var = (S**2).sum()
    for thresh in [0.5, 0.8, 0.9, 0.95, 0.99]:
        cumvar = np.cumsum(S**2) / total_var
        k_eff = np.searchsorted(cumvar, thresh) + 1
        print(f"  {thresh:.0%} variance explained by top {k_eff} components")

    # ── Pairwise cosine similarity between runs (top-1 directions) ──
    print(f"\nTop-1 direction cosine similarities across runs:")
    top1_dirs = []
    for res in all_run_results:
        if len(res['directions']) > 0:
            top1_dirs.append(res['directions'][0])
    if len(top1_dirs) > 1:
        top1_matrix = torch.stack(top1_dirs).float()
        cos_mat = (top1_matrix @ top1_matrix.T).numpy()
        print(f"  Mean |cos|: {np.abs(cos_mat[np.triu_indices(len(top1_dirs), k=1)]).mean():.4f}")
        print(f"  Min |cos|:  {np.abs(cos_mat[np.triu_indices(len(top1_dirs), k=1)]).min():.4f}")

    # ── Save results ──
    results = {
        'args': vars(args),
        'target_prompt': args.target_prompt,
        'target_activation': target_activation.cpu(),
        'all_run_results': all_run_results,
        'direction_matrix': D,
        'singular_values': S,
        'svd_U': U,
        'svd_Vh': Vh,
        'random_response': random_response,
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")

    # ── Plot ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Singular value spectrum
    ax = axes[0, 0]
    ax.plot(range(1, len(S)+1), S, 'o-', markersize=4)
    ax.set_xlabel('Component')
    ax.set_ylabel('Singular value')
    ax.set_title(f'Singular value spectrum ({total_dirs} directions from {args.n_runs} runs)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, min(len(S), 30) + 0.5)

    # 2. Cumulative variance explained
    ax = axes[0, 1]
    cumvar = np.cumsum(S**2) / total_var
    ax.plot(range(1, len(cumvar)+1), cumvar, 'o-', markersize=4)
    ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90%')
    ax.axhline(0.95, color='orange', linestyle='--', alpha=0.5, label='95%')
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative variance explained')
    ax.set_title('Cumulative variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, min(len(cumvar), 30) + 0.5)
    ax.set_ylim(0, 1.05)

    # 3. Normalized singular values (σ_i / σ_1)
    ax = axes[1, 0]
    ax.plot(range(1, len(S)+1), S / S[0], 'o-', markersize=4)
    ax.set_xlabel('Component')
    ax.set_ylabel('σ_i / σ_1')
    ax.set_title('Normalized singular values')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, min(len(S), 30) + 0.5)

    # 4. Per-run direction counts + magnitudes
    ax = axes[1, 1]
    for run_idx, res in enumerate(all_run_results):
        mags = res['magnitudes']
        xs = range(1, len(mags)+1)
        ax.plot(xs, mags, 'o-', markersize=4, alpha=0.6, label=f'Run {run_idx+1}')
    ax.set_xlabel('Direction index (within run)')
    ax.set_ylabel('α (magnitude)')
    ax.set_title('Magnitudes per run')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Reproducibility SVD: '{args.target_prompt}'\n"
                 f"{args.model} layer {args.layer}, k={args.k}, "
                 f"{args.n_runs} runs × {args.n_restarts} restarts × {args.n_steps} steps",
                 fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'svd_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    main()
