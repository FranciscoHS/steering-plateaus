#!/usr/bin/env python3
"""
Reproducibility test for MELBO-like decomposition with KL-divergence
reconstruction instead of L2 norm.

Directions are found the same way (maximize response, orthogonality penalty),
but after each direction is found, magnitudes are re-optimized to minimize
KL(original_logits || reconstructed_logits). The residual is updated based on
these KL-optimal magnitudes, so subsequent direction search is guided by
behavioral reconstruction error rather than L2 error.

Usage:
    python scripts/jacobian/reproducibility_svd_kl.py \
        --target_prompt "The Eiffel tower is in" \
        --results_dir results/reproducibility_svd_kl/eiffel
"""

import argparse
import math
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np

from src.model import load_model, forward_from_layer, forward_from_layer_to_layer, _get_blocks
from scripts.dictionary_learning.case_study_decomp import (
    spherical_perturb_batch, decompose_one_direction
)


def compute_kl(model, context, activation, reconstruction, layer, device):
    """Compute KL(original || reconstructed) on next-token logits."""
    with torch.no_grad():
        orig_logits = forward_from_layer(
            model, context, activation.unsqueeze(0).unsqueeze(0), layer)
        recon_logits = forward_from_layer(
            model, context, reconstruction.unsqueeze(0).unsqueeze(0), layer)
        orig_log_probs = F.log_softmax(orig_logits, dim=-1)
        recon_log_probs = F.log_softmax(recon_logits, dim=-1)
        kl = F.kl_div(recon_log_probs, orig_log_probs.exp(), reduction='batchmean')
    return kl.item()


def optimize_magnitudes_kl(model, context, activation, directions, layer,
                           device, n_steps=100, lr=0.05):
    """
    Optimize magnitudes {α_i} to minimize KL(original || Σ α_i d_i).

    Returns optimized magnitudes and final KL divergence.
    """
    k = len(directions)
    D = torch.stack(directions).float()  # [k, d]

    # Compute original logits once
    with torch.no_grad():
        orig_logits = forward_from_layer(
            model, context, activation.unsqueeze(0).unsqueeze(0), layer)
        orig_probs = F.softmax(orig_logits, dim=-1).detach()

    # Initialize with L2 projections
    alphas = (D @ activation).clone().detach()
    alphas.requires_grad_(True)
    optimizer = torch.optim.Adam([alphas], lr=lr)

    best_kl = float('inf')
    best_alphas = alphas.detach().clone()

    for step in range(n_steps):
        optimizer.zero_grad()
        recon = (alphas.unsqueeze(-1) * D).sum(0)  # [d]

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            recon_logits = forward_from_layer(
                model, context, recon.unsqueeze(0).unsqueeze(0), layer)

        recon_log_probs = F.log_softmax(recon_logits, dim=-1)
        kl = F.kl_div(recon_log_probs, orig_probs, reduction='batchmean')
        kl.backward()
        optimizer.step()

        if kl.item() < best_kl:
            best_kl = kl.item()
            best_alphas = alphas.detach().clone()

    return best_alphas.tolist(), best_kl


def run_single_decomposition_kl(model, stacked_contexts, stacked_activations,
                                baselines, target_context, target_activation,
                                act_norm, layer, end_layer, args, opt_seed, device):
    """Run one full greedy decomposition with KL-div magnitude optimization."""
    theta = math.radians(args.theta_deg)

    residual = target_activation.clone()
    directions = []
    magnitudes = []
    responses = []
    kl_history = []

    for i in range(args.k):
        residual_frac = residual.norm().item() / act_norm

        # Find direction by maximizing response (same as L2 version)
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
        response_best = best_across['response']

        directions.append(d_best)
        responses.append(response_best)

        # Re-optimize ALL magnitudes to minimize KL divergence
        opt_mags, kl = optimize_magnitudes_kl(
            model, target_context, target_activation, directions, layer,
            device, n_steps=args.kl_steps, lr=args.kl_lr
        )
        magnitudes = opt_mags
        kl_history.append(kl)

        # Update residual based on KL-optimal reconstruction
        reconstruction = sum(m * d for m, d in zip(magnitudes, directions))
        residual = target_activation - reconstruction

        print(f"    d_{i+1}: α_kl=[{', '.join(f'{m:.2f}' for m in magnitudes)}], "
              f"resp={response_best:.4f}, KL={kl:.6f}, ||resid||={residual.norm():.2f}")

        # Stop if KL is very small
        if kl < args.kl_threshold:
            print(f"    Stopping: KL={kl:.6f} < {args.kl_threshold}")
            break

    return directions, magnitudes, responses, kl_history


def main():
    parser = argparse.ArgumentParser(description='Reproducibility test with KL-div reconstruction')
    parser.add_argument('--target_prompt', required=True)
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=7)
    # Decomposition params (direction-finding, same as L2 version)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--theta_deg', type=float, default=10.0)
    parser.add_argument('--lambda_orth', type=float, default=None)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--n_restarts', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    # KL magnitude optimization params
    parser.add_argument('--kl_steps', type=int, default=100,
                        help='Optimization steps for KL magnitude fitting')
    parser.add_argument('--kl_lr', type=float, default=0.05,
                        help='Learning rate for KL magnitude optimization')
    parser.add_argument('--kl_threshold', type=float, default=0.001,
                        help='Stop when KL divergence drops below this')
    # Reproducibility params
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--base_seed', type=int, default=42)
    parser.add_argument('--results_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    theta = math.radians(args.theta_deg)

    print(f"=== Reproducibility SVD test (KL-div reconstruction) ===")
    print(f"Prompt: '{args.target_prompt}'")
    print(f"Config: model={args.model}, layer={args.layer}, k={args.k}")
    print(f"  n_runs={args.n_runs}, n_restarts={args.n_restarts}, n_steps={args.n_steps}")
    print(f"  KL: kl_steps={args.kl_steps}, kl_lr={args.kl_lr}, kl_threshold={args.kl_threshold}")

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

    # ── Baseline KL (zero reconstruction) ──
    zero_recon = torch.zeros_like(target_activation)
    kl_zero = compute_kl(model, target_context, target_activation, zero_recon,
                         args.layer, device)
    print(f"  KL(original || zero): {kl_zero:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # RUN DECOMPOSITION N TIMES
    # ══════════════════════════════════════════════════════════════════════
    all_directions = []
    all_run_results = []

    for run_idx in range(args.n_runs):
        opt_seed = args.base_seed + (run_idx + 1) * 10000
        t0 = time.time()
        print(f"\n--- Run {run_idx+1}/{args.n_runs} (opt_seed={opt_seed}) ---")

        dirs, mags, resps, kl_hist = run_single_decomposition_kl(
            model, stacked_contexts, stacked_activations, baselines,
            target_context, target_activation, act_norm,
            args.layer, end_layer, args, opt_seed, device
        )

        elapsed = time.time() - t0
        n_found = len(dirs)
        print(f"  Found {n_found} directions in {elapsed:.0f}s, final KL={kl_hist[-1]:.6f}")

        all_directions.extend(dirs)
        all_run_results.append({
            'opt_seed': opt_seed,
            'directions': [d.cpu() for d in dirs],
            'magnitudes': mags,
            'responses': resps,
            'kl_history': kl_hist,
        })

    # ══════════════════════════════════════════════════════════════════════
    # SVD ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SVD ANALYSIS")
    print(f"{'='*60}")

    D = torch.stack(all_directions).cpu().float()
    total_dirs = D.shape[0]
    print(f"Direction matrix: {D.shape[0]} directions × {D.shape[1]} dimensions")

    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    S = S.numpy()

    print(f"\nSingular values (top 20):")
    total_var = (S**2).sum()
    for i, s in enumerate(S[:20]):
        cumvar = (S[:i+1]**2).sum() / total_var
        print(f"  σ_{i+1} = {s:.4f}  (cumulative variance: {cumvar:.1%})")

    for thresh in [0.5, 0.8, 0.9, 0.95, 0.99]:
        cumvar = np.cumsum(S**2) / total_var
        k_eff = np.searchsorted(cumvar, thresh) + 1
        print(f"  {thresh:.0%} variance explained by top {k_eff} components")

    # ── Pairwise subspace overlap ──
    print(f"\nPairwise subspace overlap:")
    run_directions = [r['directions'] for r in all_run_results]
    n_runs = len(run_directions)

    for k_trunc in [5, 10, 20]:
        k_used = min(k_trunc, min(len(dirs) for dirs in run_directions))
        matrices = []
        for dirs in run_directions:
            Ui = torch.stack(dirs[:k_used]).float()
            matrices.append(Ui)

        overlap = np.zeros((n_runs, n_runs))
        for i in range(n_runs):
            for j in range(n_runs):
                cross = matrices[i] @ matrices[j].T
                overlap[i, j] = (cross ** 2).sum().item() / k_used

        mask = ~np.eye(n_runs, dtype=bool)
        off = overlap[mask]
        print(f"  top-{k_used}: mean={off.mean():.4f}, min={off.min():.4f}, "
              f"max={off.max():.4f} (random={k_used/d_model:.4f}, "
              f"ratio={off.mean()/(k_used/d_model):.1f}x)")

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
        'kl_zero': kl_zero,
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
    ax.set_title(f'Singular value spectrum ({total_dirs} dirs from {args.n_runs} runs)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, min(len(S), 30) + 0.5)

    # 2. KL divergence vs number of directions
    ax = axes[0, 1]
    for run_idx, res in enumerate(all_run_results):
        kl_hist = res['kl_history']
        ax.plot(range(1, len(kl_hist)+1), kl_hist, 'o-', markersize=4,
                alpha=0.6, label=f'Run {run_idx+1}')
    ax.set_xlabel('Number of directions')
    ax.set_ylabel('KL divergence')
    ax.set_title('KL(original || reconstruction) per run')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 3. Cumulative variance explained
    ax = axes[1, 0]
    cumvar = np.cumsum(S**2) / total_var
    ax.plot(range(1, len(cumvar)+1), cumvar, 'o-', markersize=4)
    ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90%')
    ax.axhline(0.95, color='orange', linestyle='--', alpha=0.5, label='95%')
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative variance explained')
    ax.set_title('Cumulative variance (SVD)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, min(len(cumvar), 30) + 0.5)
    ax.set_ylim(0, 1.05)

    # 4. Per-run magnitudes
    ax = axes[1, 1]
    for run_idx, res in enumerate(all_run_results):
        mags = res['magnitudes']
        xs = range(1, len(mags)+1)
        ax.plot(xs, mags, 'o-', markersize=4, alpha=0.6, label=f'Run {run_idx+1}')
    ax.set_xlabel('Direction index')
    ax.set_ylabel('α (KL-optimized magnitude)')
    ax.set_title('Magnitudes per run (KL-optimized)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Reproducibility (KL-div): '{args.target_prompt}'\n"
                 f"{args.model} layer {args.layer}, k={args.k}, "
                 f"{args.n_runs} runs", fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'svd_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    main()
