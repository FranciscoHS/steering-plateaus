#!/usr/bin/env python3
"""
Two-stage decomposition: find directions via MELBO, then select by KL.

Stage 1: Find 20 candidate directions using existing response-based
  decomposition (MELBO-like: maximize plateau-breaking, enforce orthogonality).
Stage 2: Greedy forward selection — pick directions one at a time to minimize
  KL(p_target || model(reconstruction)), where p_target is the model's output
  distribution at the original activation.

The number of directions needed to get KL below threshold is the effective
dimensionality at this activation.

Usage:
    python scripts/jacobian/kl_selection.py \
        --target_prompt "The Eiffel tower is in" \
        --results_dir results/kl_selection/eiffel
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
from scripts.jacobian.case_study_decomp import (
    spherical_perturb_batch, decompose_one_direction
)


def optimize_coeffs(model, context, directions, activation, target_probs,
                    layer, device, n_steps=100, lr=0.01):
    """
    Optimize scalar coefficients to minimize KL(target || model(Σ c_i d_i)).

    Args:
        directions: list of direction tensors
        activation: original activation (used to initialize coefficients)
        target_probs: softmax output at original activation

    Returns:
        best_coeffs: list of floats
        best_kl: float
        log: dict with optimization trajectory
    """
    k = len(directions)
    D = torch.stack(directions).float()  # [k, d]

    # Initialize with L2 projections
    coeffs = (D @ activation).clone().detach()
    coeffs.requires_grad_(True)
    optimizer = torch.optim.Adam([coeffs], lr=lr)

    best_kl = float('inf')
    best_coeffs = coeffs.detach().clone()
    log = {'kl': [], 'grad_norm': []}

    for step in range(n_steps):
        optimizer.zero_grad()
        recon = (coeffs.unsqueeze(-1) * D).sum(0)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            recon_logits = forward_from_layer(
                model, context, recon.unsqueeze(0).unsqueeze(0), layer)

        recon_log_probs = F.log_softmax(recon_logits, dim=-1)
        kl = F.kl_div(recon_log_probs, target_probs, reduction='batchmean')
        kl.backward()

        log['kl'].append(kl.item())
        log['grad_norm'].append(coeffs.grad.norm().item())

        optimizer.step()

        if kl.item() < best_kl:
            best_kl = kl.item()
            best_coeffs = coeffs.detach().clone()

    return best_coeffs.tolist(), best_kl, log


def main():
    parser = argparse.ArgumentParser(description='KL-based direction selection')
    parser.add_argument('--target_prompt', required=True)
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=7)
    # Stage 1: direction finding
    parser.add_argument('--k', type=int, default=20,
                        help='Number of candidate directions to find')
    parser.add_argument('--theta_deg', type=float, default=10.0)
    parser.add_argument('--lambda_orth', type=float, default=None)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--n_restarts', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--min_residual_frac', type=float, default=0.01)
    # Stage 2: KL selection
    parser.add_argument('--kl_threshold', type=float, default=0.01)
    parser.add_argument('--coeff_steps', type=int, default=30)
    parser.add_argument('--coeff_lr', type=float, default=0.01)
    # General
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    theta = math.radians(args.theta_deg)

    print(f"=== KL-based direction selection ===")
    print(f"Prompt: '{args.target_prompt}'")
    print(f"Config: model={args.model}, layer={args.layer}, k={args.k}")

    # ── Load model ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    end_layer = n_layers - 2

    # ── Extract target activation ──
    target_tokens = tokenizer(args.target_prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**target_tokens, output_hidden_states=True)
    target_hidden = outputs.hidden_states[args.layer + 1]
    target_context = target_hidden[:, :-1, :].detach()
    target_activation = target_hidden[0, -1, :].detach().float()
    act_norm = target_activation.norm().item()
    print(f"  ||activation|| = {act_norm:.2f}, d_model={d_model}")

    # ── Compute target distribution ──
    with torch.no_grad():
        target_logits = forward_from_layer(
            model, target_context,
            target_activation.unsqueeze(0).unsqueeze(0), args.layer)
        target_probs = F.softmax(target_logits, dim=-1).detach()

    # ── Response setup (self mode) ──
    stacked_contexts = target_context
    stacked_activations = target_activation.unsqueeze(0)

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

    if args.lambda_orth is None:
        args.lambda_orth = act_norm * random_response * 10

    # ── Baseline KL ──
    zero_recon = torch.zeros_like(target_activation)
    with torch.no_grad():
        zero_logits = forward_from_layer(
            model, target_context, zero_recon.unsqueeze(0).unsqueeze(0), args.layer)
        zero_log_probs = F.log_softmax(zero_logits, dim=-1)
        kl_zero = F.kl_div(zero_log_probs, target_probs, reduction='batchmean').item()
    print(f"  KL(target || zero) = {kl_zero:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Find candidate directions via MELBO
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"STAGE 1: Finding {args.k} candidate directions")
    print(f"{'='*60}")

    t0 = time.time()
    residual = target_activation.clone()
    candidates = []
    candidate_magnitudes = []
    candidate_responses = []

    for i in range(args.k):
        residual_frac = residual.norm().item() / act_norm
        if residual_frac < args.min_residual_frac:
            print(f"  Stopping: ||residual||/||a|| = {residual_frac:.4f}")
            break

        best_across = {'loss': float('inf'), 'd': None, 'alpha': None, 'response': None}
        for r in range(args.n_restarts):
            torch.manual_seed(args.seed + i * 1000 + r)
            d, alpha, response, loss = decompose_one_direction(
                model, stacked_contexts, stacked_activations, baselines,
                residual, args.layer, end_layer, candidates,
                theta, args.lambda_orth, args.n_steps, args.lr, device
            )
            if loss < best_across['loss']:
                best_across = {'loss': loss, 'd': d, 'alpha': alpha, 'response': response}

        d_best = best_across['d']
        alpha_best = best_across['alpha']
        response_best = best_across['response']

        candidates.append(d_best)
        candidate_magnitudes.append(alpha_best)
        candidate_responses.append(response_best)
        residual = residual - alpha_best * d_best

        print(f"  d_{i+1}: α={alpha_best:.3f}, response={response_best:.4f}")

    stage1_time = time.time() - t0
    print(f"Found {len(candidates)} candidates in {stage1_time:.0f}s")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Greedy forward selection by KL
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"STAGE 2: Greedy KL selection (threshold={args.kl_threshold})")
    print(f"{'='*60}")

    t0 = time.time()
    n_candidates = len(candidates)
    selected_indices = []
    coefficients = []
    kl_history = []
    opt_logs = []

    for round_idx in range(n_candidates):
        best_kl = float('inf')
        best_idx = None
        best_coeffs = None

        for i in range(n_candidates):
            if i in selected_indices:
                continue

            trial_dirs = [candidates[j] for j in selected_indices] + [candidates[i]]
            trial_coeffs, kl, log = optimize_coeffs(
                model, target_context, trial_dirs, target_activation,
                target_probs, args.layer, device,
                n_steps=args.coeff_steps, lr=args.coeff_lr
            )

            if kl < best_kl:
                best_kl = kl
                best_idx = i
                best_coeffs = trial_coeffs
                best_log = log

        selected_indices.append(best_idx)
        coefficients = best_coeffs
        kl_history.append(best_kl)
        opt_logs.append(best_log)

        log = best_log
        print(f"  Round {round_idx+1}: added d_{best_idx+1} "
              f"(α_melbo={candidate_magnitudes[best_idx]:.3f}, "
              f"resp={candidate_responses[best_idx]:.4f}), "
              f"KL={best_kl:.6f}, "
              f"opt: init={log['kl'][0]:.4f}->final={log['kl'][-1]:.4f}, "
              f"grad_norm: init={log['grad_norm'][0]:.4f}->final={log['grad_norm'][-1]:.4f}")

        if best_kl < args.kl_threshold:
            print(f"  -> KL below threshold, stopping")
            break

    stage2_time = time.time() - t0

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Effective dimensionality: {len(selected_indices)} directions")
    print(f"Final KL: {kl_history[-1]:.6f}")
    print(f"KL(target || zero): {kl_zero:.4f}")
    print(f"Selection order (1-indexed):")
    for rank, idx in enumerate(selected_indices):
        print(f"  {rank+1}. d_{idx+1}: α_melbo={candidate_magnitudes[idx]:.3f}, "
              f"response={candidate_responses[idx]:.4f}, "
              f"coeff={coefficients[rank]:.3f}, "
              f"KL_after={kl_history[rank]:.6f}")
    print(f"Stage 1: {stage1_time:.0f}s, Stage 2: {stage2_time:.0f}s")

    # ── Save ──
    results = {
        'args': vars(args),
        'target_prompt': args.target_prompt,
        'target_activation': target_activation.cpu(),
        'candidates': [d.cpu() for d in candidates],
        'candidate_magnitudes': candidate_magnitudes,
        'candidate_responses': candidate_responses,
        'selected_indices': selected_indices,
        'coefficients': coefficients,
        'kl_history': kl_history,
        'opt_logs': opt_logs,
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

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. KL vs number of selected directions
    ax = axes[0]
    ax.plot(range(1, len(kl_history)+1), kl_history, 'o-', markersize=5)
    ax.axhline(kl_zero, color='red', linestyle='--', alpha=0.5,
               label=f'KL(target||zero)={kl_zero:.2f}')
    ax.axhline(args.kl_threshold, color='green', linestyle='--', alpha=0.5,
               label=f'Threshold={args.kl_threshold}')
    ax.set_xlabel('Number of selected directions')
    ax.set_ylabel('KL divergence')
    ax.set_title('KL vs effective dimensionality')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Selection order vs MELBO order
    ax = axes[1]
    melbo_order = list(range(1, n_candidates+1))
    selection_order = [idx+1 for idx in selected_indices]
    ax.plot(range(1, len(selection_order)+1), selection_order, 'o-', markersize=5)
    ax.set_xlabel('KL selection rank')
    ax.set_ylabel('MELBO extraction order')
    ax.set_title('Selection order vs MELBO order')
    ax.grid(True, alpha=0.3)

    # 3. Candidate responses (bar), colored by whether selected early/late/not
    ax = axes[2]
    colors = ['gray'] * n_candidates
    for rank, idx in enumerate(selected_indices):
        if rank < 5:
            colors[idx] = 'tab:blue'
        elif rank < 10:
            colors[idx] = 'tab:orange'
        else:
            colors[idx] = 'tab:green'
    ax.bar(range(1, n_candidates+1), candidate_responses, color=colors)
    ax.set_xlabel('MELBO direction index')
    ax.set_ylabel('Response')
    ax.set_title('Candidate responses\n(blue=selected 1-5, orange=6-10, green=11+)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f"KL selection: '{args.target_prompt}'\n"
                 f"{args.model} layer {args.layer}, "
                 f"{len(selected_indices)} dirs for KL<{args.kl_threshold}",
                 fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'kl_selection.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    main()
