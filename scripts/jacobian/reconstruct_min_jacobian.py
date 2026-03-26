#!/usr/bin/env python3
"""Reconstruct an activation from MELBO directions by minimizing Jacobian norm.

Given N directions found by iterative MELBO, find coefficients c_1..c_N such that
a = sum(c_i * d_i) has minimal Jacobian Frobenius norm (= looks "on the plateau").

Uses the Hutchinson trace estimator via VJP with create_graph=True for second-order
gradients, following the pattern in run_delta_optimization_v8.py.

Usage:
    python scripts/jacobian/reconstruct_min_jacobian.py \
        --direction_pkl results/iterative_melbo_hf/qwen3-1.7b_L7_meas-2_angle0.1_act0/results.pkl
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
    load_model, forward_from_layer_to_final,
    _get_blocks, _get_final_norm, _get_lm_head,
)
from src.data import load_fineweb_fixed_length


def estimate_jacobian_norm(model, ctx, a, layer, n_probes, device):
    """Estimate ||J||_F at point a, differentiable w.r.t. a.

    Returns:
        frob_sq: scalar tensor (differentiable) — estimated ||J||_F^2
    """
    d_model = a.shape[-1]
    point = a.reshape(1, 1, d_model)
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(dtype=model_dtype)
    point_d = point.to(dtype=model_dtype)

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        final_act = forward_from_layer_to_final(model, ctx_d, point_d, layer).float()

    frob_sq = torch.tensor(0.0, device=device)
    for k in range(n_probes):
        u = torch.randn_like(final_act)
        s = (final_act * u).sum()
        g_a = torch.autograd.grad(s, a, create_graph=True, retain_graph=True)[0]
        frob_sq = frob_sq + (g_a ** 2).sum()
    frob_sq = frob_sq / n_probes

    return frob_sq


def estimate_jacobian_norm_detached(model, ctx, a, layer, n_probes, device):
    """Estimate ||J||_F at point a, no gradient tracking. For evaluation."""
    d_model = a.shape[-1]
    point = a.detach().reshape(1, 1, d_model)
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(dtype=model_dtype)

    estimates = []
    for k in range(n_probes):
        point_var = point.clone().requires_grad_(True)
        point_d = point_var.to(dtype=model_dtype)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            final_act = forward_from_layer_to_final(model, ctx_d, point_d, layer).float()
        u = torch.randn_like(final_act)
        s = (final_act * u).sum()
        g = torch.autograd.grad(s, point_var)[0]
        estimates.append((g ** 2).sum().item())

    frob_sq = np.mean(estimates)
    return np.sqrt(max(frob_sq, 0)), np.std(estimates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--direction_pkl', type=str, required=True,
                        help='Path to results.pkl from iterative_melbo_hf')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (default: from pkl)')
    parser.add_argument('--layer', type=int, default=None,
                        help='Injection layer (default: from pkl)')
    parser.add_argument('--n_probes', type=int, default=10,
                        help='Hutchinson probes per optimization step')
    parser.add_argument('--n_eval_probes', type=int, default=50,
                        help='Hutchinson probes for final evaluation')
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--norm_lambda', type=float, default=1.0,
                        help='Weight of norm penalty')
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--act_idx', type=int, default=0)
    parser.add_argument('--n_load', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load directions
    print(f"Loading directions from {args.direction_pkl}")
    with open(args.direction_pkl, 'rb') as f:
        dir_data = pickle.load(f)

    directions = dir_data['directions']
    melbo_args = dir_data['args']
    n_dirs = len(directions)
    model_name = args.model or melbo_args['model']
    layer = args.layer if args.layer is not None else melbo_args['inject_layer']
    print(f"  {n_dirs} directions, model={model_name}, layer={layer}")

    if args.results_dir is None:
        args.results_dir = f'results/reconstruct_min_jacobian/{model_name}_L{layer}_{n_dirs}dirs'
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {model_name}")
    model, tokenizer, device = load_model(model_name)
    d_model = model._plateau_info['d_model']

    # Build direction matrix D [d_model, n_dirs]
    D = torch.stack([d.float() for d in directions], dim=1).to(device)  # [d_model, n_dirs]
    print(f"Direction matrix D: {D.shape}")

    # Load activation + context (same as MELBO used)
    print(f"Loading {args.n_load} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_load, tokenizer, seq_len=args.seq_len, seed=args.seed)

    print(f"Extracting layer-{layer} activations...")
    all_contexts = []
    all_activations = []
    batch_size = 64
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].float().cpu())
        all_activations.append(hidden[:, -1, :].float().cpu())
    all_contexts = torch.cat(all_contexts, 0)
    all_activations = torch.cat(all_activations, 0)

    ctx = all_contexts[args.act_idx].unsqueeze(0).to(device)  # [1, seq-1, d]
    real_act = all_activations[args.act_idx].to(device)        # [d]
    R_target = all_activations.norm(dim=-1).mean().item()
    real_norm = real_act.norm().item()
    print(f"Real activation norm: {real_norm:.2f}, target norm (mean): {R_target:.2f}")

    # Project real activation onto directions for reference
    real_coeffs = (D.T @ real_act).cpu().numpy()  # [n_dirs]
    real_proj_norm = np.sqrt((real_coeffs ** 2).sum())
    print(f"Real activation projection onto {n_dirs} dirs: norm={real_proj_norm:.2f}, "
          f"fraction of variance: {real_proj_norm**2 / real_norm**2:.4f}")

    # === Reference Jacobian norms ===
    print(f"\nComputing reference Jacobian norms ({args.n_eval_probes} probes)...")

    print("  Real activation...")
    real_jac_norm, real_jac_std = estimate_jacobian_norm_detached(
        model, ctx, real_act, layer, args.n_eval_probes, device)
    print(f"  ||J||_F (real) = {real_jac_norm:.1f} (std={real_jac_std:.1f})")

    print("  Random activation (same norm)...")
    rand_act = F.normalize(torch.randn(d_model, device=device), dim=0) * R_target
    rand_jac_norm, rand_jac_std = estimate_jacobian_norm_detached(
        model, ctx, rand_act, layer, args.n_eval_probes, device)
    print(f"  ||J||_F (random) = {rand_jac_norm:.1f} (std={rand_jac_std:.1f})")

    # === Optimization ===
    print(f"\n{'='*60}")
    print(f"Optimizing {n_dirs} coefficients, {args.n_restarts} restarts, "
          f"{args.n_steps} steps")
    print(f"norm_lambda={args.norm_lambda}, R_target={R_target:.2f}")
    print(f"{'='*60}")

    best_frob = float('inf')
    best_c = None
    best_trajectory = None

    for restart in range(args.n_restarts):
        print(f"\n--- Restart {restart+1}/{args.n_restarts} ---")

        # Initialize c so ||Dc|| ≈ R_target
        # Since directions are orthonormal, ||Dc|| = ||c||, so init ||c|| = R_target
        c = torch.randn(n_dirs, device=device)
        c = c / c.norm() * R_target
        c = c.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([c], lr=args.lr)
        trajectory = []

        for step in range(args.n_steps):
            optimizer.zero_grad()

            a = D @ c  # [d_model]

            # Jacobian norm (differentiable)
            frob_sq = estimate_jacobian_norm(
                model, ctx, a, layer, args.n_probes, device)

            # Norm penalty
            a_norm = a.norm()
            norm_loss = args.norm_lambda * (a_norm - R_target) ** 2

            loss = frob_sq + norm_loss
            loss.backward()
            optimizer.step()

            frob_val = frob_sq.item()
            norm_val = a_norm.item()
            trajectory.append({
                'step': step,
                'frob_sq': frob_val,
                'frob_norm': np.sqrt(max(frob_val, 0)),
                'a_norm': norm_val,
                'loss': loss.item(),
            })

            if step % 50 == 0 or step == args.n_steps - 1:
                print(f"  step {step}: ||J||_F={np.sqrt(max(frob_val,0)):.1f}, "
                      f"||a||={norm_val:.2f}, loss={loss.item():.1f}")

        final_frob = np.sqrt(max(trajectory[-1]['frob_sq'], 0))
        if final_frob < best_frob:
            best_frob = final_frob
            best_c = c.detach().clone()
            best_trajectory = trajectory
            print(f"  ** New best: ||J||_F={final_frob:.1f}")

    # === Evaluation ===
    print(f"\n{'='*60}")
    print(f"EVALUATION")
    print(f"{'='*60}")

    a_recon = (D @ best_c).detach()
    recon_norm = a_recon.norm().item()

    print(f"\nReconstructed activation: norm={recon_norm:.2f}")
    print(f"Coefficients: min={best_c.min().item():.2f}, max={best_c.max().item():.2f}, "
          f"mean={best_c.mean().item():.2f}")

    # Cosine similarity to real activation
    cos_to_real = F.cosine_similarity(a_recon.unsqueeze(0), real_act.unsqueeze(0)).item()
    print(f"Cosine similarity to real activation: {cos_to_real:.4f}")

    # Jacobian norm with more probes
    print(f"\nJacobian norms ({args.n_eval_probes} probes):")
    recon_jac_norm, recon_jac_std = estimate_jacobian_norm_detached(
        model, ctx, a_recon, layer, args.n_eval_probes, device)
    print(f"  Reconstructed: ||J||_F = {recon_jac_norm:.1f} (std={recon_jac_std:.1f})")
    print(f"  Real:          ||J||_F = {real_jac_norm:.1f} (std={real_jac_std:.1f})")
    print(f"  Random:        ||J||_F = {rand_jac_norm:.1f} (std={rand_jac_std:.1f})")

    # Logit entropy
    print(f"\nLogit entropy:")
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(dtype=model_dtype)
    for name, act_vec in [('reconstructed', a_recon), ('real', real_act), ('random', rand_act)]:
        with torch.no_grad():
            pt = act_vec.reshape(1, 1, d_model).to(dtype=model_dtype)
            final = forward_from_layer_to_final(model, ctx_d, pt, layer).float()
            logits = _get_lm_head(model)(_get_final_norm(model)(final.unsqueeze(1)).squeeze(1))
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * probs.log()).sum().item()
        print(f"  {name}: H = {entropy:.2f} nats")

    # Top tokens
    print(f"\nTop-5 tokens:")
    for name, act_vec in [('reconstructed', a_recon), ('real', real_act)]:
        with torch.no_grad():
            pt = act_vec.reshape(1, 1, d_model).to(dtype=model_dtype)
            final = forward_from_layer_to_final(model, ctx_d, pt, layer).float()
            logits = _get_lm_head(model)(_get_final_norm(model)(final.unsqueeze(1)).squeeze(1))
            probs = F.softmax(logits, dim=-1)
            top5 = probs[0].topk(5)
        tokens = [tokenizer.decode([t]) for t in top5.indices.tolist()]
        probs_str = ', '.join(f'{t!r}:{p:.3f}' for t, p in zip(tokens, top5.values.tolist()))
        print(f"  {name}: {probs_str}")

    # === Save ===
    results = {
        'coefficients': best_c.cpu(),
        'reconstructed_activation': a_recon.cpu(),
        'real_activation': real_act.cpu(),
        'random_activation': rand_act.cpu(),
        'direction_matrix': D.cpu(),
        'trajectory': best_trajectory,
        'jacobian_norms': {
            'reconstructed': (recon_jac_norm, recon_jac_std),
            'real': (real_jac_norm, real_jac_std),
            'random': (rand_jac_norm, rand_jac_std),
        },
        'cos_to_real': cos_to_real,
        'args': vars(args),
        'melbo_args': melbo_args,
    }
    pkl_path = os.path.join(args.results_dir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # === Plot ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: optimization trajectory
    ax = axes[0]
    steps = [t['step'] for t in best_trajectory]
    frob_norms = [t['frob_norm'] for t in best_trajectory]
    ax.plot(steps, frob_norms, 'b-', lw=1.5)
    ax.axhline(real_jac_norm, color='green', ls='--', lw=1.5, label=f'Real ({real_jac_norm:.0f})')
    ax.axhline(rand_jac_norm, color='red', ls='--', lw=1.5, label=f'Random ({rand_jac_norm:.0f})')
    ax.set_xlabel('Step')
    ax.set_ylabel('||J||_F')
    ax.set_title('Jacobian norm during optimization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Middle: activation norm trajectory
    ax = axes[1]
    norms = [t['a_norm'] for t in best_trajectory]
    ax.plot(steps, norms, 'b-', lw=1.5)
    ax.axhline(R_target, color='green', ls='--', lw=1.5, label=f'Target ({R_target:.1f})')
    ax.axhline(real_norm, color='orange', ls='--', lw=1.5, label=f'Real ({real_norm:.1f})')
    ax.set_xlabel('Step')
    ax.set_ylabel('||a||')
    ax.set_title('Activation norm during optimization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: coefficient magnitudes
    ax = axes[2]
    coeffs = best_c.cpu().numpy()
    ax.bar(range(n_dirs), coeffs)
    ax.set_xlabel('Direction index')
    ax.set_ylabel('Coefficient')
    ax.set_title(f'Optimized coefficients (n={n_dirs})')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'{model_name} L{layer}: Jacobian-minimizing reconstruction from {n_dirs} MELBO dirs\n'
                 f'||J||_F: recon={recon_jac_norm:.0f}, real={real_jac_norm:.0f}, '
                 f'random={rand_jac_norm:.0f}',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(args.results_dir, 'reconstruction.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == '__main__':
    main()
