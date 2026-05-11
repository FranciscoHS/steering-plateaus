#!/usr/bin/env python3
"""
Greedy Jacobian-guided feature decomposition of activations.

Decomposes activation a into k features {(d_i, alpha_i)} by iteratively solving:
  min_{d,alpha}: -alpha + lambda_jac * ||J(r - alpha*d)||^2_F
                        + lambda_orth * sum_{j<i} (d . d_j)^2
  s.t. ||d|| = 1
where r is the current residual and J = d(final_layer_act) / d(layer_act).

Uses VJP-based Hutchinson estimator for ||J||^2_F, with create_graph=True
for second-order optimization (differentiating through the Jacobian norm).

Usage:
    # Calibrate first to see term scales
    python scripts/dictionary_learning/feature_decomp.py --calibrate_only

    # Run decomposition
    python scripts/dictionary_learning/feature_decomp.py --k 5

    # With explicit lambdas
    python scripts/dictionary_learning/feature_decomp.py --k 5 --lambda_jac 0.001 --lambda_orth 500
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model, forward_from_layer_to_final
from src.data import load_fineweb_fixed_length
from src.analysis import compute_jacobian_norm_vjp


def extract_single_activation(model, token_ids, layer, device):
    """Get context (prefix at layer L) and last-token activation."""
    tokens = token_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    hidden = outputs.hidden_states[layer + 1]
    context = hidden[:, :-1, :].detach()
    activation = hidden[0, -1, :].detach().float()
    return context, activation


def decompose_one_direction(model, context, residual, layer, prev_directions,
                            lambda_jac, lambda_orth, n_probes, n_steps, lr, device):
    """
    Find direction d minimizing the combined objective, with alpha = d^T residual.

    Uses create_graph=True on the VJP so the Jacobian norm is differentiable
    w.r.t. d (second-order autograd).
    """
    d_model = residual.shape[0]

    # Random initialization
    d_raw = torch.randn(d_model, device=device, dtype=torch.float32)
    d_raw.requires_grad_(True)

    optimizer = torch.optim.Adam([d_raw], lr=lr)

    # Fixed probe vectors for stability within this run
    probes = [torch.randn(1, d_model, device=device, dtype=torch.float32)
              for _ in range(n_probes)]

    best = {'loss': float('inf'), 'd': None, 'alpha': None}
    log = {'loss': [], 'alpha': [], 'jac': [], 'orth': []}

    r = residual.detach()

    for step in range(n_steps):
        optimizer.zero_grad()

        d_hat = d_raw / (d_raw.norm() + 1e-8)
        alpha = (d_hat @ r)  # projection — differentiable w.r.t. d_raw

        # Point after removal: x = r - (d^T r) * d
        x_point = (r - alpha * d_hat).reshape(1, 1, d_model)

        # Forward through remaining layers (MATH backend for 2nd-order compat)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            final_act = forward_from_layer_to_final(model, context, x_point, layer)

        # Stochastic ||J||^2_F via VJP with create_graph
        jac_est = torch.tensor(0.0, device=device)
        for u in probes:
            s = (final_act * u).sum()
            g = torch.autograd.grad(s, x_point, create_graph=True,
                                    retain_graph=True)[0]
            jac_est = jac_est + g.pow(2).sum()
        jac_est = jac_est / n_probes

        # Orthogonality penalty
        orth = torch.tensor(0.0, device=device)
        for d_prev in prev_directions:
            orth = orth + (d_hat @ d_prev).pow(2)

        loss = -alpha + lambda_jac * jac_est + lambda_orth * orth
        loss.backward()
        torch.nn.utils.clip_grad_norm_([d_raw], 1.0)
        optimizer.step()

        lv = loss.item()
        log['loss'].append(lv)
        log['alpha'].append(alpha.item())
        log['jac'].append(jac_est.item())
        log['orth'].append(orth.item())

        if lv < best['loss']:
            best['loss'] = lv
            best['d'] = d_hat.detach().clone()
            best['alpha'] = alpha.item()

    return best['d'], best['alpha'], best['loss'], log


def main():
    parser = argparse.ArgumentParser(description='Jacobian-guided feature decomposition')
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=7)
    parser.add_argument('--k', type=int, default=100, help='Max number of features to extract')
    parser.add_argument('--min_residual_frac', type=float, default=0.05,
                        help='Stop when ||residual|| / ||activation|| < this (default: 0.05)')
    parser.add_argument('--lambda_jac', type=float, default=None)
    parser.add_argument('--lambda_orth', type=float, default=None)
    parser.add_argument('--n_probes', type=int, default=5,
                        help='VJP probes per step for Jacobian estimate')
    parser.add_argument('--n_steps', type=int, default=100,
                        help='Optimization steps per restart')
    parser.add_argument('--n_restarts', type=int, default=5,
                        help='Random restarts per direction')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Which FineWeb sample to decompose')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default='results/feature_decomp')
    parser.add_argument('--calibrate_only', action='store_true',
                        help='Only measure term scales, do not run decomposition')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Config: model={args.model}, layer={args.layer}, k={args.k}")
    print(f"  n_probes={args.n_probes}, n_steps={args.n_steps}, "
          f"n_restarts={args.n_restarts}, lr={args.lr}")

    # Load model
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"  d_model={d_model}, device={device}")

    # Load data and extract activation
    token_ids_list = load_fineweb_fixed_length(
        args.sample_idx + 1, tokenizer, seq_len=args.seq_len, seed=args.seed
    )
    context, activation = extract_single_activation(
        model, token_ids_list[args.sample_idx], args.layer, device
    )
    act_norm = activation.norm().item()
    print(f"  ||activation|| = {act_norm:.2f}")

    # Calibration: measure Jacobian norms at key points
    print("\nCalibrating...")
    t0 = time.time()
    jac_at_act, _ = compute_jacobian_norm_vjp(
        model, context, activation.to(device), args.layer, n_probes=50
    )
    jac_at_half, _ = compute_jacobian_norm_vjp(
        model, context, (0.5 * activation).to(device), args.layer, n_probes=50
    )
    jac_at_zero, _ = compute_jacobian_norm_vjp(
        model, context, torch.zeros(d_model, device=device), args.layer, n_probes=50
    )
    print(f"  ||J(a)||_F   = {jac_at_act:.1f}  (squared: {jac_at_act**2:.1f})")
    print(f"  ||J(a/2)||_F = {jac_at_half:.1f}")
    print(f"  ||J(0)||_F   = {jac_at_zero:.1f}")
    print(f"  ({time.time()-t0:.1f}s)")

    # Auto-set lambdas: balance -alpha ~ lambda_jac * ||J||^2_F
    if args.lambda_jac is None:
        args.lambda_jac = act_norm / (jac_at_act ** 2 + 1e-8)
        print(f"  Auto lambda_jac = {args.lambda_jac:.6f}")
    if args.lambda_orth is None:
        args.lambda_orth = act_norm * 10
        print(f"  Auto lambda_orth = {args.lambda_orth:.2f}")

    cal = {
        'act_norm': act_norm,
        'jac_at_act': jac_at_act,
        'jac_at_half': jac_at_half,
        'jac_at_zero': jac_at_zero,
        'lambda_jac': args.lambda_jac,
        'lambda_orth': args.lambda_orth,
    }

    if args.calibrate_only:
        import json
        cal_path = os.path.join(args.results_dir, 'calibration.json')
        with open(cal_path, 'w') as f:
            json.dump(cal, f, indent=2)
        print(f"\nCalibration saved to {cal_path}")
        return

    # Greedy decomposition
    print(f"\n{'='*60}")
    print(f"Decomposing: k<={args.k}, stop at {args.min_residual_frac:.0%} residual, "
          f"lambda_jac={args.lambda_jac:.6f}, lambda_orth={args.lambda_orth:.1f}")
    print(f"{'='*60}")

    residual = activation.clone()
    directions = []
    magnitudes = []
    jac_norms_after = []
    all_logs = []

    for i in range(args.k):
        residual_frac = residual.norm().item() / act_norm
        if residual_frac < args.min_residual_frac:
            print(f"\n--- Stopping: ||residual||/||a|| = {residual_frac:.4f} "
                  f"< {args.min_residual_frac} ---")
            break
        print(f"\n--- Direction {i+1}/{args.k} | ||residual|| = {residual.norm():.2f} "
              f"({residual_frac:.1%} of original) ---")

        best_across = {'loss': float('inf'), 'd': None, 'alpha': None, 'log': None}

        t0 = time.time()
        for r in range(args.n_restarts):
            torch.manual_seed(args.seed + i * 1000 + r)
            d, alpha, loss, log = decompose_one_direction(
                model, context, residual, args.layer, directions,
                args.lambda_jac, args.lambda_orth, args.n_probes,
                args.n_steps, args.lr, device
            )
            if loss < best_across['loss']:
                best_across = {'loss': loss, 'd': d, 'alpha': alpha, 'log': log}

            print(f"  restart {r+1}/{args.n_restarts}: "
                  f"alpha={alpha:.3f}, loss={loss:.3f}, "
                  f"jac={log['jac'][-1]:.3f}, orth={log['orth'][-1]:.5f}")

        elapsed = time.time() - t0
        d_best = best_across['d']
        alpha_best = best_across['alpha']

        directions.append(d_best)
        magnitudes.append(alpha_best)
        all_logs.append(best_across['log'])

        # Update residual
        residual = residual - alpha_best * d_best

        # Evaluate Jacobian at new residual (non-differentiable, for reporting)
        jac_after, _ = compute_jacobian_norm_vjp(
            model, context, residual.to(device), args.layer, n_probes=50
        )
        jac_norms_after.append(jac_after)
        print(f"  -> best alpha={alpha_best:.3f}, "
              f"||J(residual)||_F = {jac_after:.1f} "
              f"(baseline={jac_at_act:.1f}) [{elapsed:.0f}s]")

    # Cosine similarity matrix
    n = len(directions)
    cosines = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            cosines[a, b] = (directions[a] @ directions[b]).item()

    # Reconstruction quality
    recon = sum(m * d for m, d in zip(magnitudes, directions))
    recon_error = (activation - recon).norm().item() / act_norm

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline ||J||_F = {jac_at_act:.1f}")
    for i in range(args.k):
        print(f"  d_{i+1}: alpha={magnitudes[i]:.3f}, "
              f"||J||_F after = {jac_norms_after[i]:.1f}")
    print(f"Reconstruction error (relative): {recon_error:.4f}")
    print(f"Final ||residual|| = {residual.norm():.2f} "
          f"(started at {act_norm:.2f})")
    print(f"\nCosine similarities:\n{cosines.round(3)}")

    # Save results
    results = {
        'args': vars(args),
        'calibration': cal,
        'directions': [d.cpu() for d in directions],
        'magnitudes': magnitudes,
        'jac_baseline': jac_at_act,
        'jac_norms_after': jac_norms_after,
        'cosines': cosines,
        'recon_error': recon_error,
        'residual_norm': residual.norm().item(),
        'activation': activation.cpu(),
        'context': context.cpu(),
        'opt_logs': all_logs,
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
    ax.set_ylabel('alpha')
    ax.set_title('Feature magnitudes')
    ax.grid(True, alpha=0.3)

    # Jacobian norm decay
    ax = axes[0, 1]
    jvals = [jac_at_act] + jac_norms_after
    ax.plot(range(n+1), jvals, 'o-')
    ax.set_xlabel('Features removed')
    ax.set_ylabel('||J||_F')
    ax.set_title('Jacobian norm after successive removals')
    ax.set_xticks(range(n+1))
    ax.grid(True, alpha=0.3)

    # Cosine similarities
    ax = axes[1, 0]
    im = ax.imshow(cosines, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Direction cosine similarities')
    for a in range(n):
        for b in range(n):
            ax.text(b, a, f'{cosines[a,b]:.2f}', ha='center', va='center',
                    fontsize=8)
    plt.colorbar(im, ax=ax)

    # Optimization curves
    ax = axes[1, 1]
    for i, log in enumerate(all_logs):
        ax.plot(log['loss'], label=f'd_{i+1}', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Optimization curves (best restart)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Feature decomposition: {args.model} layer {args.layer}, k={args.k}',
                 fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    main()
