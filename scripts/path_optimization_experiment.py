#!/usr/bin/env python3
"""
Path optimization: find paths in activation space that minimize Jacobian norm
while keeping path length short.

The path lives in layer 2 activation space (last token only) of Qwen3-0.6B.
It's parameterized by N free knots, with spherical interpolation between
consecutive waypoints (endpoints + knots). The loss is:

    L = softmax_jac_loss + (path_length / L_baseline)

where softmax_jac_loss = (1/alpha) * log(mean(exp(alpha * ||J||_F^2 / J_baseline)))
is a soft-max over sampled Jacobian norms (approximating max for large alpha),
and J_baseline and L_baseline are measured along the slerp path between
the two endpoints, so both terms start at ~1.

The Jacobian is d(layer_end activation) / d(layer_start activation) for
the last token position, estimated via Hutchinson VJP.

Usage:
    python scripts/path_optimization_experiment.py
    python scripts/path_optimization_experiment.py --n_knots 10 --n_steps 300
    python scripts/path_optimization_experiment.py --resume
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from src.model import (load_model, extract_activations, forward_from_layer,
                       forward_from_layer_to_layer)
from src.analysis import compute_logit_entropy


# =========================================================================
# Slerp utilities (differentiable)
# =========================================================================

def slerp_point(v0, v1, t):
    """
    Differentiable slerp between two vectors at parameter t.

    Spherical interpolation for direction, linear interpolation for magnitude.
    All operations are differentiable w.r.t. v0 and v1.

    Args:
        v0: [d_model] tensor
        v1: [d_model] tensor
        t: scalar (float or 0-d tensor)

    Returns:
        [d_model] tensor
    """
    norm0 = v0.norm()
    norm1 = v1.norm()

    u0 = v0 / norm0
    u1 = v1 / norm1

    dot = torch.clamp(torch.dot(u0, u1), -1.0, 1.0)
    theta = torch.acos(dot)

    if theta.abs() < 1e-6:
        # Nearly parallel — fall back to linear
        direction = u0
    else:
        sin_theta = torch.sin(theta)
        direction = (torch.sin((1 - t) * theta) / sin_theta) * u0 + \
                    (torch.sin(t * theta) / sin_theta) * u1

    mag = (1 - t) * norm0 + t * norm1
    return direction * mag


def get_waypoints(start_act, knots, end_act):
    """
    Assemble the full waypoint sequence: [start, knots..., end].

    Args:
        start_act: [d_model] tensor (fixed)
        knots: [n_knots, d_model] tensor (optimizable)
        end_act: [d_model] tensor (fixed)

    Returns:
        [n_knots + 2, d_model] tensor
    """
    return torch.cat([start_act.unsqueeze(0), knots, end_act.unsqueeze(0)], dim=0)


def compute_chord_length(waypoints):
    """
    Compute total chord length between consecutive waypoints.

    Differentiable w.r.t. waypoints.

    Args:
        waypoints: [n_waypoints, d_model] tensor

    Returns:
        total_length: scalar tensor
    """
    diffs = waypoints[1:] - waypoints[:-1]
    return diffs.norm(dim=1).sum()


def sample_path_point(waypoints, seg_idx, t):
    """
    Sample a single point along the piecewise-slerp path.

    Differentiable w.r.t. waypoints.

    Args:
        waypoints: [n_waypoints, d_model] tensor
        seg_idx: which segment (0 to n_segments-1)
        t: parameter within segment [0, 1]

    Returns:
        [d_model] tensor
    """
    return slerp_point(waypoints[seg_idx], waypoints[seg_idx + 1], t)


# =========================================================================
# Jacobian norm computation
# =========================================================================

def estimate_jac_norm(model, context, point, start_layer, end_layer, n_probes=100):
    """
    Estimate ||J||_F for J mapping start_layer -> end_layer (last token only).

    Non-differentiable version for baseline/evaluation (more probes, no graph).
    Like compute_jacobian_norm_vjp but uses forward_from_layer_to_layer
    so the Jacobian is consistent with the optimization loss.

    Args:
        model: language model (parameters frozen)
        context: [1, seq_len-1, d_model]
        point: [d_model] tensor
        start_layer, end_layer: layer indices
        n_probes: number of Hutchinson probes

    Returns:
        frob_norm: estimated ||J||_F
        frob_sq_std: std of per-probe estimates
    """
    d_model = point.shape[-1]
    device = point.device

    estimates = []
    for k in range(n_probes):
        point_var = point.clone().reshape(1, 1, d_model).requires_grad_(True)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            final_act = forward_from_layer_to_layer(
                model, context, point_var, start_layer, end_layer
            )

        u = torch.randn_like(final_act)
        s = (final_act * u).sum()
        g = torch.autograd.grad(s, point_var)[0]

        estimates.append((g ** 2).sum().item())

    estimates = np.array(estimates)
    frob_sq = np.mean(estimates)
    frob_norm = np.sqrt(max(frob_sq, 0))
    frob_sq_std = np.std(estimates)

    return frob_norm, frob_sq_std


def jacobian_norm_at_point(model, context, point, start_layer, end_layer, n_probes=5):
    """
    Compute a differentiable estimate of ||J||_F^2 at a single point.

    Uses Hutchinson VJP estimator with create_graph=True so gradients
    flow back to the point (and through it to the knots).

    Adapted from scripts/run_delta_optimization.py optimization_step pattern.

    Args:
        model: language model (parameters frozen)
        context: [1, seq_len-1, d_model] prefix at start_layer
        point: [d_model] tensor (may carry gradients from knots through slerp)
        start_layer: layer index to start from
        end_layer: layer index to stop at (inclusive)
        n_probes: number of Hutchinson probes

    Returns:
        frob_sq: scalar tensor (differentiable estimate of ||J||_F^2)
    """
    d_model = point.shape[0]
    device = point.device

    point_3d = point.reshape(1, 1, d_model)

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        final_act = forward_from_layer_to_layer(model, context, point_3d, start_layer, end_layer)

    # Hutchinson VJP: E[||J^T u||^2] = ||J||_F^2
    frob_sq = torch.tensor(0.0, device=device)
    for k in range(n_probes):
        u = torch.randn_like(final_act)
        s = (final_act * u).sum()
        g = torch.autograd.grad(s, point, create_graph=True, retain_graph=True)[0]
        frob_sq = frob_sq + (g ** 2).sum()

    return frob_sq / n_probes


# =========================================================================
# Baseline computation
# =========================================================================

def compute_slerp_baseline(model, context, start_act, end_act,
                           start_layer, end_layer, n_points=20, n_probes=50):
    """
    Compute Jacobian norm stats and path length for the slerp baseline.

    Uses the non-differentiable VJP estimator (compute_jacobian_norm_vjp)
    for accuracy, since this is just a one-time measurement.

    Args:
        model: language model
        context: [1, seq_len-1, d_model]
        start_act, end_act: [d_model] tensors
        start_layer, end_layer: layer indices
        n_points: number of points to sample along slerp
        n_probes: Hutchinson probes per point

    Returns:
        j_baseline: mean ||J||_F^2 along slerp path
        l_baseline: total chord length of slerp path
        slerp_norms: list of ||J||_F values at each point
    """
    device = start_act.device

    # Generate slerp waypoints (dense, for accurate length)
    n_dense = 100
    t_dense = torch.linspace(0, 1, n_dense, device=device)
    dense_points = torch.stack([slerp_point(start_act, end_act, t) for t in t_dense])
    diffs = dense_points[1:] - dense_points[:-1]
    l_baseline = diffs.norm(dim=1).sum().item()

    # Measure Jacobian norm at n_points evenly spaced points
    t_values = torch.linspace(0, 1, n_points, device=device)
    slerp_norms = []
    slerp_norms_sq = []

    print(f"  Computing slerp baseline ({n_points} points, {n_probes} probes)...")
    for i, t in enumerate(t_values):
        point = slerp_point(start_act.detach(), end_act.detach(), t.item())
        frob_norm, _ = estimate_jac_norm(
            model, context, point, start_layer, end_layer, n_probes=n_probes
        )
        slerp_norms.append(frob_norm)
        slerp_norms_sq.append(frob_norm ** 2)
        print(f"    {i+1}/{n_points}: t={t.item():.2f}, ||J||_F={frob_norm:.1f}", end='\r')

    print()

    j_baseline = np.mean(slerp_norms_sq)  # mean ||J||_F^2

    print(f"  Slerp baseline: mean ||J||_F = {np.mean(slerp_norms):.1f}, "
          f"mean ||J||_F^2 = {j_baseline:.1f}, path length = {l_baseline:.1f}")

    return j_baseline, l_baseline, slerp_norms


# =========================================================================
# Optimization
# =========================================================================

def optimization_step(model, context, knots, start_act, end_act,
                      start_layer, end_layer, n_jac_samples, n_probes,
                      j_baseline, l_baseline, alpha=3.0):
    """
    One optimization step.

    Samples random points along the path, estimates Jacobian norm at each,
    computes path length, and returns the combined normalized loss.

    The Jacobian loss uses log-mean-exp (soft-max) aggregation:
        L_jac = (1/alpha) * log(mean(exp(alpha * ||J||_F^2 / J_baseline)))
    This approximates max(||J||_F^2) for large alpha, penalizing the worst
    points much more heavily than the mean.

    Args:
        model: language model
        context: [1, seq_len-1, d_model]
        knots: nn.Parameter [n_knots, d_model]
        start_act, end_act: [d_model] tensors (fixed)
        start_layer, end_layer: layer indices
        n_jac_samples: number of points to sample for Jacobian
        n_probes: Hutchinson probes per point
        j_baseline: normalization constant for Jacobian term
        l_baseline: normalization constant for path length term
        alpha: soft-max temperature (higher = closer to max)

    Returns:
        total_loss: scalar (for logging)
        jac_loss_val: Jacobian term contribution (for logging)
        length_loss_val: path length term contribution (for logging)
    """
    device = knots.device
    waypoints = get_waypoints(start_act, knots, end_act)
    n_segments = waypoints.shape[0] - 1

    # --- Jacobian norm term (soft-max aggregation) ---
    # Collect normalized Jacobian norms at sampled points, keeping graphs
    frob_sq_normalized = []
    for i in range(n_jac_samples):
        seg_idx = torch.randint(0, n_segments, (1,)).item()
        t = torch.rand(1).item()

        point = sample_path_point(waypoints, seg_idx, t)
        frob_sq = jacobian_norm_at_point(
            model, context, point, start_layer, end_layer, n_probes
        )
        frob_sq_normalized.append(frob_sq / j_baseline)

    # Log-mean-exp: (1/alpha) * log(mean(exp(alpha * x_i)))
    stacked = torch.stack(frob_sq_normalized)
    jac_loss = (1.0 / alpha) * (
        torch.logsumexp(alpha * stacked, dim=0) - np.log(len(stacked))
    )
    jac_loss.backward()
    jac_loss_val = jac_loss.item()

    # --- Path length term ---
    waypoints = get_waypoints(start_act, knots, end_act)
    length = compute_chord_length(waypoints)
    length_loss = length / l_baseline
    length_loss.backward()
    length_loss_val = length_loss.item()

    total_loss = jac_loss_val + length_loss_val
    return total_loss, jac_loss_val, length_loss_val


# =========================================================================
# Post-optimization analysis
# =========================================================================

def evaluate_path_dense(model, tokenizer, context, waypoints,
                        start_layer, end_layer, n_points=100, n_probes=50):
    """
    Dense evaluation of a path: Jacobian norm, logit entropy, top predictions.

    Args:
        model: language model
        tokenizer: tokenizer
        context: [1, seq_len-1, d_model]
        waypoints: [n_waypoints, d_model]
        start_layer, end_layer: layer indices
        n_points: number of evenly-spaced evaluation points
        n_probes: Hutchinson probes per point

    Returns:
        dict with 't_values', 'jac_norms', 'entropies', 'predictions'
    """
    device = waypoints.device
    n_segments = waypoints.shape[0] - 1

    t_global = np.linspace(0, 1, n_points)
    jac_norms = []
    entropies = []
    predictions = []

    print(f"  Evaluating path ({n_points} points, {n_probes} probes)...")
    for i, tg in enumerate(t_global):
        # Map global t to segment + local t
        seg_float = tg * n_segments
        seg_idx = min(int(seg_float), n_segments - 1)
        t_local = seg_float - seg_idx

        with torch.no_grad():
            point = slerp_point(
                waypoints[seg_idx].detach(),
                waypoints[seg_idx + 1].detach(),
                t_local
            )

        # Jacobian norm (non-differentiable, accurate)
        frob_norm, _ = estimate_jac_norm(
            model, context, point, start_layer, end_layer, n_probes=n_probes
        )
        jac_norms.append(frob_norm)

        # Logit entropy
        point_3d = point.reshape(1, 1, -1)
        entropy = compute_logit_entropy(model, context, point_3d, layer=start_layer)
        entropies.append(entropy)

        # Top predictions at a few points
        if i % (n_points // 10) == 0 or i == n_points - 1:
            with torch.no_grad():
                logits = forward_from_layer(model, context, point_3d, start_layer)
                probs = torch.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs.squeeze(0), 5)
                tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
                predictions.append({
                    't': tg,
                    'tokens': tokens,
                    'probs': top_probs.cpu().numpy(),
                    'entropy': entropy,
                })

        print(f"    {i+1}/{n_points}: t={tg:.2f}, ||J||_F={frob_norm:.1f}, "
              f"H={entropy:.2f}", end='\r')

    print()

    return {
        't_values': t_global,
        'jac_norms': np.array(jac_norms),
        'entropies': np.array(entropies),
        'predictions': predictions,
    }


def plot_results(slerp_eval, opt_eval, history, save_dir):
    """Generate all analysis plots."""

    # 1. Jacobian norm comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(slerp_eval['t_values'], slerp_eval['jac_norms'],
            'b-', linewidth=2, label='Slerp path', alpha=0.7)
    ax.plot(opt_eval['t_values'], opt_eval['jac_norms'],
            'r-', linewidth=2, label='Optimized path', alpha=0.7)
    ax.set_xlabel('Path parameter t')
    ax.set_ylabel('||J||_F')
    ax.set_title('Jacobian Frobenius Norm Along Path')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'jacobian_norm_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Logit entropy comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(slerp_eval['t_values'], slerp_eval['entropies'],
            'b-', linewidth=2, label='Slerp path', alpha=0.7)
    ax.plot(opt_eval['t_values'], opt_eval['entropies'],
            'r-', linewidth=2, label='Optimized path', alpha=0.7)
    ax.set_xlabel('Path parameter t')
    ax.set_ylabel('Logit Entropy')
    ax.set_title('Logit Entropy Along Path')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'entropy_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Optimization loss curve
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['total_loss'])
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['jac_loss'], label='Jacobian')
    axes[1].plot(history['length_loss'], label='Path length')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Normalized Loss')
    axes[1].set_title('Loss Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['grad_norm'])
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Gradient Norm')
    axes[2].set_title('Gradient Norm')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'optimization_history.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {save_dir}")


def print_path_predictions(predictions, label):
    """Print top predictions at sampled points along a path."""
    print(f"\n{label}:")
    for pred in predictions:
        tokens_str = ', '.join(
            f"'{t}' ({p:.3f})" for t, p in zip(pred['tokens'][:3], pred['probs'][:3])
        )
        print(f"  t={pred['t']:.2f}: {tokens_str}  (H={pred['entropy']:.2f})")


# =========================================================================
# Checkpointing
# =========================================================================

def save_checkpoint(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Checkpoint loaded: {path}")
    return data


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='Path optimization to minimize Jacobian norm')
    parser.add_argument('--model', type=str, default='qwen3-0.6b')
    parser.add_argument('--prompt_a', type=str, default='The house was big')
    parser.add_argument('--prompt_b', type=str, default='The house was in')
    parser.add_argument('--start_layer', type=int, default=2)
    parser.add_argument('--n_knots', type=int, default=10)
    parser.add_argument('--n_probes', type=int, default=5,
                        help='Hutchinson probes during optimization')
    parser.add_argument('--n_probes_baseline', type=int, default=50,
                        help='Hutchinson probes for baseline measurement')
    parser.add_argument('--n_probes_eval', type=int, default=50,
                        help='Hutchinson probes for post-optimization evaluation')
    parser.add_argument('--n_jac_samples', type=int, default=15,
                        help='Points sampled per step for Jacobian')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm (0 to disable)')
    parser.add_argument('--alpha', type=float, default=3.0,
                        help='Soft-max temperature for Jacobian loss (higher = closer to max)')
    parser.add_argument('--results_dir', type=str, default='results/path_optimization_v2')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.results_dir, 'checkpoint.pkl')

    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Prompts: '{args.prompt_a}' -> '{args.prompt_b}'")
    print(f"  Start layer: {args.start_layer}")
    print(f"  Knots: {args.n_knots}, Jacobian samples/step: {args.n_jac_samples}")
    print(f"  Probes (opt/baseline/eval): {args.n_probes}/{args.n_probes_baseline}/{args.n_probes_eval}")
    print(f"  Optimization: lr={args.lr}, steps={args.n_steps}, grad_clip={args.grad_clip}")
    print(f"  Soft-max alpha: {args.alpha}")
    print("-" * 60)

    # Resume logic
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path)
        stage = checkpoint.get('stage', 'start')
        print(f"Resuming from stage: {stage}")
    else:
        checkpoint = {'stage': 'start', 'args': vars(args)}
        stage = 'start'

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    # Determine end layer (second-to-last)
    blocks = None
    if hasattr(model, 'transformer'):
        n_layers = len(model.transformer.h)
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        n_layers = len(model.model.layers)
    else:
        raise ValueError(f"Can't determine number of layers for {type(model)}")

    end_layer = n_layers - 2  # second-to-last
    print(f"Model has {n_layers} layers (0-{n_layers-1})")
    print(f"Jacobian: layer {args.start_layer} -> layer {end_layer} (second-to-last)")

    # =================================================================
    # Stage 0: Extract activations
    # =================================================================
    if stage == 'start':
        print(f"\n=== Stage 0: Extracting activations ===")

        contexts_a, activations_a = extract_activations(
            model, tokenizer, [args.prompt_a], args.start_layer, device
        )
        contexts_b, activations_b = extract_activations(
            model, tokenizer, [args.prompt_b], args.start_layer, device
        )

        context = contexts_a[0].to(device)   # [1, seq_len-1, d_model]
        start_act = activations_a[0].to(device)  # [d_model]
        end_act = activations_b[0].to(device)    # [d_model]

        dist = (end_act - start_act).norm().item()
        cos_sim = torch.dot(start_act / start_act.norm(),
                           end_act / end_act.norm()).item()
        angle = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))

        print(f"  Context shape: {context.shape}")
        print(f"  Activation dim: {start_act.shape[0]}")
        print(f"  Endpoint distance: {dist:.2f}")
        print(f"  Endpoint angle: {angle:.1f} degrees")
        print(f"  Endpoint norms: A={start_act.norm().item():.2f}, B={end_act.norm().item():.2f}")

        checkpoint['context'] = context.cpu()
        checkpoint['start_act'] = start_act.cpu()
        checkpoint['end_act'] = end_act.cpu()
        checkpoint['end_layer'] = end_layer
        checkpoint['n_layers'] = n_layers
        checkpoint['stage'] = 'activations_ready'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'activations_ready'

    # =================================================================
    # Stage 1: Slerp baseline
    # =================================================================
    if stage == 'activations_ready':
        print(f"\n=== Stage 1: Computing slerp baseline ===")

        context = checkpoint['context'].to(device)
        start_act = checkpoint['start_act'].to(device)
        end_act = checkpoint['end_act'].to(device)

        j_baseline, l_baseline, slerp_norms = compute_slerp_baseline(
            model, context, start_act, end_act,
            args.start_layer, end_layer,
            n_points=20, n_probes=args.n_probes_baseline
        )

        checkpoint['j_baseline'] = j_baseline
        checkpoint['l_baseline'] = l_baseline
        checkpoint['slerp_norms'] = slerp_norms
        checkpoint['stage'] = 'baseline_done'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'baseline_done'

    # =================================================================
    # Stage 2: Optimization
    # =================================================================
    if stage in ['baseline_done', 'optimization_in_progress']:
        print(f"\n=== Stage 2: Path optimization ===")

        context = checkpoint['context'].to(device)
        start_act = checkpoint['start_act'].to(device)
        end_act = checkpoint['end_act'].to(device)
        j_baseline = checkpoint['j_baseline']
        l_baseline = checkpoint['l_baseline']

        print(f"  Baselines: J={j_baseline:.1f}, L={l_baseline:.1f}")

        # Initialize or resume knots
        opt_state = checkpoint.get('opt_state', None)
        if opt_state is not None:
            knots = opt_state['knots'].to(device)
            knots = nn.Parameter(knots)
            start_step = opt_state['step'] + 1
            history = opt_state['history']
            print(f"  Resuming from step {start_step}")
        else:
            # Initialize knots along the slerp path
            n_waypoints = args.n_knots + 2
            t_init = torch.linspace(0, 1, n_waypoints, device=device)
            all_waypoints = torch.stack([
                slerp_point(start_act.detach(), end_act.detach(), t.item())
                for t in t_init
            ])
            # Inner knots only (exclude endpoints)
            knots_init = all_waypoints[1:-1].clone()
            knots = nn.Parameter(knots_init)
            start_step = 0
            history = {
                'total_loss': [], 'jac_loss': [], 'length_loss': [],
                'grad_norm': [],
            }

        optimizer = torch.optim.Adam([knots], lr=args.lr)
        if opt_state is not None and 'optimizer_state' in opt_state:
            optimizer.load_state_dict(opt_state['optimizer_state'])

        print(f"  Knots shape: {knots.shape}")
        print(f"  Starting optimization ({args.n_steps} steps)...")

        t0 = time.time()
        for step in range(start_step, args.n_steps):
            optimizer.zero_grad()

            total_loss, jac_loss_val, length_loss_val = optimization_step(
                model, context, knots, start_act, end_act,
                args.start_layer, end_layer,
                args.n_jac_samples, args.n_probes,
                j_baseline, l_baseline,
                alpha=args.alpha,
            )

            grad_norm = knots.grad.norm().item() if knots.grad is not None else 0.0

            # Gradient clipping
            if args.grad_clip > 0 and knots.grad is not None:
                torch.nn.utils.clip_grad_norm_([knots], args.grad_clip)

            optimizer.step()

            history['total_loss'].append(total_loss)
            history['jac_loss'].append(jac_loss_val)
            history['length_loss'].append(length_loss_val)
            history['grad_norm'].append(grad_norm)

            if (step + 1) % 10 == 0:
                elapsed = time.time() - t0
                steps_done = step - start_step + 1
                rate = steps_done / elapsed
                eta = (args.n_steps - step - 1) / rate if rate > 0 else 0
                print(f"  Step {step+1}/{args.n_steps}  loss={total_loss:.3f} "
                      f"(J={jac_loss_val:.3f}, L={length_loss_val:.3f})  "
                      f"grad={grad_norm:.4f}  ({rate:.2f} steps/s, ETA {eta:.0f}s)")

            if (step + 1) % 50 == 0:
                checkpoint['opt_state'] = {
                    'knots': knots.detach().cpu(),
                    'step': step,
                    'history': history,
                    'optimizer_state': optimizer.state_dict(),
                }
                checkpoint['stage'] = 'optimization_in_progress'
                save_checkpoint(checkpoint, checkpoint_path)

        # Save final state
        checkpoint['opt_state'] = {
            'knots': knots.detach().cpu(),
            'step': args.n_steps - 1,
            'history': history,
            'optimizer_state': optimizer.state_dict(),
        }
        checkpoint['stage'] = 'optimization_done'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'optimization_done'

        total_time = time.time() - t0
        print(f"\n  Optimization done in {total_time:.1f}s ({total_time/60:.1f} min)")

    # =================================================================
    # Stage 3: Evaluation
    # =================================================================
    if stage == 'optimization_done':
        print(f"\n=== Stage 3: Evaluation ===")

        context = checkpoint['context'].to(device)
        start_act = checkpoint['start_act'].to(device)
        end_act = checkpoint['end_act'].to(device)
        opt_knots = checkpoint['opt_state']['knots'].to(device)
        history = checkpoint['opt_state']['history']

        # Evaluate slerp path (dense)
        print("Evaluating slerp path...")
        n_slerp_waypoints = args.n_knots + 2
        t_slerp = torch.linspace(0, 1, n_slerp_waypoints, device=device)
        slerp_waypoints = torch.stack([
            slerp_point(start_act.detach(), end_act.detach(), t.item())
            for t in t_slerp
        ])

        slerp_eval = evaluate_path_dense(
            model, tokenizer, context, slerp_waypoints,
            args.start_layer, end_layer,
            n_points=100, n_probes=args.n_probes_eval
        )

        # Evaluate optimized path (dense)
        print("Evaluating optimized path...")
        opt_waypoints = get_waypoints(start_act.detach(), opt_knots.detach(), end_act.detach())

        opt_eval = evaluate_path_dense(
            model, tokenizer, context, opt_waypoints,
            args.start_layer, end_layer,
            n_points=100, n_probes=args.n_probes_eval
        )

        # Path lengths
        slerp_length = compute_chord_length(slerp_waypoints).item()
        opt_length = compute_chord_length(opt_waypoints).item()

        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  Slerp ||J||_F: mean={slerp_eval['jac_norms'].mean():.1f}, "
              f"max={slerp_eval['jac_norms'].max():.1f}")
        print(f"  Optimized ||J||_F: mean={opt_eval['jac_norms'].mean():.1f}, "
              f"max={opt_eval['jac_norms'].max():.1f}")
        print(f"  Slerp path length: {slerp_length:.1f}")
        print(f"  Optimized path length: {opt_length:.1f} "
              f"({opt_length/slerp_length:.2f}x slerp)")
        print(f"  Slerp entropy: mean={slerp_eval['entropies'].mean():.2f}")
        print(f"  Optimized entropy: mean={opt_eval['entropies'].mean():.2f}")

        # Print predictions
        print_path_predictions(slerp_eval['predictions'], "Slerp path predictions")
        print_path_predictions(opt_eval['predictions'], "Optimized path predictions")

        # Plot
        plot_results(slerp_eval, opt_eval, history, args.results_dir)

        # Save results
        results = {
            'args': vars(args),
            'n_layers': checkpoint.get('n_layers'),
            'end_layer': checkpoint.get('end_layer'),
            'j_baseline': checkpoint['j_baseline'],
            'l_baseline': checkpoint['l_baseline'],
            'slerp_norms_baseline': checkpoint['slerp_norms'],
            'slerp_eval': slerp_eval,
            'opt_eval': opt_eval,
            'opt_knots': opt_knots.cpu(),
            'start_act': start_act.cpu(),
            'end_act': end_act.cpu(),
            'history': history,
            'slerp_length': slerp_length,
            'opt_length': opt_length,
        }
        results_path = os.path.join(args.results_dir, 'results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved: {results_path}")

        checkpoint['stage'] = 'done'
        save_checkpoint(checkpoint, checkpoint_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
