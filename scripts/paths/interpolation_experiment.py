#!/usr/bin/env python3
"""
Activation plateau interpolation experiment.

Tests whether interpolating between two prompts' activations at a given layer
shows plateau -> boundary -> plateau structure in logit space.

This is the fundamental activation plateau experiment: if the model's output
is piecewise-linear in activation space, interpolating between two prompts
that land in different "basins" should produce an S-curve in relative distance
rather than a straight line d = t.

Interpolation method: spherical interpolation for direction, linear
interpolation for L2 magnitude (following Stefan Heimersheim's approach).
Only the last token position is interpolated; the prefix is kept fixed.

Usage:
    python scripts/interpolation_experiment.py
    python scripts/interpolation_experiment.py --model qwen3-0.6b --layer 1
    python scripts/interpolation_experiment.py --prompt_a "The house was big" --prompt_b "The house was in"
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer


def load_model(model_name, device=None):
    """Load model via transformer_lens."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name} via transformer_lens...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_layers} layers, d_model={d_model}, {n_params/1e6:.0f}M parameters")

    return model, device


def get_activations(model, prompt, layer):
    """
    Get activations at a specified layer for a prompt.

    Returns:
        activations: [seq_len, d_model] tensor (post-layer residual stream)
        tokens: tokenized input [seq_len]
    """
    hook_point = f"blocks.{layer}.hook_resid_post"

    _, cache = model.run_with_cache(prompt)
    act = cache[hook_point].squeeze(0)  # [seq_len, d_model]
    tokens = model.to_tokens(prompt).squeeze(0)

    return act, tokens


def slerp_with_linear_magnitude(v0, v1, t):
    """
    Spherical interpolation for direction, linear interpolation for magnitude.

    Following Stefan Heimersheim's approach: use slerp to determine the direction
    of interpolated points, then rescale so L2 magnitude is linearly interpolated
    between the two endpoints. This prevents vector norms from becoming unusually
    small (which happens with naive slerp when vectors are far apart).

    Args:
        v0: Start vector [d_model]
        v1: End vector [d_model]
        t: Interpolation parameter (0 = v0, 1 = v1)

    Returns:
        Interpolated vector [d_model]
    """
    norm0 = v0.norm()
    norm1 = v1.norm()

    u0 = v0 / norm0
    u1 = v1 / norm1

    dot = torch.clamp(torch.dot(u0, u1), -1.0, 1.0)
    theta = torch.acos(dot)

    if theta.abs() < 1e-6:
        # Nearly parallel — slerp degenerates, use linear
        direction = u0
    else:
        sin_theta = torch.sin(theta)
        direction = (torch.sin((1 - t) * theta) / sin_theta) * u0 + \
                    (torch.sin(t * theta) / sin_theta) * u1

    # Linear magnitude interpolation
    mag = (1 - t) * norm0 + t * norm1

    return direction * mag


def inject_last_token_and_get_logits(model, prompt, prefix_act, last_token_act, layer):
    """
    Inject activations at a layer (prefix + last token) and get output logits.

    Args:
        model: HookedTransformer model
        prompt: Input prompt (needed for token structure / earlier layers)
        prefix_act: Prefix activations [seq_len-1, d_model]
        last_token_act: Last token activation [d_model]
        layer: Layer to inject at

    Returns:
        logits: [vocab_size] tensor (logits for last token position)
    """
    hook_point = f"blocks.{layer}.hook_resid_post"

    # Reconstruct full activation: [1, seq_len, d_model]
    full_act = torch.cat([prefix_act, last_token_act.unsqueeze(0)], dim=0).unsqueeze(0)

    def injection_hook(activation, hook):
        activation[:, :, :] = full_act
        return activation

    with torch.no_grad():
        logits = model.run_with_hooks(
            prompt,
            fwd_hooks=[(hook_point, injection_hook)]
        )

    return logits[0, -1, :]  # [vocab_size]


def run_interpolation(model, prompt_a, prompt_b, layer, n_steps=100):
    """
    Slerp-interpolate between two prompts' last-token activations and measure
    output distance.

    Only the last token position is interpolated; the prefix ("The house was")
    is taken from prompt_a and kept fixed. Interpolation uses spherical
    direction + linear magnitude.

    Args:
        model: HookedTransformer model
        prompt_a: First prompt (endpoint A, t=0)
        prompt_b: Second prompt (endpoint B, t=1)
        layer: Layer to interpolate at
        n_steps: Number of interpolation steps

    Returns:
        t_values: array of interpolation parameters [0, 1]
        rel_distances: array of relative distances (0 near A, 1 near B)
        logits_A: logits at endpoint A
        logits_B: logits at endpoint B
    """
    # Get activations for both prompts
    print(f"Getting activations for prompt A: '{prompt_a}'")
    act_A, tokens_A = get_activations(model, prompt_a, layer)
    print(f"  Tokens: {[model.to_string(t.item()) for t in tokens_A]}")
    print(f"  Shape: {act_A.shape}")

    print(f"Getting activations for prompt B: '{prompt_b}'")
    act_B, tokens_B = get_activations(model, prompt_b, layer)
    print(f"  Tokens: {[model.to_string(t.item()) for t in tokens_B]}")
    print(f"  Shape: {act_B.shape}")

    # Check shapes match
    if act_A.shape != act_B.shape:
        raise ValueError(
            f"Activation shapes don't match: {act_A.shape} vs {act_B.shape}. "
            f"Prompts must tokenize to the same length. "
            f"Prompt A has {act_A.shape[0]} tokens, prompt B has {act_B.shape[0]} tokens."
        )

    # Split into prefix (shared) and last token (interpolated)
    prefix_act = act_A[:-1, :]   # [seq_len-1, d_model] — from prompt_a, held fixed
    last_A = act_A[-1, :]        # [d_model]
    last_B = act_B[-1, :]        # [d_model]

    # Report distances
    last_token_dist = (last_B - last_A).norm().item()
    angle = torch.acos(torch.clamp(
        torch.dot(last_A / last_A.norm(), last_B / last_B.norm()), -1.0, 1.0
    )).item()
    print(f"Last-token activation distance: {last_token_dist:.4f}")
    print(f"Last-token activation angle: {np.degrees(angle):.2f}°")
    print(f"Last-token norms: A={last_A.norm().item():.4f}, B={last_B.norm().item():.4f}")

    # Get logits at endpoints
    logits_A = inject_last_token_and_get_logits(model, prompt_a, prefix_act, last_A, layer)
    logits_B = inject_last_token_and_get_logits(model, prompt_a, prefix_act, last_B, layer)

    # Interpolate
    t_values = np.linspace(0, 1, n_steps)
    rel_distances = []

    print(f"Running slerp interpolation ({n_steps} steps)...")
    for i, t in enumerate(t_values):
        interp_last = slerp_with_linear_magnitude(last_A, last_B, t)

        logits = inject_last_token_and_get_logits(model, prompt_a, prefix_act, interp_last, layer)

        dist_to_A = (logits - logits_A).norm().item()
        dist_to_B = (logits - logits_B).norm().item()
        rel_dist = dist_to_A / (dist_to_A + dist_to_B + 1e-10)

        rel_distances.append(rel_dist)

        if (i + 1) % 25 == 0:
            print(f"  Step {i+1}/{n_steps}: t={t:.3f}, d={rel_dist:.4f}")

    rel_distances = np.array(rel_distances)

    return t_values, rel_distances, logits_A, logits_B


def plot_results(t_values, rel_distances, prompt_a, prompt_b, model_name, layer,
                 save_path=None):
    """Plot interpolation results: relative distance vs interpolation parameter."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    ax.plot(t_values, rel_distances, 'b-', linewidth=2, label='Observed')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Linear (d = t)')
    ax.set_xlabel("Interpolation parameter t", fontsize=12)
    ax.set_ylabel("Relative distance d", fontsize=12)
    ax.set_title(
        f"Interpolation: '{prompt_a}' → '{prompt_b}'\n"
        f"Model: {model_name}, Layer {layer}",
        fontsize=12
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()


def print_top_predictions(model, logits, label, k=5):
    """Print top-k token predictions from logits."""
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k)

    print(f"\n{label}:")
    for prob, idx in zip(top_probs, top_indices):
        token = model.to_string(idx.item())
        print(f"  '{token}': {prob.item():.4f}")


def main():
    parser = argparse.ArgumentParser(description='Activation plateau interpolation experiment')
    parser.add_argument('--prompt_a', type=str, default="The house was big",
                        help='First prompt (endpoint A)')
    parser.add_argument('--prompt_b', type=str, default="The house was in",
                        help='Second prompt (endpoint B)')
    parser.add_argument('--model', type=str, default='qwen3-0.6b',
                        help='Model name for transformer_lens')
    parser.add_argument('--layer', type=int, default=None,
                        help='Layer to interpolate at (default: layer 1)')
    parser.add_argument('--n_steps', type=int, default=100,
                        help='Number of interpolation steps')
    parser.add_argument('--save_dir', type=str, default='results/interpolation',
                        help='Directory to save results')
    parser.add_argument('--no_save', action='store_true',
                        help='Skip saving results')

    args = parser.parse_args()

    # Load model
    model, device = load_model(args.model)

    # Default layer: layer 1 (early, matching original GPT-2 experiment)
    layer = args.layer if args.layer is not None else 1
    print(f"\nInterpolating at layer {layer} (of {model.cfg.n_layers})")

    # Run interpolation
    print(f"\nPrompt A: '{args.prompt_a}'")
    print(f"Prompt B: '{args.prompt_b}'")
    print("=" * 60)

    t_values, rel_distances, logits_A, logits_B = run_interpolation(
        model, args.prompt_a, args.prompt_b, layer, args.n_steps
    )

    # Print endpoint predictions
    print_top_predictions(model, logits_A, f"Predictions at endpoint A ('{args.prompt_a}')")
    print_top_predictions(model, logits_B, f"Predictions at endpoint B ('{args.prompt_b}')")

    # Summary statistics
    max_dev = np.max(np.abs(rel_distances - t_values))
    area_diff = np.trapz(np.abs(rel_distances - t_values), t_values)

    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  d at t=0: {rel_distances[0]:.4f} (should be ~0)")
    print(f"  d at t=1: {rel_distances[-1]:.4f} (should be ~1)")
    print(f"  Max deviation from linear: {max_dev:.4f}")
    print(f"  Area between curve and linear: {area_diff:.4f}")

    if max_dev > 0.05:
        print(f"\n  -> Significant deviation from linear!")
        print(f"  -> Suggests plateau-like structure in activation space.")
    else:
        print(f"\n  -> Close to linear (d ~ t).")
        print(f"  -> No clear plateau structure at this layer.")

    # Save results
    if not args.no_save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize model name for filenames
        model_short = args.model.replace('/', '_')

        # Save data
        results = {
            't_values': t_values,
            'rel_distances': rel_distances,
            'logits_A': logits_A,
            'logits_B': logits_B,
            'prompt_a': args.prompt_a,
            'prompt_b': args.prompt_b,
            'model': args.model,
            'layer': layer,
            'n_steps': args.n_steps,
        }
        data_path = save_dir / f"interp_{model_short}_layer{layer}.pt"
        torch.save(results, data_path)
        print(f"\nSaved data to {data_path}")

        # Save plot
        fig_path = save_dir / f"interp_{model_short}_layer{layer}.png"
        plot_results(t_values, rel_distances, args.prompt_a, args.prompt_b,
                     args.model, layer, save_path=str(fig_path))

    print("\nDone.")


if __name__ == '__main__':
    main()
