#!/usr/bin/env python3
"""
Minimal steering experiment.

Tests whether steering along a direction vector shows plateau-like behavior.

Usage:
    python scripts/steering_experiment.py
    python scripts/steering_experiment.py --model gpt2-large --n_steps 100
"""

import argparse
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import load_model, get_layer0_output, forward_from_layer1


def run_steering_experiment(model, tokenizer, device,
                            prompt_a="The house was big",
                            prompt_b="The house was in",
                            n_steps=50):
    """
    Run steering experiment between two prompts.

    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        device: torch device
        prompt_a: Source prompt (will be steered from)
        prompt_b: Target prompt (defines steering direction)
        n_steps: Number of steering magnitudes to test

    Returns:
        alphas: array of steering magnitudes [0, 1]
        rel_distances: array of relative distances (0 at A, 1 at B)
        logits_A: logits at α=0 (endpoint A)
        logits_B: logits at α=1 (endpoint B)
    """
    # Get layer 1 activations for both prompts
    resid_a, _ = get_layer0_output(model, tokenizer, prompt_a, device)
    resid_b, _ = get_layer0_output(model, tokenizer, prompt_b, device)

    # Separate prefix (context) from last token
    prefix_resid = resid_a[:, :-1, :]  # "The house was"
    act_a = resid_a[:, -1:, :]         # "big"
    act_b = resid_b[:, -1:, :]         # "in"

    # Steering vector
    steering_vec = act_b - act_a

    # Sweep over steering magnitudes
    alphas = np.linspace(0, 1, n_steps)

    # Compute endpoint logits
    with torch.no_grad():
        logits_A = forward_from_layer1(model, prefix_resid, act_a).squeeze(0)
        logits_B = forward_from_layer1(model, prefix_resid, act_b).squeeze(0)

    rel_distances = []
    all_logits = []

    for alpha in alphas:
        # Steer: act_a + α * (act_b - act_a)
        steered_act = act_a + alpha * steering_vec

        with torch.no_grad():
            logits = forward_from_layer1(model, prefix_resid, steered_act).squeeze(0)

        # Relative distance: d = ||x - A|| / (||x - A|| + ||x - B||)
        dist_to_A = torch.norm(logits - logits_A).item()
        dist_to_B = torch.norm(logits - logits_B).item()
        rel_dist = dist_to_A / (dist_to_A + dist_to_B + 1e-10)

        rel_distances.append(rel_dist)
        all_logits.append(logits)

    rel_distances = np.array(rel_distances)

    return alphas, rel_distances, logits_A, logits_B


def plot_steering_results(alphas, rel_distances, prompt_a, prompt_b, save_path=None):
    """Plot relative distance vs steering magnitude."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Relative distance vs alpha
    ax1.plot(alphas, rel_distances, 'b-', linewidth=2)
    ax1.set_xlabel("Steering magnitude α", fontsize=12)
    ax1.set_ylabel("Relative distance d", fontsize=12)
    ax1.set_title(f"Steering: '{prompt_a}' → '{prompt_b}'", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Add reference line for linear behavior (d = α)
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Linear (d = α)')
    ax1.legend()

    # Plot 2: Derivative (velocity) - shows where changes happen
    velocity = np.diff(rel_distances) / np.diff(alphas)
    ax2.plot(alphas[:-1] + np.diff(alphas)/2, velocity, 'g-', linewidth=2)
    ax2.set_xlabel("Steering magnitude α", fontsize=12)
    ax2.set_ylabel("d(rel_distance)/dα", fontsize=12)
    ax2.set_title("Rate of change (velocity)", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Linear (=1)')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def print_top_predictions(model, tokenizer, logits, label, k=5):
    """Print top-k token predictions."""
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k)

    print(f"\n{label}:")
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.decode([idx.item()])
        print(f"  '{token}': {prob.item():.4f}")


def main():
    parser = argparse.ArgumentParser(description='Minimal steering experiment')
    parser.add_argument('--prompt_a', type=str, default="The house was big",
                        help='Source prompt')
    parser.add_argument('--prompt_b', type=str, default="The house was in",
                        help='Target prompt (defines steering direction)')
    parser.add_argument('--model', type=str, default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    parser.add_argument('--n_steps', type=int, default=50,
                        help='Number of steering magnitudes')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save plot')

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    print(f"Using device: {device}")

    print(f"\nRunning steering experiment:")
    print(f"  From: '{args.prompt_a}'")
    print(f"  To:   '{args.prompt_b}'")
    print(f"  Steps: {args.n_steps}")
    print("-" * 50)

    alphas, rel_distances, logits_A, logits_B = run_steering_experiment(
        model, tokenizer, device,
        prompt_a=args.prompt_a,
        prompt_b=args.prompt_b,
        n_steps=args.n_steps
    )

    # Print summary statistics
    velocity = np.diff(rel_distances) / np.diff(alphas)
    print(f"\nResults (relative distance: d = ||x-A|| / (||x-A|| + ||x-B||)):")
    print(f"  d at α=0: {rel_distances[0]:.4f} (should be 0)")
    print(f"  d at α=1: {rel_distances[-1]:.4f} (should be 1)")
    print(f"  Max deviation from linear: {np.max(np.abs(rel_distances - alphas)):.4f}")
    print(f"  Mean velocity: {np.mean(velocity):.4f} (linear = 1)")
    print(f"  Max velocity: {np.max(velocity):.4f}")
    print(f"  Min velocity: {np.min(velocity):.4f}")

    # Print predictions at start and end
    print_top_predictions(model, tokenizer, logits_A, "Predictions at α=0 (endpoint A)")
    print_top_predictions(model, tokenizer, logits_B, "Predictions at α=1 (endpoint B)")

    # Plot - include model name in save path if provided
    save_path = args.save
    if save_path and '{model}' in save_path:
        save_path = save_path.replace('{model}', args.model)
    plot_steering_results(alphas, rel_distances, args.prompt_a, args.prompt_b, save_path=save_path)


if __name__ == '__main__':
    main()
