#!/usr/bin/env python3
"""
Steering overshoot experiment.

Tests what happens when we steer beyond the target (α > 1).

Usage:
    python scripts/steering_overshoot.py --model gpt2-large
"""

import argparse
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import load_model, get_layer0_output, forward_from_layer1


def run_overshoot_experiment(model, tokenizer, device,
                              prompt_a="The house was big",
                              prompt_b="The house was in",
                              alpha_max=2.0,
                              n_steps=200):
    """
    Run steering experiment with overshoot (α beyond [0,1]).

    Returns:
        results: dict with alphas, all_logits, logits_A, logits_B, distances_from_A
    """
    # Get layer 1 activations for both prompts
    resid_a, _ = get_layer0_output(model, tokenizer, prompt_a, device)
    resid_b, _ = get_layer0_output(model, tokenizer, prompt_b, device)

    # Separate prefix (context) from last token
    prefix_resid = resid_a[:, :-1, :]
    act_a = resid_a[:, -1:, :]
    act_b = resid_b[:, -1:, :]

    # Steering vector
    steering_vec = act_b - act_a

    # Sweep over steering magnitudes [0, alpha_max]
    alphas = np.linspace(0, alpha_max, n_steps)

    # Compute endpoint logits
    with torch.no_grad():
        logits_A = forward_from_layer1(model, prefix_resid, act_a).squeeze(0)
        logits_B = forward_from_layer1(model, prefix_resid, act_b).squeeze(0)

    all_logits = []
    distances_from_A = []

    for alpha in alphas:
        steered_act = act_a + alpha * steering_vec

        with torch.no_grad():
            logits = forward_from_layer1(model, prefix_resid, steered_act).squeeze(0)

        all_logits.append(logits.cpu())
        distances_from_A.append(torch.norm(logits - logits_A).item())

    return {
        'alphas': alphas,
        'all_logits': all_logits,
        'logits_A': logits_A.cpu(),
        'logits_B': logits_B.cpu(),
        'distances_from_A': np.array(distances_from_A),
    }


def sample_continuation(model, tokenizer, logits, n_tokens=10):
    """Sample n_tokens continuation from logits using greedy decoding."""
    tokens = []
    current_logits = logits

    for _ in range(n_tokens):
        # Greedy: take argmax
        next_token = torch.argmax(current_logits).item()
        tokens.append(next_token)

        # For simplicity, just return the greedy sequence from initial logits
        # (Not running the model autoregressively - just showing what the model
        # would predict at this single position)
        break

    # Actually, let's just return top-10 tokens at this position
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, n_tokens)

    return [(tokenizer.decode([idx.item()]), prob.item())
            for idx, prob in zip(top_indices, top_probs)]


def plot_overshoot_results(results, prompt_a, prompt_b, model_name, save_path=None):
    """Plot raw distance from A vs steering magnitude."""
    alphas = results['alphas']
    distances = results['distances_from_A']

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(alphas, distances, 'b-', linewidth=2)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='α=1 (target B)')
    ax.set_xlabel("Steering magnitude α", fontsize=12)
    ax.set_ylabel("Distance from A (||logits - logits_A||)", fontsize=12)
    ax.set_title(f"Steering overshoot: '{prompt_a}' → '{prompt_b}' ({model_name})", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Steering overshoot experiment')
    parser.add_argument('--prompt_a', type=str, default="The house was big")
    parser.add_argument('--prompt_b', type=str, default="The house was in")
    parser.add_argument('--model', type=str, default='gpt2-large',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    parser.add_argument('--alpha_max', type=float, default=2.0,
                        help='Maximum steering magnitude')
    parser.add_argument('--n_steps', type=int, default=200,
                        help='Number of steering magnitudes')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save plot')
    parser.add_argument('--save_data', type=str, default=None,
                        help='Path to save raw data (torch .pt file)')

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    print(f"Using device: {device}")

    print(f"\nRunning overshoot experiment:")
    print(f"  From: '{args.prompt_a}'")
    print(f"  To:   '{args.prompt_b}'")
    print(f"  α range: [0, {args.alpha_max}]")
    print(f"  Steps: {args.n_steps}")
    print("-" * 50)

    results = run_overshoot_experiment(
        model, tokenizer, device,
        prompt_a=args.prompt_a,
        prompt_b=args.prompt_b,
        alpha_max=args.alpha_max,
        n_steps=args.n_steps
    )

    # Save raw data if requested
    if args.save_data:
        save_data_path = args.save_data
        if '{model}' in save_data_path:
            save_data_path = save_data_path.replace('{model}', args.model)
        torch.save(results, save_data_path)
        print(f"Saved data to {save_data_path}")

    # Print summary
    alphas = results['alphas']
    distances = results['distances_from_A']

    print(f"\nResults:")
    print(f"  Distance at α=0: {distances[0]:.4f}")
    idx_1 = np.argmin(np.abs(alphas - 1.0))
    print(f"  Distance at α=1: {distances[idx_1]:.4f}")
    print(f"  Distance at α={args.alpha_max}: {distances[-1]:.4f}")

    # Sample tokens at key α values
    sample_alphas = [0, 0.5, 1.0, 1.5, 2.0]
    sample_alphas = [a for a in sample_alphas if a <= args.alpha_max]

    print(f"\nTop-10 token predictions at selected α values:")
    for target_alpha in sample_alphas:
        idx = np.argmin(np.abs(alphas - target_alpha))
        logits = results['all_logits'][idx]
        tokens = sample_continuation(model, tokenizer, logits, n_tokens=10)

        print(f"\n  α = {alphas[idx]:.2f}:")
        for token, prob in tokens:
            print(f"    '{token}': {prob:.4f}")

    # Plot
    save_path = args.save
    if save_path and '{model}' in save_path:
        save_path = save_path.replace('{model}', args.model)
    plot_overshoot_results(results, args.prompt_a, args.prompt_b, args.model, save_path=save_path)


if __name__ == '__main__':
    main()
