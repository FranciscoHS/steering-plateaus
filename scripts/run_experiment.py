#!/usr/bin/env python3
"""
Example experiment script for steering activation plateaus.

Runs path optimization between two prompts and analyzes the results.

Usage:
    python scripts/run_experiment.py --prompt_a "The house was big" --prompt_b "The house was large"
"""

import argparse
import sys
sys.path.insert(0, '.')

import torch
from src import (
    load_model,
    get_layer0_output,
    slerp_path,
    optimize_path,
    compute_path_length,
    compute_velocity_profile,
    peak_to_average_ratio,
    plot_velocity_profiles,
    plot_path_lengths,
    plot_optimization_history,
)


def main():
    parser = argparse.ArgumentParser(description='Run steering plateau experiment')
    parser.add_argument('--prompt_a', type=str, default="The house was big",
                        help='Starting prompt')
    parser.add_argument('--prompt_b', type=str, default="The house was large",
                        help='Ending prompt')
    parser.add_argument('--model', type=str, default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    parser.add_argument('--n_steps', type=int, default=50,
                        help='Number of path points')
    parser.add_argument('--lambda_output', type=float, default=1.0,
                        help='Weight on output energy term')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate')
    parser.add_argument('--iterations', type=int, default=200,
                        help='Optimization iterations')
    parser.add_argument('--save_prefix', type=str, default=None,
                        help='Prefix for saving results (default: no save)')

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    print(f"Using device: {device}")

    print(f"\nExperiment: '{args.prompt_a}' → '{args.prompt_b}'")
    print(f"Parameters: n_steps={args.n_steps}, λ={args.lambda_output}, "
          f"lr={args.lr}, iters={args.iterations}")
    print("-" * 60)

    # Get layer 0 outputs
    resid_a, _ = get_layer0_output(model, tokenizer, args.prompt_a, device)
    resid_b, _ = get_layer0_output(model, tokenizer, args.prompt_b, device)

    # Separate prefix (context) from target (last token)
    prefix_resid = resid_a[:, :-1, :]
    start_vec = resid_a[:, -1:, :]
    end_vec = resid_b[:, -1:, :]

    # Generate slerp baseline
    print("\nGenerating slerp baseline...")
    initial_path = slerp_path(start_vec, end_vec, args.n_steps, device)

    # Optimize path
    print("\nOptimizing path...")
    optimized_path, history = optimize_path(
        model, prefix_resid, start_vec, end_vec,
        n_steps=args.n_steps,
        lambda_output=args.lambda_output,
        lr=args.lr,
        iterations=args.iterations,
        verbose=True
    )

    # Compute velocity profiles
    print("\nComputing velocity profiles...")
    vel_initial = compute_velocity_profile(model, prefix_resid, initial_path)
    vel_optimized = compute_velocity_profile(model, prefix_resid, optimized_path)

    par_initial = peak_to_average_ratio(vel_initial)
    par_optimized = peak_to_average_ratio(vel_optimized)

    print(f"\nResults:")
    print(f"  Initial PAR:   {par_initial:.2f}")
    print(f"  Optimized PAR: {par_optimized:.2f}")

    # Compute path lengths
    len_initial, steps_initial = compute_path_length(initial_path)
    len_optimized, steps_optimized = compute_path_length(optimized_path)
    straight_dist = torch.norm(end_vec - start_vec).item()

    print(f"\nPath lengths:")
    print(f"  Straight line:  {straight_dist:.4f}")
    print(f"  Slerp path:     {len_initial:.4f}")
    print(f"  Optimized path: {len_optimized:.4f}")

    # Plot results
    save_prefix = args.save_prefix

    plot_velocity_profiles(
        {'Slerp': vel_initial, 'Optimized': vel_optimized},
        title=f"'{args.prompt_a}' → '{args.prompt_b}'",
        save_path=f"{save_prefix}_velocity.png" if save_prefix else None
    )

    plot_path_lengths(
        {'Slerp': steps_initial, 'Optimized': steps_optimized},
        n_steps=args.n_steps,
        straight_line_dist=straight_dist,
        save_path=f"{save_prefix}_lengths.png" if save_prefix else None
    )

    plot_optimization_history(
        history,
        save_path=f"{save_prefix}_history.png" if save_prefix else None
    )


if __name__ == '__main__':
    main()
