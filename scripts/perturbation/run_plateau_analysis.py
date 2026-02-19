#!/usr/bin/env python3
"""
Real vs Random activation comparison experiment.

Compares Jacobian structure (top singular values, Frobenius norm) and logit entropy
between real activations from FineWeb and covariance-matched random activations.

Usage:
    python scripts/run_plateau_analysis.py --layer 6 --n_real 100 --n_cov_samples 2000
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model, extract_activations
from src.data import load_fineweb_samples, sample_random_activations
from src.analysis import (
    compute_top_singular_values,
    estimate_frobenius_norm,
    compute_logit_entropy,
)
from src.visualization import plot_distribution_comparison


def save_checkpoint(data, path):
    """Save checkpoint to pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path):
    """Load checkpoint from pickle file."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Checkpoint loaded: {path}")
    return data


def main():
    parser = argparse.ArgumentParser(description='Real vs Random activation analysis')
    parser.add_argument('--model', type=str, default='gpt2-large',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    parser.add_argument('--layer', type=int, default=6,
                        help='Layer to analyze (0-indexed)')
    parser.add_argument('--n_real', type=int, default=100,
                        help='Number of real activations to analyze')
    parser.add_argument('--n_cov_samples', type=int, default=2000,
                        help='Number of samples for covariance estimation')
    parser.add_argument('--n_random', type=int, default=100,
                        help='Number of random activations to generate')
    parser.add_argument('--k_singular', type=int, default=5,
                        help='Number of top singular values to compute')
    parser.add_argument('--n_frob_samples', type=int, default=50,
                        help='Number of random directions for Frobenius norm estimation')
    parser.add_argument('--results_dir', type=str, default='results/plateau_analysis',
                        help='Directory for results')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.results_dir, 'checkpoint.pkl')

    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Layer: {args.layer}")
    print(f"  Real samples: {args.n_real}")
    print(f"  Covariance samples: {args.n_cov_samples}")
    print(f"  Random samples: {args.n_random}")
    print(f"  Top-k singular values: {args.k_singular}")
    print("-" * 60)

    # Try to resume
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path)
        stage = checkpoint.get('stage', 'start')
        print(f"Resuming from stage: {stage}")
    else:
        checkpoint = {'stage': 'start'}
        stage = 'start'

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    print(f"Using device: {device}")

    # Stage 1: Load FineWeb and extract activations
    if stage in ['start']:
        print(f"\n=== Stage 1: Loading FineWeb samples ===")
        texts, token_counts = load_fineweb_samples(
            args.n_cov_samples, tokenizer, min_tokens=10, seed=args.seed
        )

        print(f"\nExtracting activations at layer {args.layer}...")
        contexts, all_activations = extract_activations(
            model, tokenizer, texts, args.layer, device
        )

        checkpoint['texts'] = texts
        checkpoint['token_counts'] = token_counts
        checkpoint['contexts'] = contexts
        checkpoint['all_activations'] = all_activations
        checkpoint['stage'] = 'activations_extracted'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'activations_extracted'

    # Stage 2: Generate random activations
    if stage == 'activations_extracted':
        print(f"\n=== Stage 2: Generating random activations ===")

        all_activations = checkpoint['all_activations']
        contexts = checkpoint['contexts']

        # Select real samples to analyze (first n_real)
        real_activations = all_activations[:args.n_real]
        real_contexts = contexts[:args.n_real]

        # Generate random activations from full covariance
        random_activations = sample_random_activations(
            all_activations, args.n_random, seed=args.seed
        )

        # Pick fixed context for random activations (first one)
        fixed_context = contexts[0]

        checkpoint['real_activations'] = real_activations
        checkpoint['real_contexts'] = real_contexts
        checkpoint['random_activations'] = random_activations
        checkpoint['fixed_context'] = fixed_context
        checkpoint['stage'] = 'random_generated'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'random_generated'

    # Stage 3: Compute metrics for real activations
    if stage == 'random_generated':
        print(f"\n=== Stage 3: Computing metrics for real activations ===")

        real_activations = checkpoint['real_activations']
        real_contexts = checkpoint['real_contexts']

        real_metrics = {
            'singular_values': [],
            'frobenius_norm': [],
            'entropy': [],
        }

        for i in range(len(real_activations)):
            print(f"Real activation {i+1}/{len(real_activations)}", end='\r')

            act = real_activations[i].to(device).reshape(1, 1, -1)
            ctx = real_contexts[i].to(device)

            # Top singular values
            svs, _ = compute_top_singular_values(
                model, ctx, act, layer=args.layer, k=args.k_singular
            )
            real_metrics['singular_values'].append(svs)

            # Frobenius norm
            frob = estimate_frobenius_norm(
                model, ctx, act, args.layer, n_samples=args.n_frob_samples
            )
            real_metrics['frobenius_norm'].append(frob)

            # Entropy
            ent = compute_logit_entropy(model, ctx, act, args.layer)
            real_metrics['entropy'].append(ent)

        print()  # newline after progress

        checkpoint['real_metrics'] = real_metrics
        checkpoint['stage'] = 'real_computed'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'real_computed'

    # Stage 4: Compute metrics for random activations
    if stage == 'real_computed':
        print(f"\n=== Stage 4: Computing metrics for random activations ===")

        random_activations = checkpoint['random_activations']
        fixed_context = checkpoint['fixed_context'].to(device)

        random_metrics = {
            'singular_values': [],
            'frobenius_norm': [],
            'entropy': [],
        }

        for i in range(len(random_activations)):
            print(f"Random activation {i+1}/{len(random_activations)}", end='\r')

            act = random_activations[i].to(device).reshape(1, 1, -1)

            # Top singular values
            svs, _ = compute_top_singular_values(
                model, fixed_context, act, layer=args.layer, k=args.k_singular
            )
            random_metrics['singular_values'].append(svs)

            # Frobenius norm
            frob = estimate_frobenius_norm(
                model, fixed_context, act, args.layer, n_samples=args.n_frob_samples
            )
            random_metrics['frobenius_norm'].append(frob)

            # Entropy
            ent = compute_logit_entropy(model, fixed_context, act, args.layer)
            random_metrics['entropy'].append(ent)

        print()  # newline after progress

        checkpoint['random_metrics'] = random_metrics
        checkpoint['stage'] = 'random_computed'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'random_computed'

    # Stage 5: Analysis and visualization
    if stage == 'random_computed':
        print(f"\n=== Stage 5: Analysis and visualization ===")

        real_metrics = checkpoint['real_metrics']
        random_metrics = checkpoint['random_metrics']

        # Convert to arrays
        real_frob = np.array(real_metrics['frobenius_norm'])
        random_frob = np.array(random_metrics['frobenius_norm'])
        real_entropy = np.array(real_metrics['entropy'])
        random_entropy = np.array(random_metrics['entropy'])
        real_top_sv = np.array([sv[0] for sv in real_metrics['singular_values']])
        random_top_sv = np.array([sv[0] for sv in random_metrics['singular_values']])

        # Print summary statistics
        print(f"\nResults Summary:")
        print(f"Frobenius Norm:")
        print(f"  Real:   mean={real_frob.mean():.2f}, std={real_frob.std():.2f}")
        print(f"  Random: mean={random_frob.mean():.2f}, std={random_frob.std():.2f}")
        print(f"Entropy:")
        print(f"  Real:   mean={real_entropy.mean():.2f}, std={real_entropy.std():.2f}")
        print(f"  Random: mean={random_entropy.mean():.2f}, std={random_entropy.std():.2f}")
        print(f"Top Singular Value:")
        print(f"  Real:   mean={real_top_sv.mean():.2f}, std={real_top_sv.std():.2f}")
        print(f"  Random: mean={random_top_sv.mean():.2f}, std={random_top_sv.std():.2f}")

        # Generate plots
        plot_distribution_comparison(
            real_frob, random_frob, "Jacobian Frobenius Norm",
            title=f"Jacobian Frobenius Norm (Layer {args.layer})",
            save_path=os.path.join(args.results_dir, 'frobenius_norm.png')
        )

        plot_distribution_comparison(
            real_entropy, random_entropy, "Logit Entropy",
            title=f"Logit Entropy (Layer {args.layer})",
            save_path=os.path.join(args.results_dir, 'entropy.png')
        )

        plot_distribution_comparison(
            real_top_sv, random_top_sv, "Top Singular Value",
            title=f"Top Singular Value (Layer {args.layer})",
            save_path=os.path.join(args.results_dir, 'top_sv.png')
        )

        # Save final results
        results = {
            'args': vars(args),
            'real_metrics': real_metrics,
            'random_metrics': random_metrics,
            'summary': {
                'frobenius_norm': {
                    'real_mean': real_frob.mean(),
                    'real_std': real_frob.std(),
                    'random_mean': random_frob.mean(),
                    'random_std': random_frob.std(),
                },
                'entropy': {
                    'real_mean': real_entropy.mean(),
                    'real_std': real_entropy.std(),
                    'random_mean': random_entropy.mean(),
                    'random_std': random_entropy.std(),
                },
                'top_sv': {
                    'real_mean': real_top_sv.mean(),
                    'real_std': real_top_sv.std(),
                    'random_mean': random_top_sv.mean(),
                    'random_std': random_top_sv.std(),
                },
            }
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
