#!/usr/bin/env python3
"""
Jacobian analysis v2: all-positions patching with diagonal vs full-covariance random activations.

Key differences from v1 (run_plateau_analysis.py):
- Patches ALL positions at layer L (not just last token)
- Fixed sequence length (5 tokens) so activations are comparable
- Two flavors of random: diagonal covariance and full covariance
- Hutchinson estimator with forward-mode AD (exact, not finite-difference)
- 100k samples for Gaussian fitting (well-conditioned covariance)

Usage:
    python scripts/run_jacobian_v2.py
    python scripts/run_jacobian_v2.py --n_fit 100000 --n_eval 100 --n_probes 100
    python scripts/run_jacobian_v2.py --resume
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model, extract_activations_allpos, forward_from_layer_allpos
from src.data import load_fineweb_fixed_length, sample_random_activations_v2
from src.analysis import estimate_frobenius_norm_hutchinson
from src.visualization import plot_three_way_comparison


def save_checkpoint(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Checkpoint loaded: {path}")
    return data


def main():
    parser = argparse.ArgumentParser(description='Jacobian analysis v2: all-positions')
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_fit', type=int, default=100000,
                        help='Number of samples for Gaussian fitting')
    parser.add_argument('--n_eval', type=int, default=100,
                        help='Number of samples per category for Jacobian computation')
    parser.add_argument('--n_probes', type=int, default=100,
                        help='Number of Hutchinson probe vectors')
    parser.add_argument('--results_dir', type=str, default='results/jacobian_v2')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.results_dir, 'checkpoint.pkl')

    d_model = {'gpt2': 768, 'gpt2-medium': 1024, 'gpt2-large': 1280, 'gpt2-xl': 1600}[args.model]
    D = args.seq_len * d_model

    print(f"Configuration:")
    print(f"  Model: {args.model} (d_model={d_model})")
    print(f"  Layer: {args.layer}")
    print(f"  Seq length: {args.seq_len} (D={D})")
    print(f"  Fitting samples: {args.n_fit}")
    print(f"  Eval samples per category: {args.n_eval}")
    print(f"  Hutchinson probes: {args.n_probes}")
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
    print(f"Device: {device}")

    # =========================================================
    # Stages 1+2: Collect activations, fit Gaussians, sample
    # Combined to avoid saving the large 100k activation tensor
    # =========================================================
    if stage == 'start':
        print(f"\n=== Stage 1: Collecting {args.n_fit} real activations ===")
        t0 = time.time()

        token_ids_list = load_fineweb_fixed_length(
            args.n_fit, tokenizer, seq_len=args.seq_len, seed=args.seed
        )

        print(f"Running forward passes...")
        all_activations = extract_activations_allpos(
            model, tokenizer, token_ids_list, args.layer, device
        )
        print(f"Activations shape: {all_activations.shape}")
        print(f"Stage 1 took {time.time()-t0:.1f}s")

        print(f"\n=== Stage 2: Fitting Gaussians and sampling ===")
        t0 = time.time()

        # Real activations for eval (first n_eval)
        real_eval = all_activations[:args.n_eval]

        # Fit and sample: diagonal covariance (GPU-accelerated)
        print("Fitting diagonal covariance...")
        random_diag, stats_diag = sample_random_activations_v2(
            all_activations, args.n_eval, mode='diagonal', seed=args.seed, device=device
        )

        # Fit and sample: full covariance (GPU-accelerated)
        print("Fitting full covariance...")
        random_full, stats_full = sample_random_activations_v2(
            all_activations, args.n_eval, mode='full', seed=args.seed + 1, device=device
        )

        print(f"Stage 2 took {time.time()-t0:.1f}s")

        # Save only what we need going forward (NOT the 100k activations)
        checkpoint['token_ids_eval'] = token_ids_list[:args.n_eval]
        checkpoint['real_eval'] = real_eval
        checkpoint['random_diag'] = random_diag
        checkpoint['random_full'] = random_full
        checkpoint['stats_diag'] = {k: v for k, v in stats_diag.items() if k != 'cov'}
        checkpoint['stats_full'] = {'mode': stats_full['mode'], 'rank': stats_full.get('rank')}
        checkpoint['stage'] = 'random_sampled'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'random_sampled'

        del all_activations, token_ids_list  # free memory

    # =========================================================
    # Stage 3: Compute Frobenius norms
    # =========================================================
    if stage in ['random_sampled', 'frobenius_in_progress']:
        print(f"\n=== Stage 3: Computing Frobenius norms (Hutchinson, {args.n_probes} probes) ===")

        real_eval = checkpoint['real_eval']
        random_diag = checkpoint['random_diag']
        random_full = checkpoint['random_full']

        # Resume partial progress if available
        frob_results = checkpoint.get('frob_results', {
            'real': [], 'diag': [], 'full': [],
            'real_std': [], 'diag_std': [], 'full_std': [],
        })

        categories = [
            ('real', real_eval),
            ('diag', random_diag),
            ('full', random_full),
        ]

        for cat_name, activations in categories:
            done = len(frob_results[cat_name])
            total = len(activations)

            if done >= total:
                print(f"  {cat_name}: already done ({done}/{total})")
                continue

            print(f"  {cat_name}: computing {done}..{total}")
            t0 = time.time()

            for i in range(done, total):
                # Reshape to [1, seq_len, d_model]
                act = activations[i].to(device).reshape(1, args.seq_len, d_model)

                frob_norm, frob_std = estimate_frobenius_norm_hutchinson(
                    model, act, args.layer, n_probes=args.n_probes
                )
                frob_results[cat_name].append(frob_norm)
                frob_results[f'{cat_name}_std'].append(frob_std)

                elapsed = time.time() - t0
                rate = (i - done + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"    {cat_name} {i+1}/{total}  frob={frob_norm:.1f}  "
                      f"({rate:.2f} samples/s, ETA {eta:.0f}s)", end='\r')

                # Checkpoint every 10 samples
                if (i + 1) % 10 == 0:
                    checkpoint['frob_results'] = frob_results
                    checkpoint['stage'] = 'frobenius_in_progress'
                    save_checkpoint(checkpoint, checkpoint_path)

            print()

        checkpoint['frob_results'] = frob_results
        checkpoint['stage'] = 'frobenius_done'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'frobenius_done'

    # =========================================================
    # Stage 4: Results
    # =========================================================
    if stage == 'frobenius_done':
        print(f"\n=== Stage 4: Results ===")

        frob = checkpoint['frob_results']

        real_frob = np.array(frob['real'])
        diag_frob = np.array(frob['diag'])
        full_frob = np.array(frob['full'])

        print(f"Frobenius norm ||J||_F:")
        print(f"  Real:             mean={real_frob.mean():.1f}, std={real_frob.std():.1f}")
        print(f"  Random (diagonal): mean={diag_frob.mean():.1f}, std={diag_frob.std():.1f}")
        print(f"  Random (full cov): mean={full_frob.mean():.1f}, std={full_frob.std():.1f}")

        # Plot
        plot_three_way_comparison(
            {
                'Real': real_frob,
                'Random (diagonal)': diag_frob,
                'Random (full cov)': full_frob,
            },
            metric_name="Jacobian Frobenius Norm",
            title=f"||J||_F at Layer {args.layer} (seq_len={args.seq_len}, {args.n_probes} probes)",
            save_path=os.path.join(args.results_dir, 'frobenius_norm_v2.png')
        )

        # Save results
        results = {
            'args': vars(args),
            'frob_results': frob,
            'summary': {
                'real': {'mean': real_frob.mean(), 'std': real_frob.std()},
                'diag': {'mean': diag_frob.mean(), 'std': diag_frob.std()},
                'full': {'mean': full_frob.mean(), 'std': full_frob.std()},
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
