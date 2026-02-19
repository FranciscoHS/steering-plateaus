#!/usr/bin/env python3
"""Plot perturbation sweep results for genre directions vs random.

Shows KL divergence and L2 distance (final hidden state) as a function
of steering angle, comparing each genre's all-pairwise estimator against
random directions.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def main():
    results_dir = 'results/genre_contrastive/qwen3-0.6b'
    with open(os.path.join(results_dir, 'sweep_results.pkl'), 'rb') as f:
        sweep = pickle.load(f)

    angles = sweep['angles_deg']
    results = sweep['sweep_results']

    # Only show up to max_angle degrees
    max_angle = 60
    mask = angles <= max_angle
    angles_plot = angles[mask]

    # Genre directions (all-pairwise estimators)
    genres = ['news', 'scientific', 'recipe', 'legal', 'children',
              'review', 'wikipedia', 'dialogue', 'email', 'technical', 'advertising']

    # Random baseline: shape (n_act, n_angles, n_dirs) — average over dirs then acts
    rand_data = results['random']
    rand_kl_mean = rand_data['kl'].mean(axis=(0, 2)) if rand_data['kl'].ndim == 3 else rand_data['kl'].mean(axis=0)
    rand_l2_mean = rand_data['l2'].mean(axis=(0, 2)) if rand_data['l2'].ndim == 3 else rand_data['l2'].mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color map
    cmap = plt.cm.tab20
    colors = [cmap(i / len(genres)) for i in range(len(genres))]

    for idx, genre in enumerate(genres):
        key = f'f_allpair_{genre}'
        data = results[key]
        # Shape: (n_act, n_angles) — average over activations
        kl_mean = data['kl'].mean(axis=0)
        l2_mean = data['l2'].mean(axis=0)

        axes[0].plot(angles_plot, kl_mean[mask], label=genre, color=colors[idx], linewidth=1.5)
        axes[1].plot(angles_plot, l2_mean[mask], label=genre, color=colors[idx], linewidth=1.5)

    # Random baseline
    axes[0].plot(angles_plot, rand_kl_mean[mask], 'k--', linewidth=2, label='random', alpha=0.7)
    axes[1].plot(angles_plot, rand_l2_mean[mask], 'k--', linewidth=2, label='random', alpha=0.7)

    axes[0].set_xlabel('Angle (degrees)')
    axes[0].set_ylabel('KL divergence')
    axes[0].set_title('KL divergence vs steering angle')
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_xlim(0, max_angle)

    axes[1].set_xlabel('Angle (degrees)')
    axes[1].set_ylabel('L2 distance (final hidden)')
    axes[1].set_title('L2 distance vs steering angle')
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].set_xlim(0, max_angle)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.suptitle('Genre direction perturbation sweeps (all-pairwise estimators, layer 14)', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'genre_sweep_vs_random.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {results_dir}/genre_sweep_vs_random.png")

    # Also print numeric summary at key angles
    key_angles = [15, 30, 45, 60]
    print(f"\nKL divergence at key angles (mean over 50 FineWeb activations):")
    print(f"  {'Direction':<15s}", end='')
    for a in key_angles:
        print(f"  {a:>5d}°", end='')
    print()
    for genre in genres:
        kl_mean = results[f'f_allpair_{genre}']['kl'].mean(axis=0)
        print(f"  {genre:<15s}", end='')
        for a in key_angles:
            ai = np.argmin(np.abs(angles - a))
            print(f"  {kl_mean[ai]:>6.2f}", end='')
        print()
    print(f"  {'random':<15s}", end='')
    for a in key_angles:
        ai = np.argmin(np.abs(angles - a))
        print(f"  {rand_kl_mean[ai]:>6.2f}", end='')
    print()

    print(f"\nL2 distance at key angles:")
    print(f"  {'Direction':<15s}", end='')
    for a in key_angles:
        print(f"  {a:>5d}°", end='')
    print()
    for genre in genres:
        l2_mean = results[f'f_allpair_{genre}']['l2'].mean(axis=0)
        print(f"  {genre:<15s}", end='')
        for a in key_angles:
            ai = np.argmin(np.abs(angles - a))
            print(f"  {l2_mean[ai]:>6.1f}", end='')
        print()
    print(f"  {'random':<15s}", end='')
    for a in key_angles:
        ai = np.argmin(np.abs(angles - a))
        print(f"  {rand_l2_mean[ai]:>6.1f}", end='')
    print()


if __name__ == '__main__':
    main()
