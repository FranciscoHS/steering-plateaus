#!/usr/bin/env python3
"""Plot individual genre direction sweeps up to 120 degrees."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    results_dir = 'results/genre_contrastive/qwen3-0.6b'
    with open(os.path.join(results_dir, 'sweep_results.pkl'), 'rb') as f:
        sweep = pickle.load(f)

    angles = sweep['angles_deg']
    results = sweep['sweep_results']

    max_angle = 120
    mask = angles <= max_angle
    angles_plot = angles[mask]

    genres = ['news', 'scientific', 'recipe', 'legal', 'children',
              'review', 'wikipedia', 'dialogue', 'email', 'technical', 'advertising']

    # Random baseline
    rand_data = results['random']
    if rand_data['kl'].ndim == 3:
        rand_kl_per_act = rand_data['kl'].mean(axis=2)
        rand_l2_per_act = rand_data['l2'].mean(axis=2)
    else:
        rand_kl_per_act = rand_data['kl']
        rand_l2_per_act = rand_data['l2']
    rand_kl_mean = rand_kl_per_act.mean(axis=0)
    rand_l2_mean = rand_l2_per_act.mean(axis=0)
    rand_kl_q25 = np.percentile(rand_kl_per_act, 25, axis=0)
    rand_kl_q75 = np.percentile(rand_kl_per_act, 75, axis=0)
    rand_l2_q25 = np.percentile(rand_l2_per_act, 25, axis=0)
    rand_l2_q75 = np.percentile(rand_l2_per_act, 75, axis=0)

    cmap = plt.cm.tab20
    colors = [cmap(i / len(genres)) for i in range(len(genres))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, genre in enumerate(genres):
        key = f'f_allpair_{genre}'
        data = results[key]
        kl_mean = data['kl'].mean(axis=0)
        l2_mean = data['l2'].mean(axis=0)
        axes[0].plot(angles_plot, kl_mean[mask], label=genre, color=colors[idx], linewidth=1.5)
        axes[1].plot(angles_plot, l2_mean[mask], label=genre, color=colors[idx], linewidth=1.5)

    # Random with IQR
    axes[0].plot(angles_plot, rand_kl_mean[mask], 'k--', linewidth=2, label='random', alpha=0.7)
    axes[0].fill_between(angles_plot, rand_kl_q25[mask], rand_kl_q75[mask], color='black', alpha=0.1)
    axes[1].plot(angles_plot, rand_l2_mean[mask], 'k--', linewidth=2, label='random', alpha=0.7)
    axes[1].fill_between(angles_plot, rand_l2_q25[mask], rand_l2_q75[mask], color='black', alpha=0.1)

    axes[0].set_xlabel('Angle (degrees)')
    axes[0].set_ylabel('KL divergence')
    axes[0].set_title('KL divergence vs steering angle')
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_xlabel('Angle (degrees)')
    axes[1].set_ylabel('L2 distance (final hidden)')
    axes[1].set_title('L2 distance vs steering angle')
    axes[1].legend(fontsize=8, ncol=2)

    for ax in axes:
        ax.set_xlim(0, max_angle)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Genre direction perturbation sweeps (all-pairwise estimators, layer 14)', y=1.02)
    plt.tight_layout()
    out = os.path.join(results_dir, 'genre_sweep_vs_random_120.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")

if __name__ == '__main__':
    main()
