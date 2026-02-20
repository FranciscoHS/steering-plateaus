#!/usr/bin/env python3
"""Plot combo sweep results: KL and L2 vs angle, grouped by subset size k."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    results_dir = 'results/genre_contrastive/qwen3-0.6b'
    with open(os.path.join(results_dir, 'combo_sweep_results.pkl'), 'rb') as f:
        data = pickle.load(f)

    angles = data['angles_deg']
    results_by_k = data['results_by_k']
    rand_data = data['random']

    # Random baseline
    rand_kl = rand_data['kl'].mean(axis=(0, 2)) if rand_data['kl'].ndim == 3 else rand_data['kl'].mean(axis=0)
    rand_l2 = rand_data['l2'].mean(axis=(0, 2)) if rand_data['l2'].ndim == 3 else rand_data['l2'].mean(axis=0)

    ks = sorted(results_by_k.keys())

    # For each k, compute median KL and L2 across all combos
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(ks) - 1)) for i in range(len(ks))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, k in enumerate(ks):
        combos = results_by_k[k]
        all_kl = np.array([c['kl_mean'] for c in combos])  # (n_combos, n_angles)
        all_l2 = np.array([c['l2_mean'] for c in combos])

        median_kl = np.median(all_kl, axis=0)
        median_l2 = np.median(all_l2, axis=0)

        # Shaded region: 25th-75th percentile
        q25_kl = np.percentile(all_kl, 25, axis=0)
        q75_kl = np.percentile(all_kl, 75, axis=0)
        q25_l2 = np.percentile(all_l2, 25, axis=0)
        q75_l2 = np.percentile(all_l2, 75, axis=0)

        label = f'k={k} ({len(combos)})'
        axes[0].plot(angles, median_kl, color=colors[idx], linewidth=1.5, label=label)
        axes[1].plot(angles, median_l2, color=colors[idx], linewidth=1.5, label=label)

    # Random baseline with IQR shading
    # rand shape: (n_act, n_angles, n_dirs) — compute per-activation mean over dirs, then percentiles
    if rand_data['kl'].ndim == 3:
        rand_kl_per_act = rand_data['kl'].mean(axis=2)  # (n_act, n_angles)
        rand_l2_per_act = rand_data['l2'].mean(axis=2)
    else:
        rand_kl_per_act = rand_data['kl']
        rand_l2_per_act = rand_data['l2']
    rand_kl_q25 = np.percentile(rand_kl_per_act, 25, axis=0)
    rand_kl_q75 = np.percentile(rand_kl_per_act, 75, axis=0)
    rand_l2_q25 = np.percentile(rand_l2_per_act, 25, axis=0)
    rand_l2_q75 = np.percentile(rand_l2_per_act, 75, axis=0)
    axes[0].plot(angles, rand_kl, 'k--', linewidth=2, label='random', alpha=0.7)
    axes[0].fill_between(angles, rand_kl_q25, rand_kl_q75, color='black', alpha=0.1)
    axes[1].plot(angles, rand_l2, 'k--', linewidth=2, label='random', alpha=0.7)
    axes[1].fill_between(angles, rand_l2_q25, rand_l2_q75, color='black', alpha=0.1)

    for ax in axes:
        ax.set_xlim(0, 20)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylim(0, 0.15)
    axes[1].set_ylim(0, 100)

    axes[0].set_xlabel('Angle (degrees)')
    axes[0].set_ylabel('KL divergence')
    axes[0].set_title('KL divergence vs steering angle')
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_xlabel('Angle (degrees)')
    axes[1].set_ylabel('L2 distance (final hidden)')
    axes[1].set_title('L2 distance vs steering angle')
    axes[1].legend(fontsize=7, ncol=2)

    plt.suptitle('Perturbation sweeps: sum of k genre directions (median + IQR)', y=1.02)
    plt.tight_layout()
    out = os.path.join(results_dir, 'combo_sweep_by_k.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
