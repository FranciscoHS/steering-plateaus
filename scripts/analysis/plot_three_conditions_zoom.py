#!/usr/bin/env python3
"""Plot three conditions zoomed to 0-40 degrees: real+random, real+genre, synthetic+random."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    results_dir = 'results/genre_contrastive/qwen3-0.6b'

    with open(os.path.join(results_dir, 'sweep_results.pkl'), 'rb') as f:
        sw = pickle.load(f)
    with open(os.path.join(results_dir, 'synthetic_sweep_results.pkl'), 'rb') as f:
        sy = pickle.load(f)

    genres = ['news', 'scientific', 'recipe', 'legal', 'children',
              'review', 'wikipedia', 'dialogue', 'email', 'technical', 'advertising']

    # Condition 1: Real activations, random directions
    angles_synth = sy['angles_deg']
    real_rand_kl = sy['real']['kl'].mean(axis=2)
    real_rand_l2 = sy['real']['l2'].mean(axis=2)

    # Condition 2: Real activations, genre directions (avg across 11 genres)
    angles_genre = sw['angles_deg']
    genre_kls = []
    genre_l2s = []
    for g in genres:
        key = f'f_allpair_{g}'
        genre_kls.append(sw['sweep_results'][key]['kl'])
        genre_l2s.append(sw['sweep_results'][key]['l2'])
    real_genre_kl = np.stack(genre_kls).mean(axis=0)
    real_genre_l2 = np.stack(genre_l2s).mean(axis=0)

    # Condition 3: Synthetic activations, random directions
    synth_rand_kl = sy['synthetic']['kl'].mean(axis=2)
    synth_rand_l2 = sy['synthetic']['l2'].mean(axis=2)

    max_angle = 40
    mask_synth = angles_synth <= max_angle
    mask_genre = angles_genre <= max_angle

    conditions = [
        ('real acts, random dirs', angles_synth[mask_synth], real_rand_kl[:, mask_synth], real_rand_l2[:, mask_synth], 'tab:blue'),
        ('real acts, genre dirs', angles_genre[mask_genre], real_genre_kl[:, mask_genre], real_genre_l2[:, mask_genre], 'tab:green'),
        ('synth acts, random dirs', angles_synth[mask_synth], synth_rand_kl[:, mask_synth], synth_rand_l2[:, mask_synth], 'tab:red'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for label, angles, kl_per_act, l2_per_act, color in conditions:
        kl_mean = kl_per_act.mean(axis=0)
        kl_q25 = np.percentile(kl_per_act, 25, axis=0)
        kl_q75 = np.percentile(kl_per_act, 75, axis=0)
        l2_mean = l2_per_act.mean(axis=0)
        l2_q25 = np.percentile(l2_per_act, 25, axis=0)
        l2_q75 = np.percentile(l2_per_act, 75, axis=0)

        axes[0].plot(angles, kl_mean, color=color, linewidth=2, label=label)
        axes[0].fill_between(angles, kl_q25, kl_q75, color=color, alpha=0.12)
        axes[1].plot(angles, l2_mean, color=color, linewidth=2, label=label)
        axes[1].fill_between(angles, l2_q25, l2_q75, color=color, alpha=0.12)

    for ax in axes:
        ax.set_xlim(0, max_angle)
        ax.grid(True, alpha=0.3)

    axes[0].set_xlabel('Angle (degrees)')
    axes[0].set_ylabel('KL divergence')
    axes[0].set_title('KL divergence vs perturbation angle')
    axes[0].legend()

    axes[1].set_xlabel('Angle (degrees)')
    axes[1].set_ylabel('L2 distance (final hidden)')
    axes[1].set_title('L2 distance vs perturbation angle')
    axes[1].legend()

    plt.suptitle('Perturbation sweeps: 3 conditions, 0-40° (mean + IQR over 50 activations)', y=1.02)
    plt.tight_layout()
    out = os.path.join(results_dir, 'three_conditions_sweep_zoom40.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
