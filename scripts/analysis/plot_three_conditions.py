#!/usr/bin/env python3
"""Plot three conditions: real+random, real+genre, synthetic+random."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    results_dir = 'results/genre_contrastive/qwen3-0.6b'

    # Load genre sweep data (real activations, genre + random directions)
    with open(os.path.join(results_dir, 'sweep_results.pkl'), 'rb') as f:
        sw = pickle.load(f)

    # Load synthetic sweep data (real + synthetic activations, random directions)
    with open(os.path.join(results_dir, 'synthetic_sweep_results.pkl'), 'rb') as f:
        sy = pickle.load(f)

    genres = ['news', 'scientific', 'recipe', 'legal', 'children',
              'review', 'wikipedia', 'dialogue', 'email', 'technical', 'advertising']

    # --- Condition 1: Real activations, random directions (from synthetic sweep) ---
    # Shape: (50, 40, 10) -> mean over dirs -> (50, 40)
    angles_synth = sy['angles_deg']
    real_rand_kl = sy['real']['kl'].mean(axis=2)  # (50, 40)
    real_rand_l2 = sy['real']['l2'].mean(axis=2)

    # --- Condition 2: Real activations, genre (real) directions ---
    # Average across all 11 genres. Each genre: (50, 60)
    angles_genre = sw['angles_deg']
    genre_kls = []
    genre_l2s = []
    for g in genres:
        key = f'f_allpair_{g}'
        genre_kls.append(sw['sweep_results'][key]['kl'])
        genre_l2s.append(sw['sweep_results'][key]['l2'])
    # Stack: (11, 50, 60), then mean over genres -> (50, 60)
    real_genre_kl = np.stack(genre_kls).mean(axis=0)
    real_genre_l2 = np.stack(genre_l2s).mean(axis=0)

    # --- Condition 3: Synthetic (random) activations, random directions ---
    synth_rand_kl = sy['synthetic']['kl'].mean(axis=2)  # (50, 40)
    synth_rand_l2 = sy['synthetic']['l2'].mean(axis=2)

    # Use common angle range: 3 to 120 degrees
    max_angle = 120
    mask_genre = angles_genre <= max_angle

    # Stats for each condition
    conditions = [
        ('real acts, random dirs', angles_synth, real_rand_kl, real_rand_l2, 'tab:blue'),
        ('real acts, genre dirs', angles_genre[mask_genre], real_genre_kl[:, mask_genre], real_genre_l2[:, mask_genre], 'tab:green'),
        ('synth acts, random dirs', angles_synth, synth_rand_kl, synth_rand_l2, 'tab:red'),
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

    axes[0].set_xlabel('Angle (degrees)')
    axes[0].set_ylabel('KL divergence')
    axes[0].set_title('KL divergence vs perturbation angle')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Angle (degrees)')
    axes[1].set_ylabel('L2 distance (final hidden)')
    axes[1].set_title('L2 distance vs perturbation angle')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Perturbation sweeps: 3 conditions (mean + IQR over 50 activations)', y=1.02)
    plt.tight_layout()
    out = os.path.join(results_dir, 'three_conditions_sweep.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
