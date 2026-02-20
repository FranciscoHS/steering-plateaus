#!/usr/bin/env python3
"""Plot synthetic vs real activation perturbation sweep results."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    results_dir = 'results/genre_contrastive/qwen3-0.6b'
    with open(os.path.join(results_dir, 'synthetic_sweep_results.pkl'), 'rb') as f:
        data = pickle.load(f)

    angles = data['angles_deg']
    # Shape: (n_activations, n_angles, n_dirs)
    real_kl = data['real']['kl']
    real_l2 = data['real']['l2']
    synth_kl = data['synthetic']['kl']
    synth_l2 = data['synthetic']['l2']

    # Average over random directions first, then compute stats across activations
    real_kl_per_act = real_kl.mean(axis=2)   # (50, 40)
    real_l2_per_act = real_l2.mean(axis=2)
    synth_kl_per_act = synth_kl.mean(axis=2)
    synth_l2_per_act = synth_l2.mean(axis=2)

    real_kl_mean = real_kl_per_act.mean(axis=0)
    real_kl_q25 = np.percentile(real_kl_per_act, 25, axis=0)
    real_kl_q75 = np.percentile(real_kl_per_act, 75, axis=0)

    synth_kl_mean = synth_kl_per_act.mean(axis=0)
    synth_kl_q25 = np.percentile(synth_kl_per_act, 25, axis=0)
    synth_kl_q75 = np.percentile(synth_kl_per_act, 75, axis=0)

    real_l2_mean = real_l2_per_act.mean(axis=0)
    real_l2_q25 = np.percentile(real_l2_per_act, 25, axis=0)
    real_l2_q75 = np.percentile(real_l2_per_act, 75, axis=0)

    synth_l2_mean = synth_l2_per_act.mean(axis=0)
    synth_l2_q25 = np.percentile(synth_l2_per_act, 25, axis=0)
    synth_l2_q75 = np.percentile(synth_l2_per_act, 75, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # KL
    axes[0].plot(angles, real_kl_mean, 'b-', linewidth=2, label='real activations')
    axes[0].fill_between(angles, real_kl_q25, real_kl_q75, color='blue', alpha=0.15)
    axes[0].plot(angles, synth_kl_mean, 'r-', linewidth=2, label='synthetic (Gaussian)')
    axes[0].fill_between(angles, synth_kl_q25, synth_kl_q75, color='red', alpha=0.15)
    axes[0].set_xlabel('Angle (degrees)')
    axes[0].set_ylabel('KL divergence')
    axes[0].set_title('KL divergence vs perturbation angle')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # L2
    axes[1].plot(angles, real_l2_mean, 'b-', linewidth=2, label='real activations')
    axes[1].fill_between(angles, real_l2_q25, real_l2_q75, color='blue', alpha=0.15)
    axes[1].plot(angles, synth_l2_mean, 'r-', linewidth=2, label='synthetic (Gaussian)')
    axes[1].fill_between(angles, synth_l2_q25, synth_l2_q75, color='red', alpha=0.15)
    axes[1].set_xlabel('Angle (degrees)')
    axes[1].set_ylabel('L2 distance (final hidden)')
    axes[1].set_title('L2 distance vs perturbation angle')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Random-direction perturbation: real vs synthetic activations\n'
                 '(50 activations each, 10 random dirs, mean + IQR)', y=1.02)
    plt.tight_layout()
    out = os.path.join(results_dir, 'synthetic_vs_real_sweep.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
