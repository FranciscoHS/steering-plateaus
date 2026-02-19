#!/usr/bin/env python3
"""
Plot perturbation sweeps for programming language directions.

Usage:
    python scripts/directions/plot_programming_sweeps.py [--results_dir results/programming_contrastive/qwen3-0.6b]
"""

import sys
import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir',
                        default='results/programming_contrastive/qwen3-0.6b')
    args = parser.parse_args()

    # Load sweep results
    sweep_path = os.path.join(args.results_dir, 'sweep_results.pkl')
    print(f"Loading {sweep_path}")
    with open(sweep_path, 'rb') as f:
        data = pickle.load(f)

    angles_deg = data['angles_deg']
    sweep_results = data['sweep_results']

    # Load contrastive results for feature norms
    contrastive_path = os.path.join(args.results_dir, 'contrastive_results.pkl')
    with open(contrastive_path, 'rb') as f:
        contrastive = pickle.load(f)
    languages = contrastive['languages']

    # --- Plot 1: KL divergence for all f_hat directions + random ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Random baseline (mean ± std across directions and samples)
    rand_kl = sweep_results['random']['kl']  # [N, n_angles, n_dirs]
    rand_mean = rand_kl.mean(axis=(0, 2))
    rand_std = rand_kl.std(axis=(0, 2))
    ax.fill_between(angles_deg, rand_mean - rand_std, rand_mean + rand_std,
                     alpha=0.15, color='gray')
    ax.plot(angles_deg, rand_mean, '--', color='gray', label='random', linewidth=1.5)

    # Language directions
    colors = plt.cm.tab10(np.linspace(0, 1, len(languages)))
    for i, lang in enumerate(languages):
        key = f'f_hat_{lang}'
        if key in sweep_results:
            kl = sweep_results[key]['kl']  # [N, n_angles]
            kl_mean = kl.mean(axis=0)
            ax.plot(angles_deg, kl_mean, '-', color=colors[i], label=f'f̂_{lang}',
                    linewidth=2)

    # SVD u1
    if 'svd_u1' in sweep_results:
        kl = sweep_results['svd_u1']['kl']
        ax.plot(angles_deg, kl.mean(axis=0), ':', color='black', label='SVD u₁',
                linewidth=2)

    ax.set_xlabel('Steering angle (degrees)', fontsize=12)
    ax.set_ylabel('KL divergence', fontsize=12)
    ax.set_title('Perturbation sweeps: programming language directions vs random', fontsize=13)
    ax.legend(ncol=2, fontsize=9)
    ax.set_xlim(0, 180)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(args.results_dir, 'sweep_kl_all_languages.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    # --- Plot 2: Same but with L2 distance ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

    rand_l2 = sweep_results['random']['l2']
    rand_l2_mean = rand_l2.mean(axis=(0, 2))
    rand_l2_std = rand_l2.std(axis=(0, 2))
    ax2.fill_between(angles_deg, rand_l2_mean - rand_l2_std, rand_l2_mean + rand_l2_std,
                      alpha=0.15, color='gray')
    ax2.plot(angles_deg, rand_l2_mean, '--', color='gray', label='random', linewidth=1.5)

    for i, lang in enumerate(languages):
        key = f'f_hat_{lang}'
        if key in sweep_results:
            l2 = sweep_results[key]['l2']
            ax2.plot(angles_deg, l2.mean(axis=0), '-', color=colors[i],
                     label=f'f̂_{lang}', linewidth=2)

    if 'svd_u1' in sweep_results:
        l2 = sweep_results['svd_u1']['l2']
        ax2.plot(angles_deg, l2.mean(axis=0), ':', color='black', label='SVD u₁',
                 linewidth=2)

    ax2.set_xlabel('Steering angle (degrees)', fontsize=12)
    ax2.set_ylabel('L2 distance (final hidden state)', fontsize=12)
    ax2.set_title('L2 distance: programming language directions vs random', fontsize=13)
    ax2.legend(ncol=2, fontsize=9)
    ax2.set_xlim(0, 180)
    ax2.grid(True, alpha=0.3)

    out_path2 = os.path.join(args.results_dir, 'sweep_l2_all_languages.png')
    fig2.tight_layout()
    fig2.savefig(out_path2, dpi=150)
    print(f"Saved {out_path2}")

    # --- Print summary stats at key angles ---
    print(f"\n{'='*60}")
    print("KL DIVERGENCE AT KEY ANGLES (mean across activations)")
    print(f"{'='*60}")
    key_angles = [15, 30, 45, 90]
    angle_indices = [np.argmin(np.abs(angles_deg - a)) for a in key_angles]

    header = f"{'Direction':>18s}" + "".join(f"{a:>8d}°" for a in key_angles)
    print(header)
    print(f"{'random':>18s}" + "".join(f"{rand_mean[idx]:>9.2f}" for idx in angle_indices))
    for lang in languages:
        key = f'f_hat_{lang}'
        if key in sweep_results:
            kl = sweep_results[key]['kl'].mean(axis=0)
            print(f"{'f̂_'+lang:>18s}" + "".join(f"{kl[idx]:>9.2f}" for idx in angle_indices))
    if 'svd_u1' in sweep_results:
        kl = sweep_results['svd_u1']['kl'].mean(axis=0)
        print(f"{'SVD u₁':>18s}" + "".join(f"{kl[idx]:>9.2f}" for idx in angle_indices))

    plt.close('all')
    print("\nDone!")


if __name__ == '__main__':
    main()
