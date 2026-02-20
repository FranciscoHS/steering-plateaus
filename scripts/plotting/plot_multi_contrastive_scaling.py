#!/usr/bin/env python3
"""Plot multi-contrastive perturbation sweeps across Qwen3 model sizes."""

import pickle
import numpy as np
import matplotlib.pyplot as plt

models = ['qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b']
labels = ['0.6B', '1.7B', '4B']
model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
max_angle = 60

# Directions to plot: diff-of-means, multi-contrastive estimators, random
# Group by type for clarity
direction_groups = {
    'Diff-of-means': ['d_EM', 'd_ED', 'd_MD'],
    'Multi-contrastive': ['f_E', 'f_M', 'f_D'],
}

# One figure per model: KL vs angle for all feature directions + random
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (model, label, color) in enumerate(zip(models, labels, model_colors)):
    ax = axes[i]
    with open(f'results/multi_contrastive/{model}/sweep_results.pkl', 'rb') as f:
        d = pickle.load(f)
    angles = d['angles_deg']
    mask = angles <= max_angle
    angles_cut = angles[mask]
    results = d['sweep_results']

    # Plot random baseline (mean over activations and directions)
    rand_kl = results['random']['kl']  # [N, n_angles, n_dirs]
    rand_mean = rand_kl.mean(axis=(0, 2))[mask]
    ax.plot(angles_cut, rand_mean, 'k--', linewidth=2, alpha=0.5, label='random')

    # Plot diff-of-means directions
    dom_colors = ['#e41a1c', '#377eb8', '#4daf4a']
    dom_labels = ['d_EM (EN-ZH)', 'd_ED (EN-NL)', 'd_MD (ZH-NL)']
    for j, (dname, dlabel, dcolor) in enumerate(zip(
            direction_groups['Diff-of-means'], dom_labels, dom_colors)):
        if dname in results:
            kl = results[dname]['kl'].mean(axis=0)[mask]
            ax.plot(angles_cut, kl, '-', color=dcolor, linewidth=2, label=dlabel)

    # Plot multi-contrastive estimators
    mc_colors = ['#984ea3', '#ff7f00', '#a65628']
    mc_labels = ['f_E (English)', 'f_M (Mandarin)', 'f_D (Dutch)']
    for j, (dname, dlabel, dcolor) in enumerate(zip(
            direction_groups['Multi-contrastive'], mc_labels, mc_colors)):
        if dname in results:
            kl = results[dname]['kl'].mean(axis=0)[mask]
            ax.plot(angles_cut, kl, ':', color=dcolor, linewidth=2, label=dlabel)

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('KL divergence')
    ax.set_title(f'Qwen3-{label} (layer 14)')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(0, max_angle)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

fig.suptitle('Multi-contrastive language directions: perturbation sweeps', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('results/multi_contrastive/perturbation_sweeps_scaling.png', dpi=150, bbox_inches='tight')
print('Saved: results/multi_contrastive/perturbation_sweeps_scaling.png')
