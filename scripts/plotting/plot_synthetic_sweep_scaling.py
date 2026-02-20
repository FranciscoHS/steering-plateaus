#!/usr/bin/env python3
"""Plot synthetic sweep results across Qwen3 model sizes (0.6B, 1.7B, 4B)."""

import pickle
import numpy as np
import matplotlib.pyplot as plt

models = ['qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b']
labels = ['0.6B', '1.7B', '4B']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
max_angle = 60

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (model, label, color) in enumerate(zip(models, labels, colors)):
    with open(f'results/genre_contrastive/{model}/synthetic_sweep_results_L2.pkl', 'rb') as f:
        d = pickle.load(f)
    angles = d['angles_deg']
    mask = angles <= max_angle
    angles_cut = angles[mask]

    real_kl = d['real']['kl'].mean(axis=(0, 2))[mask]
    synth_kl = d['synthetic']['kl'].mean(axis=(0, 2))[mask]
    real_l2 = d['real']['l2'].mean(axis=(0, 2))[mask]
    synth_l2 = d['synthetic']['l2'].mean(axis=(0, 2))[mask]

    # Normalize L2 to max across both real and synthetic within the plotted range
    l2_max = max(real_l2.max(), synth_l2.max())
    real_l2_norm = real_l2 / l2_max
    synth_l2_norm = synth_l2 / l2_max

    # KL plot (unnormalized)
    axes[0].plot(angles_cut, real_kl, '-', color=color, linewidth=2, label=f'{label} real')
    axes[0].plot(angles_cut, synth_kl, '--', color=color, linewidth=2, label=f'{label} synthetic')

    # Normalized L2 plot
    axes[1].plot(angles_cut, real_l2_norm, '-', color=color, linewidth=2, label=f'{label} real')
    axes[1].plot(angles_cut, synth_l2_norm, '--', color=color, linewidth=2, label=f'{label} synthetic')

axes[0].set_xlabel('Perturbation angle (degrees)')
axes[0].set_ylabel('KL divergence')
axes[0].set_title('KL divergence (layer 2, random dirs)')
axes[0].legend(fontsize=8, ncol=2)
axes[0].set_xlim(0, max_angle)
axes[0].set_ylim(bottom=0)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Perturbation angle (degrees)')
axes[1].set_ylabel('L2 distance (normalized)')
axes[1].set_title('Normalized L2 distance (layer 2, random dirs)')
axes[1].legend(fontsize=8, ncol=2)
axes[1].set_xlim(0, max_angle)
axes[1].set_ylim(0, 1.05)
axes[1].grid(True, alpha=0.3)

fig.suptitle('Qwen3 family: real vs synthetic activation perturbation sweeps', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('results/genre_contrastive/synthetic_sweep_scaling.png', dpi=150, bbox_inches='tight')
print('Saved: results/genre_contrastive/synthetic_sweep_scaling.png')
