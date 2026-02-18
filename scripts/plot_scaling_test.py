#!/usr/bin/env python3
"""
Plots for multi-contrastive scaling verification experiment.

1. Scaling curve: measured angle vs theoretical prediction
2. Cosine similarity heatmap of pairwise directions
3. Distribution of angles at each k (violin plot)

Usage:
    python scripts/plot_scaling_test.py
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

results_dir = 'results/scaling_test/qwen3-0.6b'

with open(f'{results_dir}/scaling_results.pkl', 'rb') as f:
    data = pickle.load(f)

consistency = data['consistency_results']
svd_results = data['svd_results']
pairwise_dirs = data['pairwise_dirs']
non_en = data['non_en']
k_ref = data['k_ref']


def theoretical_cosine(k, k_ref):
    return np.sqrt(k_ref * (k - 1) / (k * (k_ref - 1)))


# ============================================================
# Plot 1: Scaling curve (main result)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: angles
ax = axes[0]
ks = sorted(consistency.keys())
ang_means = [consistency[k]['angles_deg'].mean() for k in ks]
ang_stds = [consistency[k]['angles_deg'].std() for k in ks]
ang_preds = [consistency[k]['angle_predicted'] for k in ks]

# SVD-based angles
svd_ang_means = []
for k in ks:
    if k in svd_results:
        svd_ang = np.degrees(np.arccos(np.clip(svd_results[k]['cosines_to_ref'], -1, 1)))
        svd_ang_means.append(svd_ang.mean())
    else:
        svd_ang_means.append(0)

# Theory line (smooth)
k_smooth = np.linspace(2, k_ref, 100)
ang_theory = np.degrees(np.arccos(theoretical_cosine(k_smooth, k_ref)))

ax.plot(k_smooth, ang_theory, 'k-', lw=2, label='Theory: 1/√k', zorder=3)
ax.errorbar(ks, ang_means, yerr=ang_stds, fmt='o', color='#d62728',
            capsize=4, markersize=8, lw=2, label='Mean-based estimator', zorder=4)
ax.plot(ks, svd_ang_means, 's', color='#1f77b4', markersize=7,
        label='SVD-based estimator', zorder=4)
# k=10 point at 0
ax.plot(k_ref, 0, 'o', color='#d62728', markersize=8, zorder=4)
ax.plot(k_ref, 0, 's', color='#1f77b4', markersize=7, zorder=4)

ax.set_xlabel('k (number of languages)', fontsize=12)
ax.set_ylabel('Angle to k=10 reference (°)', fontsize=12)
ax.set_title('Multi-contrastive scaling: angle convergence', fontsize=13)
ax.set_xticks(range(2, k_ref + 1))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(1.5, k_ref + 0.5)
ax.set_ylim(bottom=-2)

# Right panel: cosines
ax = axes[1]
cos_means = [consistency[k]['cosines'].mean() for k in ks]
cos_stds = [consistency[k]['cosines'].std() for k in ks]
cos_preds = [consistency[k]['cos_predicted'] for k in ks]

cos_theory = theoretical_cosine(k_smooth, k_ref)

ax.plot(k_smooth, cos_theory, 'k-', lw=2, label='Theory: √(10(k-1)/(9k))', zorder=3)
ax.errorbar(ks, cos_means, yerr=cos_stds, fmt='o', color='#d62728',
            capsize=4, markersize=8, lw=2, label='Mean-based estimator', zorder=4)
ax.plot(k_ref, 1.0, 'o', color='#d62728', markersize=8, zorder=4)

ax.set_xlabel('k (number of languages)', fontsize=12)
ax.set_ylabel('Cosine similarity to k=10 reference', fontsize=12)
ax.set_title('Multi-contrastive scaling: cosine convergence', fontsize=13)
ax.set_xticks(range(2, k_ref + 1))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(1.5, k_ref + 0.5)
ax.set_ylim(0.5, 1.05)

fig.suptitle(f'Multi-contrastive English estimator — self-consistency test\n'
             f'{data["model"]}, layer {data["layer"]} | {data["n_prompts"]} prompts, '
             f'{len(non_en)} contrast languages',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{results_dir}/scaling_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {results_dir}/scaling_curve.png")


# ============================================================
# Plot 2: Cosine similarity heatmap of pairwise directions
# ============================================================
import torch.nn.functional as F
import torch

dir_names = [f'd_E_{l}' for l in non_en] + ['f_E (k=10)']
dir_tensors = [pairwise_dirs[l] for l in non_en]
dir_tensors.append(data['ref_direction'])

n = len(dir_names)
cos_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        cos_matrix[i, j] = F.cosine_similarity(
            dir_tensors[i].unsqueeze(0),
            dir_tensors[j].unsqueeze(0)).item()

fig, ax = plt.subplots(figsize=(10, 8))
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = ax.imshow(cos_matrix, cmap='RdBu_r', norm=norm, aspect='auto')

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(dir_names, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(dir_names, fontsize=10)

for i in range(n):
    for j in range(n):
        v = cos_matrix[i, j]
        color = 'white' if abs(v) > 0.6 else 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8, color=color)

plt.colorbar(im, label='Cosine similarity')
ax.set_title(f'Cosine similarity: pairwise EN-X directions + k=10 estimator\n'
             f'{data["model"]}, layer {data["layer"]}', fontsize=13)
plt.tight_layout()
plt.savefig(f'{results_dir}/cosine_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {results_dir}/cosine_heatmap.png")


# ============================================================
# Plot 3: Distribution of angles at each k (violin/box plot)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

positions = sorted(consistency.keys())
angle_data = [consistency[k]['angles_deg'] for k in positions]

parts = ax.violinplot(angle_data, positions=positions, showmeans=True,
                      showmedians=False, widths=0.6)
for pc in parts['bodies']:
    pc.set_facecolor('#d62728')
    pc.set_alpha(0.3)
parts['cmeans'].set_color('#d62728')

# Theory line
k_smooth = np.linspace(2, max(positions), 100)
ax.plot(k_smooth, np.degrees(np.arccos(theoretical_cosine(k_smooth, k_ref))),
        'k--', lw=2, label='Theory: 1/√k')

ax.set_xlabel('k (number of languages)', fontsize=12)
ax.set_ylabel('Angle to k=10 reference (°)', fontsize=12)
ax.set_title(f'Distribution of angles across subsets at each k\n'
             f'{data["model"]}, layer {data["layer"]}', fontsize=13)
ax.set_xticks(positions)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(1.5, max(positions) + 0.5)

plt.tight_layout()
plt.savefig(f'{results_dir}/angle_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {results_dir}/angle_distributions.png")


print("\nAll plots saved!")
