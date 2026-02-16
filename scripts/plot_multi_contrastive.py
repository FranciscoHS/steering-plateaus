#!/usr/bin/env python3
"""
Plots for multi-contrastive language direction isolation experiment.

1. KL perturbation sweep: pairwise diffs vs multi-contrastive vs MELBO
2. Cosine similarity heatmap
3. Spanish steering: language fractions by direction and angle
"""

import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

results_dir = 'results/multi_contrastive/qwen3-0.6b'

# Load data
with open(f'{results_dir}/sweep_results.pkl', 'rb') as f:
    sweep_data = pickle.load(f)
with open(f'{results_dir}/optimization_results.pkl', 'rb') as f:
    opt_data = pickle.load(f)
with open(f'{results_dir}/steering_spanish.json', 'r') as f:
    steer_data = json.load(f)

angles_deg = sweep_data['angles_deg']
sweep_results = sweep_data['sweep_results']


def kl_med(name):
    d = sweep_results[name]['kl']
    if d.ndim == 3:
        d = np.median(d, axis=2)
    return np.median(d, axis=0)


# ============================================================
# Plot 1: KL perturbation sweeps
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

max_angle = 90
mask = angles_deg <= max_angle
x = angles_deg[mask]

# Random baseline for all panels
rand_med = kl_med('random')

# Panel A: Pairwise diff-of-means (positive and negative)
ax = axes[0]
ax.plot(x, rand_med[mask], color='gray', lw=1.5, ls='--', label='Random')
for name, color, label in [('d_EM', '#d62728', 'd_EM (+mean, → EN)'),
                            ('-d_EM', '#d62728', '−d_EM (−mean, → ZH)'),
                            ('d_ED', '#1f77b4', 'd_ED (→ EN)'),
                            ('-d_ED', '#1f77b4', '−d_ED (→ NL)'),
                            ('d_MD', '#2ca02c', 'd_MD (→ ZH)'),
                            ('-d_MD', '#2ca02c', '−d_MD (→ NL)')]:
    ls = '-' if name.startswith('-') else ':'
    lw = 2.5 if name.startswith('-') else 1.5
    ax.plot(x, kl_med(name)[mask], color=color, lw=lw, ls=ls, label=label)
ax.set_title('Pairwise diff-of-means', fontsize=12)
ax.set_xlabel('Perturbation angle (°)')
ax.set_ylabel('Median KL divergence')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_angle)

# Panel B: Multi-contrastive estimators
ax = axes[1]
ax.plot(x, rand_med[mask], color='gray', lw=1.5, ls='--', label='Random')
for name, color, label in [('f_E', '#ff7f0e', 'f_E (English)'),
                            ('f_M', '#d62728', 'f_M (Mandarin)'),
                            ('f_D', '#1f77b4', 'f_D (Dutch)')]:
    ax.plot(x, kl_med(name)[mask], color=color, lw=2.5, label=label)
ax.set_title('Multi-contrastive estimators', fontsize=12)
ax.set_xlabel('Perturbation angle (°)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel C: MELBO directions
ax = axes[2]
ax.plot(x, rand_med[mask], color='gray', lw=1.5, ls='--', label='Random')
for name, color, label in [('MELBO_joint_1', '#d62728', 'MELBO joint 1'),
                            ('MELBO_joint_2', '#2ca02c', 'MELBO joint 2'),
                            ('MELBO_joint_3', '#1f77b4', 'MELBO joint 3'),
                            ('MELBO_EM', '#d62728', 'MELBO EM'),
                            ('MELBO_ED', '#1f77b4', 'MELBO ED'),
                            ('MELBO_MD', '#2ca02c', 'MELBO MD')]:
    ls = '-' if 'joint' in name else ':'
    ax.plot(x, kl_med(name)[mask], color=color, lw=2.5 if 'joint' in name else 1.5,
            ls=ls, label=label)
ax.set_title('MELBO directions', fontsize=12)
ax.set_xlabel('Perturbation angle (°)')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

fig.suptitle('KL perturbation response — multi-contrastive directions\n'
             'Qwen3-0.6B, layer 14 | 50 FineWeb activations',
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{results_dir}/kl_sweeps.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {results_dir}/kl_sweeps.png")


# ============================================================
# Plot 2: Cosine similarity heatmap
# ============================================================
names = opt_data['cosine_names']
cos_matrix = opt_data['cosine_matrix']

# Use shorter labels
short_names = []
for n in names:
    n = n.replace('MELBO_joint_', 'MJ').replace('MELBO_', 'M_')
    n = n.replace('old_mean', 'old_mean').replace('old_MELBO_1', 'old_M1')
    short_names.append(n)

fig, ax = plt.subplots(figsize=(10, 8))
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = ax.imshow(cos_matrix, cmap='RdBu_r', norm=norm, aspect='auto')

ax.set_xticks(range(len(short_names)))
ax.set_yticks(range(len(short_names)))
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(short_names, fontsize=9)

# Add text annotations
for i in range(len(names)):
    for j in range(len(names)):
        v = cos_matrix[i, j]
        color = 'white' if abs(v) > 0.6 else 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color=color)

plt.colorbar(im, label='Cosine similarity')
ax.set_title('Cosine similarity between all directions\n'
             'Qwen3-0.6B, layer 14', fontsize=13)
plt.tight_layout()
plt.savefig(f'{results_dir}/cosine_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {results_dir}/cosine_heatmap.png")


# ============================================================
# Plot 3: Spanish steering — language fractions
# ============================================================
# Group directions by purpose
steer_groups = {
    'Toward Mandarin': ['-d_EM', 'f_M', '-MELBO_EM'],
    'Toward English': ['d_EM', 'f_E', '-MELBO_joint_1'],
    'Toward Dutch': ['-d_ED', 'f_D'],
}

angles = sorted(set(r['angle'] for r in steer_data))
lang_colors = {'mandarin': '#d62728', 'english': '#ff7f0e', 'dutch': '#1f77b4', 'spanish': '#2ca02c'}

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# Row 1: toward-Mandarin and toward-English directions
# Row 2: toward-Dutch directions and empty

dir_order = ['-d_EM', 'f_M', '-MELBO_EM', 'd_EM', 'f_E', '-MELBO_joint_1', '-d_ED', 'f_D']
dir_labels = {
    '-d_EM': '−d_EM\n(pairwise → ZH)',
    'f_M': 'f_M\n(multi-contr. → ZH)',
    '-MELBO_EM': '−MELBO_EM\n(optimized → ZH)',
    'd_EM': 'd_EM\n(pairwise → EN)',
    'f_E': 'f_E\n(multi-contr. → EN)',
    '-MELBO_joint_1': '−MELBO_joint_1\n(optimized)',
    '-d_ED': '−d_ED\n(pairwise → NL)',
    'f_D': 'f_D\n(multi-contr. → NL)',
}

for idx, dir_name in enumerate(dir_order):
    ax = axes[idx // 4, idx % 4]

    for lang, color in lang_colors.items():
        if lang == 'mandarin':
            vals = [np.mean([r['cjk_fraction'] for r in steer_data
                            if r['direction'] == dir_name and r['angle'] == a])
                    for a in angles]
        else:
            vals = [np.mean([r['lang_scores'][lang] for r in steer_data
                            if r['direction'] == dir_name and r['angle'] == a])
                    for a in angles]
        ax.plot(angles, vals, 'o-', color=color, lw=2, markersize=5,
                label=lang.capitalize())

    ax.set_title(dir_labels.get(dir_name, dir_name), fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-2, 47)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=8, loc='upper right')
    if idx % 4 == 0:
        ax.set_ylabel('Language fraction')
    if idx >= 4:
        ax.set_xlabel('Steering angle (°)')

fig.suptitle('Language steering from Spanish prompts (held-out language)\n'
             'Qwen3-0.6B, layer 14 | 30 prompts, 50 tokens, temp=0.7',
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{results_dir}/steering_spanish.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {results_dir}/steering_spanish.png")

print("\nAll plots saved!")
