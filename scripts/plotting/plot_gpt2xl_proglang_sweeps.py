#!/usr/bin/env python3
"""
Plot perturbation sweep curves for GPT-2 XL programming language features.
Two subplots: KL divergence and L2 distance, each showing per-language directions + random with shading.
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_base = 'results/programming_contrastive/gpt2-xl'

with open(f'{results_base}/sweep_results.pkl', 'rb') as f:
    s = pickle.load(f)

angles_deg = np.array(s['angles_deg'])
mask = np.ones(len(angles_deg), dtype=bool)  # show full range

lang_colors = {
    'java': '#e41a1c',
    'c': '#377eb8',
    'rust': '#ff7f00',
    'lisp': '#984ea3',
    'haskell': '#4daf4a',
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for mi, (metric, ylabel) in enumerate([('kl', 'KL divergence'), ('l2', 'L2 distance')]):
    ax = axes[mi]

    for lang, color in lang_colors.items():
        if lang not in s['sweep_results']:
            continue
        data = s['sweep_results'][lang][metric]
        mean_curve = data.mean(axis=0)
        ax.plot(angles_deg[mask], mean_curve[mask], color=color, linewidth=2, label=lang)

    # Random baseline with shading
    rand_data = s['sweep_results']['random'][metric]  # (N, n_angles, n_dirs)
    rand_per_dir = rand_data.mean(axis=0)  # (n_angles, n_dirs)
    rand_mean = rand_per_dir.mean(axis=1)
    rand_std = rand_per_dir.std(axis=1)
    ax.plot(angles_deg[mask], rand_mean[mask], color='gray', linewidth=2, linestyle='--', label='random')
    ax.fill_between(angles_deg[mask], (rand_mean - rand_std)[mask], (rand_mean + rand_std)[mask],
                     color='gray', alpha=0.2)

    # Set y-axis
    all_in_range = []
    for lang in lang_colors:
        if lang in s['sweep_results']:
            all_in_range.append(s['sweep_results'][lang][metric].mean(axis=0)[mask])
    all_in_range.append((rand_mean + rand_std)[mask])
    y_max = max(v.max() for v in all_in_range) * 1.05

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(bottom=0, top=y_max)

fig.suptitle('GPT-2 XL — Programming language feature sweeps (layer 2, coding prompts)', fontsize=13)
fig.tight_layout()
out = f'{results_base}/sweep_curves_kl_l2.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close(fig)
print("Done!")
