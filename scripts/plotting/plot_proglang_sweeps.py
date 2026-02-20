#!/usr/bin/env python3
"""
Plot perturbation sweep curves (KL vs angle) for programming language features.
One subplot per model, showing each language direction + random baseline.
"""

import pickle, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

models = ['qwen3-1.7b', 'qwen3-4b']
model_labels = {'qwen3-1.7b': 'Qwen3-1.7B', 'qwen3-4b': 'Qwen3-4B'}
results_base = 'results/programming_contrastive'

lang_colors = {
    'java': '#e41a1c',
    'c': '#377eb8',
    'rust': '#ff7f00',
    'go': '#4daf4a',
    'lisp': '#984ea3',
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for mi, model in enumerate(models):
    ax = axes[mi]

    sweep_path = f'{results_base}/{model}/sweep_results.pkl'
    if not os.path.exists(sweep_path):
        print(f"Skipping {model} — not found")
        continue

    with open(sweep_path, 'rb') as f:
        s = pickle.load(f)

    angles_deg = np.array(s['angles_deg'])
    mask = angles_deg <= 50

    for lang, color in lang_colors.items():
        if lang not in s['sweep_results']:
            continue
        kl = s['sweep_results'][lang]['kl']
        mean_kl = kl.mean(axis=0)
        ax.plot(angles_deg[mask], mean_kl[mask], color=color, linewidth=2, label=lang)

    # Random baseline with shading (std across random dirs, averaged over samples)
    rand_kl = s['sweep_results']['random']['kl']  # (N, n_angles, n_dirs)
    rand_per_dir = rand_kl.mean(axis=0)  # (n_angles, n_dirs)
    rand_mean = rand_per_dir.mean(axis=1)
    rand_std = rand_per_dir.std(axis=1)
    ax.plot(angles_deg[mask], rand_mean[mask], color='gray', linewidth=2, linestyle='--', label='random')
    ax.fill_between(angles_deg[mask], (rand_mean - rand_std)[mask], (rand_mean + rand_std)[mask],
                     color='gray', alpha=0.2)

    # Set y-axis to match data range in the 0-50° window
    all_kl_in_range = []
    for lang in lang_colors:
        if lang in s['sweep_results']:
            all_kl_in_range.append(s['sweep_results'][lang]['kl'].mean(axis=0)[mask])
    all_kl_in_range.append((rand_mean + rand_std)[mask])
    y_max = max(v.max() for v in all_kl_in_range) * 1.05

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('KL divergence')
    ax.set_title(model_labels[model])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    ax.set_ylim(bottom=0, top=y_max)

fig.suptitle('Programming language feature sweeps', fontsize=13)
fig.tight_layout()
out = f'{results_base}/sweep_curves_proglang.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close(fig)
print("Done!")
