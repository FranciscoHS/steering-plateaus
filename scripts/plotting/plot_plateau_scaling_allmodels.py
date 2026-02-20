#!/usr/bin/env python3
"""
Plot plateau size vs n for all models on the same plot, KL threshold = 0.1.
Two plots: one using 5-lang data (n=1..4), one using 10-lang data (n=1..9).
Includes random baseline as horizontal lines.
"""

import pickle, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def measure_threshold_crossing(mean_curve, angles_deg, threshold):
    """Find angle where curve first crosses threshold, using linear interpolation."""
    for i in range(len(mean_curve) - 1):
        if mean_curve[i] < threshold <= mean_curve[i + 1]:
            frac = (threshold - mean_curve[i]) / (mean_curve[i + 1] - mean_curve[i])
            return angles_deg[i] + frac * (angles_deg[i + 1] - angles_deg[i])
    if mean_curve[-1] < threshold:
        return angles_deg[-1]
    return angles_deg[0]


models = ['qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b']
model_colors = {'qwen3-0.6b': '#e41a1c', 'qwen3-1.7b': '#377eb8', 'qwen3-4b': '#4daf4a'}
model_labels = {'qwen3-0.6b': 'Qwen3-0.6B', 'qwen3-1.7b': 'Qwen3-1.7B', 'qwen3-4b': 'Qwen3-4B'}
threshold = 0.1

# Compute random baseline plateau sizes
random_plateaus = {}
for model in models:
    with open(f'results/allpairs_contrastive/{model}/sweep_results.pkl', 'rb') as f:
        s = pickle.load(f)
    rand_kl = s['sweep_results']['random']['kl']
    rand_kl_mean = rand_kl.mean(axis=(0, 2))
    angles = np.array(s['angles_deg'])
    random_plateaus[model] = measure_threshold_crossing(rand_kl_mean, angles, threshold)
    print(f"{model}: random baseline = {random_plateaus[model]:.1f}°")

configs = [
    ('plateau_scaling_5lang.pkl', 5, '5-lang features'),
    ('plateau_scaling.pkl', 10, '10-lang features'),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ci, (pkl_name, k_total, title_suffix) in enumerate(configs):
    ax = axes[ci]
    k_max = k_total - 1

    for model in models:
        pkl_path = f'results/allpairs_contrastive/{model}/{pkl_name}'
        if not os.path.exists(pkl_path):
            print(f"Skipping {model} / {pkl_name} — not found")
            continue

        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)

        angles_deg = d['angles_deg']

        n_values = []
        mean_plateaus = []
        std_plateaus = []

        for n in range(1, k_max + 1):
            sizes = []
            for r in d['all_results'][n]:
                mean_kl = r['kl'].mean(axis=0)
                ps = measure_threshold_crossing(mean_kl, angles_deg, threshold)
                sizes.append(ps)
            n_values.append(n)
            mean_plateaus.append(np.mean(sizes))
            std_plateaus.append(np.std(sizes))

        n_values = np.array(n_values)
        mean_plateaus = np.array(mean_plateaus)
        std_plateaus = np.array(std_plateaus)

        color = model_colors[model]
        label = model_labels[model]

        ax.errorbar(n_values, mean_plateaus, yerr=std_plateaus,
                     fmt='o-', color=color, markersize=5, linewidth=2,
                     capsize=3, label=label, zorder=2)

        # Random baseline as horizontal dashed line
        ax.axhline(y=random_plateaus[model], color=color, linestyle='--',
                    linewidth=1, alpha=0.5)

        print(f"{model} ({title_suffix}): n=1: {mean_plateaus[0]:.1f}°, n={k_max}: {mean_plateaus[-1]:.1f}°, random: {random_plateaus[model]:.1f}°")

    # Dummy entry for legend
    ax.plot([], [], '--', color='gray', linewidth=1, alpha=0.5, label='random baseline')

    ax.set_xlabel('Number of features (n)')
    ax.set_ylabel('Plateau size (degrees)')
    ax.set_title(f'{title_suffix} (n=1..{k_max})')
    ax.set_xticks(range(1, k_max + 1))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle(f'Plateau size vs n — KL threshold = {threshold}', fontsize=13)
fig.tight_layout()
out = 'results/allpairs_contrastive/plateau_scaling_allmodels_kl01.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out}")
plt.close(fig)
print("Done!")
