#!/usr/bin/env python3
"""
Plot plateau size vs n for programming language features (1.7B and 4B).
Includes random baseline as horizontal dashed lines.
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


models = ['qwen3-1.7b', 'qwen3-4b']
model_colors = {'qwen3-1.7b': '#377eb8', 'qwen3-4b': '#4daf4a'}
model_labels = {'qwen3-1.7b': 'Qwen3-1.7B', 'qwen3-4b': 'Qwen3-4B'}
results_base = 'results/programming_contrastive'

for threshold in [0.1, 0.5]:
    fig, ax = plt.subplots(figsize=(7, 5))

    for model in models:
        # Load plateau scaling
        pkl_path = f'{results_base}/{model}/plateau_scaling.pkl'
        if not os.path.exists(pkl_path):
            print(f"Skipping {model} — {pkl_path} not found")
            continue

        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)

        # Load sweep results for random baseline
        sweep_path = f'{results_base}/{model}/sweep_results.pkl'
        with open(sweep_path, 'rb') as f:
            s = pickle.load(f)

        angles_deg = d['angles_deg']
        k_max = max(d['all_results'].keys())

        # Random baseline
        rand_kl = s['sweep_results']['random']['kl']
        rand_kl_mean = rand_kl.mean(axis=(0, 2))
        rand_plateau = measure_threshold_crossing(rand_kl_mean, np.array(s['angles_deg']), threshold)

        # Recompute plateau sizes at this threshold
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

        # Exclude last point (only 1 combo)
        plot_n = n_values[:-1]
        plot_mean = mean_plateaus[:-1]
        plot_std = std_plateaus[:-1]

        ax.errorbar(plot_n, plot_mean, yerr=plot_std,
                     fmt='o-', color=color, markersize=5, linewidth=2,
                     capsize=3, label=label, zorder=2)

        # Random baseline
        ax.axhline(y=rand_plateau, color=color, linestyle='--',
                    linewidth=1, alpha=0.5)

        print(f"KL={threshold} | {model}: n=1: {mean_plateaus[0]:.1f}°±{std_plateaus[0]:.1f}°, "
              f"n={plot_n[-1]}: {plot_mean[-1]:.1f}°±{plot_std[-1]:.1f}°, random: {rand_plateau:.1f}°")

    ax.plot([], [], '--', color='gray', linewidth=1, alpha=0.5, label='random baseline')
    ax.set_xlabel('Number of features (n)')
    ax.set_ylabel('Plateau size (degrees)')
    ax.set_title(f'Programming languages — KL threshold = {threshold}')
    ax.set_xticks(range(1, k_max))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    thr_str = str(threshold).replace('.', '')
    out = f'{results_base}/plateau_scaling_proglang_kl{thr_str}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)

print("Done!")
