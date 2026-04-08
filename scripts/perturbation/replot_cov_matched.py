#!/usr/bin/env python3
"""Prettier replot of gpt2-xl cov-matched perturbation sweep (L2 only)."""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_dir = 'results/perturbation_sweep_spherical/gpt2-xl_cov_matched'
with open(f'{results_dir}/results.pkl', 'rb') as f:
    r = pickle.load(f)

angles_deg = r['angles_deg']
# Restrict to 0-40 degrees
mask = angles_deg <= 60
angles = angles_deg[mask]

blue = '#2B4E72'
orange = '#E8820C'

conditions = [
    ('real', 'Real (in-distribution)', blue),
    ('random', 'Synthetic (randomly sampled)', orange),
]

np.random.seed(42)
n_individual = 10

metrics = [
    ('l2', 'L2 distance'),
    ('kl', 'KL divergence'),
]

for metric_key, metric_label in metrics:
    for plot_type in ['median', 'individual']:
        fig, ax = plt.subplots(figsize=(7, 5))

        for cond_key, label, color in conditions:
            d = r[cond_key][metric_key][:, mask, :]
            per_act = np.median(d, axis=2)

            if plot_type == 'median':
                median = np.median(per_act, axis=0)
                q25 = np.percentile(per_act, 25, axis=0)
                q75 = np.percentile(per_act, 75, axis=0)
                ax.plot(angles, median, color=color, label=label, linewidth=2)
                ax.fill_between(angles, q25, q75, color=color, alpha=0.15)
            else:
                chosen = np.random.choice(per_act.shape[0], n_individual, replace=False)
                for j, idx in enumerate(chosen):
                    ax.plot(angles, per_act[idx], color=color, alpha=0.4, linewidth=1.5,
                            label=label if j == 0 else None)

        ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.legend(fontsize=11)
        ax.set_xlim(0, 60)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        path = f'{results_dir}/{metric_key}_sweep_{plot_type}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', transparent=True)
        plt.close()
        print(f'Saved: {path}')
