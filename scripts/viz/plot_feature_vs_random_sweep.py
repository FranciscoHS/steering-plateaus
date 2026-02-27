"""Replot feature vs random sweep from saved pkl — median lines only, no shading."""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pkl_path = 'results/feature_vs_random_sweep/qwen3-4b/feature_vs_random_sweep.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

angles_deg = data['angles_deg']
cos_rc = data['cos_ru_c']
results_kl = data['kl']
results_l2 = data['l2']
n_act = data['args']['n_act']
model = data['args']['model']
layer = data['args']['layer']

colors = {'C': '#4a86c8', 'RU': '#c84a5a', 'RU+C': '#7b4ac8'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (metric_data, metric_name) in zip(axes, [
    (results_kl, 'KL divergence (nats)'),
    (results_l2, 'L2 distance (final layer)'),
]):
    # Random
    rand = metric_data['random']
    rand_per_act = np.median(rand, axis=1)
    rand_median = np.median(rand_per_act, axis=0)
    ax.plot(angles_deg, rand_median, color='gray', linewidth=2.5, label='Random')

    # Named directions
    for name in ['C', 'RU', 'RU+C']:
        d = metric_data[name]
        median = np.median(d, axis=0)
        ax.plot(angles_deg, median, color=colors[name], linewidth=2.5, label=name)

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, angles_deg[-1])

fig.suptitle(f'{model}, layer {layer}: Feature vs Random Perturbation\n'
             f'n={n_act} activations, cos(RU,C)={cos_rc:.3f}',
             fontsize=13, y=1.02)
plt.tight_layout()
fig_path = 'results/feature_vs_random_sweep/qwen3-4b/feature_vs_random_sweep_clean.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path}")
