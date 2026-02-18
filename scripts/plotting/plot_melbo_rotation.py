#!/usr/bin/env python3
"""Improved plot for MELBO rotation results.
Only 90° (⊥ MELBO_1) to 180° (MELBO_1), plus −mean as reference."""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_dir = 'results/melbo_rotation/qwen3-0.6b'
with open(f'{results_dir}/results.pkl', 'rb') as f:
    res = pickle.load(f)

# Load -mean sweep from the previous experiment (rot_180 in {mean, orth} plane = -mean)
with open('results/direction_decomposition_v2/qwen3-0.6b/results.pkl', 'rb') as f:
    res_v2 = pickle.load(f)

angles_deg = res['angles_deg']
rotation_phis_deg = res['rotation_phis_deg']
sweep_results = res['sweep_results']

# Verify angles match between experiments
assert np.allclose(angles_deg, res_v2['angles_deg']), "Angle grids don't match!"

# Add -mean from the other experiment
sweep_results['neg_mean'] = res_v2['sweep_results']['rot_180']

max_angle = 60
mask = angles_deg <= max_angle
x = angles_deg[mask]


def get_stats(name, key):
    data = sweep_results[name][key]
    if data.ndim == 3:
        per_act = np.median(data, axis=2)
    else:
        per_act = data
    return (np.median(per_act, axis=0),
            np.percentile(per_act, 25, axis=0),
            np.percentile(per_act, 75, axis=0))


# Only rotation angles from 90° to 180°
rot_subset = [p for p in rotation_phis_deg if p >= 90]

rot_colors = {
    90:  '#888888',   # gray (orth)
    105: '#bcbd22',   # olive
    120: '#ff7f0e',   # orange
    135: '#e377c2',   # pink
    150: '#9467bd',   # purple
    165: '#1f77b4',   # blue
    180: '#d62728',   # red (MELBO_1)
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for metric_idx, (key, ylabel) in enumerate([('kl', 'KL divergence'), ('l2', 'L2 distance')]):
    ax = axes[metric_idx]

    # Random baseline
    rm, rq25, rq75 = get_stats('random', key)
    ax.fill_between(x, rq25[mask], rq75[mask], color='gray', alpha=0.12)
    ax.plot(x, rm[mask], color='gray', lw=2, ls='--', label='Random', zorder=1)

    # -mean (reference)
    med, q25, q75 = get_stats('neg_mean', key)
    ax.fill_between(x, q25[mask], q75[mask], color='#2ca02c', alpha=0.08)
    ax.plot(x, med[mask], color='#2ca02c', lw=2.5, label='−mean', zorder=2)

    # Rotation directions 90°→180°
    for phi_d in rot_subset:
        name = f'rot_{phi_d}'
        med, q25, q75 = get_stats(name, key)
        color = rot_colors.get(phi_d, 'black')

        label = f'{phi_d:.0f}°'
        if phi_d == 90:
            label += ' (⊥ MELBO₁)'
        elif phi_d == 180:
            label += ' (= MELBO₁)'

        lw = 2.5 if phi_d == 180 else 1.8
        zorder = 4 if phi_d == 180 else 3
        ax.fill_between(x, q25[mask], q75[mask], color=color, alpha=0.08)
        ax.plot(x, med[mask], color=color, lw=lw, label=label, zorder=zorder)

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_angle)

fig.suptitle(
    'Rotation from ⊥ MELBO₁ (90°) → MELBO₁ (180°) in {MELBO₁, mean} plane\n'
    'Qwen3-0.6B, layer 14 | −mean shown for comparison',
    fontsize=13)
plt.tight_layout()
plt.savefig(f'{results_dir}/melbo_rotation_v3.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {results_dir}/melbo_rotation_v3.png")
