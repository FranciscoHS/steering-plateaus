#!/usr/bin/env python3
"""
Improved rotation plot with contrastive colors and MELBO_1 mapped on.

Loads saved results from run_direction_decomposition_v2.py.
Runs locally (no GPU needed).
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load results
results_dir = 'results/direction_decomposition_v2/qwen3-0.6b'
with open(f'{results_dir}/results.pkl', 'rb') as f:
    res = pickle.load(f)

angles_deg = res['angles_deg']
rotation_phis_deg = res['rotation_phis_deg']
pc1_phi_deg = res['pc1_phi_deg']
melbo1_phi_deg = res['melbo1_phi_deg']
sweep_results = res['sweep_results']
cosines = res['signed_cosines']

max_angle = 60
mask = angles_deg <= max_angle
x = angles_deg[mask]


def get_stats(name, key):
    data = sweep_results[name][key]
    if data.ndim == 3:  # random: [N, angles, dirs]
        per_act = np.median(data, axis=2)
    else:
        per_act = data
    return (np.median(per_act, axis=0),
            np.percentile(per_act, 25, axis=0),
            np.percentile(per_act, 75, axis=0))


# --- Contrastive color scheme for rotation angles ---
# Use a discrete set of well-separated colors
# 0=mean, 30, 60, 90=orth, 120, 150, ~168=PC1, 180=-mean
rotation_colors = {
    0.0:    '#2ca02c',   # green (mean)
    30.0:   '#bcbd22',   # olive
    60.0:   '#ff7f0e',   # orange
    90.0:   '#9467bd',   # purple (orth)
    120.0:  '#e377c2',   # pink
    150.0:  '#17becf',   # cyan
    round(pc1_phi_deg, 1): '#1f77b4',  # blue (PC1)
    180.0:  '#d62728',   # red (-mean)
}


# --- Plot: Rotation + MELBO_1 ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for metric_idx, (key, ylabel) in enumerate([('kl', 'KL divergence'), ('l2', 'L2 distance')]):
    ax = axes[metric_idx]

    # Random baseline
    rm, rq25, rq75 = get_stats('random', key)
    ax.fill_between(x, rq25[mask], rq75[mask], color='gray', alpha=0.15)
    ax.plot(x, rm[mask], color='gray', lw=2, ls='--', label='Random', zorder=1)

    # Rotation directions
    for phi_d in rotation_phis_deg:
        name = f'rot_{phi_d}'
        med, q25, q75 = get_stats(name, key)
        color = rotation_colors.get(phi_d, 'black')
        is_pc1 = abs(phi_d - pc1_phi_deg) < 1.0

        label = f'{phi_d:.0f}°'
        if phi_d == 0:
            label += ' (mean)'
        elif is_pc1:
            label += ' (≈PC1)'
        elif phi_d == 90:
            label += ' (orth)'
        elif phi_d == 180:
            label += ' (−mean)'

        lw = 2.5 if is_pc1 or phi_d == 180 else 1.5
        zorder = 3 if is_pc1 else 2
        ax.fill_between(x, q25[mask], q75[mask], color=color, alpha=0.08)
        ax.plot(x, med[mask], color=color, lw=lw, label=label, zorder=zorder)

    # MELBO_1 overlay
    med, q25, q75 = get_stats('MELBO_1', key)
    ax.fill_between(x, q25[mask], q75[mask], color='#d62728', alpha=0.08)
    ax.plot(x, med[mask], color='#d62728', lw=2.5, ls='--',
            label=f'MELBO_1 ({melbo1_phi_deg:.0f}°, 92% in-plane)', zorder=4)

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_angle)

fig.suptitle(
    f'Rotation from mean→−mean in {{mean, orth}} plane + MELBO_1\n'
    f'Qwen3-0.6B, layer 14 | PC1 at {pc1_phi_deg:.0f}°, '
    f'MELBO_1 at {melbo1_phi_deg:.0f}° (both ≈ −mean)',
    fontsize=13)
plt.tight_layout()
plt.savefig(f'{results_dir}/rotation_sweep_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {results_dir}/rotation_sweep_v2.png")


# --- Also redo the decomposition plot with same improved style ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

decomp_styles = {
    'mean':               ('#2ca02c',  '-',  2.0,  'mean (diff-of-means)'),
    'PC1':                ('#1f77b4',  '-',  2.0,  'PC1'),
    'MELBO_1':            ('#d62728',  '-',  2.0,  'MELBO_1'),
    'melbo1_orth_to_mean':('#d62728',  ':',  2.0,  'MELBO_1 ⊥ mean'),
    'pc1_orth_to_mean':   ('#1f77b4',  ':',  2.0,  'PC1 ⊥ mean'),
}

for metric_idx, (key, ylabel) in enumerate([('kl', 'KL divergence'), ('l2', 'L2 distance')]):
    ax = axes[metric_idx]

    rm, rq25, rq75 = get_stats('random', key)
    ax.fill_between(x, rq25[mask], rq75[mask], color='gray', alpha=0.15)
    ax.plot(x, rm[mask], color='gray', lw=2, ls='--', label='Random')

    for name, (color, ls, lw, label) in decomp_styles.items():
        med, q25, q75 = get_stats(name, key)
        ax.fill_between(x, q25[mask], q75[mask], color=color, alpha=0.08)
        ax.plot(x, med[mask], color=color, ls=ls, lw=lw, label=label)

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_angle)

fig.suptitle(
    f'Direction decompositions | Qwen3-0.6B, layer 14\n'
    f'cos(PC1,mean)={cosines["pc1_mean"]:.3f}, '
    f'cos(MELBO_1,mean)={cosines["melbo1_mean"]:.3f}, '
    f'cos(MELBO_1,PC1)={cosines["melbo1_pc1"]:.3f}',
    fontsize=13)
plt.tight_layout()
plt.savefig(f'{results_dir}/decomposition_sweep_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {results_dir}/decomposition_sweep_v2.png")

print("\nDone!")
