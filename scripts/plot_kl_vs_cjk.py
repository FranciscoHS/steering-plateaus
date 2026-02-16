#!/usr/bin/env python3
"""
Combined plot: KL perturbation sweep (top) + CJK fraction from steering (bottom).
Shared x-axis, same colors per direction.
"""

import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load KL sweep data (from orth_comparison experiment)
with open('results/orth_comparison/qwen3-0.6b/results.pkl', 'rb') as f:
    kl_data = pickle.load(f)

# Load CJK fraction data
with open('results/steering_mandarin_fraction/qwen3-0.6b/results.json', 'r') as f:
    cjk_raw = json.load(f)

kl_angles = kl_data['angles_deg']
kl_results = kl_data['sweep_results']

max_angle = 60
kl_mask = kl_angles <= max_angle
kl_x = kl_angles[kl_mask]


def kl_stats(name):
    data = kl_results[name]['kl']
    if data.ndim == 3:
        per_act = np.median(data, axis=2)
    else:
        per_act = data
    return (np.median(per_act, axis=0),
            np.percentile(per_act, 25, axis=0),
            np.percentile(per_act, 75, axis=0))


# Compute CJK stats per (direction, angle)
cjk_angles = [0, 5, 10, 25, 45]
dir_name_map = {
    'MELBO_1': 'MELBO_1',
    '-mean': '-mean',
    'MELBO_1_orth_mean': 'MELBO_1_orth_mean',
}

cjk_stats = {}
for dir_name in dir_name_map.values():
    cjk_stats[dir_name] = {}
    for angle in cjk_angles:
        fracs = [r['cjk_fraction'] for r in cjk_raw
                 if r['direction'] == dir_name and r['angle'] == angle]
        cjk_stats[dir_name][angle] = {
            'mean': np.mean(fracs),
            'median': np.median(fracs),
            'q25': np.percentile(fracs, 25),
            'q75': np.percentile(fracs, 75),
            'std': np.std(fracs),
        }

# Consistent styling
styles = {
    'MELBO_1':          {'color': '#d62728', 'label': 'MELBO₁'},
    '-mean':            {'color': '#2ca02c', 'label': '−mean'},
    'MELBO_1_orth_mean': {'color': '#1f77b4', 'label': 'MELBO₁ ⊥ mean'},
}

fig, (ax_kl, ax_cjk) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                      gridspec_kw={'height_ratios': [1, 1]})

# === Top panel: KL divergence sweep ===

# Random baseline
rm, rq25, rq75 = kl_stats('random')
ax_kl.fill_between(kl_x, rq25[kl_mask], rq75[kl_mask], color='gray', alpha=0.12)
ax_kl.plot(kl_x, rm[kl_mask], color='gray', lw=1.5, ls='--', label='Random', zorder=1)

for dir_name, sty in styles.items():
    kl_name = dir_name if dir_name != 'MELBO_1_orth_mean' else 'melbo1_orth_to_mean'
    med, q25, q75 = kl_stats(kl_name)
    ax_kl.fill_between(kl_x, q25[kl_mask], q75[kl_mask], color=sty['color'], alpha=0.1)
    ax_kl.plot(kl_x, med[kl_mask], color=sty['color'], lw=2.5, label=sty['label'], zorder=3)

ax_kl.set_ylabel('KL divergence', fontsize=12)
ax_kl.legend(fontsize=10, loc='upper left')
ax_kl.grid(True, alpha=0.3)
ax_kl.set_xlim(0, max_angle)

# === Bottom panel: CJK fraction ===

offsets = {'MELBO_1': -0.8, '-mean': 0, 'MELBO_1_orth_mean': 0.8}

for dir_name, sty in styles.items():
    angles_arr = np.array(cjk_angles)
    medians = [cjk_stats[dir_name][a]['median'] for a in cjk_angles]
    q25s = [cjk_stats[dir_name][a]['q25'] for a in cjk_angles]
    q75s = [cjk_stats[dir_name][a]['q75'] for a in cjk_angles]

    offset = offsets[dir_name]
    x_pos = angles_arr + offset

    # Plot median as points, IQR as error bars
    lower_err = [m - q for m, q in zip(medians, q25s)]
    upper_err = [q - m for m, q in zip(medians, q75s)]

    ax_cjk.errorbar(x_pos, medians, yerr=[lower_err, upper_err],
                     fmt='o-', color=sty['color'], lw=2, markersize=8,
                     capsize=4, capthick=1.5, label=sty['label'], zorder=3)

ax_cjk.set_xlabel('Perturbation angle (degrees)', fontsize=12)
ax_cjk.set_ylabel('CJK character fraction', fontsize=12)
ax_cjk.set_ylim(-0.05, 1.05)
ax_cjk.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax_cjk.legend(fontsize=10, loc='upper left')
ax_cjk.grid(True, alpha=0.3)

# Mark the tested angles with subtle vertical lines spanning both panels
for angle in cjk_angles:
    ax_kl.axvline(angle, color='gray', alpha=0.15, lw=0.8, zorder=0)
    ax_cjk.axvline(angle, color='gray', alpha=0.15, lw=0.8, zorder=0)

fig.suptitle('KL perturbation response vs Mandarin steering effect\n'
             'Qwen3-0.6B, layer 14 | 30 prompts, 50 tokens, temp=0.7',
             fontsize=13)
plt.tight_layout()

out_path = 'results/steering_mandarin_fraction/qwen3-0.6b/kl_vs_cjk.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
