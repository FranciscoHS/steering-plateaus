#!/usr/bin/env python3
"""
Combined plot v2: KL sweep (top) + CJK fraction (bottom) with coherence shading.
Extended to 90° with coherence regions annotated.

Coherence assessment (from manual inspection of 30 prompts x 50 tokens):
  MELBO_1:
    0°: coherent English
    5°: mostly coherent English, rare CJK
    10°: coherent mixed EN/ZH — proper Chinese sentences
    25°: degraded mixed EN/ZH, code-switching, less grammatical
    45-90°: token-fragment gibberish (underscores, code tokens: PASS, MONITOR, _each)

  -mean:
    0°: coherent English
    5°: mostly coherent English
    10°: coherent Chinese (similar quality to MELBO_1@10°)
    25°: coherent Chinese, some repetition
    45-90°: token-fragment gibberish (same type as MELBO_1: pipes, underscores, code tokens)

  MELBO_1⊥mean:
    0°: coherent English
    5°: coherent English
    10°: coherent English (no CJK)
    25°: coherent Chinese — exam questions, proper grammar
    45-50°: incoherent CJK word-salad — "种植种植种植", agriculture/nature theme
    60°: still mostly CJK, English words bleeding in
    70°: bilingual mash-up, "following" obsession begins
    80-90°: repetitive English word loops ("following following following", "process process")
            — qualitatively different from MELBO_1/-mean gibberish (words, not token fragments)
"""

import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Load KL sweep data
with open('results/orth_comparison/qwen3-0.6b/results.pkl', 'rb') as f:
    kl_data = pickle.load(f)

# Load CJK fraction data (original + extended)
with open('results/steering_mandarin_fraction/qwen3-0.6b/results.json', 'r') as f:
    cjk_orig = json.load(f)
with open('results/steering_mandarin_fraction/qwen3-0.6b/results_extended.json', 'r') as f:
    cjk_ext = json.load(f)
cjk_all = cjk_orig + cjk_ext

kl_angles = kl_data['angles_deg']
kl_results = kl_data['sweep_results']

max_angle = 90
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


# CJK stats
cjk_angles = sorted(set(r['angle'] for r in cjk_all))
dir_names = ['MELBO_1', '-mean', 'MELBO_1_orth_mean']

cjk_stats = {}
for dn in dir_names:
    cjk_stats[dn] = {}
    for angle in cjk_angles:
        fracs = [r['cjk_fraction'] for r in cjk_all
                 if r['direction'] == dn and r['angle'] == angle]
        if fracs:
            cjk_stats[dn][angle] = {
                'median': np.median(fracs),
                'q25': np.percentile(fracs, 25),
                'q75': np.percentile(fracs, 75),
            }

styles = {
    'MELBO_1':          {'color': '#d62728', 'label': 'MELBO₁'},
    '-mean':            {'color': '#2ca02c', 'label': '−mean'},
    'MELBO_1_orth_mean': {'color': '#1f77b4', 'label': 'MELBO₁ ⊥ mean'},
}

fig, (ax_kl, ax_cjk) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                                      gridspec_kw={'height_ratios': [1, 1]})

# === Top panel: KL divergence ===
rm, rq25, rq75 = kl_stats('random')
ax_kl.fill_between(kl_x, rq25[kl_mask], rq75[kl_mask], color='gray', alpha=0.12)
ax_kl.plot(kl_x, rm[kl_mask], color='gray', lw=1.5, ls='--', label='Random', zorder=1)

for dn, sty in styles.items():
    kl_name = dn if dn != 'MELBO_1_orth_mean' else 'melbo1_orth_to_mean'
    med, q25, q75 = kl_stats(kl_name)
    ax_kl.fill_between(kl_x, q25[kl_mask], q75[kl_mask], color=sty['color'], alpha=0.1)
    ax_kl.plot(kl_x, med[kl_mask], color=sty['color'], lw=2.5, label=sty['label'], zorder=3)

ax_kl.set_ylabel('KL divergence', fontsize=12)
ax_kl.legend(fontsize=10, loc='upper right')
ax_kl.grid(True, alpha=0.3)
ax_kl.set_xlim(0, max_angle)

# === Bottom panel: CJK fraction with coherence shading ===

# Background shading first (lowest layer)
ax_cjk.axvspan(8, 20, color='#d62728', alpha=0.07, zorder=0)
ax_cjk.axvspan(8, 30, color='#2ca02c', alpha=0.07, zorder=0)
ax_cjk.axvspan(20, 35, color='#1f77b4', alpha=0.07, zorder=0)
ax_cjk.axvspan(35, 90, color='gray', alpha=0.04, zorder=0)
ax_cjk.axvspan(40, 63, color='#1f77b4', alpha=0.04, zorder=0)

# Data lines
offsets = {'MELBO_1': -1.0, '-mean': 0, 'MELBO_1_orth_mean': 1.0}

for dn, sty in styles.items():
    angles_with_data = sorted(cjk_stats[dn].keys())
    medians = [cjk_stats[dn][a]['median'] for a in angles_with_data]
    q25s = [cjk_stats[dn][a]['q25'] for a in angles_with_data]
    q75s = [cjk_stats[dn][a]['q75'] for a in angles_with_data]

    x_pos = np.array(angles_with_data) + offsets[dn]
    lower_err = [m - q for m, q in zip(medians, q25s)]
    upper_err = [q - m for m, q in zip(medians, q75s)]

    ax_cjk.errorbar(x_pos, medians, yerr=[lower_err, upper_err],
                     fmt='o-', color=sty['color'], lw=2, markersize=7,
                     capsize=3, capthick=1.5, label=sty['label'], zorder=2)

# Annotations LAST so they draw on top of everything
ax_cjk.annotate('coherent\nCJK', xy=(14, 0.92), fontsize=10, color='#d62728',
                ha='center', va='top', style='italic')
ax_cjk.annotate('coherent\nCJK', xy=(19, 0.77), fontsize=10, color='#2ca02c',
                ha='center', va='top', style='italic')
ax_cjk.annotate('coherent\nCJK', xy=(27.5, 1.02), fontsize=10, color='#1f77b4',
                ha='center', va='top', style='italic')
ax_cjk.annotate('token-fragment\ngibberish', xy=(62, 0.18), fontsize=10, color='#888888',
                ha='center', va='bottom', style='italic')
ax_cjk.annotate('incoherent\nCJK', xy=(51, 0.72), fontsize=10, color='#1f77b4',
                ha='center', va='top', style='italic')
ax_cjk.annotate('bilingual\nmash-up', xy=(70, 0.55), fontsize=10, color='#1f77b4',
                ha='center', va='bottom', style='italic')
ax_cjk.annotate('repetitive\nEnglish', xy=(85, 0.12), fontsize=10, color='#1f77b4',
                ha='center', va='bottom', style='italic')

ax_cjk.set_xlabel('Perturbation angle (degrees)', fontsize=12)
ax_cjk.set_ylabel('CJK character fraction', fontsize=12)
ax_cjk.set_ylim(-0.05, 1.05)
ax_cjk.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax_cjk.legend(fontsize=10, loc='upper right')
ax_cjk.grid(True, alpha=0.3)

# Subtle vertical alignment lines
for angle in cjk_angles:
    ax_kl.axvline(angle, color='gray', alpha=0.1, lw=0.5, zorder=0)
    ax_cjk.axvline(angle, color='gray', alpha=0.1, lw=0.5, zorder=0)

fig.suptitle('KL perturbation response vs Mandarin steering effect\n'
             'Qwen3-0.6B, layer 14 | 30 prompts, 50 tokens, temp=0.7 | '
             'shaded = coherent CJK region',
             fontsize=12)
plt.tight_layout()

out_path = 'results/steering_mandarin_fraction/qwen3-0.6b/kl_vs_cjk_v2.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
