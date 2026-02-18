#!/usr/bin/env python3
"""
Coherence analysis plot for Arabic→English steering (OOD test).
Arabic was NOT in the k=10 language set.

Compares d_EM (EN-ZH pairwise) vs k=10 SVD — neither has seen Arabic.

Usage:
    python scripts/plot_arabic_coherence.py
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_dir = 'results/scaling_test/qwen3-0.6b'

with open(f'{results_dir}/steering_arabic.pkl', 'rb') as f:
    data = pickle.load(f)

results = data['ar_results']
angles = data['angles']
dir_names = data['directions']  # ['d_EM', 'k=10 SVD']

styles = {
    'd_EM':      {'color': '#d62728', 'label': '$d_{EM}$ (EN−ZH)'},
    'k=10 SVD':  {'color': '#1f77b4', 'label': 'k=10 SVD'},
}

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

for col, dir_name in enumerate(dir_names):
    ax_ppl = axes[0, col]
    ax_lang = axes[1, col]
    sty = styles[dir_name]

    ppl_medians, ppl_q25, ppl_q75 = [], [], []
    ar_means, ar_stds = [], []
    eng_means, eng_stds = [], []
    cjk_means = []

    for angle in angles:
        subset = [r for r in results
                  if r['direction'] == dir_name and r['angle'] == angle]
        ppls = [r['perplexity'] for r in subset if not np.isnan(r['perplexity'])]
        ars = [r['arabic_fraction'] for r in subset]
        engs = [r['english_fraction'] for r in subset]
        cjks = [r['cjk_fraction'] for r in subset]

        ppl_medians.append(np.median(ppls))
        ppl_q25.append(np.percentile(ppls, 25))
        ppl_q75.append(np.percentile(ppls, 75))
        ar_means.append(np.mean(ars))
        ar_stds.append(np.std(ars))
        eng_means.append(np.mean(engs))
        eng_stds.append(np.std(engs))
        cjk_means.append(np.mean(cjks))

    angles_arr = np.array(angles)

    # Top: perplexity
    ax_ppl.fill_between(angles_arr, ppl_q25, ppl_q75,
                        color=sty['color'], alpha=0.15)
    ax_ppl.plot(angles_arr, ppl_medians, 'o-',
                color=sty['color'], lw=2, markersize=5)
    ax_ppl.set_title(sty['label'], fontsize=12)
    ax_ppl.set_ylabel('Perplexity (median)' if col == 0 else '', fontsize=11)
    ax_ppl.set_ylim(0, 160)
    ax_ppl.grid(True, alpha=0.3)

    # Coherence threshold
    ax_ppl.axhspan(0, 20, color='green', alpha=0.04, zorder=0)
    ax_ppl.axhline(20, color='green', alpha=0.3, ls='--', lw=0.8)
    if col == 0:
        ax_ppl.annotate('coherent', xy=(2, 17), fontsize=9, color='green',
                        style='italic', va='top')

    # Bottom: language fractions
    ax_lang.errorbar(angles_arr, eng_means, yerr=eng_stds,
                     fmt='s-', color='#2ca02c', lw=2, markersize=5,
                     capsize=3, label='English', zorder=3)
    ax_lang.errorbar(angles_arr, ar_means, yerr=ar_stds,
                     fmt='^-', color='#e377c2', lw=2, markersize=5,
                     capsize=3, label='Arabic', zorder=3)
    ax_lang.plot(angles_arr, cjk_means, 'v-',
                 color='#d62728', lw=1.5, markersize=4,
                 label='CJK', zorder=2, alpha=0.7)

    ax_lang.set_xlabel('Steering angle (degrees)', fontsize=11)
    ax_lang.set_ylabel('Character fraction' if col == 0 else '', fontsize=11)
    ax_lang.set_ylim(-0.05, 1.05)
    ax_lang.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax_lang.grid(True, alpha=0.3)
    if col == 1:
        ax_lang.legend(fontsize=9, loc='center right')

fig.suptitle('Arabic → English steering (out-of-distribution test)\n'
             'Qwen3-0.6B, layer 14 | 36 prompts, 50 tokens, temp=0.7\n'
             'Arabic was NOT in the k=10 language set',
             fontsize=13)
plt.tight_layout()

out_path = f'{results_dir}/arabic_coherence.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
