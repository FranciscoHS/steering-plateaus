#!/usr/bin/env python3
"""
Coherence analysis plot for Russian→English steering.
Compares d_ER (EN-RU diff-of-means), d_EM (EN-ZH diff-of-means),
and k=10 SVD when steering from Russian prompts.

Top row: perplexity vs angle (one panel per direction)
Bottom row: Cyrillic fraction + English fraction vs angle

Usage:
    python scripts/plot_russian_coherence.py
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_dir = 'results/scaling_test/qwen3-0.6b'

with open(f'{results_dir}/steering_russian.pkl', 'rb') as f:
    data = pickle.load(f)

results = data['ru_results']
angles = data['angles']
dir_names = data['directions']  # ['d_ER', 'd_EM', 'k=10 SVD']

styles = {
    'd_ER':      {'color': '#9467bd', 'label': '$d_{ER}$ (EN−RU)'},
    'd_EM':      {'color': '#d62728', 'label': '$d_{EM}$ (EN−ZH)'},
    'k=10 SVD':  {'color': '#1f77b4', 'label': 'k=10 SVD'},
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

for col, dir_name in enumerate(dir_names):
    ax_ppl = axes[0, col]
    ax_lang = axes[1, col]
    sty = styles[dir_name]

    ppl_medians, ppl_q25, ppl_q75 = [], [], []
    cyr_means, cyr_stds = [], []
    eng_means, eng_stds = [], []
    cjk_means = []

    for angle in angles:
        subset = [r for r in results
                  if r['direction'] == dir_name and r['angle'] == angle]
        ppls = [r['perplexity'] for r in subset if not np.isnan(r['perplexity'])]
        cyrs = [r['cyrillic_fraction'] for r in subset]
        engs = [r['english_fraction'] for r in subset]
        cjks = [r['cjk_fraction'] for r in subset]

        ppl_medians.append(np.median(ppls))
        ppl_q25.append(np.percentile(ppls, 25))
        ppl_q75.append(np.percentile(ppls, 75))
        cyr_means.append(np.mean(cyrs))
        cyr_stds.append(np.std(cyrs))
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
    ax_ppl.set_ylim(0, 50)
    ax_ppl.grid(True, alpha=0.3)

    # Coherence threshold
    ax_ppl.axhspan(0, 15, color='green', alpha=0.04, zorder=0)
    ax_ppl.axhline(15, color='green', alpha=0.3, ls='--', lw=0.8)
    if col == 0:
        ax_ppl.annotate('coherent', xy=(2, 13), fontsize=9, color='green',
                        style='italic', va='top')

    # Bottom: language fractions
    ax_lang.errorbar(angles_arr, eng_means, yerr=eng_stds,
                     fmt='s-', color='#2ca02c', lw=2, markersize=5,
                     capsize=3, label='English', zorder=3)
    ax_lang.errorbar(angles_arr, cyr_means, yerr=cyr_stds,
                     fmt='^-', color='#9467bd', lw=2, markersize=5,
                     capsize=3, label='Cyrillic', zorder=3)
    ax_lang.plot(angles_arr, cjk_means, 'v-',
                 color='#d62728', lw=1.5, markersize=4,
                 label='CJK', zorder=2, alpha=0.7)

    ax_lang.set_xlabel('Steering angle (degrees)', fontsize=11)
    ax_lang.set_ylabel('Character fraction' if col == 0 else '', fontsize=11)
    ax_lang.set_ylim(-0.05, 1.05)
    ax_lang.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax_lang.grid(True, alpha=0.3)
    if col == 2:
        ax_lang.legend(fontsize=9, loc='center right')

fig.suptitle('Russian → English steering coherence analysis\n'
             'Qwen3-0.6B, layer 14 | 36 prompts, 50 tokens, temp=0.7',
             fontsize=13)
plt.tight_layout()

out_path = f'{results_dir}/russian_coherence.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
