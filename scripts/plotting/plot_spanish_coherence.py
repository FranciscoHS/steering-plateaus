#!/usr/bin/env python3
"""
Coherence analysis plot for Spanish→English steering.
Compares d_ES (EN-ES diff-of-means), d_EM (EN-ZH diff-of-means),
and k=10 SVD when steering from Spanish prompts.

Top row: perplexity vs angle (one panel per direction)
Bottom row: English fraction + Spanish fraction vs angle

Usage:
    python scripts/plot_spanish_coherence.py
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_dir = 'results/scaling_test/qwen3-0.6b'

with open(f'{results_dir}/steering_spanish.pkl', 'rb') as f:
    data = pickle.load(f)

results = data['es_results']
angles = data['angles']
dir_names = data['directions']  # ['d_ES', 'd_EM', 'k=10 SVD']

styles = {
    'd_ES':      {'color': '#ff7f0e', 'label': '$d_{ES}$ (EN−ES)'},
    'd_EM':      {'color': '#d62728', 'label': '$d_{EM}$ (EN−ZH)'},
    'k=10 SVD':  {'color': '#1f77b4', 'label': 'k=10 SVD'},
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

for col, dir_name in enumerate(dir_names):
    ax_ppl = axes[0, col]
    ax_lang = axes[1, col]
    sty = styles[dir_name]

    # Collect stats per angle
    ppl_medians, ppl_q25, ppl_q75 = [], [], []
    eng_means, eng_stds = [], []
    spa_means, spa_stds = [], []
    cjk_means = []

    for angle in angles:
        subset = [r for r in results
                  if r['direction'] == dir_name and r['angle'] == angle]
        ppls = [r['perplexity'] for r in subset if not np.isnan(r['perplexity'])]
        engs = [r['english_fraction'] for r in subset]
        spas = [r['spanish_fraction'] for r in subset]
        cjks = [r['cjk_fraction'] for r in subset]

        ppl_medians.append(np.median(ppls))
        ppl_q25.append(np.percentile(ppls, 25))
        ppl_q75.append(np.percentile(ppls, 75))
        eng_means.append(np.mean(engs))
        eng_stds.append(np.std(engs))
        spa_means.append(np.mean(spas))
        spa_stds.append(np.std(spas))
        cjk_means.append(np.mean(cjks))

    angles_arr = np.array(angles)

    # Top: perplexity
    ax_ppl.fill_between(angles_arr, ppl_q25, ppl_q75,
                        color=sty['color'], alpha=0.15)
    ax_ppl.plot(angles_arr, ppl_medians, 'o-',
                color=sty['color'], lw=2, markersize=5)
    ax_ppl.set_title(sty['label'], fontsize=12)
    ax_ppl.set_ylabel('Perplexity (median)' if col == 0 else '', fontsize=11)
    ax_ppl.set_ylim(0, 80)
    ax_ppl.grid(True, alpha=0.3)

    # Coherence shading on perplexity panel
    ax_ppl.axhspan(0, 20, color='green', alpha=0.04, zorder=0)
    ax_ppl.axhline(20, color='green', alpha=0.3, ls='--', lw=0.8)
    if col == 0:
        ax_ppl.annotate('coherent', xy=(2, 17), fontsize=9, color='green',
                        style='italic', va='top')

    # Bottom: language fractions
    ax_lang.errorbar(angles_arr, eng_means, yerr=eng_stds,
                     fmt='s-', color='#2ca02c', lw=2, markersize=5,
                     capsize=3, label='English frac', zorder=3)
    ax_lang.errorbar(angles_arr, spa_means, yerr=spa_stds,
                     fmt='^-', color='#ff7f0e', lw=2, markersize=5,
                     capsize=3, label='Spanish frac', zorder=3)
    ax_lang.plot(angles_arr, cjk_means, 'v-',
                 color='#d62728', lw=1.5, markersize=4,
                 label='CJK frac', zorder=2)

    ax_lang.set_xlabel('Steering angle (degrees)', fontsize=11)
    ax_lang.set_ylabel('Character fraction' if col == 0 else '', fontsize=11)
    ax_lang.set_ylim(-0.05, 1.05)
    ax_lang.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax_lang.grid(True, alpha=0.3)
    if col == 2:
        ax_lang.legend(fontsize=9, loc='center right')

fig.suptitle('Spanish → English steering coherence analysis\n'
             'Qwen3-0.6B, layer 14 | 30 prompts, 50 tokens, temp=0.7',
             fontsize=13)
plt.tight_layout()

out_path = f'{results_dir}/spanish_coherence.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
