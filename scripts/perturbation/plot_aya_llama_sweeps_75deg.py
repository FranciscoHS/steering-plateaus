#!/usr/bin/env python3
"""Two-panel 1D perturbation sweeps to 75° for Aya and Llama, all
non-natural-language contrastive directions (6 SEM + 9 programming) plus
random baseline IQR.

Reads results/fig4_robustness/v2_C1d75_<aya|llama>.pkl from
extended_1d_sweeps_75deg.py.

Output: writeup/figures/sweeps_aya_llama_75deg.png/.pdf
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = 'results/fig4_robustness'
OUT_PNG = 'writeup/figures/sweeps_aya_llama_75deg.png'
OUT_PDF = 'writeup/figures/sweeps_aya_llama_75deg.pdf'

SEM = ['Gender', 'Sentiment', 'Formal', 'Literary', 'Children', 'News']
PROG = ['Python', 'JavaScript', 'Java', 'Rust', 'Haskell',
        'Cpp', 'Go', 'Lisp', 'TypeScript']
TARGETS = [
    ('aya',   'Aya-Expanse-8B'),
    ('llama', 'Llama-3.1-8B'),
]


def panel(ax, target, label, show_legend):
    pkl = os.path.join(DIR, f'v2_C1d75_{target}.pkl')
    with open(pkl, 'rb') as f: d = pickle.load(f)
    angles = d['angles_deg']
    dirs = d['directions']
    rand = d['random']

    rand_flat = rand.reshape(-1, rand.shape[-1])
    rmed = np.median(rand_flat, axis=0)
    rlo = np.percentile(rand_flat, 25, axis=0)
    rhi = np.percentile(rand_flat, 75, axis=0)
    ax.fill_between(angles, rlo, rhi, color='gray', alpha=0.22,
                    label='random (IQR)')
    ax.plot(angles, rmed, '-', color='gray', lw=1.8, label='random (median)')

    cmap_sem = plt.cm.YlOrRd(np.linspace(0.45, 0.95, len(SEM)))
    for c, name in zip(cmap_sem, SEM):
        if name not in dirs: continue
        med = np.median(dirs[name], axis=0)
        ax.plot(angles, med, '-', color=c, lw=2.0, label=name)

    cmap_prog = plt.cm.GnBu(np.linspace(0.45, 0.95, len(PROG)))
    for c, name in zip(cmap_prog, PROG):
        if name not in dirs: continue
        med = np.median(dirs[name], axis=0)
        ax.plot(angles, med, '--', color=c, lw=1.6, label=name)

    ax.set_xlim(0, 75)
    ax.set_xlabel(r'Perturbation angle $\theta$ (deg)', fontsize=11)
    ax.set_ylabel(r'$L^2$ at penultimate layer (median)', fontsize=11)
    ax.set_title(
        f'{label}  ·  perturb L{d["layer"]} → measure L{d["measure_layer"]}',
        fontsize=12, pad=8)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if show_legend:
        ax.legend(frameon=True, fontsize=8, ncol=2, loc='upper left')


def main():
    plt.rcParams['font.family'] = 'Inter'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'cm'

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6))
    for i, (tgt, lbl) in enumerate(TARGETS):
        panel(axes[i], tgt, lbl, show_legend=(i == 0))

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    print(f'Saved {OUT_PNG}')


if __name__ == '__main__':
    main()
