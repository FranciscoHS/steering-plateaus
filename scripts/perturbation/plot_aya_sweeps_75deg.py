#!/usr/bin/env python3
"""1D perturbation sweeps to 75° for Aya, all non-natural-language contrastive
directions (6 SEM + 9 programming), with random baseline IQR band.

Reads results/fig4_robustness/v2_C1d75_aya.pkl produced by
extended_1d_sweeps_75deg.py. Plots median-over-contexts L² vs angle for each
direction. SEM directions in warm hues, programming in cool hues.

Output: writeup/figures/sweeps_aya_75deg.png/.pdf
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PKL = 'results/fig4_robustness/v2_C1d75_aya.pkl'
OUT_PNG = 'writeup/figures/sweeps_aya_75deg.png'
OUT_PDF = 'writeup/figures/sweeps_aya_75deg.pdf'

SEM = ['Gender', 'Sentiment', 'Formal', 'Literary', 'Children', 'News']
PROG = ['Python', 'JavaScript', 'Java', 'Rust', 'Haskell',
        'Cpp', 'Go', 'Lisp', 'TypeScript']


def main():
    plt.rcParams['font.family'] = 'Inter'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'cm'

    with open(PKL, 'rb') as f: d = pickle.load(f)
    angles = d['angles_deg']
    dirs = d['directions']
    rand = d['random']   # (n_rand, n_act, n_steps)

    fig, ax = plt.subplots(figsize=(8.5, 5.6))

    # Random IQR (collapsed across rand × contexts)
    rand_flat = rand.reshape(-1, rand.shape[-1])
    rmed = np.median(rand_flat, axis=0)
    rlo = np.percentile(rand_flat, 25, axis=0)
    rhi = np.percentile(rand_flat, 75, axis=0)
    ax.fill_between(angles, rlo, rhi, color='gray', alpha=0.22,
                    label='random (IQR)')
    ax.plot(angles, rmed, '-', color='gray', lw=1.8, label='random (median)')

    # SEM curves (warm cmap)
    cmap_sem = plt.cm.YlOrRd(np.linspace(0.45, 0.95, len(SEM)))
    for c, name in zip(cmap_sem, SEM):
        if name not in dirs: continue
        med = np.median(dirs[name], axis=0)
        ax.plot(angles, med, '-', color=c, lw=2.0, label=name)

    # Programming curves (cool cmap, dashed to distinguish)
    cmap_prog = plt.cm.GnBu(np.linspace(0.45, 0.95, len(PROG)))
    for c, name in zip(cmap_prog, PROG):
        if name not in dirs: continue
        med = np.median(dirs[name], axis=0)
        ax.plot(angles, med, '--', color=c, lw=1.6, label=name)

    ax.set_xlim(0, 75)
    ax.set_xlabel(r'Perturbation angle $\theta$ (deg)', fontsize=12)
    ax.set_ylabel(r'$L^2$ at penultimate layer (median over 30 contexts)',
                  fontsize=12)
    ax.set_title(
        f'Aya-Expanse-8B  ·  perturb L{d["layer"]} → measure L{d["measure_layer"]}'
        f'  ·  {len(dirs)} contrastive directions',
        fontsize=12, pad=10)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, fontsize=8, ncol=2, loc='upper left')

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    print(f'Saved {OUT_PNG}')


if __name__ == '__main__':
    main()
