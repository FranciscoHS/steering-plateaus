#!/usr/bin/env python3
"""Plot 1D perturbation sweeps from matched holdout activations.

For Gender (perturbed from MALE_PROMPTS_HOLDOUT) and Sentiment (from
POSITIVE_PROMPTS_HOLDOUT), plot the median L² response across 30 holdout
prompts vs angle (0–75°). Random baseline is 10 random unit vectors swept
from the SAME activations.

Usage:
  python scripts/perturbation/plot_holdout_sweeps.py --target gemma --layer 2
"""
import os, sys, pickle, argparse
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = 'results/fig4_robustness'

LABEL = {
    'gemma':   'Gemma-2-9B',
    'qwen':    'Qwen3-1.7B',
    'llama':   'Llama-3.1-8B',
    'mistral': 'Mistral-7B-v0.3',
    'aya':     'Aya-Expanse-8B',
    'yi':      'Yi-1.5-9B',
}
SOURCE_LABEL = {
    'Gender':    'male holdout',
    'Sentiment': 'positive holdout',
}
COLORS = {
    'Gender':    '#e41a1c',
    'Sentiment': '#377eb8',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(LABEL))
    ap.add_argument('--layer', type=int, default=2)
    args = ap.parse_args()

    plt.rcParams['font.family'] = 'Inter'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'cm'

    pkl = os.path.join(DIR, f'v2_M1d_{args.target}_L{args.layer}.pkl')
    with open(pkl, 'rb') as f: d = pickle.load(f)
    angles = d['angles_deg']
    sources = d['sources']

    fig, ax = plt.subplots(figsize=(8.5, 5.6))

    for name, info in sources.items():
        c = COLORS.get(name, 'k')
        # contrastive direction sweep — median over 30 holdout activations
        med = np.median(info['l2_dir'], axis=0)
        lo = np.percentile(info['l2_dir'], 25, axis=0)
        hi = np.percentile(info['l2_dir'], 75, axis=0)
        ax.fill_between(angles, lo, hi, color=c, alpha=0.18)
        ax.plot(angles, med, '-', color=c, lw=2.4,
                label=f'{name}  ({SOURCE_LABEL.get(name, name)})')
        # random baseline from same activations (10 random dirs × 30 acts)
        rflat = info['l2_rand'].reshape(-1, info['l2_rand'].shape[-1])
        rmed = np.median(rflat, axis=0)
        rlo = np.percentile(rflat, 25, axis=0)
        rhi = np.percentile(rflat, 75, axis=0)
        ax.fill_between(angles, rlo, rhi, color=c, alpha=0.06)
        ax.plot(angles, rmed, '--', color=c, lw=1.4, alpha=0.7,
                label=f'random from {SOURCE_LABEL.get(name, name)} (median)')

    ax.set_xlim(0, 75)
    ax.set_xlabel(r'Perturbation angle $\theta$ (deg)', fontsize=12)
    ax.set_ylabel(r'$L^2$ at penultimate layer (median over 30 holdout)',
                  fontsize=12)
    ax.set_title(
        f'{LABEL[args.target]}  ·  perturb L{d["layer"]} → measure L{d["measure_layer"]}'
        f'  ·  matched-holdout activations',
        fontsize=12, pad=10)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, fontsize=9, loc='upper left')

    plt.tight_layout()
    out_png = f'writeup/figures/sweeps_holdout_{args.target}_L{args.layer}.png'
    out_pdf = f'writeup/figures/sweeps_holdout_{args.target}_L{args.layer}.pdf'
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    print(f'Saved {out_png}')


if __name__ == '__main__':
    main()
