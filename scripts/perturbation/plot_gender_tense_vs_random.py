#!/usr/bin/env python3
"""1D perturbation sweep: Gender (from male holdout) and Tense (from past
in-sample) vs random baseline. Single panel.

Usage:
  python scripts/perturbation/plot_gender_tense_vs_random.py --target gemma --layer 2
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
COLORS = {'Gender': '#e41a1c', 'Tense': '#ff7f00'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(LABEL))
    ap.add_argument('--layer', type=int, default=2)
    args = ap.parse_args()

    plt.rcParams['font.family'] = 'Inter'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'cm'

    pkl = os.path.join(DIR, f'v2_S1d_{args.target}_L{args.layer}.pkl')
    with open(pkl, 'rb') as f: d = pickle.load(f)
    angles = d['angles_deg']
    src = d['sources']

    fig, ax = plt.subplots(figsize=(8.5, 5.6))

    # Pool random baselines from Gender + Tense sources for the gray band.
    rand_concat = np.concatenate(
        [src[name]['l2_rand'].reshape(-1, src[name]['l2_rand'].shape[-1])
         for name in ('Gender', 'Tense')], axis=0)
    rmed = np.median(rand_concat, axis=0)
    rlo = np.percentile(rand_concat, 25, axis=0)
    rhi = np.percentile(rand_concat, 75, axis=0)
    ax.fill_between(angles, rlo, rhi, color='gray', alpha=0.22,
                    label='random (IQR, pooled)')
    ax.plot(angles, rmed, '-', color='gray', lw=1.8, label='random (median)')

    for name in ('Gender', 'Tense'):
        info = src[name]
        c = COLORS[name]
        med = np.median(info['l2_dir'], axis=0)
        lo = np.percentile(info['l2_dir'], 25, axis=0)
        hi = np.percentile(info['l2_dir'], 75, axis=0)
        ax.fill_between(angles, lo, hi, color=c, alpha=0.18)
        tag = '' if not info.get('in_sample') else ' (in-sample)'
        ax.plot(angles, med, '-', color=c, lw=2.4, label=f'{name}{tag}')

    ax.set_xlim(0, 75)
    ax.set_xlabel(r'Perturbation angle $\theta$ (deg)', fontsize=12)
    ax.set_ylabel(r'$L^2$ at penultimate layer (median over 30 starting acts)',
                  fontsize=12)
    ax.set_title(
        f'{LABEL[args.target]}  ·  perturb L{d["layer"]} → measure L{d["measure_layer"]}'
        f'  ·  Gender (male holdout) and Tense (past in-sample) vs random',
        fontsize=11.5, pad=10)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, fontsize=10, loc='upper left')

    plt.tight_layout()
    out_png = f'writeup/figures/sweeps_gender_tense_vs_random_{args.target}_L{args.layer}.png'
    out_pdf = f'writeup/figures/sweeps_gender_tense_vs_random_{args.target}_L{args.layer}.pdf'
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    print(f'Saved {out_png}')


if __name__ == '__main__':
    main()
