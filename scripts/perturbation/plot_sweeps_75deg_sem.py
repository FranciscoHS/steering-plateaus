#!/usr/bin/env python3
"""Single-panel 1D perturbation sweep to 75° for one model, SEM contrastive
directions only (no programming languages, no natural-language directions),
with random baseline IQR.

Usage:
  python scripts/perturbation/plot_sweeps_75deg_sem.py --target aya
  python scripts/perturbation/plot_sweeps_75deg_sem.py --target llama
  ... etc.
"""
import os, sys, pickle, argparse
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = 'results/fig4_robustness'
SEM = ['Gender', 'Sentiment', 'Formal', 'Literary', 'Children', 'News']
SEM_COLORS = {
    'Gender':    '#e41a1c',   # red
    'Sentiment': '#377eb8',   # blue
    'Formal':    '#4daf4a',   # green
    'Literary':  '#984ea3',   # purple
    'Children':  '#ff7f00',   # orange
    'News':      '#a65628',   # brown
}
LABEL = {
    'gemma':   'Gemma-2-9B',
    'qwen':    'Qwen3-1.7B',
    'llama':   'Llama-3.1-8B',
    'mistral': 'Mistral-7B-v0.3',
    'aya':     'Aya-Expanse-8B',
    'yi':      'Yi-1.5-9B',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(LABEL))
    ap.add_argument('--layer', type=int, default=None,
                    help='If set, read v2_C1d75_<target>_L<layer>.pkl')
    args = ap.parse_args()

    plt.rcParams['font.family'] = 'Inter'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'cm'

    suffix = f'_L{args.layer}' if args.layer is not None else ''
    pkl = os.path.join(DIR, f'v2_C1d75_{args.target}{suffix}.pkl')
    with open(pkl, 'rb') as f: d = pickle.load(f)
    angles = d['angles_deg']
    dirs = d['directions']
    rand = d['random']

    fig, ax = plt.subplots(figsize=(8.5, 5.6))

    rand_flat = rand.reshape(-1, rand.shape[-1])
    rmed = np.median(rand_flat, axis=0)
    rlo = np.percentile(rand_flat, 25, axis=0)
    rhi = np.percentile(rand_flat, 75, axis=0)
    ax.fill_between(angles, rlo, rhi, color='gray', alpha=0.22,
                    label='random (IQR)')
    ax.plot(angles, rmed, '-', color='gray', lw=1.8, label='random (median)')

    for name in SEM:
        if name not in dirs: continue
        med = np.median(dirs[name], axis=0)
        ax.plot(angles, med, '-', color=SEM_COLORS[name], lw=2.2, label=name)

    ax.set_xlim(0, 75)
    ax.set_xlabel(r'Perturbation angle $\theta$ (deg)', fontsize=12)
    ax.set_ylabel(r'$L^2$ at penultimate layer (median over 30 contexts)',
                  fontsize=12)
    ax.set_title(
        f'{LABEL[args.target]}  ·  perturb L{d["layer"]} → measure L{d["measure_layer"]}'
        f'  ·  {len(SEM)} SEM directions',
        fontsize=12, pad=10)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, fontsize=9, loc='upper left')

    plt.tight_layout()
    out_png = f'writeup/figures/sweeps_{args.target}{suffix}_75deg_sem.png'
    out_pdf = f'writeup/figures/sweeps_{args.target}{suffix}_75deg_sem.pdf'
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    print(f'Saved {out_png}')


if __name__ == '__main__':
    main()
