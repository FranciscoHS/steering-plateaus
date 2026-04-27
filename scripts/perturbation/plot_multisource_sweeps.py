#!/usr/bin/env python3
"""Plot multisource 1D sweeps: one figure per model, one curve per
contrastive direction (each starting from its own matched activations),
with per-source random IQR baseline.

Usage:
  python scripts/perturbation/plot_multisource_sweeps.py --target gemma --layer 2
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
PALETTE = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#a65628', '#f781bf', '#1b9e77', '#d95f02', '#7570b3',
    '#e7298a', '#66a61e', '#e6ab02', '#a6761d',
]


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
    sources = d['sources']
    if not sources:
        print('no sources yet'); return

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    # Single random baseline (median across all sources' randoms — they all
    # behave similarly per source, but plotting per-source random would be
    # too noisy. We average all to show "typical random".)
    rand_concat = np.concatenate(
        [info['l2_rand'].reshape(-1, info['l2_rand'].shape[-1])
         for info in sources.values()], axis=0)
    rmed = np.median(rand_concat, axis=0)
    rlo = np.percentile(rand_concat, 25, axis=0)
    rhi = np.percentile(rand_concat, 75, axis=0)
    ax.fill_between(angles, rlo, rhi, color='gray', alpha=0.20,
                    label='random (IQR, all sources pooled)')
    ax.plot(angles, rmed, '-', color='gray', lw=1.6, label='random (median)')

    for i, (name, info) in enumerate(sources.items()):
        c = PALETTE[i % len(PALETTE)]
        med = np.median(info['l2_dir'], axis=0)
        tag = '' if info.get('in_sample') is False else '*'   # * = in-sample
        ax.plot(angles, med, '-', color=c, lw=2.0,
                label=f'{name}{tag}')

    ax.set_xlim(0, 75)
    ax.set_xlabel(r'Perturbation angle $\theta$ (deg)', fontsize=12)
    ax.set_ylabel(r'$L^2$ at penultimate layer (median over 30 starting acts)',
                  fontsize=12)
    ax.set_title(
        f'{LABEL[args.target]}  ·  perturb L{d["layer"]} → measure L{d["measure_layer"]}'
        f'  ·  {len(sources)} directions, matched starting activations'
        f'   ( * = in-sample starting prompts)',
        fontsize=11.5, pad=10)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, fontsize=8, ncol=2, loc='upper left')

    plt.tight_layout()
    out_png = f'writeup/figures/sweeps_multisource_{args.target}_L{args.layer}.png'
    out_pdf = f'writeup/figures/sweeps_multisource_{args.target}_L{args.layer}.pdf'
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    print(f'Saved {out_png}  ({len(sources)} sources)')


if __name__ == '__main__':
    main()
