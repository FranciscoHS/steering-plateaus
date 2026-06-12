#!/usr/bin/env python3
"""Plot a mixture-order sweep (median p vs k) from a *_k_sweep*.pkl.

Works for both mixture_k_sweep_p.py (optimized atoms) and
mixture_k_sweep_contrast_p.py (contrastive dirs): both store results['by_k'] =
{k: array of per-pair p}. Draws jittered per-pair points, median, and IQR
whiskers per k, with the p=2 (ellipse / non-feature) reference line.

    python scripts/superellipse/plot_mixture_k_sweep.py --pkl .../X.pkl \\
        --title 'Contrastive mixture sweep' --out .../X.png
"""
import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--title', default='Mixture-order sweep')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    b = pickle.load(open(args.pkl, 'rb'))
    by_k = b['by_k']
    ks = sorted(by_k.keys())

    fig, ax = plt.subplots(figsize=(1.0 * len(ks) + 3, 5.5))
    jit_rng = np.random.default_rng(0)
    for x, k in enumerate(ks):
        v = np.asarray(by_k[k], float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            ax.text(x, 2.0, 'infeasible', ha='center', va='bottom',
                    fontsize=8, color='gray', rotation=90)
            continue
        jit = jit_rng.uniform(-0.28, 0.28, size=v.size)
        ax.scatter(np.full(v.size, x) + jit, v, s=9, alpha=0.35, color='C0',
                   edgecolors='none', zorder=2)
        med = np.median(v)
        q1, q3 = np.quantile(v, [0.25, 0.75])
        ax.plot([x - 0.34, x + 0.34], [med, med], color='k', lw=2.2, zorder=4)
        ax.plot([x - 0.28, x + 0.28], [q1, q1], color='k', lw=0.8, zorder=4)
        ax.plot([x - 0.28, x + 0.28], [q3, q3], color='k', lw=0.8, zorder=4)
        ax.plot([x, x], [q1, q3], color='k', lw=0.8, zorder=4)
        ax.annotate(f'{med:.2f}', (x, med), textcoords='offset points',
                    xytext=(6, 0), fontsize=8, va='center', zorder=5)

    ax.axhline(2.0, color='gray', lw=1, ls=':', label='p=2 (ellipse)')
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f'k={k}' for k in ks])
    ax.set_xlabel('mixture order k  (mix-of-k vs mix-of-k)')
    ax.set_ylabel('superellipse p')
    ax.set_title(args.title)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
