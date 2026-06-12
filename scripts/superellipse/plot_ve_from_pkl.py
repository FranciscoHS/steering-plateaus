#!/usr/bin/env python3
"""Replot centered variance-explained from an omp_eval.pkl produced by
eval_dictionary_omp_ve.py. X-axis is log-scaled but tick LABELS are the actual
K values (8, 16, 32, ...) rather than powers of 2.

Usage:
    python scripts/superellipse/plot_ve_from_pkl.py \\
        --pkl .../ve_eval_n25k_fair/omp_eval.pkl --metric centered
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--metric', choices=['centered', 'uncentered'],
                    default='centered')
    ap.add_argument('--out', default=None,
                    help='output png path (default: alongside pkl)')
    args = ap.parse_args()

    d = pickle.load(open(args.pkl, 'rb'))
    ve = d['ve_centered'] if args.metric == 'centered' else d['ve_uncentered']
    K_eval = d['K_eval']
    N = d['N']
    d_model = d['d_model']

    fig, ax = plt.subplots(figsize=(8, 5))
    if 'dict' in ve:
        ax.plot(K_eval, ve['dict'], 'o-', label=f'dict (N={N}, OMP)', color='C0')
    if 'rand_sameN' in ve:
        ax.plot(K_eval, ve['rand_sameN'], 's--',
                label=f'random (N={N}, OMP)', color='C1')
    if 'rand_wide' in ve:
        ax.plot(K_eval, ve['rand_wide'], '^--',
                label=f'random (N={d.get("rand_wide_N")}, OMP)', color='C2')
    if 'pca' in ve:
        ax.plot(K_eval, ve['pca'], 'd-.', label='PCA top-K', color='C3')
    if 'sae' in ve:
        ax.plot(d['sae_K_list'], ve['sae'], '*', markersize=14,
                label=f'Qwen-Scope SAE (N={d.get("sae_d_sae")}, TopK)',
                color='C4')

    # Log x-scale for spacing, but label ticks with the actual K values.
    ax.set_xscale('log', base=2)
    xticks = sorted(set(list(K_eval) + list(d.get('sae_K_list', []))))
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(k) for k in xticks]))
    ax.xaxis.set_minor_locator(NullLocator())

    ax.set_xlabel('K (atoms / active features)')
    ax.set_ylabel(f'fraction of variance explained ({args.metric})')
    fair = d['args'].get('center_omp', True)
    fair_s = 'free-mean OMP' if fair else 'raw OMP'
    ax.set_title(f'K-sparse reconstruction, {args.metric} VE  ({fair_s}; '
                 f'held-out FineWeb, d={d_model})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    out = args.out or os.path.join(
        os.path.dirname(args.pkl),
        f'variance_explained_{args.metric}_Klabels.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
