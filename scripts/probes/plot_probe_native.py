#!/usr/bin/env python3
"""
Plot native top-k sparse-probe accuracy vs k (mean over concepts) for the raw
ceiling and selected SAEs. Reads results.pkl from eval_probe_native.py.

Usage:
    python scripts/probes/plot_probe_native.py \
        --results results/probe_native/results.pkl \
        --saes baseline,popt --out results/probe_native/native_probe_vs_k.png
"""
import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'baseline': 'Baseline SAE', 'popt': 'SAE p-opt', 'cullmore': 'SAE cull'}
COLORS = {'baseline': 'C0', 'popt': 'C3', 'cullmore': 'C2'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/probe_native/results.pkl')
    ap.add_argument('--saes', default='baseline,popt')
    ap.add_argument('--raw_pkl', default=None,
                    help='raw_ceiling.pkl; uses its logreg_raw mean as the ceiling')
    ap.add_argument('--out', default='results/probe_native/native_probe_vs_k.png')
    args = ap.parse_args()

    r = pickle.load(open(args.results, 'rb'))
    ks = r['k_values']
    concepts = list(r['concepts'].values())
    n = len(concepts)
    if args.raw_pkl:
        raw = pickle.load(open(args.raw_pkl, 'rb'))['means']['logreg_raw']
    else:
        raw = np.mean([c['raw'] for c in concepts])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(raw, color='0.4', ls='--', lw=1.5, label=f'Raw activations ({raw:.3f})')
    for lbl in args.saes.split(','):
        key = f'sae:{lbl}'
        ys = [np.mean([c[key][k] for c in concepts]) for k in ks]
        ax.plot(ks, ys, 'o-', color=COLORS.get(lbl, None), lw=2, ms=6,
                label=f'{LABELS.get(lbl, lbl)} (k50={ys[-1]:.3f})')

    ax.set_xscale('log')
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel('sparsity k (active latents)')
    ax.set_ylabel(f'probe accuracy (mean over {n} concepts)')
    ax.set_title('Native top-k SAE sparse-code probing')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
