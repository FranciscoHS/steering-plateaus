#!/usr/bin/env python3
"""Replot a sweep_p_dict_to_sae_fullfit pkl with median+IQR only (no per-pair
lines), so the y-range is set by the summary curves and the decay is visible.

Usage:
    python scripts/superellipse/replot_dtosae_median.py \
        --pkl results/dtosae_lam100_v2/fullfit/p_dict_to_sae_fullfit.pkl \
        --out results/dtosae_lam100_v2/fullfit/p_dict_to_sae_fullfit_median.png
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
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default=None)
    args = ap.parse_args()

    d = pickle.load(open(args.pkl, 'rb'))
    ts = np.asarray(d['ts'])
    ps = np.array([np.asarray(r, dtype=float) for r in d['ps']])
    med = np.nanmedian(ps, axis=0)
    lo = np.nanquantile(ps, 0.25, axis=0)
    hi = np.nanquantile(ps, 0.75, axis=0)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.fill_between(ts, lo, hi, color='#7a3a7d', alpha=0.22, label='IQR')
    ax.plot(ts, med, '-o', color='#7a3a7d', lw=2.2, ms=4, label='median p')
    ax.axhline(2.0, color='#1F77B4', ls='--', alpha=0.7, label='p=2 (Euclidean)')
    ax.set_xlabel(r'rotation fraction $t$  (0 = optimized dirs, 1 = SAE latents)')
    ax.set_ylabel('fitted superellipse p (soft_p_estimate, eval frame)')
    n = ps.shape[0]
    ax.set_title(args.title or f'p: optimized dirs -> best-match SAE latents '
                               f'(median + IQR, n={n} pairs)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
