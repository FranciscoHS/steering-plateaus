#!/usr/bin/env python3
"""Combined summary of the diversity-penalty (lam_prev) sweep: superellipse p
and atom alignment vs lam, from the fit_pairs.pkl files and the cosine pkl.

Usage:
    python scripts/superellipse/plot_lam_sweep_summary.py \\
        --p_pkls lam0.pkl lam1.pkl ... --lams 0 1 10 50 100 \\
        --cos_pkl cos_alignment.pkl --out .../lam_summary
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p_pkls', nargs='+', required=True)
    ap.add_argument('--lams', nargs='+', type=float, required=True)
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--cos_pkl', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    p_pool, ci_lo, ci_hi, p_med, iqr_lo, iqr_hi = [], [], [], [], [], []
    for pk in args.p_pkls:
        d = pickle.load(open(pk, 'rb'))
        p_pool.append(d['p_pool'])
        ci_lo.append(d['p_pool_ci_95'][0]); ci_hi.append(d['p_pool_ci_95'][1])
        p_med.append(d['p_median'])
        iqr_lo.append(d['p_iqr'][0]); iqr_hi.append(d['p_iqr'][1])

    cos = pickle.load(open(args.cos_pkl, 'rb'))['stats']
    mean_abs = [cos[l]['mean_abs_cos'] for l in args.labels]
    shared = [cos[l]['shared_norm'] for l in args.labels]

    # x positions: log-friendly, map lam=0 to a small value left of lam=1
    xpos = [0.3 if l == 0 else l for l in args.lams]
    xt_labels = [('0' if l == 0 else f'{l:g}') for l in args.lams]

    fig, (axp, axc) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- p vs lam ---
    axp.plot(xpos, p_pool, 'o-', color='C2', lw=2, label='pooled-points p')
    axp.fill_between(xpos, ci_lo, ci_hi, color='C2', alpha=0.2,
                     label='pooled 95% CI')
    axp.plot(xpos, p_med, 's--', color='C1', lw=1.5, label='per-pair median p')
    axp.fill_between(xpos, iqr_lo, iqr_hi, color='C1', alpha=0.13, label='IQR')
    axp.axhline(2.0, color='gray', ls=':', lw=1, label='p=2')
    axp.set_xscale('log')
    axp.set_xticks(xpos); axp.set_xticklabels(xt_labels)
    axp.set_xlabel('diversity penalty  lam_prev')
    axp.set_ylabel('superellipse p')
    axp.set_title('p vs diversity penalty')
    axp.legend(fontsize=8); axp.grid(True, alpha=0.3, which='both')

    # --- alignment vs lam ---
    axc.plot(xpos, mean_abs, 'o-', color='C0', lw=2, label='mean |cos|')
    axc.plot(xpos, shared, 's-', color='C3', lw=2, label='||mean atom||')
    axc.axhline(1.0 / np.sqrt(100), color='gray', ls=':', lw=1,
                label='isotropic floor ||mean|| (1/√100)')
    axc.set_xscale('log'); axc.set_yscale('log')
    axc.set_xticks(xpos); axc.set_xticklabels(xt_labels)
    axc.set_xlabel('diversity penalty  lam_prev')
    axc.set_ylabel('atom alignment (first 100 atoms)')
    axc.set_title('alignment vs diversity penalty')
    axc.legend(fontsize=8); axc.grid(True, alpha=0.3, which='both')

    fig.suptitle('Diversity-penalty sweep  (qwen3-1.7b L14, 100 atoms each)',
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(args.out, 'lam_sweep_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


if __name__ == '__main__':
    main()
