#!/usr/bin/env python3
"""Per-pair p vs pair |cos| across the lam sweep (lam0/lam1/lam10).

Tests the degeneracy story: within a dict, fitted p should fall toward ~2 as
|cos|->1 (the mix coincides with the axes). The matched-cosine comparison
asks whether, at the SAME pair cosine, the stronger-penalty dict still has
higher p (intrinsic) or not (then the cross-dict shift is just that lam0
samples more-aligned pairs).

Reads the fit_pairs.pkl files; pure analysis.
"""
import argparse
import os
import pickle

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):
    d = pickle.load(open(path, 'rb'))
    cos = np.array([pp['cos'] for pp in d['per_pair']], float)
    p = np.array([pp['p_fit'] for pp in d['per_pair']], float)
    m = np.isfinite(cos) & np.isfinite(p)
    return cos[m], p[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkls', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--match_hi', type=float, default=0.06,
                    help='upper |cos| edge of the matched-cosine overlap band')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = {lab: load(p) for lab, p in zip(args.labels, args.pkls)}
    colors = {lab: f'C{i}' for i, lab in enumerate(args.labels)}

    print('=== per-dict: p vs |cos| correlation ===')
    for lab in args.labels:
        cos, p = data[lab]
        ac = np.abs(cos)
        sr, sp = stats.spearmanr(ac, p)
        pr, pp = stats.pearsonr(ac, p)
        print(f'  {lab:6s} n={len(p):3d}  |cos| range [{ac.min():.3f},'
              f'{ac.max():.3f}]  median p={np.median(p):.2f}  '
              f'Spearman(|cos|,p)={sr:+.3f} (p={sp:.3f})  '
              f'Pearson={pr:+.3f} (p={pp:.3f})')

    print(f'\n=== matched-cosine band |cos| <= {args.match_hi} ===')
    for lab in args.labels:
        cos, p = data[lab]
        sel = np.abs(cos) <= args.match_hi
        if sel.sum() >= 3:
            print(f'  {lab:6s} n={int(sel.sum()):3d}  '
                  f'median p={np.median(p[sel]):.2f}  '
                  f'mean p={p[sel].mean():.2f}  '
                  f'IQR=({np.quantile(p[sel],0.25):.2f},'
                  f'{np.quantile(p[sel],0.75):.2f})')
        else:
            print(f'  {lab:6s} n={int(sel.sum()):3d}  (too few for matched band)')

    # Pooled binned-median trend (all dicts together)
    allcos = np.concatenate([np.abs(data[l][0]) for l in args.labels])
    allp = np.concatenate([data[l][1] for l in args.labels])
    edges = np.array([0, 0.04, 0.08, 0.12, 0.18, 0.26, 0.5])
    centers, med, lo, hi, ns = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (allcos >= a) & (allcos < b)
        if s.sum() >= 3:
            centers.append((a + b) / 2); med.append(np.median(allp[s]))
            lo.append(np.quantile(allp[s], 0.25))
            hi.append(np.quantile(allp[s], 0.75)); ns.append(int(s.sum()))
    print('\n=== pooled p vs |cos| (binned median) ===')
    for c, mv, n in zip(centers, med, ns):
        print(f'  |cos|~{c:.3f}  median p={mv:.2f}  (n={n})')
    sr, sp = stats.spearmanr(allcos, allp)
    print(f'  pooled Spearman(|cos|,p) = {sr:+.3f}  (p={sp:.4f})')

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for lab in args.labels:
        cos, p = data[lab]
        ax.scatter(np.abs(cos), p, s=24, alpha=0.55, color=colors[lab],
                   label=f'{lab} (median p={np.median(p):.2f})')
    ax.plot(centers, med, 'k-o', lw=2, label='pooled binned median')
    ax.fill_between(centers, lo, hi, color='k', alpha=0.1)
    ax.axhline(2.0, color='gray', ls=':', lw=1, label='p=2')
    ax.axvline(args.match_hi, color='C3', ls='--', lw=1, alpha=0.5,
               label=f'matched band edge ({args.match_hi})')
    ax.set_xlabel('|cos| between the pair of directions')
    ax.set_ylabel('fitted p (per pair)')
    ax.set_title('Per-pair p vs pair |cos|  (lam sweep)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'p_vs_cos_lamsweep.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved {plot_path}')


if __name__ == '__main__':
    main()
