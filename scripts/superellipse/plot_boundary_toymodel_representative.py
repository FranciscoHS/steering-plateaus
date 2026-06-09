#!/usr/bin/env python3
"""Boundary plots for *representative* feature x random pairs (p near median).

Reads the per-pair boundary radii already stored in fit_pairs.pkl (no model
forward needed), selects the pairs whose fitted p at a chosen tau is closest to
the median of the distribution, and draws their normalized iso-plateau
boundaries with the same panel style as plot_boundary_toymodel.py.

Usage:
    python scripts/superellipse/plot_boundary_toymodel_representative.py \\
        --fit_pkl results/dictionary/toymodel_phase1_tanh3/fit_pairs_feature_vs_random/fit_pairs.pkl \\
        --tau 0.2 --n_show 3
"""
import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.superellipse.plot_boundary_toymodel import draw_panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit_pkl', default='results/dictionary/toymodel_phase1_tanh3/'
                    'fit_pairs_feature_vs_random/fit_pairs.pkl')
    ap.add_argument('--tau', type=float, default=0.2)
    ap.add_argument('--n_show', type=int, default=3, help='how many near-median pairs')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    with open(args.fit_pkl, 'rb') as f:
        D = pickle.load(f)
    phis = np.asarray(D['phis'], dtype=np.float64)
    tau = args.tau
    if tau not in D['taus']:
        raise SystemExit(f'tau={tau} not in stored taus {D["taus"]}')

    out = args.out or os.path.join(os.path.dirname(args.fit_pkl), 'representative_boundaries')
    os.makedirs(out, exist_ok=True)

    # rank pairs by |p - median| among finite fits
    pairs = D['per_pair']
    ps = np.array([pp['p_by_tau'][tau] for pp in pairs], float)
    finite = np.isfinite(ps)
    med = float(np.median(ps[finite]))
    order = np.argsort(np.abs(ps - med))
    order = [i for i in order if finite[i]][:args.n_show]
    print(f'feature x random, tau={tau}: median p={med:.3f} over n={finite.sum()} '
          f'finite fits', flush=True)

    for rank, i in enumerate(order):
        pp = pairs[i]
        rs = np.asarray(pp['rs_by_tau'][tau], dtype=np.float64)
        r0, r1 = float(rs[0]), float(rs[-1])
        p_fit = float(pp['p_by_tau'][tau])
        xs_norm = rs * np.cos(phis) / r0
        ys_norm = rs * np.sin(phis) / r1
        f_idx = pp['f']
        print(f'  pick {rank}: pair#{i}  feature e_{f_idx}  p={p_fit:.3f}  '
              f'r0={r0:.3f} r1={r1:.3f}', flush=True)

        fig, ax = plt.subplots(figsize=(5.8, 5.8))
        draw_panel(ax, xs_norm, ys_norm, p_fit, r0, r1, 'feature', 'random')
        ax.set_title(f'feature x random (near-median)   p={p_fit:.2f}  '
                     rf'($e_{{{f_idx}}}$, median={med:.2f}, tau={tau})',
                     fontsize=11)
        plt.tight_layout()
        png = os.path.join(out, f'boundary_feature_vs_random_repr{rank}_'
                           f'feat{f_idx}_p{p_fit:.2f}_tau{tau:g}.png')
        fig.savefig(png, dpi=220, bbox_inches='tight', facecolor='white')
        fig.savefig(png.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'    saved {png}', flush=True)


if __name__ == '__main__':
    main()
