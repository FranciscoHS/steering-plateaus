#!/usr/bin/env python3
"""Two figures showing the toy-model build hits high p but does NOT recover
features.

(1) beeswarm of pairwise superellipse-p for three groups:
      direction x direction (the optimized atoms), feature x feature
      (ground-truth canonical features), random x random (null).
    p per pair taken at a single threshold tau (default 0.2).

(2) per-feature recovery: for each of the d canonical features e_i, the best
    alignment max_k |cos(atom_k, e_i)| = max_k |atom_k[i]| over the K optimized
    atoms, sorted descending, vs the same statistic for K random unit
    directions (baseline). Perfect recovery = 1.0.

Pure plotting from saved pkls + atoms.pt; runs locally.

Usage:
    python scripts/superellipse/plot_toy_pfit_recovery.py \
        --base results/dictionary/toymodel_phase1_tanh3 --tau 0.2
"""

import argparse
import os
import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def per_pair_p(pkl_path, tau):
    d = pickle.load(open(pkl_path, 'rb'))
    ps = [e['p_by_tau'][tau] for e in d['per_pair'] if tau in e['p_by_tau']]
    ps = np.array(ps, dtype=float)
    return ps[np.isfinite(ps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',
                    default='results/dictionary/toymodel_phase1_tanh3')
    ap.add_argument('--tau', type=float, default=0.2,
                    help='threshold at which to read per-pair p')
    ap.add_argument('--n_rand_seeds', type=int, default=20,
                    help='random-baseline draws for the recovery figure')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.base, 'pfit_recovery')
    os.makedirs(out, exist_ok=True)

    # ---------------- Figure 1: beeswarm of pairwise p ----------------
    groups = [
        ('direction x direction',
         f'{args.base}/optimize_features_randcontrast/fit_pairs_atom_vs_atom/fit_pairs.pkl',
         'C0'),
        ('feature x feature',
         f'{args.base}/fit_pairs_feature_vs_feature/fit_pairs.pkl', 'C2'),
        ('random x random',
         f'{args.base}/fit_pairs_random_vs_random/fit_pairs.pkl', '0.5'),
    ]
    rng = np.random.default_rng(0)
    all_ps = [per_pair_p(p, args.tau) for _, p, _ in groups]
    gmax = max(ps.max() for ps in all_ps)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for x, ((label, path, color), ps) in enumerate(zip(groups, all_ps)):
        jit = (rng.random(ps.size) - 0.5) * 0.5
        ax.scatter(x + jit, ps, s=14, color=color, alpha=0.5, zorder=2)
        med = np.median(ps)
        ax.hlines(med, x - 0.32, x + 0.32, color='k', lw=2.5, zorder=3)
        ax.text(x, gmax + 0.25, f'med={med:.2f}\nn={ps.size}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylim(top=gmax + 1.1)
    ax.axhline(2.0, color='red', ls=':', lw=1, label='p=2 (ellipse)')
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel(f'pairwise superellipse-p  (tau={args.tau})')
    ax.set_title('Toy model: optimized directions reach high p\n'
                 '(higher even than the true features)')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, axis='y', alpha=0.3)
    png1 = os.path.join(out, 'beeswarm_p.png')
    plt.savefig(png1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {png1}')

    # ---------------- Figure 2: per-feature recovery ----------------
    blob = torch.load(f'{args.base}/optimize_features_randcontrast/atoms.pt',
                      map_location='cpu')
    A = blob['dictionary'].float()
    A = A / A.norm(dim=-1, keepdim=True)              # [K, d] unit atoms
    K, d = A.shape
    # recovery(feature i) = max_k |atom_k[i]|   (atoms unit, e_i canonical)
    rec_dict = A.abs().max(dim=0).values.numpy()      # [d]
    rec_dict_sorted = np.sort(rec_dict)[::-1]

    # baseline: K random unit dirs, averaged over seeds
    rec_rand = []
    for s in range(args.n_rand_seeds):
        g = torch.Generator().manual_seed(1000 + s)
        R = torch.randn(K, d, generator=g)
        R = R / R.norm(dim=-1, keepdim=True)
        rec_rand.append(np.sort(R.abs().max(dim=0).values.numpy())[::-1])
    rec_rand = np.stack(rec_rand)                     # [seeds, d] each sorted
    rand_med = np.median(rec_rand, axis=0)
    rand_lo = np.quantile(rec_rand, 0.05, axis=0)
    rand_hi = np.quantile(rec_rand, 0.95, axis=0)

    ranks = np.arange(1, d + 1)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(ranks, rec_dict_sorted, '-', color='C0', lw=2,
            label='optimized atoms')
    ax.plot(ranks, rand_med, '--', color='0.4', lw=1.5,
            label=f'random atoms (median of {args.n_rand_seeds})')
    ax.fill_between(ranks, rand_lo, rand_hi, color='0.6', alpha=0.25,
                    label='random 5-95%')
    ax.axhline(1.0, color='red', ls=':', lw=1, label='perfect recovery')
    ax.set_xscale('log')
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('feature rank (sorted by recovery, descending)')
    ax.set_ylabel(r'per-feature recovery  $\max_k |\cos(\mathrm{atom}_k, e_i)|$')
    ax.set_title(f'Toy model: features are not recovered\n'
                 f'(K={K} atoms, d={d}; best feature ~'
                 f'{rec_dict_sorted[0]:.2f}, random ~{rand_med[0]:.2f})')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    png2 = os.path.join(out, 'feature_recovery.png')
    plt.savefig(png2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {png2}')

    # numbers
    print(f'\nbest-recovered feature: dict={rec_dict_sorted[0]:.3f}  '
          f'random_med={rand_med[0]:.3f}')
    print(f'top-{K} recovery mean: dict={rec_dict_sorted[:K].mean():.3f}  '
          f'random_med={rand_med[:K].mean():.3f}')
    print(f'p medians (tau={args.tau}): ' + '  '.join(
        f'{g[0]}={np.median(per_pair_p(g[1], args.tau)):.2f}' for g in groups))


if __name__ == '__main__':
    main()
