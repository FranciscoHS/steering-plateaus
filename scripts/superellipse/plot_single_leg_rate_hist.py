#!/usr/bin/env python3
"""Histogram of initial single-leg rotation rates dp/dtheta, comparing
direction sets. A cluster near zero = rotate-without-losing-p directions =>
privileged subspace rather than a single direction.

    python scripts/superellipse/plot_single_leg_rate_hist.py \\
        --groups "optimized=A/p_single_leg_rates.pkl" \\
                 "contrastive=scaf/p.pkl,lang/p.pkl" \\
                 "SAE=sae/p_single_leg_rates.pkl" \\
        --out results/dictionary/single_leg_rate_hist.png
"""
import argparse
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--groups', nargs='+', required=True)
ap.add_argument('--near_zero', type=float, default=0.01,
                help='|slope|<this (per deg) counts as "flat"')
ap.add_argument('--out', required=True)
args = ap.parse_args()

COLORS = ['#3a7d3a', '#c1440e', '#1f77b4', '#7d3a7d']

groups = []
allvals = []
for spec in args.groups:
    label, paths = spec.split('=', 1)
    s = []
    for p in paths.split(','):
        b = pickle.load(open(p, 'rb'))
        s.append(np.asarray(b['slopes']).ravel())
    s = np.concatenate(s)
    s = s[np.isfinite(s)]
    groups.append({'label': label, 's': s})
    allvals.append(s)
    frac = np.mean(np.abs(s) < args.near_zero)
    print(f'{label}: n={s.size}  median={np.median(s):+.4f}/deg  '
          f'IQR=({np.quantile(s,.25):+.4f},{np.quantile(s,.75):+.4f})  '
          f'frac|slope|<{args.near_zero}={frac:.3f}')

lo = np.quantile(np.concatenate(allvals), 0.01)
hi = np.quantile(np.concatenate(allvals), 0.99)
bins = np.linspace(lo, hi, 60)

fig, ax = plt.subplots(figsize=(9, 5.5))
for g, c in zip(groups, COLORS):
    frac = np.mean(np.abs(g['s']) < args.near_zero)
    ax.hist(g['s'], bins=bins, density=True, alpha=0.5, color=c,
            label=f"{g['label']} (n={g['s'].size}, med={np.median(g['s']):+.3f}, "
                  f"|·|<{args.near_zero}: {frac:.0%})")
ax.axvline(0.0, color='k', ls='--', alpha=0.6)
ax.set_xlabel('initial rotation rate  dp/dθ  (per degree, one leg fixed)')
ax.set_ylabel('density')
ax.set_title('Distribution of single-leg p-decay rates toward random tangents')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(args.out, dpi=160, bbox_inches='tight', facecolor='white')
print(f'Saved {args.out}')
