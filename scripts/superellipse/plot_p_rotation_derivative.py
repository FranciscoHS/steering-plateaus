#!/usr/bin/env python3
"""Compare how fast p decays under rotation-toward-random across direction
sets (optimized / contrastive / SAE), using the p(t) curves saved by
sweep_p_vs_rotation_llm.py.

For each set we pool all its p_vs_rotation_llm.pkl files, flatten to
per-curve p(t) [n_curves, n_t], and take the local derivative dp/dt at every
t via central differences (np.gradient over the ts grid). We then plot:
  (1) p(t)  median +- IQR per set,
  (2) dp/dt(t)  median +- IQR per set  (the "rate of change" along the curve),
  (3) distribution of the t=0 slope per set.

All input pkls must share the same ts grid.

    python scripts/superellipse/plot_p_rotation_derivative.py \\
        --groups "optimized=resultsA/p_vs_rotation_llm.pkl" \\
                 "contrastive=scaffold/p.pkl,langcode/p.pkl" \\
                 "SAE=sae/p_vs_rotation_llm.pkl" \\
        --out results/dictionary/p_rotation_derivative.png
"""
import argparse
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--groups', nargs='+', required=True,
                help='each is "label=pkl1[,pkl2,...]"')
ap.add_argument('--out', required=True)
args = ap.parse_args()

COLORS = ['#3a7d3a', '#c1440e', '#1f77b4', '#7d3a7d', '#888888']

groups = []
ts_ref = None
for spec in args.groups:
    label, paths = spec.split('=', 1)
    curves = []
    for p in paths.split(','):
        b = pickle.load(open(p, 'rb'))
        ts = np.asarray(b['ts'])
        if ts_ref is None:
            ts_ref = ts
        elif not np.allclose(ts, ts_ref):
            raise ValueError(f'ts grid mismatch in {p}')
        ps = np.asarray(b['ps'])               # [pairs, draws, t]
        curves.append(ps.reshape(-1, ps.shape[-1]))
    P = np.concatenate(curves, axis=0)         # [n_curves, n_t]
    P = P[np.isfinite(P).all(axis=1)]          # drop any curve with a nan
    dP = np.gradient(P, ts_ref, axis=1)        # dp/dt at each t (central diff)
    groups.append({'label': label, 'P': P, 'dP': dP})
    print(f'{label}: {P.shape[0]} curves  p(0) med={np.median(P[:,0]):.2f}  '
          f'p(1) med={np.median(P[:,-1]):.2f}  '
          f'dp/dt|0 med={np.median(dP[:,0]):.2f}')

ts = ts_ref
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# (1) p(t)
ax = axes[0]
for g, c in zip(groups, COLORS):
    med = np.median(g['P'], axis=0)
    q1, q3 = np.quantile(g['P'], 0.25, 0), np.quantile(g['P'], 0.75, 0)
    ax.fill_between(ts, q1, q3, color=c, alpha=0.18)
    ax.plot(ts, med, '-o', color=c, lw=2, ms=4,
            label=f"{g['label']} (n={g['P'].shape[0]})")
ax.axhline(2.0, color='k', ls='--', alpha=0.5)
ax.set_xlabel('rotation fraction t'); ax.set_ylabel('fitted p')
ax.set_title('p(t): candidate -> random'); ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (2) dp/dt(t)
ax = axes[1]
for g, c in zip(groups, COLORS):
    med = np.median(g['dP'], axis=0)
    q1, q3 = np.quantile(g['dP'], 0.25, 0), np.quantile(g['dP'], 0.75, 0)
    ax.fill_between(ts, q1, q3, color=c, alpha=0.18)
    ax.plot(ts, med, '-o', color=c, lw=2, ms=4, label=g['label'])
ax.axhline(0.0, color='k', ls='--', alpha=0.5)
ax.set_xlabel('rotation fraction t'); ax.set_ylabel('dp/dt')
ax.set_title('local rate of change dp/dt'); ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (3) t=0 slope distribution
ax = axes[2]
for g, c in zip(groups, COLORS):
    s0 = g['dP'][:, 0]
    ax.hist(s0, bins=20, density=True, alpha=0.5, color=c,
            label=f"{g['label']} (med={np.median(s0):.2f})")
ax.set_xlabel('dp/dt at t=0'); ax.set_ylabel('density')
ax.set_title('initial slope distribution'); ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(args.out, dpi=160, bbox_inches='tight', facecolor='white')
print(f'Saved {args.out}')
