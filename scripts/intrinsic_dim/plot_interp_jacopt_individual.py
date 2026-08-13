#!/usr/bin/env python3
"""Plot individual relative-distance curves for jac-opt interpolation pairs.

Reads results/intrinsic_dim/interp_jacopt_pairs/results.pkl (from
interp_jacopt_pairs.py) and writes one PNG per pair for the first N pairs.
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = 'results/intrinsic_dim/interp_jacopt_pairs'
N_PLOTS = 10

with open(os.path.join(DIR, 'results.pkl'), 'rb') as f:
    r = pickle.load(f)
t, rel, pairs = r['t'], r['rel_dist'], r['pairs']

for p in range(N_PLOTS):
    i, j = pairs[p]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, rel[p], color='C4', lw=2)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('interpolation t (slerp)')
    ax.set_ylabel(r'$d_A/(d_A+d_B)$  (logits)')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f'jac-opt pair {i} -> {j}  '
                 f'(endpoint logit dist {r["endpoint_dist"][p]:.0f})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(DIR, f'pair_{p:02d}_i{i}_j{j}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')
