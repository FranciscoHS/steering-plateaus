#!/usr/bin/env python3
"""Histogram of found-direction alignment with contrastive directions.

Mirrors dict_sae_alignment.py's histogram, but for the contrastive-direction
alignment produced by dict_vs_contrastive_align.py. Reads that script's pkl
(found_best_cos = per-atom best |cos| over the contrastive set; rand_max =
random-unit-direction null) and plots found vs null densities.

Usage:
    python scripts/superellipse/plot_contrastive_align_hist.py \\
        --pkl .../dict_vs_contrastive/dict_vs_contrastive.pkl \\
        --out .../dict_vs_contrastive/contrastive_alignment_hist.png
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--pkl', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()

d = pickle.load(open(args.pkl, 'rb'))
found = np.asarray(d['found_best_cos'])     # per-atom best |cos|, [K]
rand_max = np.asarray(d['rand_max'])        # null max |cos|, [n_random]
nC = len(d['names'])
K = len(found)

lo = min(found.min(), rand_max.min())
hi = max(found.max(), rand_max.max())
bins = np.linspace(lo, hi, 60)
r95 = np.quantile(rand_max, 0.95)
n_above = int((found > r95).sum())

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(rand_max, bins=bins, density=True, color='gray', alpha=0.5,
        label=f'random null (n={len(rand_max)})')
ax.hist(found, bins=bins, density=True, color='C1', alpha=0.6,
        label=f'found atoms (n={K})')
ax.axvline(r95, color='k', ls='--', lw=1, label=f'random 95th pct = {r95:.3f}')
ax.set_xlabel('max |cosine| with any contrastive direction (best match)')
ax.set_ylabel('density')
ax.set_title(f'Found-direction alignment with contrastive directions '
             f'({nC} dirs)\n{K} atoms vs {len(rand_max)} random  |  '
             f'{n_above}/{K} above random 95th pct')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.savefig(args.out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {args.out}  ({n_above}/{K} above r95={r95:.3f})')
