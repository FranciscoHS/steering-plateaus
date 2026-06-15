#!/usr/bin/env python3
"""Overlay the per-atom best-SAE-latent |cos| histograms of two (or more)
dictionaries, plus the random-direction null from the first pkl.

    python scripts/superellipse/plot_sae_align_overlay.py \\
        --pkls A/sae_alignment.pkl B/sae_alignment.pkl \\
        --labels "with response term" "no response term" \\
        --out overlay.png
"""
import argparse
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--pkls', nargs='+', required=True)
ap.add_argument('--labels', nargs='+', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
assert len(args.pkls) == len(args.labels)

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 1, 51)
rand = None
for path, label in zip(args.pkls, args.labels):
    with open(path, 'rb') as f:
        b = pickle.load(f)
    v = np.asarray(b['found_max_cos'])
    ax.hist(v, bins=bins, density=True, alpha=0.5,
            label=f'{label} (n={len(v)}, med={np.median(v):.3f})')
    if rand is None:
        rand = np.asarray(b['rand_max'])
ax.hist(rand, bins=bins, density=True, histtype='step', lw=2, color='k',
        label=f'random null (n={len(rand)}, med={np.median(rand):.3f})')
ax.set_xlabel('best |cos| with any SAE latent')
ax.set_ylabel('density')
ax.set_title('Best SAE-latent alignment per atom')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f'Saved {args.out}')
