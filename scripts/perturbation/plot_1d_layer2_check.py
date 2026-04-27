#!/usr/bin/env python3
"""Plot 1D slices at layer 2: PT / Gender / Rust vs random baseline."""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PKL = 'results/paper_L2_gemma2_9b/sweep_1d_L2.pkl'
OUT = 'writeup/figures/fig2_1d_slices_L2_check.png'
THRESH = 50.0

with open(PKL, 'rb') as f:
    d = pickle.load(f)

angles = d['angles_deg']
curves = d['curves']

# Separate contrastive vs random
contrastive = [(k, v) for k, v in curves.items() if not k.startswith('Random')]
random_mats = np.concatenate([v for k, v in curves.items() if k.startswith('Random')], axis=0)

fig, ax = plt.subplots(figsize=(7.5, 5.0))

# Random band
rnd_med = np.median(random_mats, axis=0)
rnd_lo = np.percentile(random_mats, 25, axis=0)
rnd_hi = np.percentile(random_mats, 75, axis=0)
ax.fill_between(angles, rnd_lo, rnd_hi, color='gray', alpha=0.25,
                label='random IQR (6 dirs × 30 acts)')
ax.plot(angles, rnd_med, '-', color='gray', lw=2.0, label='random median')

for (name, mat), color in zip(contrastive,
                               ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']):
    med = np.median(mat, axis=0)
    lo = np.percentile(mat, 25, axis=0)
    hi = np.percentile(mat, 75, axis=0)
    ax.fill_between(angles, lo, hi, color=color, alpha=0.15)
    ax.plot(angles, med, '-', color=color, lw=2.0, label=name)

ax.axhline(THRESH, ls='--', color='k', lw=1.0, alpha=0.6, label=f'threshold = {THRESH:g}')
ax.set_xlabel('Perturbation angle θ (deg)')
ax.set_ylabel('L2 distance at penultimate layer')
ax.set_title(f"Gemma-2-9B  L{d['layer']} perturb → L{d['measure_layer']} measure  "
             f"(median over {d['n_activations']} activations)")
ax.legend(frameon=True, fontsize=9)
ax.set_xlim(0, angles[-1])
ax.set_ylim(bottom=0)
plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved {OUT}')

# Crossings
print('\nCrossings at L2 = 50:')
def crossing(angles, med, thr):
    above = np.where(med >= thr)[0]
    if len(above) == 0:
        return None
    k = above[0]
    if k == 0: return angles[0]
    v0, v1 = med[k-1], med[k]
    t = (thr - v0) / (v1 - v0)
    return angles[k-1] + t * (angles[k] - angles[k-1])

for name, mat in contrastive:
    c = crossing(angles, np.median(mat, axis=0), THRESH)
    print(f'  {name:12s}: {c:.2f} deg' if c else f'  {name:12s}: no crossing')
c = crossing(angles, rnd_med, THRESH)
print(f'  {"Random (pooled)":12s}: {c:.2f} deg' if c else f'  random never crosses')
