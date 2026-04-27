#!/usr/bin/env python3
"""Quick-look 1D slices at Gemma-2-9B L20 perturb: PT, Gender, Rust vs random.

Pulls 1D cross-sections from existing isoplateau 2D sweeps and plots L2(θ).
The goal is to see whether the plateau-then-break shape is visually clear at L20.
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = 'results/mandarin_haskell/gemma-2-9b'
OUT = 'writeup/figures/fig2_1d_slices_check.png'

def load(pkl):
    with open(pkl, 'rb') as f:
        return pickle.load(f)

pt_gender = load(f'{BASE}/sweep_isoplateau.pkl')           # (PT, Gender)
fr_rust   = load(f'{BASE}/sweep_isoplateau_french_rust.pkl')  # (French, Rust)
rnd       = load(f'{BASE}/sweep_isoplateau_random.pkl')       # (Random1, Random2)

# PT: varying axis 1 (first direction), Gender at α1=0 means axis 1 index 0 → slice[:,0]
pt_curves    = pt_gender['l2'][:, :, 0]    # (N_ACT, N_ANG)  -- varying PT
gender_curves= pt_gender['l2'][:, 0, :]    # varying Gender
rust_curves  = fr_rust['l2'][:, 0, :]      # varying Rust (second direction)
# random: two random directions available — both 60-deg sweeps
rnd1 = rnd['l2'][:, :, 0]
rnd2 = rnd['l2'][:, 0, :]
rnd_all = np.concatenate([rnd1, rnd2], axis=0)  # 60 traces

angles_small = pt_gender['angles_deg']  # 0..20
angles_rnd   = rnd['angles_deg']        # 0..60

THRESH = 50.0

fig, ax = plt.subplots(figsize=(7.5, 5.0))
# random: median + 25-75 IQR band
rnd_med = np.median(rnd_all, axis=0)
rnd_lo  = np.percentile(rnd_all, 25, axis=0)
rnd_hi  = np.percentile(rnd_all, 75, axis=0)
ax.fill_between(angles_rnd, rnd_lo, rnd_hi, color='gray', alpha=0.25, label='random IQR (pooled)')
ax.plot(angles_rnd, rnd_med, '-', color='gray', lw=2.0, label='random median')

for curves, label, color in [
    (pt_curves,     'Portuguese', '#1f77b4'),
    (gender_curves, 'Gender',     '#d62728'),
    (rust_curves,   'Rust',       '#2ca02c'),
]:
    med = np.median(curves, axis=0)
    lo  = np.percentile(curves, 25, axis=0)
    hi  = np.percentile(curves, 75, axis=0)
    ax.fill_between(angles_small, lo, hi, color=color, alpha=0.15)
    ax.plot(angles_small, med, '-', color=color, lw=2.0, label=label)

ax.axhline(THRESH, ls='--', color='k', lw=1.0, alpha=0.6, label=f'threshold = {THRESH:g}')
ax.set_xlabel('Perturbation angle θ (deg)')
ax.set_ylabel('L2 distance at penultimate layer')
ax.set_title('Gemma-2-9B  L20 perturb → L40 measure  (median over 30 activations)')
ax.legend(frameon=True, fontsize=9)
ax.set_xlim(0, 60)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved {OUT}')

# Also print the angle at which each curve crosses threshold=50
print('\nCrossing angles (threshold L2=50):')
for curves, name, angles in [
    (pt_curves, 'Portuguese', angles_small),
    (gender_curves, 'Gender', angles_small),
    (rust_curves, 'Rust', angles_small),
    (rnd_all, 'Random (pooled)', angles_rnd),
]:
    med = np.median(curves, axis=0)
    above = np.where(med >= THRESH)[0]
    if len(above) == 0:
        print(f'  {name:20s}: never crosses within {angles[-1]:.0f} deg (max L2 = {med.max():.1f})')
    else:
        k = above[0]
        if k == 0:
            print(f'  {name:20s}: crosses before first step')
        else:
            v0, v1 = med[k-1], med[k]
            t = (THRESH - v0) / (v1 - v0)
            theta = angles[k-1] + t * (angles[k] - angles[k-1])
            print(f'  {name:20s}: crosses at θ = {theta:.2f} deg')
