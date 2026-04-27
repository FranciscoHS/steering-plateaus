#!/usr/bin/env python3
"""Fig 2 variant including the 'Formal' contrastive direction."""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PKL = 'results/pair_exploration_L2_v2/sweep_1d.pkl'
OUT = 'writeup/figures/fig2_sensitivity_L2_with_formal.png'
OUT_PDF = 'writeup/figures/fig2_sensitivity_L2_with_formal.pdf'
THRESH = 50.0

SELECTED = [
    ('Portuguese', '#1f77b4'),
    ('Gender',     '#d62728'),
    ('Formal',     '#9467bd'),
]
XMAX = 25.0

with open(PKL, 'rb') as f:
    d = pickle.load(f)

angles = d['angles_deg']
curves = d['curves']

random_mats = np.concatenate(
    [v for k, v in curves.items() if k.startswith('Random_')], axis=0)

fig, ax = plt.subplots(figsize=(7.0, 4.8))

rnd_med = np.median(random_mats, axis=0)
rnd_lo  = np.percentile(random_mats, 25, axis=0)
rnd_hi  = np.percentile(random_mats, 75, axis=0)
ax.fill_between(angles, rnd_lo, rnd_hi, color='gray', alpha=0.22,
                label='random (IQR)')
ax.plot(angles, rnd_med, '-', color='gray', lw=2.0, label='random median')

for name, color in SELECTED:
    mat = curves[name]
    med = np.median(mat, axis=0)
    lo  = np.percentile(mat, 25, axis=0)
    hi  = np.percentile(mat, 75, axis=0)
    ax.fill_between(angles, lo, hi, color=color, alpha=0.15)
    ax.plot(angles, med, '-', color=color, lw=2.2, label=name)

ax.axhline(THRESH, ls='--', color='k', lw=1.0, alpha=0.55,
           label=f'threshold ($L^2 = {THRESH:g}$)')

ax.set_xlabel(r'Perturbation angle $\theta$ (deg)', fontsize=12)
ax.set_ylabel(r'$L^2$ distance at penultimate layer', fontsize=12)
ax.set_title(f"Gemma-2-9B · perturb L{d['layer']} → measure L{d['measure_layer']}",
             fontsize=12)
ax.legend(frameon=True, fontsize=9, loc='upper left')
ax.set_xlim(0, XMAX)
# auto-fit y to the visible angle range across selected curves + random IQR
ymax = 0.0
mask = angles <= XMAX
for name, _ in SELECTED:
    ymax = max(ymax, float(np.percentile(curves[name][:, mask], 75, axis=0).max()))
ymax = max(ymax, float(rnd_hi[mask].max()))
ax.set_ylim(0, ymax * 1.05)
ax.grid(alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
print(f'Saved {OUT}')

def crossing(angles, med, thr):
    above = np.where(med >= thr)[0]
    if len(above) == 0: return None
    k = above[0]
    if k == 0: return angles[0]
    v0, v1 = med[k-1], med[k]
    return angles[k-1] + (thr - v0) / (v1 - v0) * (angles[k] - angles[k-1])

print('\nPlateau-breaking angles (median curve crosses L2 = 50):')
for name, _ in SELECTED:
    c = crossing(angles, np.median(curves[name], axis=0), THRESH)
    print(f'  {name:12s}: {c:.2f} deg')
print(f'  {"Random":12s}: {crossing(angles, rnd_med, THRESH):.2f} deg')
