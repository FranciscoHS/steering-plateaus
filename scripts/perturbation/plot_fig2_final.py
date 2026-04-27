#!/usr/bin/env python3
"""Fig 2: L2 vs perturbation angle for Portuguese, Gender, Python + random.

Gemma-2-9B, perturb@L2, measure@penultimate. Median over 30 activations,
IQR shading. Random baseline is pooled over multiple random unit directions.
Threshold dashed horizontal line.
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PKL = 'results/paper_L2_gemma2_9b/sweep_1d_L2.pkl'
OUT = 'writeup/figures/fig2_sensitivity_L2.png'
OUT_PDF = 'writeup/figures/fig2_sensitivity_L2.pdf'
THRESH = 50.0

SELECTED = [
    ('Portuguese', '#1f77b4'),
    ('Gender',     '#d62728'),
    ('Python',     '#2ca02c'),
]

with open(PKL, 'rb') as f:
    d = pickle.load(f)

angles = d['angles_deg']
curves = d['curves']

random_mats = np.concatenate(
    [v for k, v in curves.items() if k.startswith('Random')], axis=0)

fig, ax = plt.subplots(figsize=(7.0, 4.8))

# random band (pooled over directions × activations)
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
ax.legend(frameon=True, fontsize=10, loc='upper left')
ax.set_xlim(0, angles[-1])
ax.set_ylim(bottom=0)
ax.grid(alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
print(f'Saved {OUT}')
print(f'Saved {OUT_PDF}')

# Crossings for caption
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
c = crossing(angles, rnd_med, THRESH)
print(f'  {"Random":12s}: {c:.2f} deg')
