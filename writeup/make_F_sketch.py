#!/usr/bin/env python3
"""Tiny inline sketch of the response function F(xi) shape."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = np.linspace(0, 3, 400)
x_star = 1.0
p = 1.6
baseline = 0.02
slope_pre = 0.05
F = np.where(
    x < x_star,
    baseline + slope_pre * x,
    baseline + slope_pre * x + (x - x_star) ** p,
)

fig, ax = plt.subplots(figsize=(4, 2.6))
ax.plot(x, F, color='black', lw=2.0)

ax.set_xlabel(r'$\sum_i w_i\, |d_i^\top v|^n$', fontsize=12)
ax.set_ylabel(r'$F$', fontsize=12)

ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlim(0, 3)
ax.set_ylim(-0.05, F.max() * 1.1)
ax.margins(x=0.0)

plt.tight_layout()
plt.savefig('writeup/F_sketch.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('writeup/F_sketch.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print('Saved writeup/F_sketch.png and writeup/F_sketch.pdf')
