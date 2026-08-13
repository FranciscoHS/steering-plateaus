#!/usr/bin/env python3
"""Histogram of per-point activation norms: real vs jac-optimized.

Reads the saved tensors (no model needed) and writes
results/intrinsic_dim/norm_hist.png.
"""

import os

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = 'results/intrinsic_dim'

real = torch.load(os.path.join(DATA_DIR, 'jac_filtered_data.pt'),
                  map_location='cpu', weights_only=False)['X_real'].float()
opt = torch.load(os.path.join(DATA_DIR, 'jac_optimized_data.pt'),
                 map_location='cpu', weights_only=False)['X_opt'].float()

n_real = real.norm(dim=-1).numpy()
n_opt = opt.norm(dim=-1).numpy()
n_opt_resc = n_opt * (n_real.mean() / n_opt.mean())   # global mean-norm rescale
for name, v in [('real', n_real), ('jac-opt', n_opt),
                ('jac-opt rescaled', n_opt_resc)]:
    print(f'{name:18s} n={len(v)}  mean={v.mean():.2f}  sd={v.std():.2f}  '
          f'min={v.min():.2f}  max={v.max():.2f}')

bins = np.linspace(0, max(n_real.max(), n_opt.max()) * 1.05, 60)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(n_real, bins=bins, density=True, alpha=0.55, color='#4477aa',
        label=f'real (n={len(n_real)})')
ax.hist(n_opt, bins=bins, density=True, alpha=0.55, color='#ee7733',
        label=f'jac-optimized (n={len(n_opt)})')
ax.hist(n_opt_resc, bins=bins, density=True, histtype='step', lw=1.8,
        color='#228833',
        label='jac-optimized, rescaled to real mean norm')
ax.set_xlabel(r'$\|a\|_2$')
ax.set_ylabel('density')
ax.set_title('Activation norm distributions — qwen3-1.7b L2')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(DATA_DIR, 'norm_hist.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
