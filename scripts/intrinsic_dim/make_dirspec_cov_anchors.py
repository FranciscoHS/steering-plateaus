#!/usr/bin/env python3
"""Build anchor sets for the direction-specificity and covariance tests
(qwen3-1.7b L2).

Question: in gauss + c*w_hat, which ingredient is load-bearing?
  - Direction specificity: same fresh cov-matched Gaussian points, shifted by
    c along RANDOM unit directions instead of w_hat. If plateaus appear for
    generic directions, w_hat is not special.
  - Covariance: ISOTROPIC norm-matched points (random unit dirs x norms
    resampled from real) shifted by c*w_hat. If plateaus appear, the matched
    covariance is not load-bearing either.

Uses the saved plane (w, c) from jac_along_normal/results.pkl and the same
fresh Gaussian rows 200:400 as the original shifted-anchor set.

Output: results/intrinsic_dim/dirspec_cov_anchors.pt
  X_rdir1, X_rdir2   gauss + c*r_i, r_i random unit dirs (seeds 101, 102)
  X_iso_shift        isotropic norm-matched + c*w_hat  (seed 103)
  X_iso              the unshifted isotropic points, for reference
  r1, r2, w, c       the directions and shift used
"""

import os
import pickle

import torch

DATA_DIR = 'results/intrinsic_dim'
N_POINTS = 200

filt = torch.load(os.path.join(DATA_DIR, 'jac_filtered_data.pt'),
                  map_location='cpu', weights_only=False)
with open(os.path.join(DATA_DIR, 'jac_along_normal', 'results.pkl'),
          'rb') as f:
    r = pickle.load(f)

w = torch.from_numpy(r['w']).float()
w_hat = w / w.norm()
c = float(r['c'])
X_real = filt['X_real'].float()
G = filt['X_gauss'].float()[N_POINTS:2 * N_POINTS]   # same fresh rows as X_shift
d = G.shape[1]

dirs = []
for seed in (101, 102):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    v = v / v.norm()
    dirs.append(v)
    print(f'random dir seed {seed}: |cos(r, w_hat)| = '
          f'{float((v @ w_hat).abs()):.4f}')

g = torch.Generator().manual_seed(103)
iso = torch.randn(N_POINTS, d, generator=g)
iso = iso / iso.norm(dim=-1, keepdim=True)
real_norms = X_real.norm(dim=-1)
idx = torch.randint(0, len(real_norms), (N_POINTS,), generator=g)
X_iso = iso * real_norms[idx].unsqueeze(-1)

blob = {
    'X_rdir1': G + c * dirs[0],
    'X_rdir2': G + c * dirs[1],
    'X_iso_shift': X_iso + c * w_hat,
    'X_iso': X_iso,
    'r1': dirs[0], 'r2': dirs[1], 'w': w, 'c': c,
}
for k in ('X_rdir1', 'X_rdir2', 'X_iso_shift', 'X_iso'):
    print(f'{k}: {tuple(blob[k].shape)}  '
          f'mean||a||={blob[k].norm(dim=-1).mean():.2f}')
out = os.path.join(DATA_DIR, 'dirspec_cov_anchors.pt')
torch.save(blob, out)
print(f'Saved {out}')
