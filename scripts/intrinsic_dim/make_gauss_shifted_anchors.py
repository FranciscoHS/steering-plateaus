#!/usr/bin/env python3
"""Build the gauss + c*w_hat anchor set for the 1D perturbation sweep.

Reconstructs exactly the shifted points measured in
jac_norm_along_separating_normal.py: fresh cov-matched Gaussian points
(X_gauss rows 200:400; rows 0:200 are the jac-opt starting points) plus the
constant shift c*w_hat along the separating normal saved in
jac_along_normal/results.pkl.

Output: results/intrinsic_dim/gauss_shifted_anchors.pt
  X_shift  [200, d]  shifted points
  X_fresh  [200, d]  the unshifted Gaussian points, for reference
  w, b, c            the plane and shift used
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
G = filt['X_gauss'].float()[N_POINTS:2 * N_POINTS]
X_shift = G + c * w_hat

out = os.path.join(DATA_DIR, 'gauss_shifted_anchors.pt')
torch.save({'X_shift': X_shift, 'X_fresh': G, 'w': w, 'b': r['b'], 'c': c},
           out)
print(f'X_shift: {tuple(X_shift.shape)}  '
      f'mean||a||={X_shift.norm(dim=-1).mean():.2f}  (c={c:.2f})')
print(f'Saved {out}')
