#!/usr/bin/env python3
"""Dose-response anchors: gauss + alpha*c*w_hat for several alpha.

Also prints the shift-magnitude diagnostics Francisco asked for: c relative
to activation norms, and the w_hat-projection statistics of gauss / real /
jac-opt points (is c large compared to the coordinate it moves?).

Output: results/intrinsic_dim/gauss_shift_dose_anchors.pt
  X_a025, X_a050, X_a100, X_a200   gauss + alpha*c*w_hat
  (unshifted gauss = X_fresh in gauss_shifted_anchors.pt; alpha=1 equals
  X_shift there, rebuilt here for self-containedness)
"""

import os
import pickle

import torch

DATA_DIR = 'results/intrinsic_dim'
ALPHAS = [0.25, 0.5, 1.0, 2.0]
N_POINTS = 200

b = torch.load(os.path.join(DATA_DIR, 'gauss_shifted_anchors.pt'),
               map_location='cpu', weights_only=False)
filt = torch.load(os.path.join(DATA_DIR, 'jac_filtered_data.pt'),
                  map_location='cpu', weights_only=False)
opt = torch.load(os.path.join(DATA_DIR, 'jac_optimized_data.pt'),
                 map_location='cpu', weights_only=False)
w, c, G = b['w'].float(), float(b['c']), b['X_fresh'].float()
w_hat = w / w.norm()
X_real = filt['X_real'].float()
X_opt = opt['X_opt'].float()

print(f'c = {c:.2f} (the added vector is c*w_hat, so its norm IS c)')
print(f'mean||gauss|| = {G.norm(dim=-1).mean():.2f}   '
      f'mean||real|| = {X_real.norm(dim=-1).mean():.2f}   '
      f'c/mean||gauss|| = {c / G.norm(dim=-1).mean():.3f}')
for name, X in [('gauss', G), ('real', X_real), ('jac-opt', X_opt)]:
    p = X @ w_hat
    print(f'  {name:8s} projection on w_hat: mean={p.mean():+.2f}  '
          f'std={p.std():.2f}')

blob = {}
for a in ALPHAS:
    key = f'X_a{int(a * 100):03d}'
    blob[key] = G + a * c * w_hat
    print(f'{key} (alpha={a}): mean||a||='
          f'{blob[key].norm(dim=-1).mean():.2f}')
blob.update({'w': w, 'c': c, 'alphas': ALPHAS})
out = os.path.join(DATA_DIR, 'gauss_shift_dose_anchors.pt')
torch.save(blob, out)
print(f'Saved {out}')
