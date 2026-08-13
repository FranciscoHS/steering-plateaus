#!/usr/bin/env python3
"""Rotation variant of the separating-normal shift (qwen3-1.7b L2).

Instead of translating gauss points by c*w_hat, rotate each point toward
w_hat at fixed norm: per point a, let w_perp = normalize(w_hat - (w_hat.a_hat)a_hat)
(nearly w_hat itself, since cos(a, w_hat) is small), then

    a_theta = ||a|| * (cos(theta) * a_hat + sin(theta) * w_perp)

Norm is exactly preserved. Uses the same fresh Gaussian rows 200:400 and the
same plane as the translation experiments.

Output: results/intrinsic_dim/gauss_rotated_anchors.pt
  X_rot10, X_rot20, X_rot30   rotations by 10/20/30 degrees
"""

import os
import pickle

import torch

DATA_DIR = 'results/intrinsic_dim'
THETAS_DEG = [10.0, 20.0, 30.0]
N_POINTS = 200

b = torch.load(os.path.join(DATA_DIR, 'gauss_shifted_anchors.pt'),
               map_location='cpu', weights_only=False)
w = b['w'].float()
w_hat = w / w.norm()
G = b['X_fresh'].float()

norms = G.norm(dim=-1, keepdim=True)
a_hat = G / norms
comp = (a_hat @ w_hat).unsqueeze(-1)                    # cos(a, w_hat), [N,1]
w_perp = w_hat.unsqueeze(0) - comp * a_hat
w_perp = w_perp / w_perp.norm(dim=-1, keepdim=True)
print(f'|cos(a, w_hat)|: mean={comp.abs().mean():.3f}  '
      f'max={comp.abs().max():.3f}')

blob = {'w': w, 'thetas_deg': THETAS_DEG}
for th in THETAS_DEG:
    t = torch.deg2rad(torch.tensor(th))
    X = norms * (torch.cos(t) * a_hat + torch.sin(t) * w_perp)
    key = f'X_rot{int(th)}'
    blob[key] = X
    print(f'{key}: mean||a||={X.norm(dim=-1).mean():.2f}  '
          f'proj on w_hat: mean={ (X @ w_hat).mean():.2f}')
out = os.path.join(DATA_DIR, 'gauss_rotated_anchors.pt')
torch.save(blob, out)
print(f'Saved {out}')
