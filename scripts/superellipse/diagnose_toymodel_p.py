#!/usr/bin/env python3
"""Why does feature x feature give p<2 here but p>2 in fsec-anonymous?

Same phase1_tanh3 checkpoint, several feature pairs (e_j, e_k), p fit several
ways to isolate which methodological knob flips p across 2:

  col polar_abs    : fsec method  -> polar first-crossing of a FIXED ABSOLUTE
                     threshold tau, fine radial grid. (reproduces npz p~3.5)
  col cart_base_big: my method as-run -> Cartesian grid, robust_p_fit at
                     0.5*base (base=min per-axis max within window), alpha_max
                     auto (~27.6).
  col cart_base_sm : my method but small window (alpha_max=8).
  col cart_abs_big : Cartesian grid, FIXED ABSOLUTE thresh, alpha_max~27.6.
  col cart_abs_sm  : Cartesian grid, FIXED ABSOLUTE thresh, alpha_max=8.
  col cart_abs_fine: Cartesian grid 121, FIXED ABSOLUTE thresh, alpha_max=8.

Prints per-column median/IQR over the feature pairs.
"""
import argparse
import math
import os
import sys
sys.path.insert(0, '.')

import numpy as np
import torch

from scripts.superellipse.canonical.superellipse import (
    fit_p_one_pass, fit_superellipse,
)
from scripts.superellipse.fit_pairs_toymodel import load_toy_model


@torch.no_grad()
def polar_p(model, d, S, u, v, tau, alpha_max=8.0, n_alpha=1500, n_phi=80):
    """fsec method: per-angle first-crossing radius of fixed abs tau, fit n."""
    device = next(model.parameters()).device
    x_clean = torch.zeros(d, device=device)
    for s in S:
        x_clean[s] = 1.0
    base = model(x_clean.unsqueeze(0))[0]
    alphas = torch.linspace(0.0, alpha_max, n_alpha, device=device)
    phis = np.linspace(0.0, math.pi / 2, n_phi)
    phis_t = torch.tensor(phis, dtype=torch.float32, device=device)
    dirs = (torch.cos(phis_t).unsqueeze(1) * u.unsqueeze(0)
            + torch.sin(phis_t).unsqueeze(1) * v.unsqueeze(0))   # [n_phi, d]
    x = x_clean.view(1, 1, d) + alphas.view(1, n_alpha, 1) * dirs.view(n_phi, 1, d)
    l2 = (model(x.view(-1, d)) - base.unsqueeze(0)).norm(p=2, dim=1).view(n_phi, n_alpha)
    a_cpu = alphas.cpu().numpy()
    rs = np.full(n_phi, np.nan)
    above = (l2 > tau).int()
    any_above = above.any(dim=1).cpu().numpy()
    first = above.argmax(dim=1).cpu().numpy()
    rs[any_above] = a_cpu[first[any_above]]
    if not (np.isfinite(rs[0]) and np.isfinite(rs[-1])):
        return np.nan
    xs = rs * np.cos(phis) / rs[0]
    ys = rs * np.sin(phis) / rs[-1]
    m = (xs > 1e-3) & (ys > 1e-3) & np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 10:
        return np.nan
    return fit_superellipse(xs[m], ys[m])['p']


@torch.no_grad()
def cart_grid(model, d, S, u, v, alpha_max, n_grid):
    """Cartesian L2 grid in physical alpha (angles == physical alpha)."""
    device = next(model.parameters()).device
    x_clean = torch.zeros(d, device=device)
    for s in S:
        x_clean[s] = 1.0
    base = model(x_clean.unsqueeze(0))[0]
    a = torch.linspace(0.0, alpha_max, n_grid, device=device)
    ai = a.view(n_grid, 1, 1) * u.view(1, 1, -1)
    aj = a.view(1, n_grid, 1) * v.view(1, 1, -1)
    x = (x_clean.view(1, 1, -1) + ai + aj).reshape(n_grid * n_grid, -1)
    l2 = (model(x) - base.unsqueeze(0)).norm(p=2, dim=1).view(n_grid, n_grid)
    return a.cpu().numpy(), l2.cpu().numpy()


def cart_p_base(angles, grid, frac=0.5, max_alpha=None):
    base = min(grid[:, 0].max(), grid[0, :].max())
    thresh = frac * base
    ma = max_alpha if max_alpha is not None else float(angles.max())
    return fit_p_one_pass(angles, grid, thresh, ma, exact_geodesic=False)['p']


def cart_p_abs(angles, grid, tau, max_alpha=None):
    ma = max_alpha if max_alpha is not None else float(angles.max())
    return fit_p_one_pass(angles, grid, tau, ma, exact_geodesic=False)['p']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='~/Research/denoising/runs/phase1_tanh3/model.pt')
    ap.add_argument('--denoising_repo', default='~/Research/denoising')
    ap.add_argument('--n_pairs', type=int, default=30)
    ap.add_argument('--tau', type=float, default=0.1, help='fixed absolute L2 threshold')
    ap.add_argument('--alpha_big', type=float, default=27.556)
    ap.add_argument('--alpha_small', type=float, default=8.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    model, cfg = load_toy_model(args.ckpt, args.denoising_repo, device)
    d = cfg['d']
    S = (0, 1)
    rng = np.random.default_rng(args.seed)
    inactive = [i for i in range(d) if i not in S]

    cols = {k: [] for k in ['polar_abs', 'cart_base_big', 'cart_base_sm',
                            'cart_abs_big', 'cart_abs_sm', 'cart_abs_fine']}
    for _ in range(args.n_pairs):
        j, k = rng.choice(inactive, size=2, replace=False)
        u = torch.zeros(d, device=device); u[j] = 1.0
        v = torch.zeros(d, device=device); v[k] = 1.0

        cols['polar_abs'].append(polar_p(model, d, S, u, v, args.tau,
                                         alpha_max=args.alpha_small))

        a_big, g_big = cart_grid(model, d, S, u, v, args.alpha_big, 31)
        a_sm, g_sm = cart_grid(model, d, S, u, v, args.alpha_small, 31)
        a_fine, g_fine = cart_grid(model, d, S, u, v, args.alpha_small, 121)

        cols['cart_base_big'].append(cart_p_base(a_big, g_big))
        cols['cart_base_sm'].append(cart_p_base(a_sm, g_sm))
        cols['cart_abs_big'].append(cart_p_abs(a_big, g_big, args.tau))
        cols['cart_abs_sm'].append(cart_p_abs(a_sm, g_sm, args.tau))
        cols['cart_abs_fine'].append(cart_p_abs(a_fine, g_fine, args.tau))

    print(f'\nphase1_tanh3  d={d}  S={S}  n_pairs={args.n_pairs}  '
          f'tau_abs={args.tau}  alpha_big={args.alpha_big}  '
          f'alpha_small={args.alpha_small}\n')
    print(f'{"column":16s} {"median":>8s} {"mean":>8s} {"IQR_lo":>8s} {"IQR_hi":>8s} {"n":>4s}')
    for name, vals in cols.items():
        v = np.array(vals, float); v = v[np.isfinite(v)]
        if len(v):
            print(f'{name:16s} {np.median(v):8.3f} {np.mean(v):8.3f} '
                  f'{np.quantile(v,.25):8.3f} {np.quantile(v,.75):8.3f} {len(v):4d}')
        else:
            print(f'{name:16s} {"--":>8s} (no finite fits)')


if __name__ == '__main__':
    main()
