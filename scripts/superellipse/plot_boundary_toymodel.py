#!/usr/bin/env python3
"""Iso-plateau boundary plots for the trained denoising toy model, four cases.

For a chosen direction pair (u, v) we sweep phi in [0, pi/2], take
    dir(phi) = cos(phi)*u + sin(phi)*v,
sweep the radius alpha, and record the first-crossing radius r(phi) where the
output-L2 response first exceeds a fixed absolute threshold tau. We then plot,
in raw (alpha-along-u, alpha-along-v) coordinates:
  - the data boundary points (r cos phi, r sin phi)
  - the p=2 ellipse and p->inf rectangle with semi-axes (r(0), r(pi/2))
  - the fitted superellipse |x/r0|^n + |y/r1|^n = 1

This reuses load_toy_model / boundary_radii / fit_from_radii from
fit_pairs_toymodel.py (the *trained* TwoLayerNet that the optimized atoms came
from), and the panel-drawing style from the denoising repo's plot_toy_boundary.

Four cases, one PNG each:
  feature_vs_feature   u=e_j, v=e_k            (two true features)
  feature_vs_random    u=e_j, v=random         (S-zeroed, orthogonalized to u)
  atom_vs_atom         u=atom_a, v=atom_b      (optimized high-p directions)
  feature_vs_candidate u=e_j, v=atom_a         (true feature vs optimized atom)

Usage:
    python scripts/superellipse/plot_boundary_toymodel.py \\
        --ckpt ~/Research/denoising/runs/phase1_tanh3/model.pt \\
        --denoising_repo ~/Research/denoising \\
        --atoms results/dictionary/toymodel_phase1_tanh3/optimize_features_randcontrast/atoms.pt \\
        --tau 0.2 --device cpu
"""
import argparse
import math
import os
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.superellipse.fit_pairs_toymodel import (
    load_toy_model, boundary_radii, fit_n,
)


def superellipse_first_quadrant(n, n_pts=300):
    """First-quadrant unit superellipse |x|^n + |y|^n = 1, parameterised by
    t in [0, pi/2]: x = cos(t)^(2/n), y = sin(t)^(2/n)."""
    t = np.linspace(0.0, np.pi / 2, n_pts)
    n = max(float(n), 1e-6)
    return np.cos(t) ** (2.0 / n), np.sin(t) ** (2.0 / n)


def make_unit(vec, d, device):
    v = torch.as_tensor(vec, dtype=torch.float32, device=device)
    return v / v.norm()


def feature_vec(idx, d, device):
    v = torch.zeros(d, device=device)
    v[idx] = 1.0
    return v


def orthonormalize(v, u):
    """v made orthogonal to (unit) u and renormalized."""
    v = v - (v @ u) * u
    return v / v.norm()


def load_atoms(path, d, S, device):
    """Load optimized atoms, orthogonalize each against the anchor x_clean and
    renormalize (matches fit_pairs_toymodel_atoms convention)."""
    blob = torch.load(os.path.expanduser(path), map_location=device,
                      weights_only=False)
    A = blob['dictionary'].to(device=device, dtype=torch.float32)   # [K, d]
    x_clean = torch.zeros(d, device=device)
    for s in S:
        x_clean[s] = 1.0
    x_hat = x_clean / x_clean.norm()
    A = A - (A @ x_hat).unsqueeze(-1) * x_hat
    A = A / A.norm(dim=-1, keepdim=True)
    return A


def boundary_pair(model, d, S, u, v, tau, phis, alpha_max, n_alpha):
    """First-crossing radii r(phi) at this single tau, plus fitted n."""
    rs_dict = boundary_radii(model, d, S, u, v, phis, [tau], alpha_max, n_alpha)
    rs = np.asarray(rs_dict[tau], dtype=np.float64)
    if not (np.isfinite(rs[0]) and np.isfinite(rs[-1])):
        raise SystemExit(
            'boundary did not reach tau on at least one axis — '
            'increase --alpha_max or lower --tau')
    r0, r1 = float(rs[0]), float(rs[-1])
    xs_norm = rs * np.cos(phis) / r0
    ys_norm = rs * np.sin(phis) / r1
    n_fit = float(fit_n(xs_norm, ys_norm))
    cos_uv = float((u * v).sum().item())
    return rs, r0, r1, n_fit, cos_uv, xs_norm, ys_norm


def draw_panel(ax, xs_norm, ys_norm, n_fit, r0, r1, ulab, vlab):
    """Normalized boundary: axes are alpha*cos(phi)/r0 and alpha*sin(phi)/r1, so
    the two axis intercepts both sit at 1 and the reference curves are unit
    superellipses."""
    xi, yi = superellipse_first_quadrant(20)
    ax.plot(xi, yi, ls=(0, (1.8, 2.5)), color='#D9822B', lw=1.6, alpha=0.95,
            label=r'Per-feature threshold ($p\to\infty$)')
    xc, yc = superellipse_first_quadrant(2)
    ax.plot(xc, yc, '-', color='#1F77B4', lw=2.6, alpha=0.95,
            label=r'Euclidean ($p=2$)')
    if np.isfinite(n_fit):
        xf, yf = superellipse_first_quadrant(n_fit)
        ax.plot(xf, yf, '-', color='#3a7d3a', lw=2.0, alpha=0.95, zorder=4,
                label=rf'Super-ellipse fit ($p={n_fit:.2f}$)')
    ax.plot(xs_norm, ys_norm, 'o', color='#3a3a3a', markersize=4.0, alpha=0.85,
            markeredgecolor='none', label='Data', zorder=5)
    ax.set_xticks(np.arange(0, 1.18 + 1e-6, 0.2))
    ax.set_yticks(np.arange(0, 1.18 + 1e-6, 0.2))
    ax.grid(True, which='major', color='#bbbbbb', lw=0.5, alpha=0.45)
    ax.set_xlim(-0.03, 1.15)
    ax.set_ylim(-0.03, 1.15)
    ax.set_aspect('equal')
    ax.set_xlabel(rf'$\alpha\cos\varphi\,/\,r_0$ (along {ulab}, $r_0$={r0:.2f})', fontsize=11)
    ax.set_ylabel(rf'$\alpha\sin\varphi\,/\,r_1$ (along {vlab}, $r_1$={r1:.2f})', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, loc='lower left', frameon=True, framealpha=0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='~/Research/denoising/runs/phase1_tanh3/model.pt')
    ap.add_argument('--denoising_repo', default='~/Research/denoising')
    ap.add_argument('--atoms', default='results/dictionary/toymodel_phase1_tanh3/'
                    'optimize_features_randcontrast/atoms.pt')
    ap.add_argument('--tau', type=float, default=0.2,
                    help='fixed absolute L2 threshold')
    ap.add_argument('--n_phis', type=int, default=120)
    ap.add_argument('--alpha_max', type=float, default=15.0)
    ap.add_argument('--n_alpha', type=int, default=2500)
    # representative directions per case
    ap.add_argument('--ff_pair', default='2,3', help='feature x feature (j,k)')
    ap.add_argument('--fr_feature', type=int, default=2, help='feature x random: feature j')
    ap.add_argument('--rand_seed', type=int, default=0, help='random-leg seed')
    ap.add_argument('--aa_pair', default='3,11', help='atom x atom (a,b), high-p')
    ap.add_argument('--fc_feature', type=int, default=2, help='feature x candidate: feature j')
    ap.add_argument('--fc_atom', type=int, default=3, help='feature x candidate: atom a')
    ap.add_argument('--cr_atom', type=int, default=3, help='candidate x random: atom a')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='results/dictionary/toymodel_phase1_tanh3/boundary_plots')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    model, cfg = load_toy_model(args.ckpt, args.denoising_repo, device)
    d = cfg['d']
    S = tuple(range(cfg['sparsity']))
    print(f'd={d} H={cfg["hidden_size"]} act={cfg["activation"]} S={S} '
          f'tau={args.tau} alpha_max={args.alpha_max}', flush=True)

    A = load_atoms(args.atoms, d, S, device)
    phis = np.linspace(0.0, math.pi / 2, args.n_phis)
    rng = np.random.default_rng(args.rand_seed)

    # --- build the four (u, v) cases ---------------------------------------
    jf, kf = (int(x) for x in args.ff_pair.split(','))
    aa, ab = (int(x) for x in args.aa_pair.split(','))

    def random_dir(u):
        """Unit random direction, S-coords zeroed, orthogonalized to u."""
        w = rng.standard_normal(d)
        w[list(S)] = 0.0
        return orthonormalize(make_unit(w, d, device), u)

    # feature x random direction (S-zeroed, orthogonalized to the feature)
    u_fr = feature_vec(args.fr_feature, d, device)
    v_rand = random_dir(u_fr)
    # candidate (atom) x random direction (S-zeroed, orthogonalized to the atom)
    u_cr = A[args.cr_atom]
    v_cr_rand = random_dir(u_cr)

    cases = [
        ('feature_vs_feature', feature_vec(jf, d, device), feature_vec(kf, d, device),
         rf'$e_{{{jf}}}$', rf'$e_{{{kf}}}$'),
        ('feature_vs_random', u_fr, v_rand,
         rf'$e_{{{args.fr_feature}}}$', 'random'),
        ('atom_vs_atom', A[aa], A[ab],
         rf'atom$_{{{aa}}}$', rf'atom$_{{{ab}}}$'),
        ('feature_vs_candidate', feature_vec(args.fc_feature, d, device), A[args.fc_atom],
         rf'$e_{{{args.fc_feature}}}$', rf'atom$_{{{args.fc_atom}}}$'),
        ('candidate_vs_random', u_cr, v_cr_rand,
         rf'atom$_{{{args.cr_atom}}}$', 'random'),
    ]

    for name, u, v, ulab, vlab in cases:
        rs, r0, r1, n_fit, cos_uv, xs_norm, ys_norm = boundary_pair(
            model, d, S, u, v, args.tau, phis, args.alpha_max, args.n_alpha)
        print(f'[{name}] r0={r0:.3f} r1={r1:.3f} p_fit={n_fit:.3f} '
              f'cos(u,v)={cos_uv:+.4f}', flush=True)
        fig, ax = plt.subplots(figsize=(5.8, 5.8))
        draw_panel(ax, xs_norm, ys_norm, n_fit, r0, r1, ulab, vlab)
        ax.set_title(f'{name}   p={n_fit:.2f}  (tau={args.tau})', fontsize=11)
        plt.tight_layout()
        png = os.path.join(args.out, f'boundary_{name}_tau{args.tau:g}.png')
        fig.savefig(png, dpi=220, bbox_inches='tight', facecolor='white')
        fig.savefig(png.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'  saved {png}', flush=True)


if __name__ == '__main__':
    main()
