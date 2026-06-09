#!/usr/bin/env python3
"""p versus rotation away from a candidate direction (trained toy model).

Hold one candidate (optimized atom) direction u = atom_a fixed. Take a second
candidate v = atom_b and rotate it along the geodesic (slerp) toward a target:
  - a true feature  e_f   (target='feature')   -> plot 1
  - a random direction    (target='random')    -> plot 2
At each rotation fraction t in [0, 1] we fit the superellipse exponent p of the
(u, v(t)) boundary. t=0 is candidate x candidate; t=1 is candidate x target.

We repeat over many (atom_a, atom_b, target) triples and plot p vs t with the
median +- IQR band, mirroring random_sparse_misalignment_sweep.py's
lp_vs_angle. Reuses the trained-TwoLayerNet machinery from fit_pairs_toymodel.py
and the atom loader from plot_boundary_toymodel.py.

Usage:
    python scripts/superellipse/sweep_p_vs_rotation_toymodel.py \\
        --tau 0.2 --n_pairs 20 --n_t 13 --device cpu
"""
import argparse
import math
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.superellipse.fit_pairs_toymodel import (
    load_toy_model, boundary_radii, fit_from_radii,
)
from scripts.superellipse.plot_boundary_toymodel import load_atoms, feature_vec, make_unit


def slerp(a, b, t):
    """Geodesic interpolation between unit vectors a, b on the sphere.
    t=0 -> a, t=1 -> b. Returns a unit vector."""
    dot = float(torch.clamp((a * b).sum(), -1.0, 1.0))
    omega = math.acos(dot)
    if omega < 1e-6:
        return a.clone(), omega
    so = math.sin(omega)
    v = (math.sin((1.0 - t) * omega) / so) * a + (math.sin(t * omega) / so) * b
    return v / v.norm(), omega


def run_target(model, d, S, A, target_mode, args, rng, device, phis, ts):
    """Sweep p vs rotation for one target type. Returns (ps[n_pairs, n_t],
    omegas[n_pairs])."""
    K = A.shape[0]
    inactive = [i for i in range(d) if i not in S]
    ps = np.full((args.n_pairs, len(ts)), np.nan)
    omegas = np.full(args.n_pairs, np.nan)

    for pi in range(args.n_pairs):
        a, b = rng.choice(K, size=2, replace=False)
        u = A[a]
        atom_b = A[b]
        if target_mode == 'feature':
            f = int(rng.choice(inactive))
            target = feature_vec(f, d, device)
        else:  # random direction, S-zeroed
            w = rng.standard_normal(d); w[list(S)] = 0.0
            target = make_unit(w, d, device)
        for ti, t in enumerate(ts):
            v, omega = slerp(atom_b, target, float(t))
            omegas[pi] = math.degrees(omega)
            rs = boundary_radii(model, d, S, u, v, phis, [args.tau],
                                args.alpha_max, args.n_alpha)[args.tau]
            ps[pi, ti] = fit_from_radii(rs, phis)
        if (pi + 1) % 5 == 0 or pi == 0:
            print(f'  [{target_mode}] pair {pi+1}/{args.n_pairs} '
                  f'(atoms {a},{b}) omega={omegas[pi]:.1f}deg  '
                  f'p[t=0]={ps[pi,0]:.2f} p[t=1]={ps[pi,-1]:.2f}', flush=True)
    return ps, omegas


def plot_target(ts, ps, omegas, target_mode, tau, out):
    med = np.nanmedian(ps, axis=0)
    q1 = np.nanquantile(ps, 0.25, axis=0)
    q3 = np.nanquantile(ps, 0.75, axis=0)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for pi in range(ps.shape[0]):
        ax.plot(ts, ps[pi], '-', color='#888888', lw=0.6, alpha=0.30)
    ax.fill_between(ts, q1, q3, color='#3a7d3a', alpha=0.22, label='IQR')
    ax.plot(ts, med, '-o', color='#3a7d3a', lw=2.2, ms=4, label='median p')
    ax.axhline(2.0, color='#1F77B4', ls='--', alpha=0.7, label='p=2 (Euclidean)')
    ax.set_xlabel(rf'rotation fraction $t$  (0 = candidate$\times$candidate, '
                  rf'1 = candidate$\times${target_mode})', fontsize=11)
    ax.set_ylabel('fitted superellipse p', fontsize=12)
    ax.set_title(f'p vs rotation candidate -> {target_mode}   '
                 f'(tau={tau}, N={ps.shape[0]} pairs, '
                 f'median geodesic angle={np.nanmedian(omegas):.0f} deg)',
                 fontsize=10.5)
    ax.grid(True, color='#cccccc', lw=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, loc='best')
    plt.tight_layout()
    png = os.path.join(out, f'p_vs_rotation_candidate_to_{target_mode}_tau{tau:g}.png')
    fig.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(png.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  saved {png}', flush=True)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='~/Research/denoising/runs/phase1_tanh3/model.pt')
    ap.add_argument('--denoising_repo', default='~/Research/denoising')
    ap.add_argument('--atoms', default='results/dictionary/toymodel_phase1_tanh3/'
                    'optimize_features_randcontrast/atoms.pt')
    ap.add_argument('--tau', type=float, default=0.2)
    ap.add_argument('--n_pairs', type=int, default=20, help='atom pairs per target')
    ap.add_argument('--n_t', type=int, default=13, help='rotation grid points in [0,1]')
    ap.add_argument('--n_phis', type=int, default=80)
    ap.add_argument('--alpha_max', type=float, default=15.0)
    ap.add_argument('--n_alpha', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='results/dictionary/toymodel_phase1_tanh3/p_vs_rotation')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    model, cfg = load_toy_model(args.ckpt, args.denoising_repo, device)
    d = cfg['d']; S = tuple(range(cfg['sparsity']))
    A = load_atoms(args.atoms, d, S, device)
    phis = np.linspace(0.0, math.pi / 2, args.n_phis)
    ts = np.linspace(0.0, 1.0, args.n_t)
    print(f'd={d} K={A.shape[0]} S={S} tau={args.tau} n_pairs={args.n_pairs} '
          f'n_t={args.n_t}', flush=True)

    results = {}
    for target_mode in ['feature', 'random']:
        rng = np.random.default_rng(args.seed)  # same atom pairs for both targets
        ps, omegas = run_target(model, d, S, A, target_mode, args, rng, device, phis, ts)
        plot_target(ts, ps, omegas, target_mode, args.tau, args.out)
        results[target_mode] = {'ps': ps, 'omegas': omegas}

    with open(os.path.join(args.out, f'p_vs_rotation_tau{args.tau:g}.pkl'), 'wb') as f:
        pickle.dump({'ts': ts, 'tau': args.tau, 'args': vars(args),
                     'results': results}, f)
    print(f'saved {os.path.join(args.out, f"p_vs_rotation_tau{args.tau:g}.pkl")}', flush=True)


if __name__ == '__main__':
    main()
