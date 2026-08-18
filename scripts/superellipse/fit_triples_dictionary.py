#!/usr/bin/env python3
"""Fit superellipsoid-p for triples of directions (hyperspherical rays).

3-direction generalization of fit_pairs_dictionary.py. For each triple
(d1, d2, d3), Gram-Schmidt the three directions against the anchor and each
other, sweep rays

    d_hat(phi, theta) = sin(phi)cos(theta) d1p + sin(phi)sin(theta) d2p
                        + cos(phi) d3p,       phi, theta in [0, 90] deg,

each at geodesic radii r in [0, max_angle] (exp-map, same chart as the 2D
sweeps), and record the downstream L2 response. Protocol mirrors the pair
fits:

  - threshold T = threshold_frac * min over the 3 axis rays of max_r R;
  - per-ray first-crossing intercept r*(phi, theta), linear interp;
  - exact-geodesic normalization x_i = (sin r* / sin t_i) * u_i with t_i the
    axis intercepts and u_i the ray's direction cosines;
  - fit p in |x1|^p + |x2|^p + |x3|^p = 1 on interior rays
    (all coords in (0.05, 1.5)), bounded scalar minimization;
  - median over held-out anchors; bootstrap over anchors for the CI.

The coordinate-plane rays (phi=90 row; theta=0 and theta=90 columns) give
the three 2D face fits from the same sweep, so each triple also reports its
pairwise p's (d1-d2, d1-d3, d2-d3) for the pair-vs-triple comparison.

Usage:
    python scripts/superellipse/fit_triples_dictionary.py \\
        --dict_path .../dictionary.pt --model qwen3-1.7b --layer 14 \\
        --n_triples 50 --n_anchors 20 --out results/fit_triples/<tag>
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar

from src.model import load_model, _get_blocks, forward_from_layer_to_layer
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)
from scripts.superellipse.canonical.superellipse import axis_intercept


# ---------- geometry ----------

def build_rays(n_phi, n_theta):
    """Deduplicated ray direction cosines on the (phi, theta) octant grid.

    Returns (U [n_rays, 3] float64, axis_idx dict with keys 1, 2, 3).
    u = (sin phi cos theta, sin phi sin theta, cos phi); the phi=0 row
    collapses to the single d3-axis ray.
    """
    phis = np.deg2rad(np.linspace(0.0, 90.0, n_phi))
    thetas = np.deg2rad(np.linspace(0.0, 90.0, n_theta))
    rays, seen = [], set()
    for phi in phis:
        for th in thetas:
            u = (np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th),
                 np.cos(phi))
            key = tuple(np.round(u, 9))
            if key in seen:
                continue
            seen.add(key)
            rays.append(u)
    U = np.array(rays, dtype=np.float64)
    axis_idx = {}
    for k, target in enumerate(np.eye(3)):
        d = np.abs(U - target).sum(axis=1)
        i = int(np.argmin(d))
        if d[i] > 1e-6:
            raise RuntimeError(f'axis ray {k+1} not on the grid')
        axis_idx[k + 1] = i
    return U, axis_idx


def gram_schmidt_tangent(a_hat, dirs):
    """Orthonormalize `dirs` [k, d] against a_hat and each other (in order)."""
    out = []
    basis = [a_hat]
    for d in dirs:
        v = d.clone()
        for b in basis:
            v = v - (v @ b) * b
        n = v.norm()
        if n < 1e-8:
            raise RuntimeError('degenerate direction in Gram-Schmidt')
        v = v / n
        basis.append(v)
        out.append(v)
    return torch.stack(out)


def sweep_rays_one_anchor(model, ctx, anchor, dirs, perturb_layer,
                          measure_layer, device, U, radii_rad,
                          batch_size=256):
    """L2 response grid [n_rays, n_radii] for one anchor.

    dirs: [3, d] raw directions (orthonormalized here, in order).
    Reference is the r=0 point (exp-map at r=0 returns the anchor exactly),
    so radii_rad must contain 0.
    """
    model_dtype = next(model.parameters()).dtype
    a32 = anchor.to(device=device, dtype=torch.float32)
    R = a32.norm()
    a_hat = a32 / R
    D = gram_schmidt_tangent(a_hat, [d.to(device=device, dtype=torch.float32)
                                     for d in dirs])           # [3, d]
    Ut = torch.from_numpy(U.astype(np.float32)).to(device)      # [n_rays, 3]
    T = Ut @ D                                                  # [n_rays, d]

    radii_t = torch.from_numpy(radii_rad.astype(np.float32)).to(device)
    cos_r = torch.cos(radii_t)                                  # [n_r]
    sin_r = torch.sin(radii_t)
    # [n_rays, n_r, d]
    pts = R * (cos_r.view(1, -1, 1) * a_hat.view(1, 1, -1)
               + sin_r.view(1, -1, 1) * T.unsqueeze(1))
    n_rays, n_r, d_model = pts.shape
    pts_flat = pts.reshape(-1, d_model).to(model_dtype)

    ctx_dev = ctx.to(device=device, dtype=model_dtype)
    meas = torch.empty(pts_flat.shape[0], d_model, dtype=torch.float32,
                       device=device)
    for st in range(0, pts_flat.shape[0], batch_size):
        ed = min(st + batch_size, pts_flat.shape[0])
        meas[st:ed] = forward_from_layer_to_layer(
            model, ctx_dev, pts_flat[st:ed].unsqueeze(1),
            perturb_layer, measure_layer).float()

    zero_idx = int(np.argmin(np.abs(radii_rad)))
    ref = meas[0 * n_r + zero_idx].unsqueeze(0)
    l2 = (meas - ref).norm(dim=-1).cpu().numpy()
    return l2.reshape(n_rays, n_r)


# ---------- fitting (numpy, no model) ----------

def fit_superellipsoid(X, bounds=(0.5, 10.0)):
    """Fit p in sum_i |x_i|^p = 1 to normalized points X [m, 3].

    Interior mask: all coords in (0.05, 1.5), matching the 2D convention.
    Returns dict with p, mean_radial_frac, max_radial_frac, n_pts.
    """
    X = np.asarray(X, dtype=np.float64)
    mask = np.all((X > 0.05) & (X < 1.5), axis=1)
    X = X[mask]
    if len(X) < 4:
        return {'p': np.nan, 'mean_radial_frac': np.nan,
                'max_radial_frac': np.nan, 'n_pts': int(len(X))}

    def res(p):
        return float(np.sum((np.sum(np.abs(X) ** p, axis=1) - 1.0) ** 2))

    r = minimize_scalar(res, bounds=bounds, method='bounded')
    p_fit = float(r.x)
    lp = np.sum(np.abs(X) ** p_fit, axis=1) ** (1.0 / p_fit)
    frac = np.abs(lp - 1.0)
    return {'p': p_fit, 'mean_radial_frac': float(frac.mean()),
            'max_radial_frac': float(frac.max()), 'n_pts': int(len(X))}


def fit_superellipse_2d(x, y, bounds=(0.5, 10.0)):
    """2D face fit, same convention as canonical fit_superellipse."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = (x > 0.05) & (y > 0.05) & (x < 1.5) & (y < 1.5)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {'p': np.nan, 'n_pts': int(len(x))}

    def res(p):
        return float(np.sum((np.abs(x) ** p + np.abs(y) ** p - 1.0) ** 2))

    r = minimize_scalar(res, bounds=bounds, method='bounded')
    return {'p': float(r.x), 'n_pts': int(len(x))}


def fit_triple_from_grid(radii_deg, R_grid, U, axis_idx, threshold_frac=0.5,
                         edge_eps_deg=1.0):
    """Full fit pipeline on one anchor-reduced response grid [n_rays, n_r].

    Returns dict with the 3D fit, the three 2D face fits, threshold,
    axis intercepts, and the normalized points.
    """
    ax = [axis_idx[1], axis_idx[2], axis_idx[3]]
    base = min(R_grid[a].max() for a in ax)
    T = threshold_frac * base
    t = np.array([axis_intercept(radii_deg, R_grid[a], T) for a in ax])
    out = {'threshold': float(T), 'base': float(base), 'intercepts_deg': t,
           'fit3': {'p': np.nan, 'n_pts': 0}, 'faces': {},
           'n_rays_crossed': 0}
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        return out

    # per-ray intercepts
    r_star = np.array([axis_intercept(radii_deg, R_grid[k], T)
                       for k in range(R_grid.shape[0])])
    grid_max = float(radii_deg.max())
    ok = np.isfinite(r_star) & (r_star < grid_max - edge_eps_deg)
    out['n_rays_crossed'] = int(ok.sum())

    # exact-geodesic normalized coords x_i = (sin r* / sin t_i) * u_i
    sin_t = np.sin(np.deg2rad(t))                             # [3]
    X = (np.sin(np.deg2rad(r_star))[:, None] / sin_t[None, :]) * U
    X_ok = X[ok]
    U_ok = U[ok]
    out['r_star_deg'] = r_star
    out['X'] = X_ok

    out['fit3'] = fit_superellipsoid(X_ok)

    # face fits from the coordinate-plane rays (third cosine == 0)
    for (i, j, k), name in (((0, 1, 2), 'd1d2'), ((0, 2, 1), 'd1d3'),
                            ((1, 2, 0), 'd2d3')):
        plane = np.abs(U_ok[:, k]) < 1e-9
        out['faces'][name] = fit_superellipse_2d(X_ok[plane, i],
                                                 X_ok[plane, j])
    return out


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', default=None,
                    help='required if any leg sources from "dict"')
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_triples', type=int, default=50)
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--triple_seed', type=int, default=0)
    ap.add_argument('--atom_max_idx', type=int, default=None)
    ap.add_argument('--atom_min_idx', type=int, default=0)
    ap.add_argument('--leg1_source', default='dict',
                    choices=['dict', 'scaffold', 'random'])
    ap.add_argument('--leg2_source', default='dict',
                    choices=['dict', 'scaffold', 'random'])
    ap.add_argument('--leg3_source', default='dict',
                    choices=['dict', 'scaffold', 'random'])
    ap.add_argument('--random_seed', type=int, default=12345,
                    help="seed for 'random' leg direction sampling")
    ap.add_argument('--max_abs_cos', type=float, default=None,
                    help='resample triples until all pairwise |cos| are '
                         'below this (None = no filter)')
    ap.add_argument('--n_phi', type=int, default=9,
                    help='phi grid points in [0, 90] deg (incl. endpoints)')
    ap.add_argument('--n_theta', type=int, default=9)
    ap.add_argument('--n_radii', type=int, default=31,
                    help='geodesic radii in [0, max_angle] (incl. 0)')
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--threshold_frac', type=float, default=0.5)
    ap.add_argument('--n_bootstrap', type=int, default=100,
                    help='anchor-bootstrap reps per triple (0 = point only)')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.triple_seed)
    leg_sources = (args.leg1_source, args.leg2_source, args.leg3_source)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    d_model = model._plateau_info['d_model']
    print(f'Device: {device} d_model={d_model} '
          f'L{args.layer}->{measure_layer}', flush=True)

    # Direction pools
    pools = {}
    if 'dict' in leg_sources:
        if args.dict_path is None:
            raise SystemExit("a leg uses 'dict' but --dict_path not given")
        blob = torch.load(args.dict_path, map_location='cpu',
                          weights_only=False)
        D = blob['dictionary'].float()
        D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
        N = D.shape[0]
        amax = min(args.atom_max_idx if args.atom_max_idx is not None else N, N)
        amin = max(0, args.atom_min_idx)
        pools['dict'] = D[amin:amax]
        print(f'Dictionary: {N} atoms, sampling [{amin}, {amax})', flush=True)
    if 'scaffold' in leg_sources:
        from scripts.superellipse.learn_feature_dir_vs_contrastive import (
            load_or_build_scaffold,
        )
        from scripts.superellipse.build_dictionary import SEMANTIC_NAMES
        S = load_or_build_scaffold(model, tokenizer, device, args.model,
                                   args.layer, SEMANTIC_NAMES)
        pools['scaffold'] = (S / (S.norm(dim=-1, keepdim=True) + 1e-12)).cpu()
        print(f'Scaffold: {pools["scaffold"].shape[0]} dirs', flush=True)
    if 'random' in leg_sources:
        gr = torch.Generator().manual_seed(args.random_seed)
        Rd = torch.randn(max(args.n_triples * 3, 1024), d_model, generator=gr)
        pools['random'] = Rd / (Rd.norm(dim=-1, keepdim=True) + 1e-12)

    # Anchors
    print(f'Extracting {args.n_anchors} anchors (seed={args.held_seed})',
          flush=True)
    tok = load_fineweb_fixed_length(args.n_anchors, tokenizer,
                                    seq_len=args.seq_len,
                                    seed=args.held_seed)
    ctx, act = extract_context_and_activation(model, tok, args.layer, device)
    contexts = [ctx[i:i + 1].float().cpu() for i in range(ctx.shape[0])]
    activations = act.float().cpu()

    # Rays + radii
    U, axis_idx = build_rays(args.n_phi, args.n_theta)
    radii_deg = np.linspace(0.0, args.max_angle, args.n_radii)
    radii_rad = np.deg2rad(radii_deg)
    n_interior = int(np.sum(np.all(U > 1e-9, axis=1)))
    print(f'Rays: {U.shape[0]} total ({n_interior} interior), '
          f'{args.n_radii} radii to {args.max_angle} deg '
          f'-> {U.shape[0] * args.n_radii} pts/anchor', flush=True)

    # Sample triples: (idx1, idx2, idx3) into the leg pools; distinct indices
    # whenever two legs share a pool.
    def sample_triple():
        while True:
            idxs = [int(rng.integers(pools[s].shape[0])) for s in leg_sources]
            coll = any(leg_sources[a] == leg_sources[b] and idxs[a] == idxs[b]
                       for a in range(3) for b in range(a + 1, 3))
            if coll:
                continue
            ds = [pools[leg_sources[k]][idxs[k]] for k in range(3)]
            cos = [float(ds[a] @ ds[b])
                   for a, b in ((0, 1), (0, 2), (1, 2))]
            if (args.max_abs_cos is not None
                    and max(abs(c) for c in cos) > args.max_abs_cos):
                continue
            return tuple(idxs), ds, cos

    triples = [sample_triple() for _ in range(args.n_triples)]
    print(f'Sampled {len(triples)} triples '
          f'(legs: {"/".join(leg_sources)})', flush=True)

    per_triple = []
    t_start = time.time()
    boot_rng = np.random.RandomState(args.triple_seed + 1)
    for n, (idxs, ds, cos) in enumerate(triples):
        grids = np.zeros((len(activations), U.shape[0], args.n_radii),
                         dtype=np.float32)
        for k in range(len(activations)):
            with torch.no_grad():
                grids[k] = sweep_rays_one_anchor(
                    model, contexts[k], activations[k], ds,
                    args.layer, measure_layer, device, U, radii_rad,
                    batch_size=args.batch_size)

        med = np.median(grids, axis=0)
        fit = fit_triple_from_grid(radii_deg, med, U, axis_idx,
                                   threshold_frac=args.threshold_frac)

        p_boot = []
        for _ in range(args.n_bootstrap):
            bi = boot_rng.randint(0, grids.shape[0], size=grids.shape[0])
            fb = fit_triple_from_grid(radii_deg, np.median(grids[bi], axis=0),
                                      U, axis_idx,
                                      threshold_frac=args.threshold_frac)
            if np.isfinite(fb['fit3']['p']):
                p_boot.append(fb['fit3']['p'])
        p_boot = np.array(p_boot)
        p_lo = float(np.percentile(p_boot, 5)) if len(p_boot) else np.nan
        p_hi = float(np.percentile(p_boot, 95)) if len(p_boot) else np.nan

        face_p = {k: v['p'] for k, v in fit['faces'].items()}
        per_triple.append({
            'idxs': idxs, 'cos': cos,
            'p3': fit['fit3']['p'], 'p3_lo': p_lo, 'p3_hi': p_hi,
            'n_pts3': fit['fit3']['n_pts'],
            'mean_radial_frac': fit['fit3']['mean_radial_frac'],
            'faces': face_p,
            'intercepts_deg': fit['intercepts_deg'],
            'threshold': fit['threshold'],
            'n_rays_crossed': fit['n_rays_crossed'],
            'r_star_deg': fit.get('r_star_deg'),
            'grids': grids,
        })

        done = n + 1
        rate = (time.time() - t_start) / done
        eta = rate * (len(triples) - done)
        fstr = ' '.join(f'{k}={v:.2f}' for k, v in face_p.items())
        print(f'[{done:>3}/{len(triples)}] {idxs}  '
              f'cos=({cos[0]:+.2f},{cos[1]:+.2f},{cos[2]:+.2f})  '
              f'p3={fit["fit3"]["p"]:.3f} [{p_lo:.2f},{p_hi:.2f}] '
              f'(n={fit["fit3"]["n_pts"]})  faces: {fstr}  '
              f'rate={rate:.1f}s/triple ETA={eta/60:.1f}min', flush=True)

        # incremental checkpoint (grids included; ~100KB/triple)
        with open(os.path.join(args.out, 'fit_triples.pkl'), 'wb') as f:
            pickle.dump({'args': vars(args), 'radii_deg': radii_deg,
                         'U': U, 'axis_idx': axis_idx,
                         'per_triple': per_triple}, f)

    # Summary
    p3 = np.array([t['p3'] for t in per_triple], dtype=float)
    finite = np.isfinite(p3)
    faces_all = np.array([[t['faces'][k] for k in ('d1d2', 'd1d3', 'd2d3')]
                          for t in per_triple], dtype=float)
    p3_med = float(np.median(p3[finite])) if finite.any() else np.nan
    print('\n=== Triple-level summary ===')
    print(f'  p3:      median={p3_med:.3f}  '
          f'IQR=({np.nanquantile(p3, 0.25):.3f}, '
          f'{np.nanquantile(p3, 0.75):.3f})  n={int(finite.sum())}')
    print(f'  face p:  median={np.nanmedian(faces_all):.3f}  '
          f'(per-face medians: '
          f'{np.nanmedian(faces_all[:, 0]):.2f}/'
          f'{np.nanmedian(faces_all[:, 1]):.2f}/'
          f'{np.nanmedian(faces_all[:, 2]):.2f})')

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.hist(p3[finite], bins=25, edgecolor='black', alpha=0.8, label='p3')
    ax.hist(faces_all[np.isfinite(faces_all)].ravel(), bins=25,
            histtype='step', lw=2, color='C1', label='face p (2D)')
    ax.axvline(2.0, color='gray', ls=':', label='p=2')
    ax.axvline(p3_med, color='C0', lw=2)
    ax.set_xlabel('fitted p'); ax.set_ylabel('# triples')
    ax.set_title(f'p3 over {len(per_triple)} triples '
                 f'(legs {"/".join(leg_sources)})')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    face_mean = np.nanmean(faces_all, axis=1)
    ax.scatter(face_mean, p3, s=20, alpha=0.7)
    lim = [0.5, max(np.nanmax(p3), np.nanmax(face_mean), 4) * 1.05]
    ax.plot(lim, lim, color='gray', ls=':', lw=1)
    ax.set_xlabel('mean face p (pairwise, same sweep)')
    ax.set_ylabel('p3 (triple)')
    ax.set_title('triple p vs mean pairwise p')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{args.model} L{args.layer}->{measure_layer}  '
                 f'triple fits (n_anchors={args.n_anchors}, '
                 f'{U.shape[0]} rays x {args.n_radii} radii)', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'p3_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved {os.path.join(args.out, "fit_triples.pkl")}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
