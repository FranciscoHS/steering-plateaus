#!/usr/bin/env python3
"""p versus rotation away from a candidate direction (real LLM).

LLM port of sweep_p_vs_rotation_toymodel.py. Take a candidate PAIR
(u0 = atom_a, v0 = atom_b) from --dict_path and rotate BOTH legs along the
geodesic (slerp) toward their own independent random unit targets (r_u, r_v):
the "two random directions". At rotation fraction t in [0, 1],
  u(t) = slerp(u0, r_u, t),  v(t) = slerp(v0, r_v, t),
and we fit the superellipse exponent p of the (u(t), v(t)) pair with the
*standard LLM p estimator* (pair_p from fit_p_fixed_direction.py: an
n_anchors-anchor 2D geodesic sweep -> median L2 grid -> boundary_for_set).
t=0 is candidate x candidate; t=1 is random x random.

We repeat over many (atom_a, atom_b) pairs (x n_draws random-target draws) and
plot median p +- IQR vs t, mirroring the toy-model figure. Run the same script
on the optimized dictionary and on the contrastive direction sets to compare
the decay curves.

Run on a GPU pod.

Usage:
    python scripts/superellipse/sweep_p_vs_rotation_llm.py \\
        --dict_path results/dictionary/qwen3-1.7b_L14_scaffoldcontrast_resample/dictionary.pt \\
        --model qwen3-1.7b --layer 14 \\
        --n_pairs 12 --n_targets 2 --n_t 9 --n_anchors 20 \\
        --out results/.../p_vs_rotation_llm_dict
"""
import argparse
import math
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

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)
from scripts.superellipse.fit_p_fixed_direction import pair_p


def slerp(a, b, t):
    """Geodesic interpolation between unit vectors a, b (torch, same device).
    t=0 -> a, t=1 -> b. Returns (unit vector, geodesic angle omega in rad)."""
    dot = float(torch.clamp((a * b).sum(), -1.0, 1.0))
    omega = math.acos(dot)
    if omega < 1e-6:
        return a.clone(), omega
    so = math.sin(omega)
    v = (math.sin((1.0 - t) * omega) / so) * a + (math.sin(t * omega) / so) * b
    return v / v.norm(), omega


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True,
                    help='dictionary.pt whose atoms are the candidate dirs')
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_pairs', type=int, default=20,
                    help='(atom_a, atom_b) candidate pairs')
    ap.add_argument('--n_draws', type=int, default=1,
                    help='random-target draws per pair (each draw = two random '
                         'directions, one per leg)')
    ap.add_argument('--n_t', type=int, default=9,
                    help='rotation grid points in [0,1]')
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--n_steps_grid', type=int, default=31)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--pair_seed', type=int, default=0,
                    help='seed for choosing pairs + random targets')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}

    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    print(f'Device: {device}  L{args.layer}->{measure_layer}', flush=True)

    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(dtype=torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    N, d_model = D.shape
    names = blob.get('names', None)
    print(f'Dictionary: {N} atoms, d_model={d_model}'
          + (f' (labeled)' if names else ''), flush=True)

    print(f'Extracting {args.n_anchors} anchors (seed={args.held_seed})',
          flush=True)
    tok = load_fineweb_fixed_length(args.n_anchors, tokenizer,
                                    seq_len=args.seq_len, seed=args.held_seed)
    ctx, act = extract_context_and_activation(model, tok, args.layer, device)
    contexts = [ctx[i:i + 1].float().cpu() for i in range(ctx.shape[0])]
    activations = act.float().cpu()

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps_grid)
    angles_rad = np.deg2rad(angles_deg)
    ts = np.linspace(0.0, 1.0, args.n_t)

    rng = np.random.default_rng(args.pair_seed)
    g = torch.Generator(device='cpu').manual_seed(args.pair_seed + 7)

    # ps[pair, draw, t]; both legs rotate toward their own random target.
    ps = np.full((args.n_pairs, args.n_draws, args.n_t), np.nan)
    omega_u = np.full((args.n_pairs, args.n_draws), np.nan)
    omega_v = np.full((args.n_pairs, args.n_draws), np.nan)
    pair_idx = np.zeros((args.n_pairs, 2), dtype=int)
    cos_ab = np.full(args.n_pairs, np.nan)

    t0 = time.time()
    for pi in range(args.n_pairs):
        a, b = rng.choice(N, size=2, replace=False)
        pair_idx[pi] = (a, b)
        u0 = D[a]
        v0 = D[b]
        cos_ab[pi] = float((u0 * v0).sum())
        for di in range(args.n_draws):
            wu = torch.randn(d_model, generator=g, dtype=torch.float32)
            wv = torch.randn(d_model, generator=g, dtype=torch.float32)
            r_u = wu / wu.norm()
            r_v = wv / wv.norm()
            for ti, t in enumerate(ts):
                u, om_u = slerp(u0, r_u, float(t))
                v, om_v = slerp(v0, r_v, float(t))
                omega_u[pi, di] = math.degrees(om_u)
                omega_v[pi, di] = math.degrees(om_v)
                p, _ = pair_p(model, contexts, activations, u, v,
                              args.layer, measure_layer, device,
                              angles_deg, angles_rad)
                ps[pi, di, ti] = p
            print(f'  pair {pi+1}/{args.n_pairs} (atoms {a},{b}) '
                  f'draw {di+1}/{args.n_draws}  '
                  f'omega_u={omega_u[pi,di]:.1f} omega_v={omega_v[pi,di]:.1f}deg  '
                  f'p[t=0]={ps[pi,di,0]:.2f} p[t=1]={ps[pi,di,-1]:.2f}  '
                  f'elapsed={time.time()-t0:.0f}s', flush=True)
        with open(os.path.join(args.out, 'p_vs_rotation_llm.pkl'), 'wb') as f:
            pickle.dump({'args': vars(args), 'ts': ts, 'N': N,
                         'measure_layer': measure_layer, 'ps': ps,
                         'omega_u': omega_u, 'omega_v': omega_v,
                         'pair_idx': pair_idx,
                         'cos_ab': cos_ab, 'names': names}, f)

    # --- plot: median +- IQR over flattened (pair, target) vs t ---
    flat = ps.reshape(-1, args.n_t)
    med = np.nanmedian(flat, axis=0)
    q1 = np.nanquantile(flat, 0.25, axis=0)
    q3 = np.nanquantile(flat, 0.75, axis=0)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for row in flat:
        ax.plot(ts, row, '-', color='#888888', lw=0.6, alpha=0.30)
    ax.fill_between(ts, q1, q3, color='#3a7d3a', alpha=0.22, label='IQR')
    ax.plot(ts, med, '-o', color='#3a7d3a', lw=2.2, ms=4, label='median p')
    ax.axhline(2.0, color='#1F77B4', ls='--', alpha=0.7, label='p=2 (Euclidean)')
    ax.set_xlabel(r'rotation fraction $t$  '
                  r'(0 = candidate$\times$candidate, 1 = random$\times$random)',
                  fontsize=11)
    ax.set_ylabel('fitted superellipse p', fontsize=12)
    tag = os.path.basename(os.path.dirname(args.dict_path)) or 'dict'
    med_omega = np.nanmedian(np.concatenate([omega_u.ravel(), omega_v.ravel()]))
    ax.set_title(f'p vs rotation candidate -> random (both legs)   '
                 f'{args.model} L{args.layer}->{measure_layer}\n'
                 f'{tag}  (N={flat.shape[0]} curves, '
                 f'median geo angle={med_omega:.0f}deg, '
                 f'n_anchors={args.n_anchors})', fontsize=9.5)
    ax.grid(True, color='#cccccc', lw=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, loc='best')
    plt.tight_layout()
    png = os.path.join(args.out, 'p_vs_rotation_llm.png')
    fig.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\nSaved {png}', flush=True)
    print(f'Saved {os.path.join(args.out, "p_vs_rotation_llm.pkl")}', flush=True)


if __name__ == '__main__':
    main()
