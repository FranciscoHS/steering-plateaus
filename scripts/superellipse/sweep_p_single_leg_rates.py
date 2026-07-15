#!/usr/bin/env python3
"""Initial rate of change of p when rotating ONE leg of a pair a little, in
many random tangent directions.

p is a property of a PAIR. To ask "how fast does p change as I move a single
direction", we fix one leg v and rotate the other leg u by a small geodesic
angle theta toward an independent random tangent direction e (one per draw):

    u(theta) = cos(theta) * u0 + sin(theta) * e,
    e = normalize(r - (r.u0) u0),   r ~ uniform unit vector,

and fit the initial slope dp/dtheta of p(u(theta), v0) over a few small
theta (incl. theta=0, whose p is shared across all draws for that pair).

Because random e's are ~orthogonal in high-d, different draws can decay at
very different rates. The histogram of slopes over many (pair, draw) reveals
whether the rate distribution is structured: a cluster near zero = tangent
directions you can rotate along WITHOUT losing p, i.e. the privileged object
is a SUBSPACE, not a single direction.

Run on a GPU pod.

    python scripts/superellipse/sweep_p_single_leg_rates.py \\
        --dict_path .../dictionary.pt --model qwen3-1.7b --layer 14 \\
        --n_pairs 12 --n_targets 40 --rot_degs 0,3,6,9 --n_anchors 20 \\
        --out .../p_single_leg_rates
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

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)
from scripts.superellipse.fit_p_fixed_direction import pair_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_pairs', type=int, default=12,
                    help='within-dict (u,v) pairs; v fixed, u rotated')
    ap.add_argument('--n_targets', type=int, default=40,
                    help='random tangent directions per pair')
    ap.add_argument('--rot_degs', default='0,3,6,9',
                    help='small rotation angles (deg) incl. 0 for the slope fit')
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--n_steps_grid', type=int, default=31)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--pair_seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    rot_degs = np.array([float(x) for x in args.rot_degs.split(',')])
    assert rot_degs[0] == 0.0, 'rot_degs must start at 0'
    rot_rads = np.deg2rad(rot_degs)

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
    print(f'Dictionary: {N} atoms, d_model={d_model}', flush=True)

    print(f'Extracting {args.n_anchors} anchors (seed={args.held_seed})',
          flush=True)
    tok = load_fineweb_fixed_length(args.n_anchors, tokenizer,
                                    seq_len=args.seq_len, seed=args.held_seed)
    ctx, act = extract_context_and_activation(model, tok, args.layer, device)
    contexts = [ctx[i:i + 1].float().cpu() for i in range(ctx.shape[0])]
    activations = act.float().cpu()

    # 2D superellipse-fit sweep grid (NOT the rotation angles).
    fit_degs = np.linspace(0.0, args.max_angle, args.n_steps_grid)
    fit_rads = np.deg2rad(fit_degs)

    rng = np.random.default_rng(args.pair_seed)
    g = torch.Generator(device='cpu').manual_seed(args.pair_seed + 7)

    n_rot = len(rot_degs)
    slopes = np.full((args.n_pairs, args.n_targets), np.nan)
    p_curves = np.full((args.n_pairs, args.n_targets, n_rot), np.nan)
    p0_arr = np.full(args.n_pairs, np.nan)
    pair_idx = np.zeros((args.n_pairs, 2), dtype=int)

    t0 = time.time()
    for pi in range(args.n_pairs):
        a, b = rng.choice(N, size=2, replace=False)
        pair_idx[pi] = (a, b)
        u0 = D[a]
        v0 = D[b]
        # baseline p at theta=0, shared across all targets for this pair
        p0, _ = pair_p(model, contexts, activations, u0, v0,
                       args.layer, measure_layer, device, fit_degs, fit_rads)
        p0_arr[pi] = p0
        for di in range(args.n_targets):
            w = torch.randn(d_model, generator=g, dtype=torch.float32)
            w = w / w.norm()
            e = w - (w * u0).sum() * u0          # tangent component at u0
            e = e / e.norm()
            ps = np.empty(n_rot)
            ps[0] = p0
            for ri in range(1, n_rot):
                th = float(rot_rads[ri])
                u = math.cos(th) * u0 + math.sin(th) * e
                u = u / u.norm()
                p, _ = pair_p(model, contexts, activations, u, v0,
                              args.layer, measure_layer, device,
                              fit_degs, fit_rads)
                ps[ri] = p
            p_curves[pi, di] = ps
            # initial slope dp/dtheta (per degree) via least squares
            slopes[pi, di] = np.polyfit(rot_degs, ps, 1)[0]
        med = np.nanmedian(slopes[pi])
        print(f'  pair {pi+1}/{args.n_pairs} (atoms {a},{b})  p0={p0:.2f}  '
              f'slope med={med:+.4f}/deg  '
              f'frac|slope|<0.01: '
              f'{np.mean(np.abs(slopes[pi])<0.01):.2f}  '
              f'elapsed={time.time()-t0:.0f}s', flush=True)
        with open(os.path.join(args.out, 'p_single_leg_rates.pkl'), 'wb') as f:
            pickle.dump({'args': vars(args), 'rot_degs': rot_degs,
                         'slopes': slopes, 'p_curves': p_curves,
                         'p0': p0_arr, 'pair_idx': pair_idx, 'N': N,
                         'measure_layer': measure_layer}, f)

    print(f'\nDONE in {time.time()-t0:.0f}s. '
          f'overall slope median={np.nanmedian(slopes):+.4f}/deg', flush=True)
    print(f'Saved {os.path.join(args.out, "p_single_leg_rates.pkl")}',
          flush=True)


if __name__ == '__main__':
    main()
