#!/usr/bin/env python3
"""Full-circle 2D heatmaps at min-Jacobian reconstructed activations.

Takes results.pkl files from reconstruct_min_jacobian.py (which contain a
Jacobian-minimized synthetic activation, plus the real and random-activation
references for the same context) and sweeps the tangent plane at each of
these anchors along pairs of RANDOM directions, exactly like
plot_fullcircle_pair_heatmap.py does for direction pairs at real anchors.

Question: does a synthetic activation whose ||J||_F matches real activations
also show the plateau-like circular/elliptical low-response region that real
in-distribution activations show?

The context is rebuilt from the reconstruction run's args (same FineWeb
load: n_load / seq_len / seed / act_idx), and the recomputed real activation
is checked against the stored one.

Usage:
    python scripts/jacobian/plot_fullcircle_minjac_heatmap.py \\
        --results_pkls results/reconstruct_min_jacobian/qwen3-1.7b_L7_27dirs/results.pkl \\
                       results/reconstruct_min_jacobian/qwen3-1.7b_L7_27dirs_random/results.pkl \\
        --out results/fullcircle_heatmaps/minjac_L7
"""
import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')

from src.model import load_model
from src.data import load_fineweb_fixed_length
from scripts.superellipse.plot_fullcircle_pair_heatmap import (
    sweep_full_grid, plot_grid,
)
from src.model import _get_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_pkls', nargs='+', required=True,
                    help='results.pkl files from reconstruct_min_jacobian.py '
                         '(must share model/layer/context args)')
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--max_angle', type=float, default=90.0)
    ap.add_argument('--n_steps_grid', type=int, default=61)
    ap.add_argument('--n_dir_pairs', type=int, default=2,
                    help='number of random direction pairs per anchor')
    ap.add_argument('--dir_seed', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--tokens_cache', default=None,
                    help='explicit FineWeb token-cache .pt to rebuild the '
                         'context from (for runs that predate the '
                         'tokenizer-keyed cache names)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if args.n_steps_grid % 2 == 0:
        raise SystemExit('--n_steps_grid must be odd')
    os.makedirs(args.out, exist_ok=True)

    blobs = []
    for p in args.results_pkls:
        with open(p, 'rb') as f:
            blobs.append((p, pickle.load(f)))

    # Context args must agree across pkls (they share real/random anchors).
    r0 = blobs[0][1]
    ra = r0['args']
    model_name = ra.get('model') or r0['melbo_args']['model']
    layer = (ra.get('layer') if ra.get('layer') is not None
             else r0['melbo_args']['inject_layer'])
    for p, r in blobs[1:]:
        for k in ('act_idx', 'seed', 'seq_len', 'n_load'):
            if r['args'].get(k) != ra.get(k):
                raise SystemExit(f'{p}: context arg {k} differs from first pkl')

    print(f'Model {model_name}, perturb layer {layer}', flush=True)
    model, tokenizer, device = load_model(model_name)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    print(f'Measure layer {measure_layer}', flush=True)

    # Rebuild the context the reconstruction used (act_idx-th sequence of the
    # same deterministic FineWeb load).
    if args.tokens_cache:
        token_ids_list = torch.load(args.tokens_cache, weights_only=True)
        print(f'Loaded {len(token_ids_list)} sequences from '
              f'{args.tokens_cache}', flush=True)
    else:
        token_ids_list = load_fineweb_fixed_length(
            ra['n_load'], tokenizer, seq_len=ra['seq_len'], seed=ra['seed'])
    tokens = token_ids_list[ra['act_idx']].unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)
        hidden = out.hidden_states[layer + 1]
    ctx = hidden[:, :-1, :].float().cpu()
    real_act_now = hidden[0, -1, :].float().cpu()

    real_stored = r0['real_activation'].float()
    cos_check = F.cosine_similarity(real_act_now.unsqueeze(0),
                                    real_stored.unsqueeze(0)).item()
    print(f'Recomputed real activation vs stored: cos={cos_check:.6f}', flush=True)
    if cos_check < 0.999:
        raise SystemExit('Context reconstruction mismatch — aborting')

    # Anchor set: each pkl's reconstructed activation, labeled by its basis,
    # plus the shared real and random anchors (from the first pkl).
    anchors = []
    for p, r in blobs:
        a = r['args']
        basis = ('random' if a.get('random_directions')
                 else 'covmatched' if a.get('cov_matched') else 'melbo')
        jn = r['jacobian_norms']['reconstructed'][0]
        anchors.append((f'recon_{basis}', r['reconstructed_activation'].float(),
                        jn))
    anchors.append(('real', real_stored,
                    r0['jacobian_norms']['real'][0]))
    anchors.append(('random', r0['random_activation'].float(),
                    r0['jacobian_norms']['random'][0]))

    # Random orthonormal direction pairs (shared across anchors).
    g = torch.Generator().manual_seed(args.dir_seed)
    dir_pairs = []
    for k in range(args.n_dir_pairs):
        M = torch.randn(d_model, 2, generator=g)
        Q, _ = torch.linalg.qr(M)
        dir_pairs.append((Q[:, 0].contiguous(), Q[:, 1].contiguous()))

    angles_deg = np.linspace(-args.max_angle, args.max_angle,
                             args.n_steps_grid)
    angles_rad = np.deg2rad(angles_deg)

    saved = {'args': vars(args), 'angles_deg': angles_deg,
             'context_args': ra, 'grids': {}}
    for name, act, jn in anchors:
        for k, (d1, d2) in enumerate(dir_pairs):
            t0 = time.time()
            with torch.no_grad():
                grid = sweep_full_grid(model, ctx, act, d1, d2,
                                       layer, measure_layer, device,
                                       angles_rad,
                                       batch_size=args.batch_size)
            saved['grids'][(name, k)] = grid
            print(f'{name} pair{k}: ||J||={jn:.0f}, '
                  f'{time.time() - t0:.0f}s', flush=True)
            plot_grid(grid, angles_deg, f'random dir {2*k}',
                      f'random dir {2*k+1}', measure_layer,
                      f'{name} anchor (||J||_F={jn:.0f}) — '
                      f'random pair {k}, L{layer}',
                      os.path.join(args.out, f'heatmap_{name}_pair{k}.png'))

    pkl_path = os.path.join(args.out, 'minjac_grids.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(saved, f)
    print(f'saved {pkl_path}', flush=True)


if __name__ == '__main__':
    main()
