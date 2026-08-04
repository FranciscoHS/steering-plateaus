#!/usr/bin/env python3
"""Full-circle 2D heatmaps at freely Jacobian-optimized Gaussian points.

Anchors come from intrinsic_dim_jac_optimized.py results: cov-matched
Gaussian samples in qwen3-1.7b layer-2 activation space, gradient-descended
in the FULL 2048-dim space (no basis constraint) to minimize ||J||_F with a
light distance penalty. For each anchor type — optimized point, its Gaussian
starting point, and the real activation sharing the same carrier context —
we sweep the tangent plane along pairs of random directions and plot the
full-circle L2 heatmap, as in plot_fullcircle_pair_heatmap.py.

Question: do low-Jacobian synthetic points show the plateau-like circular
low-response region that real in-distribution activations show?

Contexts are re-extracted exactly as the jac_filtered/jac_optimized scripts
did (FineWeb-Edu sample-10BT streaming, shuffle seed 42, seq_len 10); the
rebuilt real activations are verified against the stored X_real rows.

Usage:
    python scripts/jacobian/plot_fullcircle_jacopt_heatmap.py \\
        --out results/fullcircle_heatmaps/jacopt_L2
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

from src.model import load_model, _get_blocks
from scripts.superellipse.plot_fullcircle_pair_heatmap import (
    sweep_full_grid, plot_grid,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opt_data', default='results/intrinsic_dim/jac_optimized_data.pt')
    ap.add_argument('--filtered_data', default='results/intrinsic_dim/jac_filtered_data.pt')
    ap.add_argument('--source_layer', type=int, default=2)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--n_anchors', type=int, default=5,
                    help='anchors per type (optimized / gaussian / real)')
    ap.add_argument('--types', default='opt,gauss,real',
                    help="comma-separated anchor types to sweep "
                         "(subset of opt,gauss,real)")
    ap.add_argument('--n_dir_pairs', type=int, default=2)
    ap.add_argument('--dir_seed', type=int, default=0)
    ap.add_argument('--max_angle', type=float, default=90.0)
    ap.add_argument('--n_steps_grid', type=int, default=61)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--skip_check', action='store_true',
                    help='proceed even if rebuilt contexts do not match the '
                         'stored real activations (anchors are then paired '
                         'with DIFFERENT contexts than they were optimized '
                         'for)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if args.n_steps_grid % 2 == 0:
        raise SystemExit('--n_steps_grid must be odd')
    os.makedirs(args.out, exist_ok=True)

    opt = torch.load(args.opt_data, map_location='cpu', weights_only=False)
    filt = torch.load(args.filtered_data, map_location='cpu', weights_only=False)
    X_opt = opt['X_opt'].float()
    X_gauss = opt['X_gauss_subset'].float()
    X_real = filt['X_real'].float()
    # history['jac'] holds per-step MEANS over the 200 points (scalars), so
    # per-point final norms are not available — label all opt anchors with
    # the final mean.
    jac_opt_final = float(opt['history']['jac'][-1])
    jac_real = filt['real_jac_norms']
    jac_gauss = filt['gauss_jac_norms']

    # fp32 like the optimization run (anchors were optimized under fp32).
    print('Loading Qwen/Qwen3-1.7B (float32)', flush=True)
    model, tokenizer, device = load_model('Qwen/Qwen3-1.7B',
                                          dtype=torch.float32)
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    print(f'Perturb L{args.source_layer} -> measure L{measure_layer}',
          flush=True)

    # Rebuild carrier contexts exactly as intrinsic_dim_jac_filtered.py did.
    print('Streaming FineWeb-Edu for contexts...', flush=True)
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceFW/FineWeb-Edu', 'sample-10BT',
                      split='train', streaming=True)
    ds = ds.shuffle(seed=42)
    texts = []
    for ex in ds:
        if len(ex['text']) >= args.seq_len * 3:
            texts.append(ex['text'])
        if len(texts) >= args.n_anchors:
            break
    tokens = tokenizer(texts, return_tensors='pt', padding='max_length',
                       truncation=True, max_length=args.seq_len).to(device)
    with torch.no_grad():
        out = model(**tokens, output_hidden_states=True)
    hidden = out.hidden_states[args.source_layer + 1]
    contexts = hidden[:, :-1, :].float().cpu()
    real_now = hidden[:, -1, :].float().cpu()

    cos = F.cosine_similarity(real_now, X_real[:args.n_anchors], dim=-1)
    print(f'Rebuilt real activations vs stored: cos min={cos.min():.6f}',
          flush=True)
    if cos.min() < 0.99:
        if args.skip_check:
            print('WARNING: context mismatch, proceeding anyway '
                  '(--skip_check)', flush=True)
        else:
            raise SystemExit('Context reconstruction mismatch — aborting')

    d_model = X_opt.shape[1]
    g = torch.Generator().manual_seed(args.dir_seed)
    dir_pairs = []
    for k in range(args.n_dir_pairs):
        M = torch.randn(d_model, 2, generator=g)
        Q, _ = torch.linalg.qr(M)
        dir_pairs.append((Q[:, 0].contiguous(), Q[:, 1].contiguous()))

    angles_deg = np.linspace(-args.max_angle, args.max_angle,
                             args.n_steps_grid)
    angles_rad = np.deg2rad(angles_deg)

    saved = {'args': vars(args), 'angles_deg': angles_deg, 'grids': {}}
    for i in range(args.n_anchors):
        ctx = contexts[i:i + 1]
        anchor_set = [
            ('opt', X_opt[i], jac_opt_final),
            ('gauss', X_gauss[i], float(jac_gauss[i])),
            ('real', X_real[i], float(jac_real[i])),
        ]
        anchor_set = [a for a in anchor_set
                      if a[0] in args.types.split(',')]
        for name, act, jn in anchor_set:
            for k, (d1, d2) in enumerate(dir_pairs):
                t0 = time.time()
                with torch.no_grad():
                    grid = sweep_full_grid(model, ctx, act, d1, d2,
                                           args.source_layer, measure_layer,
                                           device, angles_rad,
                                           batch_size=args.batch_size)
                saved['grids'][(name, i, k)] = grid
                print(f'{name} anchor{i} pair{k}: ||J||={jn:.0f} '
                      f'||a||={act.norm():.1f} {time.time() - t0:.0f}s',
                      flush=True)
                plot_grid(grid, angles_deg, f'random dir {2*k}',
                          f'random dir {2*k+1}', measure_layer,
                          f'{name} anchor {i} (||J||_F={jn:.0f}, '
                          f'||a||={act.norm():.0f}) — random pair {k}, '
                          f'L{args.source_layer}',
                          os.path.join(args.out,
                                       f'heatmap_{name}_a{i}_pair{k}.png'))

    pkl_path = os.path.join(args.out, 'jacopt_grids.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(saved, f)
    print(f'saved {pkl_path}', flush=True)


if __name__ == '__main__':
    main()
