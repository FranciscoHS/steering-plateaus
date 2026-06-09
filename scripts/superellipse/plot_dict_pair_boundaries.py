#!/usr/bin/env python3
"""Boundary panels for specific dictionary atom pairs (by index).

fit_pairs_dictionary.py only plots its first N sampled pairs; this lets us draw
the iso-plateau boundary for arbitrary (i, j) atom pairs -- e.g. the worst-fit
candidate x candidate pairs -- reusing the same per-anchor 2D sweep + exact-
geodesic superellipse panel.

Usage:
    python scripts/superellipse/plot_dict_pair_boundaries.py \\
        --dict_path .../dictionary_snap20k.pt --model qwen3-1.7b --layer 14 \\
        --pairs "10854,20425;12186,13086;1092,16107;8495,11971" \\
        --out .../worst_cc_boundaries
"""
import argparse, os, sys
sys.path.insert(0, '.')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import extract_context_and_activation
from scripts.superellipse.canonical.sweep_core import run_2d_per_anchors
from scripts.superellipse.plot_boundary_pair import boundary_for_set, draw_panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--n_steps_grid', type=int, default=31)
    ap.add_argument('--pairs', required=True,
                    help='semicolon-separated i,j atom-index pairs')
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pairs = [tuple(int(x) for x in p.split(',')) for p in args.pairs.split(';') if p.strip()]
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = args.measure_layer if args.measure_layer is not None else n_layers - 2

    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(device=device, dtype=torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    print(f'Dictionary: {D.shape[0]} atoms; layer {args.layer}->{measure_layer}', flush=True)

    tok = load_fineweb_fixed_length(args.n_anchors, tokenizer, seq_len=args.seq_len, seed=args.held_seed)
    ctx, act = extract_context_and_activation(model, tok, args.layer, device)
    contexts = [ctx[i:i+1].float().cpu() for i in range(ctx.shape[0])]
    activations = act.float().cpu()

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps_grid)
    angles_rad = np.deg2rad(angles_deg)

    for (i, j) in pairs:
        d1 = D[i].cpu(); d2 = D[j].cpu()
        r = run_2d_per_anchors(model, contexts, activations, d1, d2,
                               args.layer, measure_layer, device, angles_rad,
                               mode='geodesic', single_batch_ref=True)
        median_grid = np.median(r['l2'], axis=0)
        b = boundary_for_set(angles_deg, median_grid, exact=True)
        p = b['fit']['p']; npts = b['fit']['n_pts']; dropped = int((~b['keep']).sum())
        print(f'pair ({i},{j}): p_fit={p:.2f}  n_pts={npts}  dropped={dropped}', flush=True)
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        draw_panel(ax, b, f'atoms ({i}, {j})   p={p:.2f}', d1lab=f'atom {i}', d2lab=f'atom {j}')
        plt.tight_layout()
        png = os.path.join(args.out, f'boundary_atoms_{i}_{j}_p{p:.2f}.png')
        fig.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
        fig.savefig(png.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'  saved {png}', flush=True)


if __name__ == '__main__':
    main()
