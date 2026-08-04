#!/usr/bin/env python3
"""Full-circle 2D response heatmaps for direction pairs.

For each pair (d1, d2), sweep the tangent plane at each anchor over a full
square grid of exp-map angles (alpha1, alpha2) in [-max_angle, +max_angle]
(all four quadrants, unlike the Q1-only fit sweeps) and record the L^2
distance of the measurement-layer activation vs unperturbed. Plots a heatmap
per pair (median over anchors, plus the first few single anchors), with the
measured iso-threshold contour and — when a fit_pairs cache is given — the
Q1 superellipse (fitted p) and p=2 ellipse overlays, in the style of the
"feature plane" example figure.

Directions come from the same pools as fit_pairs_dictionary.py:
'dict' (a learned dictionary .pt) or 'scaffold' (the fsec contrastive dirs).

Usage:
    python scripts/superellipse/plot_fullcircle_pair_heatmap.py \\
        --dict_path .../dictionary_snap20k.pt \\
        --leg1_source dict --leg2_source dict --pairs "10,15;16,17" \\
        --fit_cache results/dictionary/fit_pairs_snap20k_first20_allpairs/fit_pairs.pkl \\
        --out results/fullcircle_heatmaps/snap20k
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

from src.model import load_model, _get_blocks, forward_from_layer_to_layer
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)
from scripts.superellipse.canonical.parametrize import (
    orthonormalize, exp_map_2d,
)


def sweep_full_grid(model, ctx, anchor, d1, d2, perturb_layer, measure_layer,
                    device, alphas_rad, batch_size=256):
    """L2 grid over the full (alpha1, alpha2) square for one anchor.

    Returns (n, n) float32. Reference is the (0,0) grid point itself
    (exp-map at r=0 returns the anchor exactly), so alphas_rad must
    contain 0.
    """
    model_dtype = next(model.parameters()).dtype
    a32 = anchor.to(device=device, dtype=torch.float32)
    a_hat = a32 / a32.norm()
    d1p, d2p = orthonormalize(a_hat, d1.to(device=device, dtype=torch.float32),
                              d2.to(device=device, dtype=torch.float32))
    alphas_t = torch.from_numpy(alphas_rad.astype(np.float32)).to(device)
    A1, A2 = torch.meshgrid(alphas_t, alphas_t, indexing='ij')
    pts = exp_map_2d(a32, d1p, d2p, A1.flatten(), A2.flatten()).to(model_dtype)

    n = len(alphas_rad)
    ctx_dev = ctx.to(device=device, dtype=model_dtype)
    meas = torch.empty(pts.shape[0], a32.shape[-1], dtype=torch.float32,
                       device=device)
    for st in range(0, pts.shape[0], batch_size):
        ed = min(st + batch_size, pts.shape[0])
        meas[st:ed] = forward_from_layer_to_layer(
            model, ctx_dev, pts[st:ed].unsqueeze(1),
            perturb_layer, measure_layer).float()

    zero_idx = int(np.argmin(np.abs(alphas_rad)))
    ref = meas[zero_idx * n + zero_idx].unsqueeze(0)
    l2 = (meas - ref).norm(dim=-1).cpu().numpy()
    return l2.reshape(n, n)


def lookup_fit(fit_cache, pair):
    """Return (p_fit, intercepts_deg, threshold) for `pair` from a
    fit_pairs.pkl cache, using the '50'-level cell with the largest
    max_alpha_deg. None if the pair is absent."""
    for e in fit_cache['per_pair']:
        if tuple(e['pair']) not in (tuple(pair), tuple(pair[::-1])):
            continue
        best = None
        for key, cell in e.get('fit_cells', {}).items():
            if key == '__levels__' or key[0] != '50':
                continue
            if best is None or cell['max_alpha_deg'] > best['max_alpha_deg']:
                best = cell
        if best is None:
            return None
        icpt = best['intercepts']
        if tuple(e['pair']) == tuple(pair[::-1]):
            icpt = icpt[::-1]
        return {'p': e['p_fit'], 'intercepts': icpt,
                'threshold': best['threshold']}
    return None


def superellipse_q1(a, b, p, n=200):
    """Q1 arc of |x/a|^p + |y/b|^p = 1."""
    t = np.linspace(0, np.pi / 2, n)
    x = a * np.cos(t) ** (2.0 / p)
    y = b * np.sin(t) ** (2.0 / p)
    return x, y


def plot_grid(grid, angles_deg, label1, label2, measure_layer, title, path,
              fit=None):
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.pcolormesh(angles_deg, angles_deg, grid.T, shading='auto',
                       cmap='magma')
    plt.colorbar(im, ax=ax, label=f'L2 distance at layer {measure_layer}')
    ax.axhline(0, color='white', lw=0.5, alpha=0.4)
    ax.axvline(0, color='white', lw=0.5, alpha=0.4)
    ax.plot(0, 0, 'wo', ms=4)

    # Iso contour: cache threshold if available, else 50% of the smaller
    # positive-axis-end response (same '50'-level convention as the fits).
    if fit is not None:
        threshold = fit['threshold']
    else:
        threshold = 0.5 * min(grid[-1, len(grid) // 2],
                              grid[len(grid) // 2, -1])
    cs = ax.contour(angles_deg, angles_deg, grid.T, levels=[threshold],
                    colors='white', linewidths=1.5)
    handles = [plt.Line2D([], [], color='white', label='measured iso contour')]
    if fit is not None:
        a, b = fit['intercepts']
        x, y = superellipse_q1(a, b, fit['p'])
        ax.plot(x, y, '--', color='cyan', lw=2,
                label=f"superellipse (p={fit['p']:.2f})")
        x2, y2 = superellipse_q1(a, b, 2.0)
        ax.plot(x2, y2, ':', color='lime', lw=2, label='ellipse (p=2)')
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles + h, ['measured iso contour'] + l,
              loc='lower left', fontsize=8, framealpha=0.8)

    ax.set_xlabel(f'tangent-plane coordinate 1: {label1} (deg)', fontsize=11)
    ax.set_ylabel(f'tangent-plane coordinate 2: {label2} (deg)', fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  saved {path}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--max_angle', type=float, default=90.0)
    ap.add_argument('--n_steps_grid', type=int, default=61,
                    help='per axis over [-max_angle, max_angle]; odd so 0 '
                         'is on the grid (required for the reference point)')
    ap.add_argument('--dict_path', default=None)
    ap.add_argument('--leg1_source', default='dict',
                    choices=['dict', 'scaffold'])
    ap.add_argument('--leg2_source', default='dict',
                    choices=['dict', 'scaffold'])
    ap.add_argument('--scaffold_names', default=None,
                    help='comma-separated fsec direction names for the '
                         'scaffold pool (default: the 17 SEMANTIC_NAMES); '
                         'pair indices refer to this list')
    ap.add_argument('--pairs', required=True,
                    help="'i,j;k,l' — indices into leg1/leg2 pools")
    ap.add_argument('--fit_cache', default=None,
                    help='fit_pairs.pkl with p fits for these pairs '
                         '(enables superellipse overlay)')
    ap.add_argument('--plot_anchors', type=int, default=4,
                    help='also plot this many single-anchor heatmaps per pair')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if args.n_steps_grid % 2 == 0:
        raise SystemExit('--n_steps_grid must be odd (0 must be on the grid)')
    os.makedirs(args.out, exist_ok=True)
    pairs = [tuple(int(x) for x in p.split(','))
             for p in args.pairs.split(';') if p.strip()]

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    print(f'Device: {device} L{args.layer}->{measure_layer}', flush=True)

    pools = {}
    if 'dict' in (args.leg1_source, args.leg2_source):
        blob = torch.load(args.dict_path, map_location='cpu',
                          weights_only=False)
        D = blob['dictionary'].float()
        D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
        pools['dict'] = (D, [f'atom {i}' for i in range(D.shape[0])])
        print(f'Dictionary: {D.shape[0]} atoms', flush=True)
    if 'scaffold' in (args.leg1_source, args.leg2_source):
        # Deferred import: pulls in the fsec-anonymous repo machinery.
        from scripts.superellipse.learn_feature_dir_vs_contrastive import (
            load_or_build_scaffold,
        )
        from scripts.superellipse.build_dictionary import SEMANTIC_NAMES
        names = (args.scaffold_names.split(',') if args.scaffold_names
                 else list(SEMANTIC_NAMES))
        S = load_or_build_scaffold(model, tokenizer, device, args.model,
                                   args.layer, names)
        S = (S / (S.norm(dim=-1, keepdim=True) + 1e-12)).cpu()
        pools['scaffold'] = (S, names)
        print(f'Scaffold: {S.shape[0]} contrastive dirs', flush=True)

    fit_cache = None
    if args.fit_cache:
        with open(args.fit_cache, 'rb') as f:
            fit_cache = pickle.load(f)

    print(f'Extracting {args.n_anchors} anchors (seed={args.held_seed})',
          flush=True)
    tok = load_fineweb_fixed_length(args.n_anchors, tokenizer,
                                    seq_len=args.seq_len,
                                    seed=args.held_seed)
    ctx, act = extract_context_and_activation(model, tok, args.layer, device)
    contexts = [ctx[i:i + 1].float().cpu() for i in range(ctx.shape[0])]
    activations = act.float().cpu()

    angles_deg = np.linspace(-args.max_angle, args.max_angle,
                             args.n_steps_grid)
    angles_rad = np.deg2rad(angles_deg)

    saved = {'args': vars(args), 'angles_deg': angles_deg, 'per_pair': {}}
    for (i, j) in pairs:
        D1, names1 = pools[args.leg1_source]
        D2, names2 = pools[args.leg2_source]
        d1, d2 = D1[i], D2[j]
        label1, label2 = names1[i], names2[j]
        cos = float(d1 @ d2)
        fit = lookup_fit(fit_cache, (i, j)) if fit_cache else None
        tag = f'{args.leg1_source}{i}_{args.leg2_source}{j}'
        print(f'pair ({i},{j}) [{label1} x {label2}] cos={cos:.3f}'
              + (f' p={fit["p"]:.2f}' if fit else ''), flush=True)

        t0 = time.time()
        grids = np.zeros((len(activations), args.n_steps_grid,
                          args.n_steps_grid), dtype=np.float32)
        for k in range(len(activations)):
            with torch.no_grad():
                grids[k] = sweep_full_grid(
                    model, contexts[k], activations[k], d1, d2,
                    args.layer, measure_layer, device, angles_rad,
                    batch_size=args.batch_size)
        print(f'  sweep done in {time.time() - t0:.0f}s', flush=True)

        med = np.median(grids, axis=0)
        saved['per_pair'][(i, j)] = {
            'grids': grids, 'median': med, 'cos': cos, 'fit': fit,
            'labels': (label1, label2),
        }

        ptxt = f", p={fit['p']:.2f}" if fit else ''
        plot_grid(med, angles_deg, label1, label2, measure_layer,
                  f'{label1} x {label2} — median over {len(grids)} anchors '
                  f'(cos={cos:.2f}{ptxt})',
                  os.path.join(args.out, f'heatmap_{tag}_median.png'),
                  fit=fit)
        for k in range(min(args.plot_anchors, len(grids))):
            plot_grid(grids[k], angles_deg, label1, label2, measure_layer,
                      f'{label1} x {label2} — anchor {k} (cos={cos:.2f}{ptxt})',
                      os.path.join(args.out, f'heatmap_{tag}_anchor{k}.png'),
                      fit=fit)

    pkl_path = os.path.join(args.out, 'fullcircle_grids.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(saved, f)
    print(f'saved {pkl_path}', flush=True)


if __name__ == '__main__':
    main()
