#!/usr/bin/env python3
"""Fit superellipse-p for each found dictionary atom against the *exact*
random contrast directions it was optimized against.

The random-contrast build (build_dictionary.py --contrast_source random) did
NOT store the per-atom contrast dirs, but they are deterministically
reproducible: random_contrast(k) seeds a CUDA generator with
seed*contrast_seed_base + k and draws [n_contrast, d_model] unit vectors.

This regenerates them and fits p for every (atom d_k, its contrast d_c) pair.
Low p here would mean the optimizer actually achieved sub-additivity against
the directions it saw; p~2 would mean the proxy didn't translate into a
super/sub-additive iso-contour even on its own training partners.

Usage:
    python scripts/superellipse/fit_atom_vs_owncontrast.py \\
        --dict_dir results/dictionary/qwen3-1.7b_L14_K10_randcontrast \\
        --out results/fit_atom_vs_owncontrast_K10
"""

import argparse
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
from scripts.superellipse.canonical.sweep_core import run_2d_per_anchors
from scripts.superellipse.canonical.superellipse import (
    robust_p_fit, fit_superellipse,
)
from scripts.superellipse.plot_boundary_pair import boundary_for_set
from scripts.superellipse.fit_pairs_dictionary import (
    summarize_fit, normalized_points,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_dir', required=True,
                    help='dir with dictionary.pt + metadata.pkl from the '
                         'random-contrast build')
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--n_steps_grid', type=int, default=31)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--n_bootstrap', type=int, default=100)
    ap.add_argument('--n_pool_bootstrap', type=int, default=2000)
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
    d_model = model._plateau_info['d_model']
    print(f'Device: {device} d_model={d_model} L{args.layer}->{measure_layer}',
          flush=True)

    # Dictionary + build args (seed/contrast_seed_base/n_contrast).
    blob = torch.load(os.path.join(args.dict_dir, 'dictionary.pt'),
                      map_location='cpu')
    D = blob['dictionary'].to(device=device, dtype=torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    N = D.shape[0]
    with open(os.path.join(args.dict_dir, 'metadata.pkl'), 'rb') as f:
        meta = pickle.load(f)
    bargs = meta['args']
    seed = bargs['seed']
    csb = bargs['contrast_seed_base']
    n_contrast = bargs['n_contrast']
    assert bargs['contrast_source'] == 'random', \
        'this script only applies to random-contrast builds'
    print(f'Dictionary: {N} atoms  | regen contrast: seed={seed} '
          f'seed_base={csb} n_contrast={n_contrast}', flush=True)

    # Must match build_dictionary.random_contrast exactly (CUDA generator).
    def random_contrast(k):
        g = torch.Generator(device=device).manual_seed(seed * csb + k)
        dc = torch.randn(n_contrast, d_model, generator=g, device=device)
        return (dc / dc.norm(dim=-1, keepdim=True)).to(torch.float32)

    # Anchors: ctx + act from held-out FineWeb (same as fit_pairs_dictionary).
    print(f'Extracting {args.n_anchors} anchors (seed={args.held_seed})',
          flush=True)
    tok = load_fineweb_fixed_length(args.n_anchors, tokenizer,
                                    seq_len=args.seq_len, seed=args.held_seed)
    ctx, act = extract_context_and_activation(model, tok, args.layer, device)
    contexts = [ctx[i:i + 1].float().cpu() for i in range(ctx.shape[0])]
    activations = act.float().cpu()
    print(f'  ctx per item={tuple(contexts[0].shape)}  '
          f'activations={tuple(activations.shape)}', flush=True)

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps_grid)
    angles_rad = np.deg2rad(angles_deg)

    per_pair = []
    p_fits, p_los, p_his = [], [], []
    pool_xs, pool_ys = [], []
    t_start = time.time()
    total = N * n_contrast
    done = 0
    for k in range(N):
        contrast = random_contrast(k)            # [n_contrast, d]
        d1 = D[k].cpu()
        for m in range(n_contrast):
            d2 = contrast[m].cpu()
            r = run_2d_per_anchors(model, contexts, activations, d1, d2,
                                   args.layer, measure_layer, device,
                                   angles_rad, mode='geodesic',
                                   single_batch_ref=True)
            l2_grid = r['l2']
            median_grid = np.median(l2_grid, axis=0)
            fit = robust_p_fit(angles_deg, l2_grid,
                               n_bootstrap=args.n_bootstrap,
                               exact_geodesic=True)
            p_lo, p_hi = summarize_fit(fit)
            b = boundary_for_set(angles_deg, median_grid, exact=True)
            p_fit = b['fit']['p']
            cos_ij = float((d1.to(device) * d2.to(device)).sum().item())
            per_pair.append({
                'atom': k, 'contrast_m': m, 'cos': cos_ij,
                'p_fit': p_fit, 'p_lo': p_lo, 'p_hi': p_hi,
                'n_pts': int(b['fit']['n_pts']),
            })
            p_fits.append(p_fit); p_los.append(p_lo); p_his.append(p_hi)
            xs, ys = normalized_points(angles_deg, median_grid)
            pool_xs.append(xs); pool_ys.append(ys)
            done += 1
            if done % 10 == 0 or done == 1:
                rate = (time.time() - t_start) / done
                eta = rate * (total - done)
                print(f'  [{done:>3}/{total}] atom {k}-c{m}  cos={cos_ij:+.3f}'
                      f'  p_fit={p_fit:.3f} [{p_lo:.3f},{p_hi:.3f}]  '
                      f'rate={rate:.1f}s/pair  ETA={eta/60:.1f}min', flush=True)

    p_fits = np.array(p_fits, float)
    finite = np.isfinite(p_fits)
    p_med = float(np.median(p_fits[finite]))
    p_iqr = (float(np.quantile(p_fits[finite], 0.25)),
             float(np.quantile(p_fits[finite], 0.75)))

    valid = [i for i in range(len(pool_xs)) if pool_xs[i].size > 0]
    xs_all = (np.concatenate([pool_xs[i] for i in valid]) if valid
              else np.zeros(0))
    ys_all = (np.concatenate([pool_ys[i] for i in valid]) if valid
              else np.zeros(0))
    p_pool = (float(fit_superellipse(xs_all, ys_all)['p'])
              if xs_all.size else float('nan'))
    bs_rng = np.random.default_rng(0)
    p_boot = []
    for _ in range(args.n_pool_bootstrap):
        idx = bs_rng.integers(0, len(valid), len(valid))
        xs_b = np.concatenate([pool_xs[valid[i]] for i in idx])
        ys_b = np.concatenate([pool_ys[valid[i]] for i in idx])
        if xs_b.size:
            p_boot.append(fit_superellipse(xs_b, ys_b)['p'])
    p_boot = np.array(p_boot, float)
    p_pool_lo = float(np.quantile(p_boot, 0.025)) if p_boot.size else float('nan')
    p_pool_hi = float(np.quantile(p_boot, 0.975)) if p_boot.size else float('nan')

    cos_all = np.array([pp['cos'] for pp in per_pair])
    print('\n=== Atom-vs-own-contrast summary ===')
    print(f'  pairs={int(finite.sum())}  |cos| mean={np.abs(cos_all).mean():.4f}'
          f'  max={np.abs(cos_all).max():.4f}')
    print(f'  per-pair p_fit:  median={p_med:.3f}  IQR=({p_iqr[0]:.3f}, '
          f'{p_iqr[1]:.3f})  min={np.min(p_fits[finite]):.3f}  '
          f'max={np.max(p_fits[finite]):.3f}')
    print(f'  pooled-points p: {p_pool:.3f}  95% CI [{p_pool_lo:.3f}, '
          f'{p_pool_hi:.3f}]  (bootstrap n={args.n_pool_bootstrap})')

    out_pkl = os.path.join(args.out, 'fit_atom_vs_owncontrast.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'args': vars(args), 'build_args': bargs,
            'angles_deg': angles_deg, 'per_pair': per_pair,
            'p_fits': p_fits, 'p_median': p_med, 'p_iqr': p_iqr,
            'p_pool': p_pool, 'p_pool_ci_95': (p_pool_lo, p_pool_hi),
            'measure_layer': measure_layer,
        }, f)
    print(f'\nSaved {out_pkl}', flush=True)


if __name__ == '__main__':
    main()
