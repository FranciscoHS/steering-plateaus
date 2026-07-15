#!/usr/bin/env python3
"""Compare the two superellipse-p fit procedures on identical pairs.

For the same random sample of alive-alive atom pairs from each dictionary
(popt and baseline), fit p with BOTH pipelines:
  - grid fit: pair_p (fit_p_fixed_direction) — 31x31 angle grid, 20 anchors
    (seed 2024), contour extraction, box-filter, least-squares. The canonical
    estimator behind the rotation/build-order plots.
  - ray fit: soft_p_estimate (soft_p_proxy) — 13 rays x 40 radii, 8 anchors
    (seed 999), soft threshold crossing, Gauss-Newton (p in [0.5,10]). The
    estimator behind eval_sae_full_p (2.13 / 2.83 numbers).
Each pipeline runs at its own standard settings, so this reproduces the
historical comparison (pipeline + anchor set differ together).

Usage:
    python -u scripts/superellipse/compare_p_estimators.py \\
        --popt .../l10_lam1/sae_weights.pt \\
        --baseline .../l10_doubled_k32_2M/sae_weights.pt \\
        --align_pkl .../alignment.pkl --n_pairs 50 \\
        --model Qwen/Qwen3-1.7B --layer 10 --out .../compare_p_estimators
"""
import argparse
import importlib.util
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


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

pcull = _load('scripts/dictionary_learning/train_sae_pcull.py', 'pcull')
spp = _load('scripts/superellipse/soft_p_proxy.py', 'soft_p_proxy')


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--popt', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--align_pkl', required=True,
                    help='alignment.pkl with fires_a (popt) / fires_b (baseline)')
    ap.add_argument('--n_pairs', type=int, default=50)
    ap.add_argument('--pair_seed', type=int, default=0)
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--layer', type=int, default=10)
    ap.add_argument('--measure_layer', type=int, default=None)
    # grid-fit settings (fit_p_fixed_direction defaults)
    ap.add_argument('--grid_n_anchors', type=int, default=20)
    ap.add_argument('--grid_seed', type=int, default=2024)
    ap.add_argument('--n_steps_grid', type=int, default=31)
    # ray-fit settings (eval_sae_full_p defaults)
    ap.add_argument('--ray_n_anchor', type=int, default=8)
    ap.add_argument('--ray_seed', type=int, default=999)
    ap.add_argument('--n_rays', type=int, default=13)
    ap.add_argument('--n_radii', type=int, default=40)
    ap.add_argument('--p_cap', type=float, default=10.0)
    ap.add_argument('--chunk', type=int, default=512)
    # shared
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    print(f'model={args.model} L{args.layer}->{measure_layer}', flush=True)

    al = pickle.load(open(os.path.expanduser(args.align_pkl), 'rb'))
    dicts = {}
    for label, path, fkey in [('popt', args.popt, 'fires_a'),
                              ('baseline', args.baseline, 'fires_b')]:
        W = torch.load(path, map_location='cpu',
                       weights_only=False)['state_dict']['W_dec'].float()
        W = W / (W.norm(dim=-1, keepdim=True) + 1e-12)
        alive = np.where(np.asarray(al[fkey]) > 0)[0]
        dicts[label] = (W, alive)
        print(f'{label}: {W.shape[0]} atoms, {alive.size} alive', flush=True)

    rng = np.random.default_rng(args.pair_seed)
    pairs = {}
    for label, (W, alive) in dicts.items():
        idx = rng.choice(alive, size=(args.n_pairs, 2), replace=True)
        resample = idx[:, 0] == idx[:, 1]
        while resample.any():
            idx[resample, 1] = rng.choice(alive, size=int(resample.sum()))
            resample = idx[:, 0] == idx[:, 1]
        pairs[label] = idx

    # anchors: grid pipeline (its standard 20 anchors, seed 2024)
    tok = load_fineweb_fixed_length(args.grid_n_anchors, tokenizer,
                                    seq_len=args.seq_len, seed=args.grid_seed)
    ctx_g, act_g = extract_context_and_activation(model, tok, args.layer, device)
    contexts_g = [ctx_g[k:k + 1].float().cpu() for k in range(ctx_g.shape[0])]
    acts_g = act_g.float().cpu()

    # anchors: ray pipeline (its standard 8 anchors, seed 999)
    acts, ctxs, f_refs = pcull.build_anchor_pool(
        model, tokenizer, args.layer, args.ray_n_anchor, args.seq_len,
        args.ray_seed, device, measure_layer)
    acts_r = torch.stack(acts)
    ctx_r = torch.cat(ctxs, 0)
    frefs_r = torch.stack(f_refs)
    phis_rad, radii_rad = spp.default_phis_radii(
        args.n_rays, args.n_radii, args.max_angle)

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps_grid)
    angles_rad = np.deg2rad(angles_deg)

    results = {}
    t0 = time.time()
    for label, (W, _) in dicts.items():
        idx = pairs[label]
        p_grid = np.full(args.n_pairs, np.nan)
        p_ray = np.full(args.n_pairs, np.nan)
        for k in range(args.n_pairs):
            i, j = int(idx[k, 0]), int(idx[k, 1])
            u, v = W[i], W[j]
            p_grid[k], _ = pair_p(model, contexts_g, acts_g, u, v,
                                  args.layer, measure_layer, device,
                                  angles_deg, angles_rad)
            _, diag = spp.soft_p_estimate(
                model, acts_r, ctx_r, frefs_r, u.to(device),
                v.unsqueeze(0).to(device), args.layer, measure_layer,
                phis_rad, radii_rad, p_cap=args.p_cap,
                anchor_reduce='median', chunk=args.chunk, return_diag=True)
            p_ray[k] = float(diag['p_k'][0].item())
            if (k + 1) % 10 == 0:
                print(f'  {label} {k+1}/{args.n_pairs}  '
                      f'elapsed={time.time()-t0:.0f}s', flush=True)
        results[label] = {'pairs': idx, 'p_grid': p_grid, 'p_ray': p_ray}
        with open(os.path.join(args.out, 'compare_p_estimators.pkl'), 'wb') as f:
            pickle.dump({'args': vars(args), 'results': results,
                         'measure_layer': measure_layer}, f)

    print('\n=== medians (same pairs, both procedures) ===')
    for label, r in results.items():
        g, ry = r['p_grid'], r['p_ray']
        fin = np.isfinite(g) & np.isfinite(ry)
        print(f'  {label}: grid median={np.nanmedian(g):.3f} '
              f'IQR({np.nanquantile(g,.25):.3f},{np.nanquantile(g,.75):.3f})  |  '
              f'ray median={np.nanmedian(ry):.3f} '
              f'IQR({np.nanquantile(ry,.25):.3f},{np.nanquantile(ry,.75):.3f})  |  '
              f'per-pair corr={np.corrcoef(g[fin], ry[fin])[0,1]:.3f}  '
              f'(n_finite grid={np.isfinite(g).sum()}, ray={np.isfinite(ry).sum()})')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (label, r) in zip(axes, results.items()):
        g, ry = r['p_grid'], r['p_ray']
        ax.scatter(g, ry, s=25, alpha=0.7)
        lim = [0.5, max(np.nanmax(g), np.nanmax(ry)) * 1.05]
        ax.plot(lim, lim, 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('p (grid fit: pair_p, 20 anchors seed 2024)')
        ax.set_ylabel('p (ray fit: soft_p_estimate, 8 anchors seed 999)')
        ax.set_title(f'{label}: same {args.n_pairs} alive-alive pairs')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'compare_p_estimators.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {plot_path}', flush=True)


if __name__ == '__main__':
    main()
