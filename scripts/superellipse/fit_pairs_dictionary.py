#!/usr/bin/env python3
"""Fit superellipse-p for random pairs of dictionary atoms.

Sanity check: each atom was optimized to break plateaus against scaffold
directions. Does the same break-plateau property hold pairwise WITHIN the
dictionary? If yes, low p (≪ 2) across many random pairs supports the
"these are feature-like" reading; if not, the proxy didn't generalize.

Procedure (template: scripts/grok_mlp/fit_onevsrest_classes_mlp.py):
  - Sample n_pairs random pairs (i, j) from the dictionary.
  - For each pair: per-anchor 2D L2 grid via
    scripts/superellipse/canonical/sweep_core.run_2d_per_anchors.
  - Fit p per pair via robust_p_fit (exact-geodesic normalization);
    record bootstrap range.
  - Aggregate: distribution of per-pair p_fit + pooled-points p with
    bootstrap CI over pairs (matches pool_p_contrastive_mlp.py).

Usage:
    python scripts/superellipse/fit_pairs_dictionary.py \\
        --dict_path .../dictionary_K8000.pt \\
        --model qwen3-1.7b --layer 14 \\
        --n_pairs 100 --n_anchors 20 --held_seed 2024 \\
        --out .../fit_pairs_K8000
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

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)
from scripts.superellipse.learn_feature_dir_vs_contrastive import (
    load_or_build_scaffold,
)
from scripts.superellipse.build_dictionary import SEMANTIC_NAMES
from scripts.superellipse.canonical.sweep_core import run_2d_per_anchors
from scripts.superellipse.canonical.superellipse import (
    robust_p_fit, extract_contour, axis_intercept, fit_superellipse,
)
from scripts.superellipse.plot_boundary_pair import boundary_for_set, draw_panel


def summarize_fit(fit):
    """Collapse robust_p_fit cells into [p_lo, p_hi] over cells. Matches
    fit_superellipse_pair_v2.summarize_fit."""
    ps = []
    for k, v in fit.items():
        if k == '__levels__':
            continue
        if np.isfinite(v['p_lo']) and np.isfinite(v['p_hi']):
            ps.append(v['p_lo']); ps.append(v['p_hi'])
    p_lo = float(min(ps)) if ps else float('nan')
    p_hi = float(max(ps)) if ps else float('nan')
    return p_lo, p_hi


def normalized_points(angles_deg, median_grid):
    """Exact-geodesic normalized boundary points at thresh=0.5*base, masked
    to (0.05, 1.5). Matches pool_p_contrastive_mlp.normalized_points."""
    base = min(median_grid[:, 0].max(), median_grid[0, :].max())
    thresh = 0.5 * base
    t1 = axis_intercept(angles_deg, median_grid[:, 0], thresh)
    t2 = axis_intercept(angles_deg, median_grid[0, :], thresh)
    raw = extract_contour(angles_deg, median_grid, thresh).astype(float)
    if (raw.size == 0 or not np.isfinite(t1) or not np.isfinite(t2)
            or t1 <= 0 or t2 <= 0):
        return np.zeros(0), np.zeros(0)
    r_deg = np.hypot(raw[:, 0], raw[:, 1])
    r_safe = np.where(r_deg > 1e-12, r_deg, 1.0)
    r_rad = np.deg2rad(r_deg)
    xn = (np.sin(r_rad) / np.sin(np.deg2rad(t1))) * (raw[:, 0] / r_safe)
    yn = (np.sin(r_rad) / np.sin(np.deg2rad(t2))) * (raw[:, 1] / r_safe)
    m = (xn > 0.05) & (yn > 0.05) & (xn < 1.5) & (yn < 1.5)
    return xn[m], yn[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', default=None,
                    help='required if any leg sources from "dict"')
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_pairs', type=int, default=100)
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--pair_seed', type=int, default=0)
    ap.add_argument('--atom_max_idx', type=int, default=None,
                    help='restrict atom sampling to indices [atom_min_idx, atom_max_idx)')
    ap.add_argument('--atom_min_idx', type=int, default=0,
                    help='lower bound for dict atom sampling (build-order band)')
    ap.add_argument('--n_steps_grid', type=int, default=31,
                    help='2D sweep grid size per axis')
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--n_bootstrap', type=int, default=100,
                    help='bootstrap reps for per-pair robust_p_fit')
    ap.add_argument('--n_pool_bootstrap', type=int, default=2000,
                    help='bootstrap reps for the pair-level pooled-p CI')
    ap.add_argument('--leg1_source', default='dict',
                    choices=['dict', 'scaffold'],
                    help="first leg of each pair")
    ap.add_argument('--leg2_source', default='dict',
                    choices=['dict', 'scaffold', 'random'],
                    help="second leg of each pair: 'dict', 'scaffold' (one of "
                         "the 17 contrastive dirs), or 'random' (unit Gaussian)")
    ap.add_argument('--random_seed', type=int, default=12345,
                    help="seed for 'random' leg-2 direction sampling")
    ap.add_argument('--out', required=True)
    ap.add_argument('--plot_boundaries', type=int, default=0,
                    help='draw Fig-3-style iso-contour+superellipse panels for '
                         'the first N pairs (reuses plot_boundary_pair.draw_panel)')
    ap.add_argument('--anchor_mode', choices=['held', 'train'], default='held',
                    help="'held': n_anchors held-out FineWeb (default). 'train': "
                         "each pair uses the leg1 dict direction's single "
                         "training activation (anchor-overfitting check)")
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.pair_seed)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    d_model = model._plateau_info['d_model']
    print(f'Device: {device} d_model={d_model} n_layers={n_layers} '
          f'L{args.layer}->{measure_layer}', flush=True)

    # Dictionary (optional: only required if any leg sources from 'dict')
    if 'dict' in (args.leg1_source, args.leg2_source):
        if args.dict_path is None:
            raise SystemExit(
                "leg uses 'dict' but --dict_path was not provided")
        blob = torch.load(args.dict_path, map_location='cpu')
        D = blob['dictionary'].to(device=device, dtype=torch.float32)
        D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
        N = D.shape[0]
        print(f'Dictionary: {N} atoms', flush=True)
    else:
        D = None
        N = 0

    # Anchors: ctx + act from held-out FineWeb
    print(f'Extracting {args.n_anchors} anchors (seed={args.held_seed})',
          flush=True)
    tok = load_fineweb_fixed_length(args.n_anchors, tokenizer,
                                    seq_len=args.seq_len,
                                    seed=args.held_seed)
    ctx, act = extract_context_and_activation(model, tok, args.layer, device)
    # ctx: [B, T-1, d]   act: [B, d]
    contexts = [ctx[i:i + 1].float().cpu() for i in range(ctx.shape[0])]
    activations = act.float().cpu()

    # Train-anchor mode: for each pair use the leg1 dict direction's OWN single
    # training activation (the one it was optimized on), reconstructed from the
    # build's act pool (seed/size in metadata) + per-direction act_idx. Tests
    # whether high p shows up on the training anchor but washes out held-out.
    train_anchor = None     # dict: dict-index -> (ctx[1,T-1,d], act[d])
    if args.anchor_mode == 'train':
        if args.leg1_source != 'dict':
            raise SystemExit("--anchor_mode train requires --leg1_source dict")
        meta_path = os.path.join(os.path.dirname(args.dict_path), 'metadata.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        bargs = meta['args']
        pool_seed = bargs.get('act_pool_seed', 42)
        pool_size = bargs.get('act_pool_size', 8000)
        pool_seqlen = bargs.get('seq_len', args.seq_len)
        print(f'[train-anchor] reconstructing act pool '
              f'(seed={pool_seed} size={pool_size} seq_len={pool_seqlen})',
              flush=True)
        tok_pool = load_fineweb_fixed_length(pool_size, tokenizer,
                                             seq_len=pool_seqlen, seed=pool_seed)
        ctx_pool, act_pool = extract_context_and_activation(
            model, tok_pool, args.layer, device)
        act_idx_of = {a['k']: a['act_idx'] for a in meta['atoms']}
        train_anchor = {}
        for k, ai in act_idx_of.items():
            train_anchor[k] = (ctx_pool[ai:ai + 1].float().cpu(),
                               act_pool[ai].float().cpu())
        print(f'[train-anchor] built single anchors for {len(train_anchor)} '
              f'directions', flush=True)
    print(f'  ctx shape per item={tuple(contexts[0].shape)}  '
          f'activations={tuple(activations.shape)}', flush=True)

    # Optional scaffold (needed if either leg comes from scaffold)
    scaffold = None
    if 'scaffold' in (args.leg1_source, args.leg2_source):
        scaffold = load_or_build_scaffold(model, tokenizer, device, args.model,
                                          args.layer, SEMANTIC_NAMES)
        scaffold = scaffold / (scaffold.norm(dim=-1, keepdim=True) + 1e-12)
        print(f'Scaffold (semantic-only): shape={tuple(scaffold.shape)}',
              flush=True)

    # Optional random pool (for leg2_source='random')
    rand_dirs = None
    if args.leg2_source == 'random':
        gr = torch.Generator(device=device).manual_seed(args.random_seed)
        rand_dirs = torch.randn(args.n_pairs, d_model, generator=gr,
                                device=device, dtype=torch.float32)
        rand_dirs = rand_dirs / (rand_dirs.norm(dim=-1, keepdim=True) + 1e-12)
        print(f'Random unit dirs: shape={tuple(rand_dirs.shape)}', flush=True)

    # Sample pairs.
    # Convention: pair = (idx1, idx2) where idx1 indexes leg1 pool and
    # idx2 indexes leg2 pool.
    atom_max = args.atom_max_idx if args.atom_max_idx is not None else N
    atom_max = min(atom_max, N)
    atom_min = max(0, args.atom_min_idx)
    print(f'Sampling: leg1={args.leg1_source}  leg2={args.leg2_source}  '
          f'(dict idx range=[{atom_min}, {atom_max}))', flush=True)

    def pool_size(src):
        if src == 'dict': return atom_max - atom_min
        if src == 'scaffold': return scaffold.shape[0]
        if src == 'random': return rand_dirs.shape[0]
        raise ValueError(src)

    K1 = pool_size(args.leg1_source)
    K2 = pool_size(args.leg2_source)

    pairs = []
    if args.leg1_source == args.leg2_source:
        # Unordered, no duplicates, i != j
        seen = set()
        max_pairs = K1 * (K1 - 1) // 2
        target = min(args.n_pairs, max_pairs)
        while len(seen) < target:
            i, j = int(rng.integers(K1)), int(rng.integers(K1))
            if i == j:
                continue
            if i > j:
                i, j = j, i
            seen.add((i, j))
        pairs = sorted(seen)
    else:
        for k in range(args.n_pairs):
            pairs.append((int(rng.integers(K1)), int(rng.integers(K2))))
    print(f'Sampled {len(pairs)} pairs (leg2={args.leg2_source})', flush=True)

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps_grid)
    angles_rad = np.deg2rad(angles_deg)

    per_pair = []
    p_fits = []
    p_los, p_his = [], []
    pool_xs, pool_ys = [], []
    boundary_panels = []          # (b, title, lab1, lab2) for first N pairs
    t_start = time.time()
    def get_dir(src, idx):
        if src == 'dict': return D[atom_min + idx].cpu()
        if src == 'scaffold': return scaffold[idx].cpu()
        if src == 'random': return rand_dirs[idx].cpu()
        raise ValueError(src)

    for k, (i, j) in enumerate(pairs):
        d1 = get_dir(args.leg1_source, i)
        d2 = get_dir(args.leg2_source, j)
        if train_anchor is not None:
            ctx_i, act_i = train_anchor[atom_min + i]   # leg1 dict direction
            pair_contexts = [ctx_i]
            pair_activations = act_i.unsqueeze(0)
        else:
            pair_contexts = contexts
            pair_activations = activations
        t0 = time.time()
        r = run_2d_per_anchors(model, pair_contexts, pair_activations, d1, d2,
                               args.layer, measure_layer, device, angles_rad,
                               mode='geodesic', single_batch_ref=True)
        l2_grid = r['l2']                                  # [n_anchors, n, n]
        median_grid = np.median(l2_grid, axis=0)
        fit = robust_p_fit(angles_deg, l2_grid, n_bootstrap=args.n_bootstrap,
                           exact_geodesic=True)
        p_lo, p_hi = summarize_fit(fit)
        b = boundary_for_set(angles_deg, median_grid, exact=True)
        p_fit = b['fit']['p']
        cos_ij = float((d1.to(device) * d2.to(device)).sum().item())
        per_pair.append({
            'pair': (i, j),
            'cos': cos_ij,
            'p_fit': p_fit,
            'p_lo': p_lo,
            'p_hi': p_hi,
            'n_pts': int(b['fit']['n_pts']),
            'fit_cells': fit,
        })
        p_fits.append(p_fit)
        p_los.append(p_lo); p_his.append(p_hi)
        xs, ys = normalized_points(angles_deg, median_grid)
        pool_xs.append(xs); pool_ys.append(ys)
        if k < args.plot_boundaries:
            boundary_panels.append(
                (b, f'{args.leg1_source[:4]}{i} vs {args.leg2_source[:4]}{j}: '
                    f'p={p_fit:.2f} (cos={cos_ij:+.2f})',
                 f'{args.leg1_source[:4]}{i}', f'{args.leg2_source[:4]}{j}'))
        if (k + 1) % 10 == 0 or k == 0:
            done = k + 1
            rate = (time.time() - t_start) / done
            eta = rate * (len(pairs) - done)
            print(f'  [{done:>3}/{len(pairs)}] pair {i:4d}-{j:4d}  '
                  f'cos={cos_ij:+.3f}  p_fit={p_fit:.3f} '
                  f'[{p_lo:.3f},{p_hi:.3f}]  n_pts={b["fit"]["n_pts"]}  '
                  f'rate={rate:.1f}s/pair  ETA={eta/60:.1f}min',
                  flush=True)

    # Pair-level summary
    p_fits = np.array(p_fits, dtype=float)
    p_los = np.array(p_los, dtype=float)
    p_his = np.array(p_his, dtype=float)
    finite = np.isfinite(p_fits)
    p_med = float(np.median(p_fits[finite]))
    p_iqr = (float(np.quantile(p_fits[finite], 0.25)),
             float(np.quantile(p_fits[finite], 0.75)))

    # Pooled-points p with bootstrap over pairs
    pool_x_arr = [xs for xs in pool_xs]
    pool_y_arr = [ys for ys in pool_ys]
    valid = [k for k in range(len(pool_x_arr)) if pool_x_arr[k].size > 0]
    xs_all = np.concatenate([pool_x_arr[k] for k in valid]) if valid else np.zeros(0)
    ys_all = np.concatenate([pool_y_arr[k] for k in valid]) if valid else np.zeros(0)
    p_pool = float(fit_superellipse(xs_all, ys_all)['p']) if xs_all.size else float('nan')

    bs_rng = np.random.default_rng(args.pair_seed + 1)
    p_boot = []
    if valid:
        for _ in range(args.n_pool_bootstrap):
            idx = bs_rng.integers(0, len(valid), len(valid))
            xs_b = np.concatenate([pool_x_arr[valid[k]] for k in idx])
            ys_b = np.concatenate([pool_y_arr[valid[k]] for k in idx])
            if xs_b.size:
                p_boot.append(fit_superellipse(xs_b, ys_b)['p'])
    p_boot = np.array(p_boot, dtype=float)
    if p_boot.size:
        p_pool_lo = float(np.quantile(p_boot, 0.025))
        p_pool_hi = float(np.quantile(p_boot, 0.975))
    else:
        p_pool_lo = p_pool_hi = float('nan')

    print('\n=== Pair-level summary ===')
    print(f'  per-pair p_fit:  median={p_med:.3f}  IQR=({p_iqr[0]:.3f}, '
          f'{p_iqr[1]:.3f})  min={np.min(p_fits[finite]):.3f}  '
          f'max={np.max(p_fits[finite]):.3f}  n={int(finite.sum())}')
    print(f'  pooled-points p: {p_pool:.3f}  95% CI over pairs '
          f'[{p_pool_lo:.3f}, {p_pool_hi:.3f}]  '
          f'(bootstrap n={args.n_pool_bootstrap})')

    # Save
    out_pkl = os.path.join(args.out, 'fit_pairs.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'args': vars(args),
            'angles_deg': angles_deg,
            'per_pair': per_pair,
            'p_fits': p_fits, 'p_los': p_los, 'p_his': p_his,
            'p_median': p_med, 'p_iqr': p_iqr,
            'p_pool': p_pool, 'p_pool_ci_95': (p_pool_lo, p_pool_hi),
            'p_pool_bootstrap': p_boot,
            'measure_layer': measure_layer,
        }, f)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(p_fits[finite], bins=30, edgecolor='black', alpha=0.8)
    ax.axvline(p_med, color='C1', lw=2, label=f'median p={p_med:.2f}')
    ax.axvline(p_pool, color='C2', lw=2, ls='--',
               label=f'pooled p={p_pool:.2f}')
    ax.axvline(2.0, color='gray', lw=1, ls=':', label='p=2 (ellipse)')
    ax.axvspan(p_iqr[0], p_iqr[1], alpha=0.15, color='C1', label='IQR')
    ax.axvspan(p_pool_lo, p_pool_hi, alpha=0.15, color='C2',
               label='pooled 95% CI')
    ax.set_xlabel('fitted p (per pair)')
    ax.set_ylabel('# pairs')
    ax.set_title(f'Distribution of p over {len(pairs)} random pairs '
                 f'(N={N} atoms)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cosv = [pp['cos'] for pp in per_pair]
    ax.scatter(cosv, p_fits, c='C0', s=18, alpha=0.6)
    ax.axhline(p_med, color='C1', lw=1.5, label=f'median p={p_med:.2f}')
    ax.axhline(2.0, color='gray', lw=1, ls=':', label='p=2')
    ax.set_xlabel('cos(d_i, d_j)')
    ax.set_ylabel('fitted p')
    ax.set_title('p vs pair cosine')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{args.model} L{args.layer}->{measure_layer}  '
                 f'pair fits  (n_anchors={args.n_anchors}, '
                 f'grid={args.n_steps_grid}x{args.n_steps_grid})',
                 fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'p_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nSaved {out_pkl}')
    print(f'Saved {plot_path}')

    # Optional Fig-3-style boundary panels for the first N pairs.
    if boundary_panels:
        ncol = 2
        nrow = (len(boundary_panels) + ncol - 1) // ncol
        bfig, baxes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 6 * nrow))
        baxes = np.atleast_1d(baxes).ravel()
        for ax, (b, title, l1, l2) in zip(baxes, boundary_panels):
            draw_panel(ax, b, title, d1lab=l1, d2lab=l2)
        for ax in baxes[len(boundary_panels):]:
            ax.axis('off')
        bfig.suptitle(f'{args.model} L{args.layer}->{measure_layer}  '
                      f'eval-frame boundaries ({args.n_anchors} held-out anchors)',
                      fontsize=13)
        plt.tight_layout()
        bpath = os.path.join(args.out, 'boundaries.png')
        plt.savefig(bpath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'Saved {bpath}')
        # Also one standalone figure per pair.
        for (b, title, l1, l2) in boundary_panels:
            pfig, pax = plt.subplots(figsize=(7, 6))
            draw_panel(pax, b, title, d1lab=l1, d2lab=l2)
            plt.tight_layout()
            ppath = os.path.join(args.out, f'boundary_{l1}_{l2}.png')
            pfig.savefig(ppath, dpi=150, bbox_inches='tight',
                         facecolor='white')
            plt.close(pfig)
            print(f'Saved {ppath}')


if __name__ == '__main__':
    main()
