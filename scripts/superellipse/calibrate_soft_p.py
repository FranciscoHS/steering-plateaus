#!/usr/bin/env python3
"""Phase 0: calibrate the differentiable soft-intercept p estimator.

Two jobs, both done BEFORE spending any build compute:

1. Correlation. On a sample of existing dictionary pairs (d_i, d_j), compare
   the train estimator `soft_p_estimate` against the eval `fit_pairs` p
   (run_2d_per_anchors -> boundary_for_set) on the SAME pairs and the SAME
   anchors. Report Spearman + Pearson between train-p and eval-p. Target:
   Spearman >> the current ratio-proxy's -0.33 (aim >= 0.7).

2. Timing. Measure where per-(gradient-)step time stops being flat as the soft
   point count grows (vary n_radii), so the build's radius/ray budget can be
   set to "free" sampling below saturation.

Usage:
    python scripts/superellipse/calibrate_soft_p.py \
        --dict_path results/dictionary/qwen3-1.7b_L14_K8000_randcontrast_stepresample/dictionary.pt \
        --model qwen3-1.7b --layer 14 \
        --n_pairs 60 --n_anchors 20 \
        --out results/dictionary/calib_softp_qwen_L14
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
from scipy.stats import spearmanr, pearsonr

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)
from scripts.superellipse.canonical.sweep_core import run_2d_per_anchors
from scripts.superellipse.canonical.superellipse import robust_p_fit  # noqa: F401
from scripts.superellipse.plot_boundary_pair import boundary_for_set
from scripts.superellipse.soft_p_proxy import (
    soft_p_estimate, diag_rho_estimate, default_phis_radii,
)
from scripts.superellipse.decompose_plateau_breaking import batched_proxy_loss


def ratio_per_angle(model, acts_stack, ctx_stack, f_refs_stack, d1, d2,
                    layer, measure_layer, ratio_angles_rad):
    """Per-angle ratio R_mix/(R_f+R_c) for the pair (d1, d2) over the anchor
    stack, one value per push magnitude. The multi-magnitude objective is the
    mean of these; recording them per-angle lets us evaluate the correlation
    of any angle subset with eval-p from a single run."""
    return np.array([
        batched_proxy_loss(model, acts_stack, ctx_stack, f_refs_stack,
                           d1, d2.unsqueeze(0), a, layer, measure_layer)[0].item()
        for a in ratio_angles_rad])


def eval_p_for_pair(model, contexts, activations, d1, d2, layer,
                    measure_layer, device, angles_rad, angles_deg):
    """Eval-frame p for one pair: median 2D grid -> boundary_for_set fit."""
    r = run_2d_per_anchors(model, contexts, activations, d1, d2, layer,
                           measure_layer, device, angles_rad,
                           mode='geodesic', single_batch_ref=True)
    median_grid = np.median(r['l2'], axis=0)
    b = boundary_for_set(angles_deg, median_grid, exact=True)
    return b['fit']['p'], int(b['fit']['n_pts'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_pairs', type=int, default=60)
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--pair_seed', type=int, default=0)
    ap.add_argument('--atom_max_idx', type=int, default=None)
    # eval grid (matches fit_pairs_dictionary defaults)
    ap.add_argument('--n_steps_grid', type=int, default=31)
    ap.add_argument('--max_angle', type=float, default=60.0)
    # soft-p estimator grid
    ap.add_argument('--n_rays', type=int, default=5)
    ap.add_argument('--n_radii', type=int, default=6)
    ap.add_argument('--p_cap', type=float, default=10.0)
    ap.add_argument('--n_newton', type=int, default=20)
    ap.add_argument('--anchor_reduce', choices=['median', 'mean'],
                    default='median')
    ap.add_argument('--estimator', choices=['soft_p', 'diag_rho', 'ratio'],
                    default='soft_p',
                    help='which train metric to correlate against eval-p')
    ap.add_argument('--ratio_angles', type=float, nargs='+',
                    default=[10, 20, 30, 40, 50, 60],
                    help='push magnitudes (deg) for the multi-magnitude ratio')
    ap.add_argument('--no_timing', action='store_true',
                    help='skip the per-step timing sweep')
    ap.add_argument('--out', required=True)
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
    model_dtype = next(model.parameters()).dtype
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    d_model = model._plateau_info['d_model']
    print(f'Device: {device} d_model={d_model} L{args.layer}->{measure_layer}',
          flush=True)

    # Dictionary
    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(device=device, dtype=torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    N = D.shape[0]
    atom_max = min(args.atom_max_idx or N, N)
    print(f'Dictionary: {N} atoms (sampling from [0, {atom_max}))', flush=True)

    # Anchors
    print(f'Extracting {args.n_anchors} anchors (seed={args.held_seed})',
          flush=True)
    tok = load_fineweb_fixed_length(args.n_anchors, tokenizer,
                                    seq_len=args.seq_len, seed=args.held_seed)
    ctx, act = extract_context_and_activation(model, tok, args.layer, device)
    # eval-side (cpu list of [1,T-1,d] + [B,d])
    contexts = [ctx[i:i + 1].float().cpu() for i in range(ctx.shape[0])]
    activations = act.float().cpu()
    # estimator-side (device stacks + precomputed refs)
    acts_stack = act.to(device=device, dtype=torch.float32)        # [B,d]
    ctx_stack = ctx.to(device=device, dtype=model_dtype)           # [B,T-1,d]
    with torch.no_grad():
        pts = acts_stack.unsqueeze(1).to(model_dtype)
        f_refs_stack = forward_from_layer_to_layer(
            model, ctx_stack, pts, args.layer, measure_layer).float()  # [B,d]

    phis_rad, radii_rad = default_phis_radii(args.n_rays, args.n_radii,
                                             args.max_angle)
    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps_grid)
    angles_rad = np.deg2rad(angles_deg)

    # Sample pairs (i != j)
    seen = set()
    while len(seen) < min(args.n_pairs, atom_max * (atom_max - 1) // 2):
        i, j = int(rng.integers(atom_max)), int(rng.integers(atom_max))
        if i == j:
            continue
        seen.add((min(i, j), max(i, j)))
    pairs = sorted(seen)
    print(f'Sampled {len(pairs)} pairs', flush=True)

    ratio_angles_rad = np.deg2rad(args.ratio_angles)
    print(f'Estimator: {args.estimator}'
          + (f' (ratio_angles={args.ratio_angles})'
             if args.estimator == 'ratio' else ''), flush=True)

    # --- correlation ---
    rows = []
    t0 = time.time()
    for k, (i, j) in enumerate(pairs):
        d1 = D[i]
        d2 = D[j]
        p_eval, n_eval = eval_p_for_pair(
            model, contexts, activations, d1.cpu(), d2.cpu(),
            args.layer, measure_layer, device, angles_rad, angles_deg)
        with torch.no_grad():
            if args.estimator == 'soft_p':
                tv = soft_p_estimate(
                    model, acts_stack, ctx_stack, f_refs_stack, d1,
                    d2.unsqueeze(0), args.layer, measure_layer,
                    phis_rad, radii_rad, p_cap=args.p_cap,
                    n_newton=args.n_newton,
                    anchor_reduce=args.anchor_reduce).item()
            elif args.estimator == 'diag_rho':
                tv = diag_rho_estimate(
                    model, acts_stack, ctx_stack, f_refs_stack, d1,
                    d2.unsqueeze(0), args.layer, measure_layer,
                    phis_rad, radii_rad,
                    anchor_reduce=args.anchor_reduce).item()
            else:  # ratio (multi-magnitude)
                per_ang = ratio_per_angle(
                    model, acts_stack, ctx_stack, f_refs_stack,
                    d1, d2, args.layer, measure_layer, ratio_angles_rad)
                tv = float(per_ang.mean())
        cos_ij = float((d1 * d2).sum().item())
        row = {
            'pair': (i, j), 'cos': cos_ij,
            'p_eval': float(p_eval), 'n_eval': n_eval,
            'p_train': float(tv),
        }
        if args.estimator == 'ratio':
            row['ratio_per_angle'] = per_ang
        rows.append(row)
        if (k + 1) % 10 == 0 or k == 0:
            rate = (time.time() - t0) / (k + 1)
            print(f'  [{k+1:>3}/{len(pairs)}] {i}-{j}  cos={cos_ij:+.3f}  '
                  f'p_eval={p_eval:.3f}  train={tv:.3f}  '
                  f'rate={rate:.1f}s/pair', flush=True)

    pe = np.array([r['p_eval'] for r in rows])
    pt = np.array([r['p_train'] for r in rows])
    fin = np.isfinite(pe) & np.isfinite(pt)
    sp = spearmanr(pe[fin], pt[fin])
    pr = pearsonr(pe[fin], pt[fin])
    print('\n=== Calibration ===', flush=True)
    print(f'  n_finite={int(fin.sum())}/{len(rows)}', flush=True)
    print(f'  p_eval:  median={np.median(pe[fin]):.3f}  '
          f'[{np.min(pe[fin]):.3f}, {np.max(pe[fin]):.3f}]', flush=True)
    print(f'  p_train: median={np.median(pt[fin]):.3f}  '
          f'[{np.min(pt[fin]):.3f}, {np.max(pt[fin]):.3f}]', flush=True)
    print(f'  Spearman={sp.correlation:.3f} (p={sp.pvalue:.2e})', flush=True)
    print(f'  Pearson ={pr.statistic:.3f} (p={pr.pvalue:.2e})', flush=True)

    # --- angle-subset tracking (ratio only): how Spearman vs eval-p depends
    #     on which / how many push magnitudes we average ---
    if args.estimator == 'ratio':
        A = np.array([r['ratio_per_angle'] for r in rows])   # [n_pairs, n_ang]
        angs = list(args.ratio_angles)
        def sub_sp(cols):
            m = A[:, cols].mean(axis=1)
            f = np.isfinite(pe) & np.isfinite(m)
            return spearmanr(pe[f], m[f]).correlation
        print('\n=== angle-subset tracking (|Spearman| vs eval-p) ===',
              flush=True)
        print('  each single angle:', flush=True)
        for c, a in enumerate(angs):
            print(f'    {a:>4.0f}deg only         : {sub_sp([c]):+.3f}',
                  flush=True)
        # nested / spaced subsets
        named = {
            'all angles': list(range(len(angs))),
            'every-other (lo)': list(range(0, len(angs), 2)),
            'every-other (hi)': list(range(1, len(angs), 2)),
            'lowest 3': list(range(0, 3)),
            'highest 3': list(range(len(angs) - 3, len(angs))),
            'extremes (lo+hi)': [0, len(angs) - 1],
        }
        print('  subsets:', flush=True)
        for name, cols in named.items():
            used = [angs[c] for c in cols]
            print(f'    {name:<18s} {str(used):<24s}: {sub_sp(cols):+.3f}',
                  flush=True)

    # --- timing sweep ---
    timing = []
    if not args.no_timing and args.estimator == 'soft_p':
        print('\n=== Timing sweep (1 anchor, K=4, grad step) ===', flush=True)
        gen = torch.Generator(device=device).manual_seed(123)
        acts1 = acts_stack[:1]
        ctx1 = ctx_stack[:1]
        fref1 = f_refs_stack[:1]
        for nr in [2, 6, 18, 35]:
            ph, ra = default_phis_radii(args.n_rays, nr, args.max_angle)
            n_pts = nr + 4 * (args.n_rays - 1) * nr
            d = torch.randn(d_model, generator=gen, device=device)
            d = (d / d.norm()).requires_grad_(True)
            c = torch.randn(4, d_model, generator=gen, device=device)
            c = c / c.norm(dim=-1, keepdim=True)
            # warmup
            soft_p_estimate(model, acts1, ctx1, fref1, d, c, args.layer,
                            measure_layer, ph, ra, p_cap=args.p_cap,
                            n_newton=args.n_newton).backward()
            if device == 'cuda':
                torch.cuda.synchronize()
            t = time.time()
            reps = 3
            for _ in range(reps):
                d.grad = None
                soft_p_estimate(model, acts1, ctx1, fref1, d, c, args.layer,
                                measure_layer, ph, ra, p_cap=args.p_cap,
                                n_newton=args.n_newton).backward()
            if device == 'cuda':
                torch.cuda.synchronize()
            dt = (time.time() - t) / reps
            timing.append({'n_radii': nr, 'n_points': n_pts, 's_per_step': dt})
            print(f'  n_radii={nr:3d}  n_points={n_pts:4d}  '
                  f'{dt*1000:.1f} ms/step', flush=True)

    # --- save ---
    out_pkl = os.path.join(args.out, 'calibration.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'args': vars(args), 'rows': rows,
            'p_eval': pe, 'p_train': pt,
            'spearman': float(sp.correlation), 'pearson': float(pr.statistic),
            'timing': timing, 'measure_layer': measure_layer,
        }, f)

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.scatter(pe[fin], pt[fin], s=22, alpha=0.7, c='C0')
    # y=x only meaningful when the train metric is itself a p (same scale);
    # for the ratio it lives on a different scale, so skip it and let each
    # axis autoscale to its own data range.
    if args.estimator in ('soft_p', 'diag_rho'):
        lo = min(pe[fin].min(), pt[fin].min())
        hi = max(pe[fin].max(), pt[fin].max())
        ax.plot([lo, hi], [lo, hi], 'k:', lw=1, label='y=x')
        ax.legend(fontsize=9)
    ax.set_xlabel('eval p (boundary_for_set)')
    ax.set_ylabel(f'train metric ({args.estimator})')
    ax.set_title(f'Spearman={sp.correlation:.2f}  Pearson={pr.statistic:.2f}  '
                 f'(n={int(fin.sum())})')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if timing:
        npts = [t['n_points'] for t in timing]
        ms = [t['s_per_step'] * 1000 for t in timing]
        ax.plot(npts, ms, 'o-')
        ax.set_xlabel('soft point count (n_radii + 4*(n_rays-1)*n_radii)')
        ax.set_ylabel('ms / grad step')
        ax.set_title('Per-step time vs point count')
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')

    fig.suptitle(f'{args.model} L{args.layer}->{measure_layer}  soft-p '
                 f'calibration  (n_rays={args.n_rays}, n_radii={args.n_radii})',
                 fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'calibration.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nSaved {out_pkl}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
