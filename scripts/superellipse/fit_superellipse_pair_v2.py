#!/usr/bin/env python3
"""
Experiment 1 (canonical fit): superellipse-p for a learned (d1, d2) pair,
using the exact fsec-anonymous procedure.

Differs from fit_superellipse_pair.py (1D radial spokes + fit_n, which railed
on out-of-box data). This builds the canonical 2D iso-plateau sweep and fit:
  - exp-map chart: 31x31 grid in tangent coords (alpha_1, alpha_2) up to 60 deg
    (scripts/superellipse/canonical/parametrize.exp_map_2d via sweep_core);
  - L2 at the penultimate layer per grid point, over N anchors;
  - extract the iso-L2 contour, normalize by axis intercepts, fit |x|^p+|y|^p=1
    with point-masking to (0.05, 1.5) so out-of-box points are dropped, not
    railed (scripts/superellipse/canonical/superellipse.robust_p_fit);
  - robustness over thresholds {0.5,0.75}xbase x max_alpha {30,45,60} x bootstrap.

Runs on BOTH the activations the pair was optimized on and a fresh random set,
and reports p for both the small-angle and exact-geodesic normalizations.

Usage:
    python scripts/superellipse/fit_superellipse_pair_v2.py \
        --dict_pkl results/superellipse_dict/qwen3-1.7b_L2_a30/results.pkl
"""
import os, sys, pickle, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.canonical.sweep_core import run_2d_per_anchors
from scripts.superellipse.canonical.superellipse import (
    robust_p_fit, extract_contour, axis_intercept, superellipse_curve,
)


def build_anchors(model, token_ids_list, layer, device, batch_size=64):
    """Return (contexts_list, activations) where contexts[i] is [1, T-1, D]."""
    ctxs, acts = [], []
    for i in range(0, len(token_ids_list), batch_size):
        bt = torch.stack(token_ids_list[i:i + batch_size]).to(device)
        with torch.no_grad():
            hidden = model(bt, output_hidden_states=True).hidden_states[layer + 1]
        ctxs.append(hidden[:, :-1, :].float().cpu())
        acts.append(hidden[:, -1, :].float().cpu())
    ctx_all = torch.cat(ctxs, 0)
    act_all = torch.cat(acts, 0)
    contexts = [ctx_all[i:i + 1] for i in range(ctx_all.shape[0])]
    return contexts, act_all


def summarize_fit(fit):
    """Collapse robust_p_fit cells into overall [p_lo, p_hi] and a per-cell table."""
    ps, rows = [], []
    for k, v in fit.items():
        if k == '__levels__':
            continue
        rows.append((k, v))
        if np.isfinite(v['p_lo']) and np.isfinite(v['p_hi']):
            ps.append(v['p_lo']); ps.append(v['p_hi'])
    p_lo = float(min(ps)) if ps else np.nan
    p_hi = float(max(ps)) if ps else np.nan
    return p_lo, p_hi, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_pkl', type=str,
                        default='results/superellipse_dict/qwen3-1.7b_L2_a30/results.pkl')
    parser.add_argument('--n_steps', type=int, default=31)
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_act_fresh', type=int, default=30)
    parser.add_argument('--fresh_seed', type=int, default=999)
    parser.add_argument('--n_bootstrap', type=int, default=200)
    parser.add_argument('--mode', type=str, default='geodesic',
                        choices=['geodesic', 'additive'])
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--results_dir', type=str, default=None)
    args = parser.parse_args()

    with open(args.dict_pkl, 'rb') as f:
        dd = pickle.load(f)
    directions = dd['directions']
    da = dd['args']
    model_name = da['model']; perturb_layer = da['layer']; seq_len = da['seq_len']
    train_seed = da['seed']; n_act_train = da['n_act']
    measure_layer = dd.get('measure_layer')

    if args.results_dir is None:
        tag = 'fit_canonical' if args.mode == 'geodesic' else f'fit_canonical_{args.mode}'
        args.results_dir = os.path.join(os.path.dirname(args.dict_pkl), tag)
    os.makedirs(args.results_dir, exist_ok=True)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    print(f'Loading {model_name} (dtype={args.dtype})...')
    model, tokenizer, device = load_model(model_name, dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    if measure_layer is None:
        measure_layer = n_layers - 2
    print(f'  perturb@L{perturb_layer}, measure@L{measure_layer}, n_layers={n_layers}')
    d1 = directions[0].float()
    d2 = directions[1].float()

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)

    # Two anchor sets
    tok_opt = load_fineweb_fixed_length(max(100, n_act_train), tokenizer, seq_len, seed=train_seed)
    ctx_o, act_o = build_anchors(model, tok_opt, perturb_layer, device)
    tok_fresh = load_fineweb_fixed_length(args.n_act_fresh, tokenizer, seq_len, seed=args.fresh_seed)
    ctx_f, act_f = build_anchors(model, tok_fresh, perturb_layer, device)
    sets = {
        'optimized': (ctx_o[:n_act_train], act_o[:n_act_train]),
        'fresh': (ctx_f[:args.n_act_fresh], act_f[:args.n_act_fresh]),
    }

    results = {'args': vars(args), 'dict_args': da, 'measure_layer': measure_layer,
               'angles_deg': angles_deg, 'sets': {}}

    for set_name, (contexts, activations) in sets.items():
        print(f'\n=== Set: {set_name} ({activations.shape[0]} anchors) ===')
        r = run_2d_per_anchors(model, contexts, activations, d1, d2,
                               perturb_layer, measure_layer, device, angles_rad,
                               mode=args.mode)
        l2_grid = r['l2']  # (n_anchors, n, n)
        print(f'  l2 grid {l2_grid.shape}, max={l2_grid.max():.1f}')

        set_out = {'l2_grid': l2_grid, 'fits': {}}
        for norm_label, exact in [('small_angle', False), ('exact_geodesic', True)]:
            fit = robust_p_fit(angles_deg, l2_grid, n_bootstrap=args.n_bootstrap,
                               exact_geodesic=exact)
            p_lo, p_hi, rows = summarize_fit(fit)
            print(f'  [{norm_label}] p range [{p_lo:.2f}, {p_hi:.2f}] across '
                  f'{len(rows)} cells:')
            for k, v in rows:
                print(f'      {str(k):16s} p_med={v["p_median"]:.2f} '
                      f'[{v["p_lo"]:.2f}, {v["p_hi"]:.2f}]  n_pts={v["n_pts"]} '
                      f'intercepts=({v["intercepts"][0]:.1f},{v["intercepts"][1]:.1f})')
            set_out['fits'][norm_label] = fit
            set_out[f'p_range_{norm_label}'] = (p_lo, p_hi)
        results['sets'][set_name] = set_out

    with open(os.path.join(args.results_dir, 'fit_canonical.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # ---- plot: median L2 grid + iso-contours + fitted superellipse (exact) ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for col, set_name in enumerate(['optimized', 'fresh']):
        ax = axes[col]
        so = results['sets'][set_name]
        grid = np.median(so['l2_grid'], axis=0)  # (n, n)
        # base threshold = min(max axis response) so both axes reach it
        base = min(grid[:, 0].max(), grid[0, :].max())
        thr50 = 0.5 * base
        im = ax.pcolormesh(angles_deg, angles_deg, grid.T, shading='auto', cmap='viridis')
        plt.colorbar(im, ax=ax, fraction=0.046, label='median L2')
        cs = ax.contour(angles_deg, angles_deg, grid.T, levels=[thr50, 0.75 * base],
                        colors='white', linewidths=1.2)
        ax.clabel(cs, fmt='%.0f', fontsize=7)
        # fitted superellipse at thr50, exact normalization, scaled by intercepts
        fit = so['fits']['exact_geodesic']
        cell = fit.get(('50', 45.0))
        if cell and np.isfinite(cell['p_median']):
            t1, t2 = cell['intercepts']
            p = cell['p_median']
            xs, ys = superellipse_curve(p)
            xs = np.abs(xs) * t1; ys = np.abs(ys) * t2
            ax.plot(xs, ys, 'r--', lw=2, label=f'superellipse p={p:.2f}')
        plo, phi = so['p_range_exact_geodesic']
        ax.set_title(f'{set_name}: p in [{plo:.2f}, {phi:.2f}] (exact)')
        ax.set_xlabel('alpha_1 (deg, along d1)')
        ax.set_ylabel('alpha_2 (deg, along d2)')
        ax.legend(fontsize=9)
    fig.suptitle(f'{model_name} L{perturb_layer}->{measure_layer}: canonical superellipse fit',
                 fontsize=13)
    plt.tight_layout()
    png = os.path.join(args.results_dir, 'fit_canonical.png')
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {os.path.join(args.results_dir, "fit_canonical.pkl")}')
    print(f'Saved: {png}')


if __name__ == '__main__':
    main()
