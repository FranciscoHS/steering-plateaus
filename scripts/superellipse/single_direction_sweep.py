#!/usr/bin/env python3
"""
Experiment 3: single-direction perturbation curves (Fig 1a style).

For the learned pair, sweep the geodesic perturbation angle alpha along:
  - d1, d2 individually,
  - their equal-weight combination,
  - K random unit directions (baseline),
and plot L2-at-penultimate-layer vs alpha (median + IQR over anchors), with the
plateau-breaking threshold and per-curve breaking angles. Done for both the
optimized (training) and held-out anchor sets.

The point: do d1, d2 break the plateau EARLY relative to random directions
(i.e. are they unusually responsive single directions), and does the
combination break LATE (the super-additive cancellation)?

Reuses the canonical exp-map 1D sweep (scripts/superellipse/canonical/sweep_core).

Usage:
    python scripts/superellipse/single_direction_sweep.py \
        --dict_pkl results/superellipse_dict/qwen3-1.7b_L2_a30/results.pkl
"""
import os, sys, pickle, argparse
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.canonical.sweep_core import run_1d_per_anchors


def build_anchors(model, token_ids_list, layer, device, batch_size=64):
    ctxs, acts = [], []
    for i in range(0, len(token_ids_list), batch_size):
        bt = torch.stack(token_ids_list[i:i + batch_size]).to(device)
        with torch.no_grad():
            hidden = model(bt, output_hidden_states=True).hidden_states[layer + 1]
        ctxs.append(hidden[:, :-1, :].float().cpu())
        acts.append(hidden[:, -1, :].float().cpu())
    ctx_all = torch.cat(ctxs, 0)
    act_all = torch.cat(acts, 0)
    return [ctx_all[i:i + 1] for i in range(ctx_all.shape[0])], act_all


def crossing(angles, curve, threshold):
    if curve.max() < threshold:
        return float('nan')
    for i in range(1, len(curve)):
        if (curve[i - 1] - threshold) * (curve[i] - threshold) <= 0:
            x0, x1 = float(angles[i - 1]), float(angles[i])
            y0, y1 = float(curve[i - 1]), float(curve[i])
            return x0 if y1 == y0 else x0 + (threshold - y0) / (y1 - y0) * (x1 - x0)
    return float('nan')


def stats(g):
    return np.median(g, 0), np.percentile(g, 25, 0), np.percentile(g, 75, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_pkl', type=str,
                    default='results/superellipse_dict/qwen3-1.7b_L2_a30/results.pkl')
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--n_steps', type=int, default=61)
    ap.add_argument('--n_random', type=int, default=10)
    ap.add_argument('--n_act_fresh', type=int, default=30)
    ap.add_argument('--fresh_seed', type=int, default=999)
    ap.add_argument('--threshold_frac', type=float, default=0.50)
    ap.add_argument('--dtype', type=str, default='float32',
                    choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--results_dir', type=str, default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    with open(args.dict_pkl, 'rb') as f:
        dd = pickle.load(f)
    directions = dd['directions'].float()
    da = dd['args']
    model_name = da['model']; layer = da['layer']; seq_len = da['seq_len']
    train_seed = da['seed']; n_act_train = da['n_act']
    measure_layer = dd.get('measure_layer')

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(args.dict_pkl), 'single_direction_sweep')
    os.makedirs(args.results_dir, exist_ok=True)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    print(f'Loading {model_name} (dtype={args.dtype})...')
    model, tokenizer, device = load_model(model_name, dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    if measure_layer is None:
        measure_layer = n_layers - 2
    d_model = model._plateau_info['d_model']
    d1, d2 = directions[0], directions[1]
    combo = F.normalize(d1 + d2, dim=0)
    print(f'  perturb@L{layer}, measure@L{measure_layer}')

    g = torch.Generator().manual_seed(args.seed)
    rand_dirs = [F.normalize(torch.randn(d_model, generator=g), dim=0)
                 for _ in range(args.n_random)]

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)

    # anchor sets
    tok_opt = load_fineweb_fixed_length(max(100, n_act_train), tokenizer, seq_len, seed=train_seed)
    ctx_o, act_o = build_anchors(model, tok_opt, layer, device)
    tok_fresh = load_fineweb_fixed_length(args.n_act_fresh, tokenizer, seq_len, seed=args.fresh_seed)
    ctx_f, act_f = build_anchors(model, tok_fresh, layer, device)
    sets = {'optimized': (ctx_o[:n_act_train], act_o[:n_act_train]),
            'held_out': (ctx_f[:args.n_act_fresh], act_f[:args.n_act_fresh])}

    results = {'args': vars(args), 'angles_deg': angles_deg, 'measure_layer': measure_layer,
               'sets': {}}

    for set_name, (contexts, activations) in sets.items():
        print(f'\n=== {set_name} ({activations.shape[0]} anchors) ===')
        g1 = run_1d_per_anchors(model, contexts, activations, d1, layer, measure_layer, device, angles_rad)
        g2 = run_1d_per_anchors(model, contexts, activations, d2, layer, measure_layer, device, angles_rad)
        gc = run_1d_per_anchors(model, contexts, activations, combo, layer, measure_layer, device, angles_rad)
        rand_per_dir = np.stack([
            np.median(run_1d_per_anchors(model, contexts, activations, rd, layer, measure_layer, device, angles_rad), 0)
            for rd in rand_dirs], 0)  # (n_random, n_steps)

        m1, _, _ = stats(g1); m2, _, _ = stats(g2); mc, _, _ = stats(gc)
        T = args.threshold_frac * float(min(m1.max(), m2.max()))
        a1, a2, ac = crossing(angles_deg, m1, T), crossing(angles_deg, m2, T), crossing(angles_deg, mc, T)
        r_med = np.median(rand_per_dir, 0)
        ar = crossing(angles_deg, r_med, T)
        print(f'  T={T:.1f}  break: d1={a1:.1f} d2={a2:.1f} combo={ac:.1f} random={ar:.1f} deg')
        results['sets'][set_name] = {'g1': g1, 'g2': g2, 'gc': gc, 'rand': rand_per_dir,
                                     'T': T, 'breaks': (a1, a2, ac, ar)}

    with open(os.path.join(args.results_dir, 'single_direction_sweep.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # plot
    C_D1, C_D2, C_CB, C_RD, C_T = '#ff7f00', '#377eb8', '#3aa54a', '#888888', '#444444'
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, set_name in zip(axes, ['optimized', 'held_out']):
        so = results['sets'][set_name]
        T = so['T']; a1, a2, ac, ar = so['breaks']
        for grid, color, lab in [(so['g1'], C_D1, 'd1'), (so['g2'], C_D2, 'd2'), (so['gc'], C_CB, 'd1+d2')]:
            med, lo, hi = stats(grid)
            ax.fill_between(angles_deg, lo, hi, color=color, alpha=0.18)
            ax.plot(angles_deg, med, '-', color=color, lw=2.4, label=lab)
        r_med = np.median(so['rand'], 0); r_lo = np.percentile(so['rand'], 25, 0); r_hi = np.percentile(so['rand'], 75, 0)
        ax.fill_between(angles_deg, r_lo, r_hi, color=C_RD, alpha=0.18)
        ax.plot(angles_deg, r_med, '--', color=C_RD, lw=2.0, label=f'random (n={args.n_random})')
        ax.axhline(T, color=C_T, ls=':', lw=1.3, label=rf'threshold $L^2$={T:.0f}')
        for ang, c in [(a1, C_D1), (a2, C_D2), (ac, C_CB), (ar, C_RD)]:
            if np.isfinite(ang) and ang <= args.max_angle:
                ax.axvline(ang, color=c, ls='--', lw=1.3, alpha=0.85)
        ax.set_xlim(0, args.max_angle); ax.set_ylim(0, T * 2.2)
        ax.set_xlabel(r'Perturbation angle $\alpha$ (deg)')
        ax.set_ylabel(r'$L^2$ at penultimate layer')
        ax.set_title(f'{set_name}: break d1={a1:.0f} d2={a2:.0f} combo={ac:.0f} rand={ar:.0f} deg')
        ax.grid(alpha=0.2); ax.legend(fontsize=9, loc='upper left')
    fig.suptitle(f'{model_name} L{layer}->{measure_layer}: single-direction perturbation curves', fontsize=13)
    plt.tight_layout()
    png = os.path.join(args.results_dir, 'single_direction_sweep.png')
    plt.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\nSaved: {os.path.join(args.results_dir, "single_direction_sweep.pkl")}')
    print(f'Saved: {png}')


if __name__ == '__main__':
    main()
