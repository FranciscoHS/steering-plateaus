#!/usr/bin/env python3
"""
Fit superellipse-p for a selection of free-vs-free direction pairs from a
learned dictionary (e.g. the output of learn_against_contrastive_mlp.py).

Picks `n_pairs` pairs among the first `n_free` directions, spread across the
|cos| range (most-orthogonal -> most-overlapping), runs the canonical 2D
exp-map sweep on the optimized anchor set, fits p in |x|^p+|y|^p=1, and draws
the boundary-scatter panel (alpha_1 vs alpha_2 with the contour points and the
p=2 / p->inf / fitted-superellipse curves) for each pair in one figure.

Usage:
    python fit_free_pairs_mlp.py \
        --dict_pkl results/grok_mlp/against_contrastive/ckpt_s1000000_final_a30_nc10/results.pkl \
        --n_pairs 9
"""
import argparse
import itertools
import math
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.grok_mlp.fit_superellipse_pair_mlp import (
    build_mlp, get_h1_anchors, run_2d_per_anchors, summarize_fit,
)
from scripts.superellipse.canonical.superellipse import robust_p_fit
from scripts.superellipse.plot_boundary_pair import boundary_for_set, draw_panel


def select_pairs(candidates, gram, n_pairs):
    """Pick n_pairs from `candidates` spread across the |cos| range."""
    cos = np.array([abs(gram[i, j]) for (i, j) in candidates])
    order = np.argsort(cos)  # ascending |cos|
    pick_idx = np.unique(np.linspace(0, len(candidates) - 1, n_pairs).round().astype(int))
    return [candidates[order[k]] for k in pick_idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dict_pkl', type=str, required=True)
    p.add_argument('--mode', type=str, default='free_free',
                   choices=['free_free', 'free_contrastive', 'contrastive_contrastive'])
    p.add_argument('--n_pairs', type=int, default=9)
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--n_steps', type=int, default=31)
    p.add_argument('--max_angle', type=float, default=60.0)
    p.add_argument('--n_bootstrap', type=int, default=100)
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--results_dir', type=str, default=None)
    args = p.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    with open(args.dict_pkl, 'rb') as f:
        dd = pickle.load(f)
    directions = dd['directions'].to(device=device, dtype=dtype)
    da = dd['args']
    ckpt_path = args.ckpt if args.ckpt is not None else da['ckpt']
    inject_after = dd['inject_after']
    measure_after = dd['measure_after']
    n_free = dd.get('n_free', directions.shape[0])
    class_pairs = dd.get('class_pairs')  # class (a,b) per contrastive dir
    n_contrastive = directions.shape[0] - n_free
    full_gram = (directions @ directions.T).cpu().numpy()

    def lab(idx):
        if idx < n_free:
            return f'f{idx}'
        if class_pairs is not None:
            a, b = class_pairs[idx - n_free]
            return f'c{a}{b}'
        return f'c{idx - n_free}'

    if args.results_dir is None:
        sub = {'free_free': 'fit_free_pairs',
               'free_contrastive': 'fit_free_contrastive_pairs',
               'contrastive_contrastive': 'fit_contrastive_pairs'}[args.mode]
        args.results_dir = os.path.join(os.path.dirname(args.dict_pkl), sub)
    os.makedirs(args.results_dir, exist_ok=True)

    # Model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    depth = ckpt.get('depth', 4)
    width = ckpt.get('width', 200)
    mlp = build_mlp(depth, width, act=da.get('activation', 'GeLU')).to(device)
    mlp.load_state_dict(ckpt['model_state_dict'])
    mlp.eval()

    if args.mode == 'free_free':
        candidates = list(itertools.combinations(range(n_free), 2))
    elif args.mode == 'free_contrastive':
        candidates = [(i, n_free + k) for i in range(n_free)
                      for k in range(n_contrastive)]
    else:  # contrastive_contrastive
        candidates = list(itertools.combinations(
            range(n_free, n_free + n_contrastive), 2))
    chosen = select_pairs(candidates, full_gram, args.n_pairs)
    print(f'mode={args.mode} inject@{inject_after} measure@{measure_after}; '
          f'{len(chosen)} pairs: '
          f'{[(lab(i), lab(j), round(float(full_gram[i,j]),2)) for (i,j) in chosen]}',
          flush=True)

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)

    # Optimized anchor set (the ones the dictionary was learned on).
    anchors = get_h1_anchors(mlp, inject_after, da['n_act'], da['split'],
                             da['seed'], args.data_root, device)

    ncol = min(3, len(chosen))
    nrow = math.ceil(len(chosen) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5.5 * nrow), squeeze=False)

    per_pair = []
    for idx, (i, j) in enumerate(chosen):
        d1, d2 = directions[i], directions[j]
        l2_grid = run_2d_per_anchors(mlp, anchors, d1, d2, inject_after,
                                     measure_after, angles_rad, device)
        median_grid = np.median(l2_grid, axis=0)
        fit = robust_p_fit(angles_deg, l2_grid, n_bootstrap=args.n_bootstrap,
                           exact_geodesic=True)
        p_lo, p_hi, _ = summarize_fit(fit)
        b = boundary_for_set(angles_deg, median_grid, exact=True)
        cosij = float(full_gram[i, j])
        li, lj = lab(i), lab(j)
        print(f'  {li}-{lj} cos={cosij:+.2f}: p_fit={b["fit"]["p"]:.2f} '
              f'(range [{p_lo:.2f},{p_hi:.2f}]) n_pts={b["fit"]["n_pts"]}', flush=True)
        ax = axes[idx // ncol][idx % ncol]
        draw_panel(ax, b, f'{li}-{lj}  cos={cosij:+.2f}: p={b["fit"]["p"]:.2f}',
                   d1lab=li, d2lab=lj)
        per_pair.append({'pair': (i, j), 'cos': cosij, 'p_fit': b['fit']['p'],
                         'p_range': (p_lo, p_hi), 'l2_grid': l2_grid, 'fit': fit})

    for k in range(len(chosen), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')

    mode_lab = {'free_free': 'free-vs-free', 'free_contrastive': 'free-vs-contrastive',
                'contrastive_contrastive': 'contrastive-vs-contrastive'}[args.mode]
    fig.suptitle(f'MNIST MLP d{depth}w{width}: {mode_lab} superellipse boundary fits '
                 f'(exact-geodesic, optimized anchors)', fontsize=13)
    plt.tight_layout()
    png = os.path.join(args.results_dir, 'free_pairs_boundary.png')
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.close()

    with open(os.path.join(args.results_dir, 'fit_free_pairs.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'chosen': chosen, 'angles_deg': angles_deg,
                     'per_pair': per_pair, 'gram': full_gram}, f)

    # summary scatter: p vs |cos|
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cosv = [abs(pp['cos']) for pp in per_pair]
    pv = [pp['p_fit'] for pp in per_pair]
    ax.scatter(cosv, pv, c='C0', s=40)
    for pp in per_pair:
        ax.annotate(f"{lab(pp['pair'][0])}-{lab(pp['pair'][1])}",
                    (abs(pp['cos']), pp['p_fit']), fontsize=7, alpha=0.7)
    ax.axhline(2.0, color='gray', ls=':', label='p=2 (ellipse)')
    ax.set_xlabel('|cos(d_i, d_j)|'); ax.set_ylabel('fitted p (exact)')
    ax.set_title('Fitted p vs pair overlap'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png2 = os.path.join(args.results_dir, 'p_vs_cos.png')
    plt.savefig(png2, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nSaved: {png}\nSaved: {png2}\n'
          f'Saved: {os.path.join(args.results_dir, "fit_free_pairs.pkl")}', flush=True)


if __name__ == '__main__':
    main()
