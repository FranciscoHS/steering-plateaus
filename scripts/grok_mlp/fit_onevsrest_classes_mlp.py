#!/usr/bin/env python3
"""
Fit superellipse-p for ALL pairs of one-per-class contrastive directions.

"One direction per class" = one-vs-rest diff-of-means in h1:
    d_c = normalize( mean(h1 | class c) - grand_mean ),   c = 0..9
(grand_mean = unweighted mean of the 10 class means). Then fit p in
|x|^p+|y|^p=1 for all C(10,2)=45 class pairs, via the canonical exp-map 2D
sweep + iso-contour fit on the optimized anchor set.

Reuses class means already stored in a learn_against_contrastive_mlp.py result
pkl (no recompute needed). Outputs a 10x10 p matrix, a p-vs-|cos| scatter, and
the full 45-panel boundary grid.

Usage:
    python fit_onevsrest_classes_mlp.py \
        --dict_pkl results/grok_mlp/against_contrastive/ckpt_s1000000_final_a30_nc10/results.pkl
"""
import argparse
import itertools
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.grok_mlp.fit_superellipse_pair_mlp import (
    build_mlp, get_h1_anchors, run_2d_per_anchors, summarize_fit,
)
from scripts.superellipse.canonical.superellipse import robust_p_fit
from scripts.superellipse.plot_boundary_pair import boundary_for_set, draw_panel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dict_pkl', type=str, required=True,
                   help='a learn_against_contrastive_mlp result pkl (for class_means + metadata)')
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--n_steps', type=int, default=31)
    p.add_argument('--max_angle', type=float, default=60.0)
    p.add_argument('--n_bootstrap', type=int, default=100)
    p.add_argument('--no_panels', action='store_true', help='skip the 45-panel boundary grid')
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--results_dir', type=str, default=None)
    args = p.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    with open(args.dict_pkl, 'rb') as f:
        dd = pickle.load(f)
    if 'class_means' not in dd:
        raise SystemExit("dict pkl has no 'class_means'; rerun learn_against_contrastive_mlp.py")
    class_means = dd['class_means'].to(device=device, dtype=dtype)  # [10, width]
    da = dd['args']
    ckpt_path = args.ckpt if args.ckpt is not None else da['ckpt']
    inject_after = dd['inject_after']
    measure_after = dd['measure_after']

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(args.dict_pkl), 'fit_onevsrest_classes')
    os.makedirs(args.results_dir, exist_ok=True)

    # One-vs-rest direction per class.
    grand = class_means.mean(0, keepdim=True)
    dirs = F.normalize(class_means - grand, dim=-1)  # [10, width]
    n_cls = dirs.shape[0]
    gram = (dirs @ dirs.T).cpu().numpy()

    # Model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    depth = ckpt.get('depth', 4); width = ckpt.get('width', 200)
    mlp = build_mlp(depth, width, act=da.get('activation', 'GeLU')).to(device)
    mlp.load_state_dict(ckpt['model_state_dict'])
    mlp.eval()
    print(f'inject@{inject_after} measure@{measure_after}; {n_cls} one-vs-rest dirs; '
          f'fitting all {n_cls*(n_cls-1)//2} pairs', flush=True)

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)
    anchors = get_h1_anchors(mlp, inject_after, da['n_act'], da['split'],
                             da['seed'], args.data_root, device)

    pairs = list(itertools.combinations(range(n_cls), 2))
    p_mat = np.full((n_cls, n_cls), np.nan)
    plo_mat = np.full((n_cls, n_cls), np.nan)
    phi_mat = np.full((n_cls, n_cls), np.nan)
    per_pair = []
    boundaries = []
    for (i, j) in pairs:
        l2_grid = run_2d_per_anchors(mlp, anchors, dirs[i], dirs[j], inject_after,
                                     measure_after, angles_rad, device)
        median_grid = np.median(l2_grid, axis=0)
        fit = robust_p_fit(angles_deg, l2_grid, n_bootstrap=args.n_bootstrap, exact_geodesic=True)
        p_lo, p_hi, _ = summarize_fit(fit)
        b = boundary_for_set(angles_deg, median_grid, exact=True)
        pf = b['fit']['p']
        p_mat[i, j] = p_mat[j, i] = pf
        plo_mat[i, j] = plo_mat[j, i] = p_lo
        phi_mat[i, j] = phi_mat[j, i] = p_hi
        per_pair.append({'pair': (i, j), 'cos': float(gram[i, j]), 'p_fit': pf,
                         'p_range': (p_lo, p_hi), 'fit': fit, 'l2_grid': l2_grid})
        boundaries.append(b)
        print(f'  class {i}-{j} cos={gram[i,j]:+.2f}: p_fit={pf:.2f} '
              f'(range [{p_lo:.2f},{p_hi:.2f}]) n_pts={b["fit"]["n_pts"]}', flush=True)

    with open(os.path.join(args.results_dir, 'fit_onevsrest.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'angles_deg': angles_deg, 'dirs': dirs.cpu(),
                     'gram': gram, 'pairs': pairs, 'per_pair': per_pair,
                     'p_mat': p_mat, 'p_lo_mat': plo_mat, 'p_hi_mat': phi_mat}, f)

    # 10x10 p matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(p_mat, cmap='RdBu_r', vmin=0.5, vmax=4.0)
    for (i, j) in pairs:
        ax.text(j, i, f'{p_mat[i,j]:.1f}', ha='center', va='center', fontsize=8)
        ax.text(i, j, f'{p_mat[i,j]:.1f}', ha='center', va='center', fontsize=8)
    ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
    ax.set_xlabel('class'); ax.set_ylabel('class')
    ax.set_title(f'Fitted p, one-vs-rest class dirs (exact-geodesic)\n'
                 f'MNIST MLP d{depth}w{width}, all {len(pairs)} pairs')
    plt.colorbar(im, ax=ax, fraction=0.046, label='fitted p')
    plt.tight_layout()
    pm = os.path.join(args.results_dir, 'p_matrix.png')
    plt.savefig(pm, dpi=150, bbox_inches='tight'); plt.close()

    # p vs |cos|
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cosv = [abs(pp['cos']) for pp in per_pair]; pv = [pp['p_fit'] for pp in per_pair]
    ax.scatter(cosv, pv, c='C0', s=40)
    for pp in per_pair:
        ax.annotate(f"{pp['pair'][0]}-{pp['pair'][1]}", (abs(pp['cos']), pp['p_fit']),
                    fontsize=6, alpha=0.7)
    ax.axhline(2.0, color='gray', ls=':', label='p=2 (ellipse)')
    ax.set_xlabel('|cos(d_i, d_j)|'); ax.set_ylabel('fitted p (exact)')
    ax.set_title('Fitted p vs class-pair overlap'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pc = os.path.join(args.results_dir, 'p_vs_cos.png')
    plt.savefig(pc, dpi=150, bbox_inches='tight'); plt.close()

    saved = [pm, pc]
    if not args.no_panels:
        ncol = 5
        nrow = int(np.ceil(len(pairs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.8 * nrow), squeeze=False)
        for idx, ((i, j), b) in enumerate(zip(pairs, boundaries)):
            ax = axes[idx // ncol][idx % ncol]
            draw_panel(ax, b, f'class {i}-{j}  cos={gram[i,j]:+.2f}: p={b["fit"]["p"]:.2f}',
                       d1lab=f'cls{i}', d2lab=f'cls{j}')
        for k in range(len(pairs), nrow * ncol):
            axes[k // ncol][k % ncol].axis('off')
        fig.suptitle(f'MNIST MLP d{depth}w{width}: one-vs-rest class-dir boundary fits '
                     f'(exact-geodesic)', fontsize=14)
        plt.tight_layout()
        bp = os.path.join(args.results_dir, 'boundary_panels.png')
        plt.savefig(bp, dpi=120, bbox_inches='tight'); plt.close()
        saved.append(bp)

    finite = p_mat[np.isfinite(p_mat)]
    print(f'\np over all pairs: min={finite.min():.2f} median={np.median(finite):.2f} '
          f'max={finite.max():.2f}', flush=True)
    print('Saved: ' + '\n       '.join(saved + [os.path.join(args.results_dir, "fit_onevsrest.pkl")]),
          flush=True)


if __name__ == '__main__':
    main()
