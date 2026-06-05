#!/usr/bin/env python3
"""
Spherical L2 perturbation sweep ALONG the contrastive directions (not random).

Like perturb_mlp_spherical.py, but instead of perturbing along random directions
we perturb along each contrastive class direction and plot one L2-vs-angle curve
per class, with a random-direction baseline (median + IQR band) for comparison.

Per anchor a (= h1) and direction d (orthogonalized against a):
    a'(theta) = ||a|| ( cos theta * a_hat + sin theta * d_orth )
L2 = || f(a') - f(a) ||_2 at the last hidden layer. Curves are medians over anchors.

Directions:
  --dir_set onevsrest : d_c = normalize(class_mean_c - grand_mean), 10 (one per class)
  --dir_set classpair : the stored class-pair (cAB) contrastive dirs used in the opt

Usage:
    python perturb_contrastive_sweep_mlp.py \
        --dict_pkl results/grok_mlp/against_contrastive/ckpt_s1000000_final_a30_nc10/results.pkl
"""
import argparse
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
    build_mlp, get_h1_anchors, forward_between,
)


@torch.no_grad()
def sweep_direction(mlp, anchors, d, inject_after, measure_after, angles_rad, device):
    """L2 at the last hidden layer for a spherical sweep along global direction d,
    orthogonalized per anchor. Returns [n_anchors, n_angles]."""
    cos = torch.cos(angles_rad).unsqueeze(1)  # [n_ang, 1]
    sin = torch.sin(angles_rad).unsqueeze(1)
    out = np.zeros((len(anchors), len(angles_rad)))
    for i, a in enumerate(anchors):
        R = a.norm(); a_hat = a / R
        d_orth = d - (d @ a_hat) * a_hat
        d_orth = d_orth / d_orth.norm()
        ref = forward_between(mlp, a.unsqueeze(0), inject_after, measure_after)  # [1, w]
        pts = R * (cos * a_hat.unsqueeze(0) + sin * d_orth.unsqueeze(0))  # [n_ang, w]
        pf = forward_between(mlp, pts, inject_after, measure_after)
        out[i] = (pf - ref).norm(dim=-1).cpu().numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dict_pkl', type=str, required=True)
    p.add_argument('--dir_set', type=str, default='onevsrest',
                   choices=['onevsrest', 'classpair'])
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--n_act', type=int, default=100, help='anchors to average over')
    p.add_argument('--n_random', type=int, default=20)
    p.add_argument('--n_angles', type=int, default=60)
    p.add_argument('--max_angle', type=float, default=60.0)
    p.add_argument('--anchor_seed', type=int, default=123)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--results_dir', type=str, default=None)
    args = p.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    with open(args.dict_pkl, 'rb') as f:
        dd = pickle.load(f)
    da = dd['args']
    ckpt_path = args.ckpt if args.ckpt is not None else da['ckpt']
    inject_after = dd['inject_after']; measure_after = dd['measure_after']

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(args.dict_pkl),
                                        f'perturb_contrastive_{args.dir_set}')
    os.makedirs(args.results_dir, exist_ok=True)

    # Model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    depth = ckpt.get('depth', 4); width = ckpt.get('width', 200)
    mlp = build_mlp(depth, width, act=da.get('activation', 'GeLU')).to(device)
    mlp.load_state_dict(ckpt['model_state_dict']); mlp.eval()

    # Directions + labels
    if args.dir_set == 'onevsrest':
        cm = dd['class_means'].to(device=device, dtype=dtype)
        dirs = F.normalize(cm - cm.mean(0, keepdim=True), dim=-1)  # [10, w]
        labels = [f'class {c}' for c in range(dirs.shape[0])]
    else:
        dirs = dd['contrastive_dirs'].to(device=device, dtype=dtype)
        labels = [f'c{a}-{b}' for (a, b) in dd['class_pairs']]
    print(f'inject@{inject_after} measure@{measure_after}; {dirs.shape[0]} '
          f'{args.dir_set} dirs; {args.n_act} anchors', flush=True)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = torch.from_numpy(np.deg2rad(angles_deg)).to(device=device, dtype=dtype)

    anchors = get_h1_anchors(mlp, inject_after, args.n_act, da['split'],
                             args.anchor_seed, args.data_root, device)

    # Contrastive direction curves (median over anchors)
    dir_curves = []
    for k in range(dirs.shape[0]):
        g = sweep_direction(mlp, anchors, dirs[k], inject_after, measure_after,
                            angles_rad, device)
        dir_curves.append(np.median(g, axis=0))
    dir_curves = np.stack(dir_curves)  # [n_dir, n_angles]

    # Random baseline: n_random global unit dirs, each median over anchors
    rand_curves = []
    for _ in range(args.n_random):
        rd = torch.randn(width, generator=gen, device=device, dtype=dtype)
        rd = rd / rd.norm()
        g = sweep_direction(mlp, anchors, rd, inject_after, measure_after,
                            angles_rad, device)
        rand_curves.append(np.median(g, axis=0))
    rand_curves = np.stack(rand_curves)  # [n_random, n_angles]

    results = {'args': vars(args), 'angles_deg': angles_deg, 'labels': labels,
               'dir_curves': dir_curves, 'rand_curves': rand_curves}
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap('tab10')
    for k in range(dirs.shape[0]):
        ax.plot(angles_deg, dir_curves[k], color=cmap(k % 10), lw=2, label=labels[k])
    # random band
    r_med = np.median(rand_curves, axis=0)
    r_lo = np.percentile(rand_curves, 25, axis=0)
    r_hi = np.percentile(rand_curves, 75, axis=0)
    ax.plot(angles_deg, r_med, color='black', lw=2.2, ls='--',
            label=f'random (median, n={args.n_random})')
    ax.fill_between(angles_deg, r_lo, r_hi, color='gray', alpha=0.25, label='random IQR')
    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('L2 distance @ last hidden')
    ax.set_xlim(0, args.max_angle)
    ax.set_title(f'MNIST MLP d{depth}w{width}: L2 sweep along {args.dir_set} '
                 f'contrastive dirs vs random (median over {args.n_act} anchors)')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png = os.path.join(args.results_dir, 'sweep_contrastive.png')
    plt.savefig(png, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved: {png}\nSaved: {os.path.join(args.results_dir, "results.pkl")}', flush=True)


if __name__ == '__main__':
    main()
