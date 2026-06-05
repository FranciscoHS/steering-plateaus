#!/usr/bin/env python3
"""
Canonical superellipse-p fit for a learned (d1, d2) pair on the MNIST MLP.

Port of scripts/superellipse/fit_superellipse_pair_v2.py. The fitting geometry
(exp-map chart in scripts/superellipse/canonical/parametrize.py) and the
contour-extraction + |x|^p + |y|^p = 1 fit with bootstrap robustness
(scripts/superellipse/canonical/superellipse.py) are model-agnostic and are
reused verbatim. Only the 2D sweep is re-implemented for the MLP: a perturbed
h1 activation is pushed through forward_between(inject_after -> measure_after)
and the L2 distance to the unperturbed last-hidden response is recorded.

Procedure (per anchor a = h1):
  - 31x31 grid in tangent coords (alpha_1, alpha_2) up to 60 deg via exp_map_2d
    (orthonormal tangent directions d1_perp, d2_perp at a_hat = a/||a||);
  - L2 at the last hidden layer per grid point -> l2_grid (n_anchors, n, n);
  - extract iso-L2 contour, normalize by axis intercepts, fit p, with
    robustness over thresholds {0.5,0.75}xbase x max_alpha {30,45,60} x bootstrap;
  - report p for small-angle and exact-geodesic normalizations, on the anchor
    set the pair was optimized on and on a fresh set.

Usage:
    python fit_superellipse_pair_mlp.py \
        --dict_pkl results/grok_mlp/superellipse_dict/ckpt_s1000000_final_a30/results.pkl
"""
import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.superellipse.canonical.parametrize import orthonormalize, exp_map_2d
from scripts.superellipse.canonical.superellipse import robust_p_fit, superellipse_curve

ACT = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'Sigmoid': nn.Sigmoid, 'GeLU': nn.GELU}


def build_mlp(depth, width, in_dims=(28, 28), n_classes=10, act='GeLU'):
    layers = [nn.Flatten()]
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(int(np.prod(in_dims)), width))
            layers.append(ACT[act]())
        elif i == depth - 1:
            layers.append(nn.Linear(width, n_classes))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(ACT[act]())
    return nn.Sequential(*layers)


def hidden_act_indices(mlp):
    return [i for i, m in enumerate(mlp) if isinstance(m, tuple(ACT.values()))]


@torch.no_grad()
def forward_to(mlp, x, stop_after):
    out = x
    for m in mlp[:stop_after + 1]:
        out = m(out)
    return out


@torch.no_grad()
def forward_between(mlp, h, start_after, stop_after):
    out = h
    for m in mlp[start_after + 1:stop_after + 1]:
        out = m(out)
    return out


@torch.no_grad()
def sweep_2d_one_anchor(mlp, anchor, d1, d2, inject_after, measure_after,
                        alphas_rad, device, batch_size=512):
    """2D grid sweep in (alpha_1, alpha_2). Returns l2 grid (n, n).

    Uses the exact exp-map chart (parametrize.exp_map_2d) so ||a'|| = ||a||
    and the geodesic angle equals r = sqrt(alpha_1^2 + alpha_2^2)."""
    a = anchor.to(device)
    a_hat = a / a.norm()
    d1p, d2p = orthonormalize(a_hat, d1.to(device), d2.to(device))
    alphas_t = torch.from_numpy(alphas_rad).to(device=device, dtype=a.dtype)
    A1, A2 = torch.meshgrid(alphas_t, alphas_t, indexing='ij')
    pts = exp_map_2d(a, d1p, d2p, A1.flatten(), A2.flatten())  # (n*n, D)
    n = len(alphas_rad)
    n_total = pts.shape[0]

    ref = forward_between(mlp, a.unsqueeze(0), inject_after, measure_after)  # (1, D)
    l2 = np.zeros(n_total, dtype=np.float64)
    for st in range(0, n_total, batch_size):
        ed = min(st + batch_size, n_total)
        pf = forward_between(mlp, pts[st:ed], inject_after, measure_after)
        l2[st:ed] = (pf - ref).norm(dim=-1).cpu().numpy()
    return l2.reshape(n, n)


def run_2d_per_anchors(mlp, anchors, d1, d2, inject_after, measure_after,
                       alphas_rad, device):
    n = len(alphas_rad)
    out = np.zeros((len(anchors), n, n), dtype=np.float64)
    for i in range(len(anchors)):
        out[i] = sweep_2d_one_anchor(mlp, anchors[i], d1, d2, inject_after,
                                     measure_after, alphas_rad, device)
    return out


def get_h1_anchors(mlp, inject_after, n_act, split, seed, data_root, device):
    """First n_act h1 activations under a shuffle generator (matches the
    dict script's anchor selection when seed/split/n_act agree)."""
    ds = torchvision.datasets.MNIST(root=data_root, train=(split == 'train'),
        transform=torchvision.transforms.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=n_act, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    x, _ = next(iter(loader))
    h1 = forward_to(mlp, x.to(device), inject_after)
    return [h1[i].clone() for i in range(n_act)]


def summarize_fit(fit):
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
    p = argparse.ArgumentParser()
    p.add_argument('--dict_pkl', type=str, required=True,
                   help='results.pkl from learn_superellipse_dict_mlp.py')
    p.add_argument('--pair_i', type=int, default=0)
    p.add_argument('--pair_j', type=int, default=1)
    p.add_argument('--ckpt', type=str, default=None,
                   help='override checkpoint (default: from dict pkl args)')
    p.add_argument('--n_steps', type=int, default=31)
    p.add_argument('--max_angle', type=float, default=60.0)
    p.add_argument('--n_act_fresh', type=int, default=30)
    p.add_argument('--fresh_seed', type=int, default=999)
    p.add_argument('--n_bootstrap', type=int, default=200)
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
    n_act_train = da['n_act']
    train_seed = da['seed']
    split = da['split']

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(args.dict_pkl),
                                        f'fit_canonical_pair{args.pair_i}-{args.pair_j}')
    os.makedirs(args.results_dir, exist_ok=True)

    # Model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    depth = ckpt.get('depth', 4)
    width = ckpt.get('width', 200)
    mlp = build_mlp(depth, width, act=da.get('activation', 'GeLU')).to(device)
    mlp.load_state_dict(ckpt['model_state_dict'])
    mlp.eval()
    print(f'depth={depth} width={width} | inject after module {inject_after}, '
          f'measure after module {measure_after} | pair=({args.pair_i},{args.pair_j})',
          flush=True)

    d1 = directions[args.pair_i]
    d2 = directions[args.pair_j]
    print(f'cos(d{args.pair_i+1}, d{args.pair_j+1}) = '
          f'{(d1 @ d2 / (d1.norm()*d2.norm())).item():.3f}', flush=True)

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)

    # Two anchor sets: the ones the pair was optimized on, and a fresh set.
    opt_anchors = get_h1_anchors(mlp, inject_after, n_act_train, split,
                                 train_seed, args.data_root, device)
    fresh_anchors = get_h1_anchors(mlp, inject_after, args.n_act_fresh, split,
                                   args.fresh_seed, args.data_root, device)
    sets = {'optimized': opt_anchors, 'fresh': fresh_anchors}

    results = {'args': vars(args), 'dict_args': da, 'inject_after': inject_after,
               'measure_after': measure_after, 'angles_deg': angles_deg, 'sets': {}}

    for set_name, anchors in sets.items():
        print(f'\n=== Set: {set_name} ({len(anchors)} anchors) ===', flush=True)
        l2_grid = run_2d_per_anchors(mlp, anchors, d1, d2, inject_after,
                                     measure_after, angles_rad, device)
        print(f'  l2 grid {l2_grid.shape}, max={l2_grid.max():.3f}', flush=True)
        set_out = {'l2_grid': l2_grid, 'fits': {}}
        for norm_label, exact in [('small_angle', False), ('exact_geodesic', True)]:
            fit = robust_p_fit(angles_deg, l2_grid, n_bootstrap=args.n_bootstrap,
                               exact_geodesic=exact)
            p_lo, p_hi, rows = summarize_fit(fit)
            print(f'  [{norm_label}] p range [{p_lo:.2f}, {p_hi:.2f}] across '
                  f'{len(rows)} cells:', flush=True)
            for k, v in rows:
                print(f'      {str(k):16s} p_med={v["p_median"]:.2f} '
                      f'[{v["p_lo"]:.2f}, {v["p_hi"]:.2f}]  n_pts={v["n_pts"]} '
                      f'intercepts=({v["intercepts"][0]:.1f},{v["intercepts"][1]:.1f})',
                      flush=True)
            set_out['fits'][norm_label] = fit
            set_out[f'p_range_{norm_label}'] = (p_lo, p_hi)
        results['sets'][set_name] = set_out

    with open(os.path.join(args.results_dir, 'fit_canonical.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Plot: median L2 grid + iso-contours + fitted superellipse (exact, 50% thr)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for col, set_name in enumerate(['optimized', 'fresh']):
        ax = axes[col]
        so = results['sets'][set_name]
        grid = np.median(so['l2_grid'], axis=0)
        base = min(grid[:, 0].max(), grid[0, :].max())
        im = ax.pcolormesh(angles_deg, angles_deg, grid.T, shading='auto', cmap='viridis')
        plt.colorbar(im, ax=ax, fraction=0.046, label='median L2')
        cs = ax.contour(angles_deg, angles_deg, grid.T,
                        levels=[0.5 * base, 0.75 * base], colors='white', linewidths=1.2)
        ax.clabel(cs, fmt='%.2f', fontsize=7)
        fit = so['fits']['exact_geodesic']
        cell = fit.get(('50', 45.0))
        if cell and np.isfinite(cell['p_median']):
            t1, t2 = cell['intercepts']
            pval = cell['p_median']
            xs, ys = superellipse_curve(pval)
            xs = np.abs(xs) * t1; ys = np.abs(ys) * t2
            ax.plot(xs, ys, 'r--', lw=2, label=f'superellipse p={pval:.2f}')
        plo, phi = so['p_range_exact_geodesic']
        ax.set_title(f'{set_name}: p in [{plo:.2f}, {phi:.2f}] (exact)')
        ax.set_xlabel('alpha_1 (deg, along d1)')
        ax.set_ylabel('alpha_2 (deg, along d2)')
        ax.legend(fontsize=9)
    fig.suptitle(f'MNIST MLP d{depth}w{width} h-inject->h-measure: canonical superellipse '
                 f'fit, pair ({args.pair_i},{args.pair_j})', fontsize=13)
    plt.tight_layout()
    png = os.path.join(args.results_dir, 'fit_canonical.png')
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {os.path.join(args.results_dir, "fit_canonical.pkl")}', flush=True)
    print(f'Saved: {png}', flush=True)


if __name__ == '__main__':
    main()
