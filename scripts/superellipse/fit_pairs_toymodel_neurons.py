#!/usr/bin/env python3
"""Pairwise superellipse-p along the toy model's NEURON basis (neuron x neuron).

Same procedure as fit_pairs_toymodel_atoms.py, but the direction set is the
encoder rows of the trained TwoLayerNet (fc1.weight, [H, d]) -- each row is the
input direction that one hidden neuron reads. We perturb along pairs of neuron
directions and fit the superellipse exponent p, exactly as for feature x feature
/ atom x atom. Neuron rows are orthogonalized against the anchor x_clean and
renormalized (matching the atom convention).

Usage:
    python scripts/superellipse/fit_pairs_toymodel_neurons.py \\
        --ckpt /workspace/denoising/runs/phase1_tanh3/model.pt \\
        --denoising_repo /workspace/denoising --max_pairs 190 --device cuda
"""
import argparse
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

from scripts.superellipse.fit_pairs_toymodel import (
    load_toy_model, boundary_radii, fit_from_radii,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='~/Research/denoising/runs/phase1_tanh3/model.pt')
    ap.add_argument('--denoising_repo', default='~/Research/denoising')
    ap.add_argument('--taus', type=float, nargs='+', default=[0.1, 0.2, 0.5])
    ap.add_argument('--n_phis', type=int, default=80)
    ap.add_argument('--alpha_max', type=float, default=15.0)
    ap.add_argument('--n_alpha', type=int, default=2000)
    ap.add_argument('--max_pairs', type=int, default=190,
                    help='cap number of neuron pairs (default 190, ~random x random n)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results/dictionary/toymodel_phase1_tanh3/'
                    'fit_pairs_neuron_vs_neuron')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    model, cfg = load_toy_model(args.ckpt, args.denoising_repo, device)
    d = cfg['d']; sparsity = cfg['sparsity']; S = tuple(range(sparsity))

    # direction set = encoder rows (neuron input directions)
    E = model.fc1.weight.detach().to(device=device, dtype=torch.float32)   # [H, d]
    H = E.shape[0]
    # orthogonalize against anchor x_clean and renormalize (atom convention)
    x_clean = torch.zeros(d, device=device)
    for s in S:
        x_clean[s] = 1.0
    x_hat = x_clean / x_clean.norm()
    N = E - (E @ x_hat).unsqueeze(-1) * x_hat
    N = N / N.norm(dim=-1, keepdim=True)

    pairs = [(i, j) for i in range(H) for j in range(i + 1, H)]
    rng = np.random.default_rng(args.seed)
    if args.max_pairs is not None and len(pairs) > args.max_pairs:
        idx = rng.choice(len(pairs), size=args.max_pairs, replace=False)
        pairs = [pairs[t] for t in idx]
    phis = np.linspace(0.0, math.pi / 2, args.n_phis)
    taus = list(args.taus)
    print(f'd={d} S={S} H={H} neurons  pairs={len(pairs)}  taus={taus}  '
          f'alpha_max={args.alpha_max}', flush=True)

    p_by_tau = {t: [] for t in taus}
    cos_list = []
    per_pair = []
    for n, (i, j) in enumerate(pairs):
        u = N[i]; v = N[j]
        rs_dict = boundary_radii(model, d, S, u, v, phis, taus,
                                 args.alpha_max, args.n_alpha)
        ps = {t: fit_from_radii(rs_dict[t], phis) for t in taus}
        for t in taus:
            p_by_tau[t].append(ps[t])
        cos_ij = float((u * v).sum().item())
        cos_list.append(cos_ij)
        per_pair.append({'pair': (i, j), 'cos': cos_ij, 'p_by_tau': ps})
        if (n + 1) % 40 == 0 or n == 0:
            print(f'  [{n + 1}/{len(pairs)}]  cos={cos_ij:+.3f}  '
                  + '  '.join(f'tau{t}:p={ps[t]:.2f}' for t in taus), flush=True)

    print(f'\n=== neuron x neuron pairwise p (d={d}, H={H}, n_pairs={len(pairs)}) ===')
    summary = {}
    for t in taus:
        v = np.array(p_by_tau[t], float); v = v[np.isfinite(v)]
        if len(v):
            summary[t] = {'median': float(np.median(v)), 'mean': float(np.mean(v)),
                          'iqr': (float(np.quantile(v, .25)), float(np.quantile(v, .75))),
                          'n': int(len(v))}
            print(f'  tau={t}: median={np.median(v):.3f}  mean={np.mean(v):.3f}  '
                  f'IQR=({np.quantile(v,.25):.3f}, {np.quantile(v,.75):.3f})  '
                  f'min={v.min():.3f}  max={v.max():.3f}  n={len(v)}')
        else:
            summary[t] = None
            print(f'  tau={t}: no finite fits')
    cos_arr = np.array(cos_list)
    print(f'  pairwise |cos|: median={np.median(np.abs(cos_arr)):.3f}  '
          f'max={np.abs(cos_arr).max():.3f}')

    with open(os.path.join(args.out, 'fit_pairs.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'config': cfg, 'S': S, 'phis': phis,
                     'taus': taus, 'per_pair': per_pair,
                     'p_by_tau': {t: np.array(p_by_tau[t], float) for t in taus},
                     'cos': cos_arr, 'summary': summary}, f)

    # beeswarm across taus (matches fit_pairs_toymodel.py)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    xpos = np.arange(len(taus))
    labels = []
    for ix, t in enumerate(taus):
        v = np.array(p_by_tau[t], float); v = v[np.isfinite(v)]
        if len(v) == 0:
            labels.append(f'tau={t}\n(no fits)'); continue
        jx = xpos[ix] + (rng.random(len(v)) - 0.5) * 0.35
        ax.scatter(jx, v, s=20, alpha=0.6, edgecolor='black', linewidth=0.3,
                   color='tab:green')
        ax.hlines(np.median(v), xpos[ix] - 0.3, xpos[ix] + 0.3, color='black', linewidth=2.0)
        labels.append(f'tau={t}\nmed={np.median(v):.2f}\nmean={np.mean(v):.2f}\nN={len(v)}')
    ax.axhline(2, color='tab:purple', ls='--', alpha=0.6, label='p=2 (L2)')
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('fitted Lp exponent n'); ax.set_xlabel('output L2 threshold tau')
    ax.set_title(f'toy denoiser ({cfg["activation"]}, d={d}, H={H})  '
                 f'neuron x neuron  N={len(pairs)} pairs/col')
    ax.legend(loc='upper right', fontsize=9); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(args.out, 'p_distribution.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'\nSaved {os.path.join(args.out, "fit_pairs.pkl")}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
