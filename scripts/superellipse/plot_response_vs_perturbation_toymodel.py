#!/usr/bin/env python3
"""1-D response vs perturbation on the trained toy model, by direction type.

Plots R(alpha) = ||f(x_clean + alpha * dir) - f(x_clean)||_2 vs alpha for single
directions of four kinds:
  - feature   : true features e_j, j not in S
  - neuron    : encoder rows fc1.weight[h] (anchor-orthogonalized, unit)
  - candidate : optimized atoms (anchor-orthogonalized, unit)
  - random    : Gaussian, S-zeroed, unit
Faint line per direction + a bold median line per type. Horizontal refs at the
fit thresholds tau. This visualizes which directions are most responsive (break
the plateau with least perturbation) and the onset shape.

Neurons/atoms are anchor-orthogonalized + renormalized to match the convention
used in the p-fits (true features e_j are already orthogonal to x_clean).

Usage:
    python scripts/superellipse/plot_response_vs_perturbation_toymodel.py \\
        --ckpt /workspace/denoising/runs/phase1_tanh3/model.pt \\
        --denoising_repo /workspace/denoising --device cuda
"""
import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.superellipse.fit_pairs_toymodel import load_toy_model


@torch.no_grad()
def sweep(model, x_clean, dirs, alpha_max, n_alpha):
    """dirs: [N, d] unit directions. Returns L2[N, n_alpha]."""
    device = x_clean.device
    base = model(x_clean.unsqueeze(0))[0]
    alphas = torch.linspace(0.0, alpha_max, n_alpha, device=device)
    d = x_clean.numel()
    out_l2 = np.empty((dirs.shape[0], n_alpha), dtype=np.float64)
    for i in range(dirs.shape[0]):
        x = x_clean.view(1, d) + alphas.view(n_alpha, 1) * dirs[i].view(1, d)
        out = model(x)
        out_l2[i] = (out - base.view(1, d)).norm(p=2, dim=1).cpu().numpy()
    return out_l2, alphas.cpu().numpy()


def anchor_orthonormal(M, x_hat):
    M = M - (M @ x_hat).unsqueeze(-1) * x_hat
    return M / M.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='~/Research/denoising/runs/phase1_tanh3/model.pt')
    ap.add_argument('--denoising_repo', default='~/Research/denoising')
    ap.add_argument('--atoms', default='results/dictionary/toymodel_phase1_tanh3/'
                    'optimize_features_randcontrast/atoms.pt')
    ap.add_argument('--n_each', type=int, default=32, help='dirs per type (atoms: all)')
    ap.add_argument('--alpha_max', type=float, default=15.0)
    ap.add_argument('--n_alpha', type=int, default=600)
    ap.add_argument('--taus', type=float, nargs='+', default=[0.1, 0.2, 0.5])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results/dictionary/toymodel_phase1_tanh3/response_vs_perturbation')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device(args.device)
    model, cfg = load_toy_model(args.ckpt, args.denoising_repo, device)
    d = cfg['d']; S = tuple(range(cfg['sparsity']))
    rng = np.random.default_rng(args.seed)
    x_clean = torch.zeros(d, device=device)
    for s in S:
        x_clean[s] = 1.0
    x_hat = x_clean / x_clean.norm()
    inactive = [i for i in range(d) if i not in S]

    # --- build direction sets ---
    feats = rng.choice(inactive, size=args.n_each, replace=False)
    F = torch.zeros(args.n_each, d, device=device)
    for i, j in enumerate(feats):
        F[i, j] = 1.0

    E = model.fc1.weight.detach().to(device=device, dtype=torch.float32)  # [H, d]
    hsel = rng.choice(E.shape[0], size=min(args.n_each, E.shape[0]), replace=False)
    N = anchor_orthonormal(E[hsel], x_hat)

    A = torch.load(os.path.expanduser(args.atoms), map_location=device,
                   weights_only=False)['dictionary'].to(torch.float32)
    A = anchor_orthonormal(A, x_hat)

    Rr = torch.from_numpy(rng.standard_normal((args.n_each, d)).astype(np.float32)).to(device)
    Rr[:, list(S)] = 0.0
    Rr = Rr / Rr.norm(dim=-1, keepdim=True)

    sets = {'feature': F, 'neuron': N, 'candidate': A, 'random': Rr}
    colors = {'feature': '#1F77B4', 'neuron': '#3a7d3a',
              'candidate': '#D9822B', 'random': '#999999'}

    curves = {}
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for name, dirs in sets.items():
        l2, alphas = sweep(model, x_clean, dirs, args.alpha_max, args.n_alpha)
        curves[name] = l2
        for i in range(l2.shape[0]):
            ax.plot(alphas, l2[i], color=colors[name], lw=0.5, alpha=0.20)
        ax.plot(alphas, np.median(l2, axis=0), color=colors[name], lw=2.6,
                label=f'{name} (n={l2.shape[0]}, median)')
        print(f'{name}: n={l2.shape[0]}  median L2 at alpha_max={np.median(l2[:,-1]):.3f}',
              flush=True)
    for t in args.taus:
        ax.axhline(t, color='black', ls=':', lw=0.8, alpha=0.5)
        ax.text(args.alpha_max, t, f' tau={t}', va='center', fontsize=8, alpha=0.7)
    ax.set_xlabel(r'perturbation magnitude $\alpha$', fontsize=12)
    ax.set_ylabel(r'response  $\|f(x_0+\alpha\hat d)-f(x_0)\|_2$', fontsize=12)
    ax.set_title(f'1-D response vs perturbation by direction type '
                 f'({cfg["activation"]}, d={d}, H={cfg["hidden_size"]})', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    png = os.path.join(args.out, 'response_vs_perturbation.png')
    fig.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(png.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    with open(os.path.join(args.out, 'curves.pkl'), 'wb') as f:
        pickle.dump({'alphas': alphas, 'curves': curves, 'taus': args.taus,
                     'config': cfg}, f)
    print(f'saved {png}', flush=True)
    print(f'saved {os.path.join(args.out, "curves.pkl")}', flush=True)


if __name__ == '__main__':
    main()
