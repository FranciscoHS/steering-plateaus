#!/usr/bin/env python3
"""
Decompose WHY difference-of-activations directions show no plateau in penultimate-layer
L2, while random/contrastive do. Is it the -a_base subtraction, the residual-stream
mean/bias, or genuine feature directions?

Base activation a_base = last token of "The house was big" at layer L (gpt2-xl L2).
Real activations a_X are sampled from FineWeb (a pool). We build several direction
types, all unit-normalised, and sweep each additively, measuring L2 at the second-to-last
layer, then quantify plateau shape with the same metrics as analyze_plateau_metric.py:

  rate_init = normalized initial rate of change at m~0.15*m_max  (1=linear, <1=plateau)
  auc       = area under normalized curve                       (0.5=linear, <0.5=plateau)

Direction types:
  real_diff   : a_X - a_base                       (has -a_base; our original "diff")
  single      : a_X                                (no -a_base, but carries the mean/bias)
  mean        : mean(a_pool)                       (pure bias/mean direction; n=1)
  mixture     : a_X1 - a_X2                         (no a_base, no mean; paper's clean control)
  diff_orth   : (a_X - a_base) orthogonalised vs a_base   (removes only the a_base component)
  random      : isotropic unit vectors

Also reports overlap diagnostics: cos(a_X, mean), cos(real_diff, a_base), etc.

Usage:
    python scripts/perturbation/decompose_diff_directions.py --model gpt2-xl --layer 2
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.perturbation.sweep_diff_of_activations import get_prompt_activation, run_additive_sweep
from scripts.perturbation.run_perturbation_sweep_spherical import extract_context_and_activation
from scripts.perturbation.analyze_plateau_metric import curve_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='gpt2-xl')
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--measure_layer', type=int, default=-2)
    p.add_argument('--prefix', type=str, default='The house was')
    p.add_argument('--base_word', type=str, default='big')
    p.add_argument('--n', type=int, default=60, help='number of directions per group')
    p.add_argument('--m_max', type=float, default=180.0)
    p.add_argument('--n_steps', type=int, default=80)
    p.add_argument('--seq_len', type=int, default=5)
    p.add_argument('--results_dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.results_dir is None:
        args.results_dir = f'results/diff_of_activations/{args.model}_L{args.layer}_decompose'
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Loading {args.model}...")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    measure_layer = args.measure_layer if args.measure_layer >= 0 else n_layers + args.measure_layer
    print(f"d_model={d_model}, n_layers={n_layers}, measure_layer={measure_layer}")

    base_prompt = f'{args.prefix} {args.base_word}'
    ctx, a_base = get_prompt_activation(model, tokenizer, base_prompt, args.layer, device)
    a_base_hat = F.normalize(a_base, dim=0)
    print(f"base: '{base_prompt}'  ||a_base||={a_base.norm():.2f}")

    # Pool of real activations (2*n so we can form disjoint mixture pairs).
    n_pool = 2 * args.n
    tok_list = load_fineweb_fixed_length(n_pool, tokenizer, seq_len=args.seq_len, seed=args.seed)
    _, pool = extract_context_and_activation(model, tokenizer, tok_list, args.layer, device)  # [2n, d]
    targets = pool[:args.n]                 # a_X for diff/single/diff_orth
    others = pool[args.n:]                  # second element of mixture pairs
    mean_vec = pool.mean(0)                 # residual-stream mean/bias

    # Build unit-norm direction sets.
    real_diff = targets - a_base.unsqueeze(0)
    diff_orth = real_diff - (real_diff @ a_base_hat).unsqueeze(1) * a_base_hat.unsqueeze(0)
    mixture = targets - others
    rand = torch.randn(args.n, d_model)

    groups = {
        'real_diff':   F.normalize(real_diff, dim=-1),
        'single':      F.normalize(targets, dim=-1),
        'single-mean': F.normalize(targets - mean_vec.unsqueeze(0), dim=-1),
        'mean':        F.normalize(mean_vec, dim=0).unsqueeze(0),
        'neg_base':    F.normalize(-a_base, dim=0).unsqueeze(0),
        'mixture':     F.normalize(mixture, dim=-1),
        'diff_orth':   F.normalize(diff_orth, dim=-1),
        'random':      F.normalize(rand, dim=-1),
    }

    # Overlap diagnostics (how contaminated is each group by a_base / mean).
    mean_hat = F.normalize(mean_vec, dim=0)
    diag = {
        'cos(single, mean)':   (groups['single'] @ mean_hat).abs().mean().item(),
        'cos(single, a_base)': (groups['single'] @ a_base_hat).abs().mean().item(),
        'cos(real_diff, a_base)': (groups['real_diff'] @ a_base_hat).mean().item(),
        'cos(mixture, a_base)':   (groups['mixture'] @ a_base_hat).abs().mean().item(),
        'cos(mixture, mean)':     (groups['mixture'] @ mean_hat).abs().mean().item(),
        'cos(single-mean, a_base)': (groups['single-mean'] @ a_base_hat).abs().mean().item(),
        'cos(single-mean, mean)':   (groups['single-mean'] @ mean_hat).abs().mean().item(),
        'cos(diff_orth, a_base)': (groups['diff_orth'] @ a_base_hat).abs().mean().item(),
        'cos(random, a_base)':    (groups['random'] @ a_base_hat).abs().mean().item(),
        '||mean|| / mean||a_X||': (mean_vec.norm() / pool.norm(dim=-1).mean()).item(),
    }
    print("\nOverlap diagnostics:")
    for k, v in diag.items():
        print(f"  {k:26s} = {v:.3f}")

    mags = np.linspace(0, args.m_max, args.n_steps + 1)[1:]
    order = ['real_diff', 'single', 'single-mean', 'mean', 'neg_base', 'mixture', 'diff_orth', 'random']

    print("\nSweeping groups...")
    out = {'mags': mags, 'measure_layer': measure_layer, 'base_prompt': base_prompt,
           'diagnostics': diag, 'groups': {}}
    t0 = time.time()
    for g in order:
        l2, _ = run_additive_sweep(model, ctx, a_base, groups[g], mags, args.layer, device, measure_layer)
        l2 = l2.T  # [n_dirs, n_steps]
        m = np.array([curve_metrics(l2[i], mags) for i in range(l2.shape[0])])
        out['groups'][g] = {'l2': l2, 'rate_init': m[:, 0], 'auc': m[:, 1]}
        print(f"  {g:11s} n={l2.shape[0]:3d}  rate_init={m[:,0].mean():.3f}±{m[:,0].std():.3f}  "
              f"auc={m[:,1].mean():.3f}±{m[:,1].std():.3f}  ({time.time()-t0:.1f}s)")

    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(out, f)

    # --- Plots ---
    cmap = plt.get_cmap('tab10')
    colors = {g: cmap(i) for i, g in enumerate(order)}
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    mn = mags / mags[-1]
    ax = axes[0]
    for g in order:
        l2 = out['groups'][g]['l2']
        yn = l2 / l2[:, -1:]
        ax.plot(mn, np.median(yn, axis=0), color=colors[g], lw=2, label=g)
        if l2.shape[0] > 1:
            ax.fill_between(mn, np.percentile(yn, 25, axis=0), np.percentile(yn, 75, axis=0),
                            color=colors[g], alpha=0.10)
    ax.plot([0, 1], [0, 1], color='gray', ls=':', label='linear (no plateau)')
    ax.set_xlabel('normalized magnitude m / m_max')
    ax.set_ylabel('normalized L2 / L2(m_max)')
    ax.set_title('Median normalized response shape')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    for ax, metric, ref, reflabel in [(axes[1], 'rate_init', 1.0, 'linear=1'),
                                       (axes[2], 'auc', 0.5, 'linear=0.5')]:
        data = [out['groups'][g][metric] for g in order]
        ax.boxplot(data, positions=range(1, len(order) + 1), showfliers=False, widths=0.5)
        ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order)
        for i, g in enumerate(order):
            x = np.random.normal(i + 1, 0.06, size=len(out['groups'][g][metric]))
            ax.scatter(x, out['groups'][g][metric], color=colors[g], alpha=0.6, s=18, zorder=3)
        ax.axhline(ref, color='gray', ls=':', label=reflabel)
        ax.set_ylabel(metric); ax.set_title(f'{metric}  ({reflabel}; <ref = plateau)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

    fig.suptitle(f"{args.model} L{args.layer}: decomposing the no-plateau effect "
                 f"(base='{base_prompt}', L2@layer {measure_layer})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'decompose.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot: {path}")
    print(f"Saved data: {os.path.join(args.results_dir, 'results.pkl')}")


if __name__ == '__main__':
    main()
