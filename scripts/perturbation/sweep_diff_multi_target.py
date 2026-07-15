#!/usr/bin/env python3
"""
Multi-target difference-of-activations sweep.

Question (Francisco): the diff-of-activations directions (big->in, big->large) show
NO plateau in second-to-last-layer L2 distance -- L2 rises immediately -- whereas both
random directions and contrastive (Gender/Number) directions DO plateau. Is the
"no plateau" property specific to those targets, or does moving toward ANY other token's
activation behave this way?

Here we fix prompt_a = "The house was big" and sweep the difference direction
d_X = normalize(a_X - a_big) for several target completions X, overlaying their L2 (and
KL) response curves on a COMMON absolute perturbation-magnitude axis, against the random
band and the Gender/Number contrastive directions. All directions are unit vectors swept
over the same magnitudes m, so the comparison is at equal input-space step size.

Usage:
    python scripts/perturbation/sweep_diff_multi_target.py --model gpt2-xl --layer 2
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
from scripts.perturbation.sweep_diff_of_activations import (
    get_prompt_activation, run_additive_sweep, load_contrastive_directions,
)

DEFAULT_TARGETS = ['small', 'large', 'old', 'red', 'empty', 'expensive', 'haunted', 'on']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='gpt2-xl')
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--measure_layer', type=int, default=-2)
    p.add_argument('--prefix', type=str, default='The house was',
                   help='shared prompt prefix; base = prefix + " " + base_word')
    p.add_argument('--base_word', type=str, default='big')
    p.add_argument('--targets', type=str, default=','.join(DEFAULT_TARGETS),
                   help='comma-separated target words X (direction = a_X - a_base)')
    p.add_argument('--n_dirs', type=int, default=20, help='random directions')
    p.add_argument('--m_max', type=float, default=180.0,
                   help='max absolute perturbation magnitude on common x-axis')
    p.add_argument('--n_steps', type=int, default=80)
    p.add_argument('--contrastive_pkl', type=str,
                   default='results/paired_directions/gpt2-xl_L2/directions.pkl')
    p.add_argument('--contrastive_keys', type=str, default='Gender,Number')
    p.add_argument('--results_dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.results_dir is None:
        args.results_dir = f'results/diff_of_activations/{args.model}_L{args.layer}_multitarget'
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Loading {args.model}...")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    measure_layer = args.measure_layer if args.measure_layer >= 0 else n_layers + args.measure_layer
    print(f"d_model={d_model}, n_layers={n_layers}, measure_layer={measure_layer}")

    base_prompt = f'{args.prefix} {args.base_word}'
    ctx, a_base = get_prompt_activation(model, tokenizer, base_prompt, args.layer, device)
    a_norm = a_base.norm().item()
    print(f"base: '{base_prompt}'  ||a_base||={a_norm:.2f}")

    # Common absolute-magnitude axis (skip 0).
    mags = np.linspace(0, args.m_max, args.n_steps + 1)[1:]

    targets = [t.strip() for t in args.targets.split(',')]
    results = {'mags': mags, 'a_norm': a_norm, 'measure_layer': measure_layer,
               'base_prompt': base_prompt, 'targets': {}}

    print(f"\n=== {len(targets)} target diff directions ===")
    t0 = time.time()
    for X in targets:
        _, a_X = get_prompt_activation(model, tokenizer, f'{args.prefix} {X}', args.layer, device)
        d = a_X - a_base
        d_norm = d.norm().item()
        l2, kl = run_additive_sweep(model, ctx, a_base, F.normalize(d, dim=0).unsqueeze(0),
                                    mags, args.layer, device, measure_layer)
        results['targets'][X] = {'l2': l2[:, 0], 'kl': kl[:, 0], 'd_norm': d_norm}
        print(f"  big->{X:10s} ||d||={d_norm:.2f}  ({time.time()-t0:.1f}s)")

    print(f"=== random (n={args.n_dirs}) ===")
    rand_dirs = F.normalize(torch.randn(args.n_dirs, d_model), dim=-1)
    r_l2, r_kl = run_additive_sweep(model, ctx, a_base, rand_dirs, mags, args.layer,
                                    device, measure_layer)
    results['random'] = {'l2': r_l2, 'kl': r_kl}

    contr = None
    if args.contrastive_pkl:
        keys = args.contrastive_keys.split(',') if args.contrastive_keys else None
        cdirs, cnames = load_contrastive_directions(args.contrastive_pkl, args.layer, keys)
        c_l2, c_kl = run_additive_sweep(model, ctx, a_base, cdirs, mags, args.layer,
                                        device, measure_layer)
        contr = {'l2': c_l2, 'kl': c_kl, 'names': cnames}
        results['contrastive'] = contr

    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    diff_cmap = plt.get_cmap('autumn')
    for ax, key, name in [(axes[0], 'l2', f'L2 @ layer {measure_layer}'),
                          (axes[1], 'kl', 'KL @ final')]:
        for i, X in enumerate(targets):
            t = results['targets'][X]
            ax.plot(mags, t[key], color=diff_cmap(i / max(len(targets) - 1, 1)), lw=2,
                    label=f'big->{X} (||d||={t["d_norm"]:.0f})')
        rand = results['random'][key]
        ax.plot(mags, np.median(rand, axis=1), color='black', lw=2.5, ls='--', label='random (median)')
        ax.fill_between(mags, np.percentile(rand, 25, axis=1), np.percentile(rand, 75, axis=1),
                        color='black', alpha=0.12)
        if contr is not None:
            for j, nm in enumerate(contr['names']):
                ax.plot(mags, contr[key][:, j], lw=2, ls=':', label=f'contr:{nm}')
        ax.set_xlabel('perturbation magnitude ||m * d_hat||')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"{args.model} L{args.layer}: diff-of-activations toward various tokens "
                 f"vs random/contrastive  (base='{base_prompt}', ||a||={a_norm:.0f})", fontsize=12)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot: {path}")
    print(f"Saved data: {os.path.join(args.results_dir, 'results.pkl')}")


if __name__ == '__main__':
    main()
