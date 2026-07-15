#!/usr/bin/env python3
"""
Plateau metric: do difference-of-activations directions lack the plateau that random
and contrastive directions show?

For a base activation a (last token of "The house was big" at layer L), we perturb
additively along many unit directions and read the second-to-last-layer L2 response
L2(m) over a common magnitude axis m. We then quantify the SHAPE of each curve:

    yn = L2 / L2(m_max),   mn = m / m_max
    rate_init = yn(mn~0.15) / 0.15      normalized initial rate of change
                                        ( =1 linear, <1 plateau/slow-start, >1 saturating )
    auc       = trapz(yn, mn)           ( =0.5 linear, <0.5 convex/plateau, >0.5 concave )

Three groups of directions, compared as distributions:
  - diff_words   : a_X - a_base for X = many completions of "The house was X" (real tokens)
  - diff_fineweb : a_X - a_base for X = last-token activations of random FineWeb sequences
  - random       : isotropic unit vectors
  - contrastive  : diff-of-means concept directions (Gender, Number, Tense, ...)

Usage:
    python scripts/perturbation/analyze_plateau_metric.py --model gpt2-xl --layer 2
"""

import argparse
import importlib
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
from scripts.directions.extract_paired_directions import DEFAULT_CONCEPTS, last_token_acts

COMPLETION_WORDS = [
    'small', 'large', 'old', 'new', 'red', 'blue', 'empty', 'full', 'expensive',
    'cheap', 'haunted', 'on', 'in', 'here', 'gone', 'quiet', 'dark', 'warm',
    'modern', 'ancient', 'wooden', 'enormous', 'tiny', 'beautiful', 'cold',
    'silent', 'burning', 'flooded', 'crowded', 'abandoned',
]


def curve_metrics(l2, mags, init_frac=0.15):
    """Return (rate_init, auc) shape metrics for an L2(m) curve."""
    yn = l2 / l2[-1]
    mn = mags / mags[-1]
    k = int(np.argmin(np.abs(mn - init_frac)))
    rate_init = yn[k] / mn[k]
    auc = np.trapz(yn, mn)
    return rate_init, auc


def sweep_group(model, ctx, a_base, dirs, mags, layer, device, measure_layer):
    """Run additive sweep over a [n_dirs, d] set, return L2 curves [n_dirs, n_steps]."""
    l2, _ = run_additive_sweep(model, ctx, a_base, dirs, mags, layer, device, measure_layer)
    return l2.T  # [n_dirs, n_steps]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='gpt2-xl')
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--measure_layer', type=int, default=-2)
    p.add_argument('--prefix', type=str, default='The house was')
    p.add_argument('--base_word', type=str, default='big')
    p.add_argument('--n_fineweb', type=int, default=60)
    p.add_argument('--n_random', type=int, default=60)
    p.add_argument('--m_max', type=float, default=180.0)
    p.add_argument('--n_steps', type=int, default=80)
    p.add_argument('--seq_len', type=int, default=5)
    p.add_argument('--results_dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.results_dir is None:
        args.results_dir = f'results/diff_of_activations/{args.model}_L{args.layer}_plateau_metric'
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Loading {args.model}...")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    measure_layer = args.measure_layer if args.measure_layer >= 0 else n_layers + args.measure_layer
    print(f"d_model={d_model}, n_layers={n_layers}, measure_layer={measure_layer}")

    base_prompt = f'{args.prefix} {args.base_word}'
    ctx, a_base = get_prompt_activation(model, tokenizer, base_prompt, args.layer, device)
    print(f"base: '{base_prompt}'  ||a_base||={a_base.norm():.2f}")

    mags = np.linspace(0, args.m_max, args.n_steps + 1)[1:]

    # --- Build direction sets for each group ---
    print("\nBuilding direction sets...")
    # diff_words: completions of the same prefix
    word_acts = last_token_acts(model, tokenizer, device,
                                [f'{args.prefix} {w}' for w in COMPLETION_WORDS], args.layer)
    diff_words = F.normalize(word_acts - a_base.unsqueeze(0), dim=-1)

    # diff_fineweb: toward arbitrary real activations
    tok_list = load_fineweb_fixed_length(args.n_fineweb, tokenizer, seq_len=args.seq_len, seed=args.seed)
    _, fw_acts = extract_context_and_activation(model, tokenizer, tok_list, args.layer, device)
    diff_fineweb = F.normalize(fw_acts - a_base.unsqueeze(0), dim=-1)

    # random isotropic
    rand_dirs = F.normalize(torch.randn(args.n_random, d_model), dim=-1)

    # contrastive: diff-of-means concept directions
    contr_dirs, contr_names = [], []
    for name, mod_name in DEFAULT_CONCEPTS.items():
        pairs = importlib.import_module(mod_name).PROMPT_PAIRS
        a_acts = last_token_acts(model, tokenizer, device, [a for a, b in pairs], args.layer)
        b_acts = last_token_acts(model, tokenizer, device, [b for a, b in pairs], args.layer)
        contr_dirs.append(F.normalize(b_acts.mean(0) - a_acts.mean(0), dim=0))
        contr_names.append(name)
    contr_dirs = torch.stack(contr_dirs)

    groups = {
        'diff_words':   diff_words,
        'diff_fineweb': diff_fineweb,
        'random':       rand_dirs,
        'contrastive':  contr_dirs,
    }

    # --- Sweep each group, compute metrics ---
    print("\nSweeping groups...")
    out = {'mags': mags, 'measure_layer': measure_layer, 'base_prompt': base_prompt,
           'contr_names': contr_names, 'completion_words': COMPLETION_WORDS, 'groups': {}}
    t0 = time.time()
    for gname, dirs in groups.items():
        l2 = sweep_group(model, ctx, a_base, dirs, mags, args.layer, device, measure_layer)
        metrics = np.array([curve_metrics(l2[i], mags) for i in range(l2.shape[0])])
        out['groups'][gname] = {'l2': l2, 'rate_init': metrics[:, 0], 'auc': metrics[:, 1]}
        print(f"  {gname:13s} n={l2.shape[0]:3d}  rate_init={metrics[:,0].mean():.3f}"
              f"±{metrics[:,0].std():.3f}  auc={metrics[:,1].mean():.3f}±{metrics[:,1].std():.3f}"
              f"  ({time.time()-t0:.1f}s)")

    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(out, f)

    # --- Plots ---
    colors = {'diff_words': '#EE6677', 'diff_fineweb': '#CC3311',
              'random': 'black', 'contrastive': '#4477AA'}
    order = ['diff_words', 'diff_fineweb', 'random', 'contrastive']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: mean normalized curve per group
    ax = axes[0]
    mn = mags / mags[-1]
    for g in order:
        l2 = out['groups'][g]['l2']
        yn = l2 / l2[:, -1:]
        med = np.median(yn, axis=0)
        q25, q75 = np.percentile(yn, 25, axis=0), np.percentile(yn, 75, axis=0)
        ax.plot(mn, med, color=colors[g], lw=2, label=g)
        ax.fill_between(mn, q25, q75, color=colors[g], alpha=0.12)
    ax.plot([0, 1], [0, 1], color='gray', ls=':', label='linear (no plateau)')
    ax.set_xlabel('normalized magnitude  m / m_max')
    ax.set_ylabel('normalized L2  /  L2(m_max)')
    ax.set_title('Median normalized response shape')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panels 2,3: metric distributions (strip + box)
    for ax, metric, ref, reflabel in [(axes[1], 'rate_init', 1.0, 'linear=1'),
                                       (axes[2], 'auc', 0.5, 'linear=0.5')]:
        data = [out['groups'][g][metric] for g in order]
        ax.boxplot(data, positions=range(1, len(order) + 1), showfliers=False, widths=0.5)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order)
        for i, g in enumerate(order):
            x = np.random.normal(i + 1, 0.06, size=len(out['groups'][g][metric]))
            ax.scatter(x, out['groups'][g][metric], color=colors[g], alpha=0.6, s=18, zorder=3)
        ax.axhline(ref, color='gray', ls=':', label=reflabel)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric}  ({reflabel}; <ref = plateau)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

    fig.suptitle(f"{args.model} L{args.layer}: plateau shape metric across direction groups "
                 f"(base='{base_prompt}', L2@layer {measure_layer})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'plateau_metric.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot: {path}")
    print(f"Saved data: {os.path.join(args.results_dir, 'results.pkl')}")


if __name__ == '__main__':
    main()
