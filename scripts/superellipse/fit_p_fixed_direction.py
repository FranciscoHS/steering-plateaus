#!/usr/bin/env python3
"""Does fixing one direction compress the spread of superellipse-p?

We know p has a large spread across random pairs of dictionary atoms. Open
question: is that spread intrinsic to the *pairing* (so even a single fixed
direction, paired against many others, gives the full wide spread), or is p
partly a property of the direction itself (so fixing a direction compresses
the column and shifts its center)?

By the law of total variance:
    Var(p) = E_k[Var_j(p_kj)]        (within-direction)
           + Var_k(E_j[p_kj])        (between-direction)
so "do fixed-direction columns look compressed?" <=> "do row-means differ?".

Plot: a beeswarm with
  - column 0: random pairs (loaded straight from the cached build-order pkl;
    these are the per-pair b['fit']['p'] values, identical machinery)
  - columns 1..n_fixed: each fixes one randomly-chosen atom k and pairs it
    against n_partners random others, sorted left->right by column median.
Headline number: between-direction variance fraction over the fixed columns.

Per-pair p is computed exactly as fit_p_vs_buildorder.fit_bucket does
(anchor-median L2 grid -> boundary_for_set), so the fixed columns are directly
comparable to the cached random column. No per-pair bootstrap (we only need the
point estimate for the swarm).

Run on a GPU pod.

Usage:
    python scripts/superellipse/fit_p_fixed_direction.py \\
        --dict_path .../dictionary.pt --model qwen3-1.7b --layer 14 \\
        --random_pkl .../p_vs_buildorder.pkl \\
        --n_fixed 9 --n_partners 100 --n_anchors 20 \\
        --out .../p_fixed_direction
"""
import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)
from scripts.superellipse.canonical.sweep_core import run_2d_per_anchors
from scripts.superellipse.plot_boundary_pair import boundary_for_set


def pair_p(model, contexts, activations, d1, d2, layer, measure_layer, device,
           angles_deg, angles_rad):
    """Per-pair superellipse p, matching fit_p_vs_buildorder.fit_bucket."""
    r = run_2d_per_anchors(model, contexts, activations, d1, d2, layer,
                           measure_layer, device, angles_rad,
                           mode='geodesic', single_batch_ref=True)
    median_grid = np.median(r['l2'], axis=0)
    b = boundary_for_set(angles_deg, median_grid, exact=True)
    cos = float((d1.to(device) * d2.to(device)).sum().item())
    return float(b['fit']['p']), cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--random_pkl', required=True,
                    help='p_vs_buildorder.pkl for the same dict -> random column')
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_fixed', type=int, default=9)
    ap.add_argument('--n_partners', type=int, default=100)
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--dir_seed', type=int, default=0,
                    help='seed for choosing fixed dirs + their partners')
    ap.add_argument('--n_steps_grid', type=int, default=31)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}

    # --- random column from cache (per-pair p_fits across all buckets) ---
    rb = pickle.load(open(args.random_pkl, 'rb'))
    random_p = np.concatenate([b['p_fits'] for b in rb['buckets']])
    random_p = random_p[np.isfinite(random_p)]
    print(f'Random column: {random_p.size} cached pairs, '
          f'median={np.median(random_p):.3f}', flush=True)

    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    print(f'Device: {device}  L{args.layer}->{measure_layer}', flush=True)

    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(device=device, dtype=torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    N = D.shape[0]
    print(f'Dictionary: {N} atoms', flush=True)

    print(f'Extracting {args.n_anchors} anchors (seed={args.held_seed})',
          flush=True)
    tok = load_fineweb_fixed_length(args.n_anchors, tokenizer,
                                    seq_len=args.seq_len, seed=args.held_seed)
    ctx, act = extract_context_and_activation(model, tok, args.layer, device)
    contexts = [ctx[i:i + 1].float().cpu() for i in range(ctx.shape[0])]
    activations = act.float().cpu()

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps_grid)
    angles_rad = np.deg2rad(angles_deg)

    rng = np.random.default_rng(args.dir_seed)
    fixed_idx = rng.choice(N, size=args.n_fixed, replace=False)

    columns = []
    t0 = time.time()
    for ci, k in enumerate(fixed_idx):
        # sample n_partners distinct partners j != k
        partners = []
        seen = {int(k)}
        while len(partners) < min(args.n_partners, N - 1):
            j = int(rng.integers(N))
            if j in seen:
                continue
            seen.add(j); partners.append(j)
        dk = D[k].cpu()
        ps, coss = [], []
        for j in partners:
            p, cos = pair_p(model, contexts, activations, dk, D[j].cpu(),
                            args.layer, measure_layer, device, angles_deg,
                            angles_rad)
            ps.append(p); coss.append(cos)
        ps = np.array(ps, float); coss = np.array(coss, float)
        fin = np.isfinite(ps)
        med = float(np.median(ps[fin])) if fin.any() else float('nan')
        columns.append({'fixed_idx': int(k), 'partners': partners,
                        'p_fits': ps, 'cos': coss, 'median': med})
        print(f'[col {ci}] fixed atom {int(k)}: {fin.sum()}/{len(ps)} finite  '
              f'median={med:.3f} IQR('
              f'{np.quantile(ps[fin],0.25):.3f},{np.quantile(ps[fin],0.75):.3f})'
              f'  elapsed={time.time()-t0:.0f}s', flush=True)
        with open(os.path.join(args.out, 'p_fixed_direction.pkl'), 'wb') as f:
            pickle.dump({'args': vars(args), 'N': N,
                         'measure_layer': measure_layer,
                         'random_p': random_p, 'columns': columns}, f)

    # --- between-direction variance fraction over the fixed columns ---
    col_means = np.array([np.nanmean(c['p_fits']) for c in columns])
    within = np.array([np.nanvar(c['p_fits']) for c in columns])
    between_var = float(np.var(col_means))
    within_mean = float(np.mean(within))
    frac_between = between_var / (between_var + within_mean + 1e-12)
    print(f'\nBetween-direction var fraction (fixed cols): {frac_between:.3f}',
          flush=True)

    # --- beeswarm plot: random column + fixed cols sorted by median ---
    order = np.argsort([c['median'] for c in columns])
    sorted_cols = [columns[i] for i in order]
    labels = ['random\npairs'] + [f"#{c['fixed_idx']}" for c in sorted_cols]
    data = [random_p] + [c['p_fits'][np.isfinite(c['p_fits'])]
                         for c in sorted_cols]

    fig, ax = plt.subplots(figsize=(1.1 * len(data) + 2, 6))
    swarm_rng = np.random.default_rng(123)
    for x, vals in enumerate(data):
        color = 'C3' if x == 0 else 'C0'
        jit = swarm_rng.uniform(-0.28, 0.28, size=vals.size)
        ax.scatter(np.full(vals.size, x) + jit, vals, s=9, alpha=0.45,
                   color=color, edgecolors='none', zorder=2)
        med = np.median(vals)
        q1, q3 = np.quantile(vals, 0.25), np.quantile(vals, 0.75)
        ax.plot([x - 0.34, x + 0.34], [med, med], color='k', lw=2, zorder=3)
        ax.plot([x - 0.28, x + 0.28], [q1, q1], color='k', lw=0.8, zorder=3)
        ax.plot([x - 0.28, x + 0.28], [q3, q3], color='k', lw=0.8, zorder=3)
    ax.axhline(2.0, color='gray', lw=1, ls=':', label='p=2 (ellipse)')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('superellipse p')
    ax.set_xlabel('fixed atom (one per column, sorted by median p)')
    ax.set_title(f'p spread: random pairs vs fixed-direction columns\n'
                 f'{args.model} L{args.layer}->{measure_layer}, '
                 f'{args.n_partners} partners/col, n_anchors={args.n_anchors}  |  '
                 f'between-dir var frac = {frac_between:.3f}')
    ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'p_fixed_direction.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {plot_path}', flush=True)


if __name__ == '__main__':
    main()
