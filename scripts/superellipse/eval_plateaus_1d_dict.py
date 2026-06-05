#!/usr/bin/env python3
"""1D spherical perturbation sweep: FOUND dictionary atoms vs random.

Diagnostic on a built dictionary: perturb held-out FineWeb activations along
each found atom and along random unit directions, measuring L2 response at the
penultimate layer as a function of geodesic angle. If the found directions are
privileged, their response curves should differ systematically from random
(e.g. saturate / break the plateau earlier, or grow differently).

Reuses sweep_1d + fineweb_ctx_acts from eval_plateaus_1d.py.

Usage:
    python scripts/superellipse/eval_plateaus_1d_dict.py \\
        --dict_path results/dictionary/qwen3-1.7b_L14_K10_randcontrast_stepresample/dictionary.pt \\
        --model qwen3-1.7b --layer 14 --n_act 30 --seq_len 5 \\
        --max_angle 60 --n_steps 31 --n_random 20 \\
        --out results/dictionary/qwen3-1.7b_L14_K10_randcontrast_stepresample/plateaus_1d
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
from scripts.superellipse.eval_plateaus_1d import sweep_1d, fineweb_ctx_acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--n_act', type=int, default=30)
    ap.add_argument('--seq_len', type=int, default=5)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--n_steps', type=int, default=31)
    ap.add_argument('--n_dirs', type=int, default=None,
                    help='# found atoms to sweep (default: all in the dict)')
    ap.add_argument('--n_random', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42, help='anchor seed')
    ap.add_argument('--rand_seed', type=int, default=0)
    ap.add_argument('--dir_seed', type=int, default=1,
                    help='seed for which atoms are sampled when n_dirs < K')
    ap.add_argument('--out', required=True)
    ap.add_argument('--pairwise', action='store_true',
                    help='also sweep all equal 2-combinations normalize(d_i+d_j) '
                         'of the found directions and add a dedicated plot '
                         '(singles + pairwise mixes + random baseline)')
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    d_model = model._plateau_info['d_model']
    model_dtype = next(model.parameters()).dtype
    print(f'Device: {device} d_model={d_model} L{args.layer}->{measure_layer}',
          flush=True)

    # Found dictionary atoms
    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    K = D.shape[0]
    if args.n_dirs is not None and args.n_dirs < K:
        sel = np.random.default_rng(args.dir_seed).choice(K, args.n_dirs,
                                                          replace=False)
        D = D[torch.tensor(sorted(sel))]
    n_dirs = D.shape[0]
    print(f'Dictionary: {K} atoms, sweeping {n_dirs}', flush=True)

    # Random unit dirs (same construction as eval_plateaus_1d)
    g = torch.Generator(device=device).manual_seed(args.rand_seed)
    rand_dirs = torch.randn(args.n_random, d_model, generator=g, device=device,
                            dtype=torch.float32)
    rand_dirs = rand_dirs / rand_dirs.norm(dim=-1, keepdim=True)

    # Anchors
    print(f'Extracting {args.n_act} FineWeb anchors '
          f'(seq_len={args.seq_len}, seed={args.seed})', flush=True)
    contexts, activations = fineweb_ctx_acts(model, tokenizer, device,
                                             args.layer, args.n_act,
                                             args.seq_len, args.seed)

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)
    print(f'Sweep angles 0..{args.max_angle}° in {args.n_steps} steps',
          flush=True)

    out_found = np.zeros((n_dirs, args.n_act, args.n_steps), dtype=np.float32)
    out_random = np.zeros((args.n_random, args.n_act, args.n_steps),
                          dtype=np.float32)

    t0 = time.time()
    for k in range(n_dirs):
        d = D[k].to(device=device, dtype=torch.float32)
        for i in range(args.n_act):
            ctx = contexts[i].to(device=device, dtype=model_dtype)
            a = activations[i].to(device=device)
            out_found[k, i] = sweep_1d(model, ctx, a, d, args.layer,
                                       measure_layer, angles_rad,
                                       model_dtype, device)
        if (k + 1) % 5 == 0 or k == n_dirs - 1:
            print(f'  found {k+1}/{n_dirs}  elapsed={time.time()-t0:.1f}s',
                  flush=True)

    t1 = time.time()
    for r in range(args.n_random):
        d = rand_dirs[r]
        for i in range(args.n_act):
            ctx = contexts[i].to(device=device, dtype=model_dtype)
            a = activations[i].to(device=device)
            out_random[r, i] = sweep_1d(model, ctx, a, d, args.layer,
                                        measure_layer, angles_rad,
                                        model_dtype, device)
        if (r + 1) % 5 == 0 or r == args.n_random - 1:
            print(f'  random {r+1}/{args.n_random}  '
                  f'elapsed={time.time()-t1:.1f}s', flush=True)

    # Equal 2-combinations normalize(d_i + d_j) of the found directions.
    pair_idx = []
    out_pairs = None
    if args.pairwise:
        pair_idx = [(i, j) for i in range(n_dirs) for j in range(i + 1, n_dirs)]
        out_pairs = np.zeros((len(pair_idx), args.n_act, args.n_steps),
                             dtype=np.float32)
        print(f'Sweeping {len(pair_idx)} equal pairwise combinations',
              flush=True)
        t2 = time.time()
        for p, (i, j) in enumerate(pair_idx):
            dm = D[i].to(device) + D[j].to(device)
            dm = (dm / (dm.norm() + 1e-12)).to(torch.float32)
            for ai in range(args.n_act):
                ctx = contexts[ai].to(device=device, dtype=model_dtype)
                a = activations[ai].to(device=device)
                out_pairs[p, ai] = sweep_1d(model, ctx, a, dm, args.layer,
                                            measure_layer, angles_rad,
                                            model_dtype, device)
        print(f'  pairwise done  elapsed={time.time()-t2:.1f}s', flush=True)

    out_pkl = os.path.join(args.out, 'results.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'args': vars(args), 'angles_deg': angles_deg,
            'measure_layer': measure_layer,
            'found_l2': out_found,      # [n_dirs, n_act, n_steps]
            'random_l2': out_random,    # [n_rand, n_act, n_steps]
            'pair_l2': out_pairs,       # [n_pairs, n_act, n_steps] or None
            'pair_idx': pair_idx,
        }, f)

    # Plot: aggregate found vs random (median + IQR over all curves), and
    # per-direction faded curves.
    found_flat = out_found.reshape(-1, args.n_steps)
    random_flat = out_random.reshape(-1, args.n_steps)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for flat, color, lab, n in [
            (found_flat, 'C0', 'found atoms', n_dirs),
            (random_flat, 'C1', 'random', args.n_random)]:
        med = np.median(flat, axis=0)
        q25 = np.quantile(flat, 0.25, axis=0)
        q75 = np.quantile(flat, 0.75, axis=0)
        ax.plot(angles_deg, med, '-', color=color,
                label=f'{lab} (median over {n}×{args.n_act})')
        ax.fill_between(angles_deg, q25, q75, alpha=0.2, color=color)
    ax.set_xlabel('perturbation angle (deg)')
    ax.set_ylabel(r'L2 at measure_layer ($\|\Delta\|$)')
    ax.set_title(f'1D sweep: found vs random  {args.model} '
                 f'L{args.layer}->{measure_layer}')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for k in range(n_dirs):
        ax.plot(angles_deg, np.median(out_found[k], axis=0), color='C0',
                alpha=0.4)
    for r in range(args.n_random):
        ax.plot(angles_deg, np.median(out_random[r], axis=0), '--',
                color='gray', alpha=0.4)
    ax.plot([], [], '-', color='C0', label='found (each)')
    ax.plot([], [], '--', color='gray', label='random (each)')
    ax.set_xlabel('perturbation angle (deg)')
    ax.set_ylabel('median L2 across anchors')
    ax.set_title('Per-direction curves')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.out, 'sweep.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Dedicated pairwise plot: random baseline + single dirs + equal 2-mixes.
    if args.pairwise and out_pairs is not None:
        figp, axp = plt.subplots(figsize=(8, 6))
        rmed = np.median(random_flat, axis=0)
        rq25 = np.quantile(random_flat, 0.25, axis=0)
        rq75 = np.quantile(random_flat, 0.75, axis=0)
        axp.plot(angles_deg, rmed, '-', color='gray', lw=2, label='random')
        axp.fill_between(angles_deg, rq25, rq75, alpha=0.15, color='gray')
        for k in range(n_dirs):
            axp.plot(angles_deg, np.median(out_found[k], axis=0), '-',
                     color=f'C{k}', lw=2, label=f'd{k}')
        for p, (i, j) in enumerate(pair_idx):
            axp.plot(angles_deg, np.median(out_pairs[p], axis=0), '--',
                     lw=1.5, alpha=0.9, label=f'd{i}+d{j}')
        axp.set_xlabel('perturbation angle (deg)')
        axp.set_ylabel(r'median L2 at measure_layer ($\|\Delta\|$)')
        axp.set_title(f'Single dirs vs equal pairwise mixes vs random  '
                      f'{args.model} L{args.layer}->{measure_layer}')
        axp.legend(fontsize=8, ncol=2); axp.grid(True, alpha=0.3)
        plt.tight_layout()
        pair_path = os.path.join(args.out, 'sweep_pairwise.png')
        plt.savefig(pair_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved {pair_path}')

    # Console summary: median L2 at a few angles.
    fm = np.median(found_flat, axis=0)
    rm = np.median(random_flat, axis=0)
    print('\n=== median L2 (found / random) at sample angles ===')
    for ang in [10, 20, 30, 45, 60]:
        if ang <= args.max_angle:
            j = int(np.argmin(np.abs(angles_deg - ang)))
            print(f'  {angles_deg[j]:5.1f}°: found={fm[j]:7.3f}  '
                  f'random={rm[j]:7.3f}  ratio={fm[j]/max(rm[j],1e-9):.3f}')
    print(f'\nSaved {out_pkl}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
