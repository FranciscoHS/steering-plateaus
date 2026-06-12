#!/usr/bin/env python3
"""Mixture test, swept over mixture order k (optimized / dictionary atoms).

Generalizes the mix-vs-mix test: instead of mixing 2 atoms per side, mix k of
them. For each k we sample 2k distinct dictionary atoms, form two normalized
mixtures
    m1 = normalize(sum_{i<k} d_i),   m2 = normalize(sum_{i>=k} d_i)
and read the pairwise superellipse p(m1, m2) (same machinery as
fit_p_fixed_direction: run_2d_per_anchors -> median grid -> boundary_for_set).

k=1 is just a random pair of atoms (the unmixed baseline). Prediction
(Francisco): median p decays monotonically toward 2 as k grows — mixing more
features dilutes whatever made p high.

Atoms within each 2k-set are filtered to pairwise |cos| < cos_thresh so we mix
genuinely distinct directions (cheap for dict atoms: random pairs sit ~0.02).

Run on a GPU pod. Example:
    HF_HOME=/workspace/hf_cache python -u scripts/superellipse/mixture_k_sweep_p.py \\
        --dict_path .../scaffoldcontrast_resample_lam100_20k/dictionary.pt \\
        --model qwen3-1.7b --layer 14 --n_pairs 200 \\
        --k_list 1 2 3 4 5 6 8 10 16 \\
        --out .../scaffoldcontrast_resample_lam100_20k/mixture_k_sweep
"""
import argparse
import os
import pickle
import sys
import time
from itertools import combinations
sys.path.insert(0, '.')

import numpy as np
import torch

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)
from scripts.superellipse.fit_p_fixed_direction import pair_p


def summarize(k, ps):
    ps = np.asarray(ps, float)
    ps = ps[np.isfinite(ps)]
    if ps.size == 0:
        return f'k={k:>3}: no finite values', dict(k=k, n=0)
    q1, med, q3 = np.quantile(ps, [0.25, 0.5, 0.75])
    s = (f'k={k:>3}: median={med:.3f}  IQR=({q1:.3f},{q3:.3f})  '
         f'mean={ps.mean():.3f}  n={ps.size}')
    return s, dict(k=k, n=int(ps.size), median=float(med), q1=float(q1),
                   q3=float(q3), mean=float(ps.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--dir_seed', type=int, default=0)
    ap.add_argument('--n_pairs', type=int, default=200,
                    help='# mix-vs-mix evaluations per k')
    ap.add_argument('--k_list', type=int, nargs='+',
                    default=[1, 2, 3, 4, 5, 6, 8, 10, 16])
    ap.add_argument('--cos_thresh', type=float, default=0.1)
    ap.add_argument('--n_steps_grid', type=int, default=31)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}

    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    print(f'Device: {device}  L{args.layer}->{measure_layer}', flush=True)

    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(torch.float32)
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

    def P(d1, d2):
        p, _ = pair_p(model, contexts, activations, d1, d2, args.layer,
                      measure_layer, device, angles_deg, angles_rad)
        return p

    thr = args.cos_thresh

    def ok(idx):
        for x, y in combinations(idx, 2):
            if abs(float(D[x] @ D[y])) >= thr:
                return False
        return True

    def mix_set(idx):
        v = D[list(idx)].sum(0)
        return v / (v.norm() + 1e-12)

    rng = np.random.default_rng(args.dir_seed)
    results = {'args': vars(args), 'N': N, 'measure_layer': measure_layer,
               'by_k': {}}
    t0 = time.time()

    def checkpoint():
        with open(os.path.join(args.out, 'mixture_k_sweep.pkl'), 'wb') as f:
            pickle.dump(results, f)

    for k in args.k_list:
        ps = []
        tries = 0
        max_tries = 200 * args.n_pairs
        while len(ps) < args.n_pairs and tries < max_tries:
            tries += 1
            idx = rng.choice(N, size=2 * k, replace=False)
            if not ok(idx):
                continue
            m1 = mix_set(idx[:k])
            m2 = mix_set(idx[k:])
            ps.append(P(m1, m2))
            if len(ps) % 25 == 0:
                print(f'[k={k}] {len(ps)}/{args.n_pairs}  '
                      f'elapsed={time.time()-t0:.0f}s', flush=True)
                results['by_k'][k] = np.array(ps); checkpoint()
        results['by_k'][k] = np.array(ps)
        checkpoint()
        line, _ = summarize(k, results['by_k'][k])
        print(f'[k={k}] DONE ({tries} draws)  {line}', flush=True)

    # ---- summary ----
    lines = [f'Mixture-order sweep — {args.model} L{args.layer}->{measure_layer}',
             f'dict: {args.dict_path}  ({N} atoms)',
             f'mix-of-k vs mix-of-k, n_pairs={args.n_pairs}, '
             f'cos_thresh={args.cos_thresh}, anchors={args.n_anchors}',
             f'(k=1 is the unmixed random-pair baseline)', '']
    stats = []
    for k in args.k_list:
        line, st = summarize(k, results['by_k'][k])
        lines.append(line); stats.append(st)
    results['stats'] = stats
    checkpoint()
    report = os.path.join(args.out, 'mixture_k_sweep.txt')
    with open(report, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    print(f'\nwrote {report}\nwrote {os.path.join(args.out, "mixture_k_sweep.pkl")}',
          flush=True)


if __name__ == '__main__':
    main()
