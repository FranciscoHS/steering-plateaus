#!/usr/bin/env python3
"""Test 2 — mixture test for the superellipse-p metric.

The metric's premise: feature directions are *more* responsive than non-features
(p>2), non-features sit at p=2. A mixture of two features is itself a non-feature,
so mixing should destroy whatever made p high.

p is pairwise (the superellipse exponent of the 2D response surface of a plane),
so we measure mixtures *against mixtures*: pair m1=(a+b)/||a+b|| against
m2=(c+d)/||c+d|| with {a,b,c,d} four distinct directions, and read p(m1, m2).

Four distributions, all via the SAME pair_p machinery as fit_p_fixed_direction
(run_2d_per_anchors -> median grid -> boundary_for_set):
  - raw_opt       : random pairs of dictionary atoms (the optimized candidates)
  - mixed_opt     : mix-vs-mix over disjoint 4-tuples of dictionary atoms
  - raw_contrast  : all pairs of contrastive (diff-of-means) directions
  - mixed_contrast: mix-vs-mix over disjoint 4-tuples of contrastive directions

The sharp read (Francisco): if contrastive mixtures fall to p=2 but optimized
mixtures stay high, the optimization cannot be finding features — the whole
subspace is high-p and p can't single out feature axes (the program is cursed).

Run on a GPU pod. Example:
    HF_HOME=/workspace/hf_cache python -u scripts/superellipse/mixture_test_p.py \\
        --dict_path .../scaffoldcontrast_resample_lam100_20k/dictionary.pt \\
        --model qwen3-1.7b --layer 14 --n_mixpairs 300 --n_anchors 20 \\
        --out .../scaffoldcontrast_resample_lam100_20k/mixture_test
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
from scripts.superellipse.learn_feature_dir_vs_contrastive import (
    load_or_build_scaffold,
)

# Full fsec direction registry (registry.all_direction_names()): 17 semantic +
# 9 language + 10 code = 36. Two are near-duplicate variants (TensePresent,
# HonestyShort) that the cosine filter will drop against their siblings.
ALL_NAMES = [
    'Gender', 'Sentiment', 'Age', 'Era', 'Tense', 'TensePresent', 'Number',
    'Person', 'Status', 'Honesty', 'HonestyShort', 'Refusal', 'Formal',
    'Literary', 'Wealth', 'Health', 'Certainty', 'Portuguese', 'Chinese',
    'Spanish', 'French', 'German', 'Dutch', 'Italian', 'Japanese', 'Russian',
    'Arabic', 'Python', 'JavaScript', 'Java', 'Rust', 'Haskell', 'Cpp', 'Go',
    'Lisp', 'TypeScript',
]


def summarize(name, ps):
    ps = np.asarray(ps, float)
    ps = ps[np.isfinite(ps)]
    if ps.size == 0:
        return f'{name:>16}: no finite values', dict(n=0)
    q1, med, q3 = np.quantile(ps, [0.25, 0.5, 0.75])
    s = (f'{name:>16}: median={med:.3f}  IQR=({q1:.3f},{q3:.3f})  '
         f'mean={ps.mean():.3f}  n={ps.size}')
    return s, dict(n=int(ps.size), median=float(med), q1=float(q1), q3=float(q3),
                   mean=float(ps.mean()))


def mix(d_i, d_j):
    m = d_i + d_j
    return m / (m.norm() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--dir_seed', type=int, default=0,
                    help='seed for sampling atom tuples')
    ap.add_argument('--n_mixpairs', type=int, default=300,
                    help='# mix-vs-mix evaluations per arm (and # raw pairs)')
    ap.add_argument('--n_steps_grid', type=int, default=31)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--contrast_names', nargs='*', default=None,
                    help='fsec direction names (default: all 36 in the registry)')
    ap.add_argument('--cos_thresh', type=float, default=0.1,
                    help='only pair/mix directions with pairwise |cos| < this')
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

    # --- optimized candidates: dictionary atoms ---
    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    N = D.shape[0]
    print(f'Dictionary: {N} atoms', flush=True)

    # --- contrastive directions from fsec-anonymous ---
    names = args.contrast_names or ALL_NAMES
    C = load_or_build_scaffold(model, tokenizer, device, args.model,
                               args.layer, names).cpu().float()
    C = C / (C.norm(dim=-1, keepdim=True) + 1e-12)
    nC = C.shape[0]
    cosC = (C @ C.T).abs().numpy()                 # [nC, nC] |cos| matrix
    print(f'Contrastive directions: {nC} ({names})', flush=True)
    print(f'cos_thresh={args.cos_thresh}: keeping only pairs/tuples with '
          f'pairwise |cos| < {args.cos_thresh}', flush=True)

    # --- anchors (identical to fit_p_fixed_direction) ---
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

    rng = np.random.default_rng(args.dir_seed)
    thr = args.cos_thresh
    results = {'args': vars(args), 'N': N, 'measure_layer': measure_layer,
               'names': names, 'cosC': cosC}
    t0 = time.time()

    def checkpoint():
        with open(os.path.join(args.out, 'mixture_test.pkl'), 'wb') as f:
            pickle.dump(results, f)

    def ok_dict(idx):
        """All pairwise |cos| among dict atoms idx below threshold."""
        for x, y in combinations(idx, 2):
            if abs(float(D[x] @ D[y])) >= thr:
                return False
        return True

    def ok_contrast(idx):
        """All pairwise |cos| among contrastive dirs idx below threshold."""
        for x, y in combinations(idx, 2):
            if cosC[x, y] >= thr:
                return False
        return True

    # ---- 1. raw optimized pairs (random atom pairs, |cos|<thr) ----
    raw_opt = []
    while len(raw_opt) < args.n_mixpairs:
        i, j = rng.choice(N, size=2, replace=False)
        if not ok_dict((i, j)):
            continue
        raw_opt.append(P(D[i], D[j]))
        if len(raw_opt) % 25 == 0:
            print(f'[raw_opt] {len(raw_opt)}/{args.n_mixpairs}  '
                  f'elapsed={time.time()-t0:.0f}s', flush=True)
            results['raw_opt'] = np.array(raw_opt); checkpoint()
    results['raw_opt'] = np.array(raw_opt); checkpoint()

    # ---- 2. mixed optimized pairs (mix-vs-mix, disjoint 4-tuples, |cos|<thr) ----
    mixed_opt = []
    while len(mixed_opt) < args.n_mixpairs:
        a, b, c, d = rng.choice(N, size=4, replace=False)
        if not ok_dict((a, b, c, d)):
            continue
        mixed_opt.append(P(mix(D[a], D[b]), mix(D[c], D[d])))
        if len(mixed_opt) % 25 == 0:
            print(f'[mixed_opt] {len(mixed_opt)}/{args.n_mixpairs}  '
                  f'elapsed={time.time()-t0:.0f}s', flush=True)
            results['mixed_opt'] = np.array(mixed_opt); checkpoint()
    results['mixed_opt'] = np.array(mixed_opt); checkpoint()

    # ---- 3. raw contrastive pairs (all pairs with |cos|<thr) ----
    raw_pairs = [(i, j) for i, j in combinations(range(nC), 2)
                 if ok_contrast((i, j))]
    raw_contrast = [P(C[i], C[j]) for i, j in raw_pairs]
    results['raw_contrast'] = np.array(raw_contrast)
    results['raw_contrast_pairs'] = raw_pairs
    checkpoint()
    print(f'[raw_contrast] {len(raw_contrast)}/{nC*(nC-1)//2} pairs pass filter  '
          f'elapsed={time.time()-t0:.0f}s', flush=True)

    # ---- 4. mixed contrastive pairs (mix-vs-mix, all 4 mutually |cos|<thr) ----
    quads = []
    for a, b in combinations(range(nC), 2):
        if cosC[a, b] >= thr:
            continue
        rest = [x for x in range(nC) if x not in (a, b)]
        for c, d in combinations(rest, 2):
            if (a, b) < (c, d) and ok_contrast((a, b, c, d)):
                quads.append((a, b, c, d))
    n_valid = len(quads)
    if n_valid > args.n_mixpairs:
        sel = rng.choice(n_valid, size=args.n_mixpairs, replace=False)
        quads = [quads[s] for s in sel]
    mixed_contrast = [P(mix(C[a], C[b]), mix(C[c], C[d])) for a, b, c, d in quads]
    results['mixed_contrast'] = np.array(mixed_contrast)
    results['mixed_contrast_quads'] = quads
    checkpoint()
    print(f'[mixed_contrast] {len(mixed_contrast)} mix-pairs done '
          f'(of {n_valid} valid 4-tuples)  elapsed={time.time()-t0:.0f}s',
          flush=True)

    # ---- summary ----
    lines = [f'Mixture test (Test 2) — {args.model} L{args.layer}->{measure_layer}',
             f'dict: {args.dict_path}  ({N} atoms)',
             f'contrastive: {nC} dirs  cos_thresh={args.cos_thresh}  '
             f'anchors={args.n_anchors} max_angle={args.max_angle} '
             f'grid={args.n_steps_grid}',
             f'(directions used in pairs/mixtures filtered to pairwise '
             f'|cos| < {args.cos_thresh})', '']
    stats = {}
    for name, arr in [('raw_opt', results['raw_opt']),
                      ('mixed_opt', results['mixed_opt']),
                      ('raw_contrast', results['raw_contrast']),
                      ('mixed_contrast', results['mixed_contrast'])]:
        line, st = summarize(name, arr)
        lines.append(line); stats[name] = st
    results['stats'] = stats
    checkpoint()
    report = os.path.join(args.out, 'mixture_test.txt')
    with open(report, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    print(f'\nwrote {report}\nwrote {os.path.join(args.out, "mixture_test.pkl")}',
          flush=True)


if __name__ == '__main__':
    main()
