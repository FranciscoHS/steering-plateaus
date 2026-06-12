#!/usr/bin/env python3
"""Mixture-order sweep for CONTRASTIVE directions (sibling of mixture_k_sweep_p).

For each k, build two normalized mixtures of k contrastive (diff-of-means)
directions each and read the pairwise superellipse p(m1, m2). All 2k directions
in a draw are mutually |cos| < cos_thresh, so we mix genuinely distinct
features. Compare against mixture_k_sweep_p.py (the same sweep on optimized
dictionary atoms).

NOTE: with the |cos|<thr filter, k is capped by the largest mutually-near-
orthogonal set among the available directions (mix-of-k vs mix-of-k needs 2k of
them). With the 36-direction fsec registry that set is ~10 (languages and code
langs are internally ~0.9 correlated), so only k<=5 is reachable. Infeasible k
are reported with n=0 rather than silently dropped.

Run on a GPU pod. Example:
    HF_HOME=/workspace/hf_cache python -u \\
      scripts/superellipse/mixture_k_sweep_contrast_p.py \\
        --model qwen3-1.7b --layer 14 --n_pairs 200 --k_list 2 4 \\
        --out .../scaffoldcontrast_resample_lam100_20k/mixture_k_sweep_contrast
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
from scripts.superellipse.mixture_k_sweep_p import summarize

ALL_NAMES = [
    'Gender', 'Sentiment', 'Age', 'Era', 'Tense', 'TensePresent', 'Number',
    'Person', 'Status', 'Honesty', 'HonestyShort', 'Refusal', 'Formal',
    'Literary', 'Wealth', 'Health', 'Certainty', 'Portuguese', 'Chinese',
    'Spanish', 'French', 'German', 'Dutch', 'Italian', 'Japanese', 'Russian',
    'Arabic', 'Python', 'JavaScript', 'Java', 'Rust', 'Haskell', 'Cpp', 'Go',
    'Lisp', 'TypeScript',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--dir_seed', type=int, default=0)
    ap.add_argument('--n_pairs', type=int, default=200)
    ap.add_argument('--k_list', type=int, nargs='+', default=[2, 4])
    ap.add_argument('--cos_thresh', type=float, default=0.1)
    ap.add_argument('--contrast_names', nargs='*', default=None)
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

    names = args.contrast_names or ALL_NAMES
    C = load_or_build_scaffold(model, tokenizer, device, args.model,
                               args.layer, names).cpu().float()
    C = C / (C.norm(dim=-1, keepdim=True) + 1e-12)
    nC = C.shape[0]
    cos = (C @ C.T).numpy()
    print(f'Contrastive directions: {nC}  cos_thresh={args.cos_thresh}',
          flush=True)

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

    def mix_set(idx):
        v = C[list(idx)].sum(0)
        return v / (v.norm() + 1e-12)

    thr = args.cos_thresh
    rng = np.random.default_rng(args.dir_seed)

    def sample_valid_set(m):
        """Greedily build a set of m mutually-|cos|<thr direction indices.

        Shuffle the directions and greedily accept those that stay below thr
        against everything already chosen. Returns a list of m indices, or None
        if a shuffle can't reach m (caller retries with a fresh shuffle).
        """
        order = rng.permutation(nC)
        chosen = []
        for i in order:
            if all(abs(cos[i, j]) < thr for j in chosen):
                chosen.append(int(i))
                if len(chosen) == m:
                    return chosen
        return None

    results = {'args': vars(args), 'nC': nC, 'names': names,
               'measure_layer': measure_layer, 'by_k': {}}
    t0 = time.time()

    def checkpoint():
        with open(os.path.join(args.out, 'mixture_k_sweep_contrast.pkl'),
                  'wb') as f:
            pickle.dump(results, f)

    for k in args.k_list:
        ps = []
        # Quick feasibility probe: can we ever assemble 2k mutually-<thr dirs?
        feasible = any(sample_valid_set(2 * k) is not None for _ in range(200))
        if not feasible:
            results['by_k'][k] = np.array([])
            checkpoint()
            print(f'[k={k}] INFEASIBLE: cannot find {2*k} mutually |cos|<{thr} '
                  f'dirs among {nC}  (skipping)', flush=True)
            continue
        tries = 0
        max_tries = 500 * args.n_pairs
        while len(ps) < args.n_pairs and tries < max_tries:
            tries += 1
            s = sample_valid_set(2 * k)
            if s is None:
                continue
            m1 = mix_set(s[:k])
            m2 = mix_set(s[k:])
            ps.append(P(m1, m2))
            if len(ps) % 25 == 0:
                print(f'[k={k}] {len(ps)}/{args.n_pairs}  '
                      f'elapsed={time.time()-t0:.0f}s', flush=True)
                results['by_k'][k] = np.array(ps); checkpoint()
        results['by_k'][k] = np.array(ps)
        checkpoint()
        line, _ = summarize(k, results['by_k'][k])
        print(f'[k={k}] DONE ({tries} draws)  {line}', flush=True)

    lines = [f'Contrastive mixture-order sweep — {args.model} '
             f'L{args.layer}->{measure_layer}',
             f'{nC} contrastive dirs, mix-of-k vs mix-of-k, '
             f'n_pairs={args.n_pairs}, cos_thresh={args.cos_thresh}, '
             f'anchors={args.n_anchors}',
             f'(all 2k dirs per draw mutually |cos|<{args.cos_thresh})', '']
    stats = []
    for k in args.k_list:
        line, st = summarize(k, results['by_k'][k])
        lines.append(line); stats.append(st)
    results['stats'] = stats
    checkpoint()
    report = os.path.join(args.out, 'mixture_k_sweep_contrast.txt')
    with open(report, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    print(f'\nwrote {report}', flush=True)


if __name__ == '__main__':
    main()
