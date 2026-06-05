#!/usr/bin/env python3
"""Run 2D L^2 sweeps for pairs (c_i, d_f) where d_f is the learned feature
direction and c_i are contrastive scaffold directions. Loads model once
and processes all pairs. Output filenames follow fsec-anonymous's
sweep_2d.py convention so the canonical fit + plot scripts can consume them.

Run from fsec-anonymous's cwd (so imports resolve):
    cd /workspace/fsec-anonymous && \\
    python /workspace/steering-plateaus/scripts/superellipse/sweep_df_pairs.py \\
        --target gemma --layer 15 \\
        --results_pkl /workspace/steering-plateaus/results/feature_dir_vs_contrastive/gemma-2-9b_L15_a30/results.pkl \\
        --scaffold_pkl results/directions/dirs_gemma_L15.pkl \\
        --names Gender Age Sentiment Tense Number Formal Refusal Honesty
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, '.')
import numpy as np
import torch

from src.model import load_model, _get_blocks
from scripts.lib import registry, activations as actlib, directions as dirlib
from scripts.lib.sweep_core import run_2d_per_anchors
from scripts.lib.parametrize import eps_thresholds, CHART


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--results_pkl', required=True,
                    help='learn_feature_dir_vs_contrastive results.pkl with d_f')
    ap.add_argument('--scaffold_pkl', required=True,
                    help='fsec-anonymous dirs_<target>_L<layer>.pkl')
    ap.add_argument('--names', nargs='+', required=True,
                    help='contrastive direction names from the scaffold pkl')
    ap.add_argument('--family', default='fdc',
                    help='filename suffix tag (matches existing _dirfdc files)')
    ap.add_argument('--df_name', default='df')
    ap.add_argument('--feature_idx', type=int, default=None,
                    help='if set, take features[idx] from results_pkl as d_f '
                         'and label it d{idx+1}; otherwise use res["d_f"]')
    ap.add_argument('--feature_indices', type=int, nargs='+', default=None,
                    help='multi-feature variant of --feature_idx; loops over '
                         'each index, sharing one model load. Mutually '
                         'exclusive with --feature_idx.')
    ap.add_argument('--n_steps', type=int, default=31)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['bfloat16', 'float16', 'float32'])
    ap.add_argument('--measure_offset', type=int, default=-2)
    ap.add_argument('--out_dir', default='results/sweeps_2d')
    args = ap.parse_args()

    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
                 'float32': torch.float32}
    cfg = registry.TARGETS[args.target]
    print(f'loading {cfg["model"]} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(cfg['model'], dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers + args.measure_offset
    print(f'n_layers={n_layers} measure_layer={measure_layer}', flush=True)

    # Determine the list of (df_label, d_f) tuples to process.
    with open(args.results_pkl, 'rb') as f:
        res = pickle.load(f)
    if args.feature_indices is not None and args.feature_idx is not None:
        raise ValueError('use either --feature_idx or --feature_indices, not both')
    if args.feature_indices is not None:
        feature_iter = [(i, res['features'][i].float().to(device),
                         f'{args.df_name}{i + 1}') for i in args.feature_indices]
    elif args.feature_idx is not None:
        i = args.feature_idx
        feature_iter = [(i, res['features'][i].float().to(device),
                         f'{args.df_name}{i + 1}')]
    else:
        feature_iter = [(None, res['d_f'].float().to(device), args.df_name)]
    feature_iter = [(idx, v / v.norm(), label) for (idx, v, label) in feature_iter]
    print(f'features to process: {[label for _, _, label in feature_iter]}',
          flush=True)

    # Load scaffold; auto-extract any missing names via fsec extract_all.
    with open(args.scaffold_pkl, 'rb') as f:
        scaf = pickle.load(f)
    scaf_dirs = scaf['directions']
    scaf_signs = scaf.get('signs', {})
    missing = [n for n in args.names if n not in scaf_dirs]
    if missing:
        print(f'extending scaffold cache with {len(missing)} missing: {missing}',
              flush=True)
        # extract_all extends the on-disk pkl in-place; reload after.
        dirlib.extract_all(model, tokenizer, device, args.target,
                           args.layer, names=missing)
        with open(args.scaffold_pkl, 'rb') as f:
            scaf = pickle.load(f)
        scaf_dirs = scaf['directions']
        scaf_signs = scaf.get('signs', {})
        still_missing = [n for n in args.names if n not in scaf_dirs]
        if still_missing:
            raise KeyError(f'still missing after extract_all: {still_missing}')

    # Anchors
    print(f'loading 30 fineweb anchors at layer {args.layer}', flush=True)
    fw = actlib.fineweb_acts(model, tokenizer, device, args.target,
                             args.layer, n=30)
    contexts, activations = fw['contexts'], fw['activations']
    print(f'  N={len(activations)} ||a||={activations.norm(dim=-1).mean():.2f}',
          flush=True)

    angles_deg = np.linspace(0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)

    os.makedirs(args.out_dir, exist_ok=True)
    variant_suffix = f'_dir{args.family}'

    for feat_idx, d_f, df_label in feature_iter:
        print(f'\n--- df_label={df_label} (feature_idx={feat_idx}) ---',
              flush=True)
        for name in args.names:
            d_c = scaf_dirs[name].float().to(device)
            d_c = d_c / d_c.norm()
            # Filename: capital-first-letter names sort before lowercase df,
            # so pair is (name, df_label) → filename "<name>__<df_label>".
            a, b = name, df_label
            t0 = time.time()
            r = run_2d_per_anchors(model, contexts, activations, d_c, d_f,
                                   args.layer, measure_layer, device,
                                   angles_rad, mode='geodesic',
                                   record_cos=False, record_kl=False,
                                   single_batch_ref=False, pos=-1,
                                   full_hidden_list=None)
            l2 = r['l2']
            print(f'  {a:>10s} x {b:<10s} {time.time()-t0:.1f}s '
                  f'l2_max={l2.max():.2f}', flush=True)

            out_path = os.path.join(
                args.out_dir,
                f'sweep2d_{args.target}_L{args.layer}_{a}__{b}_fineweb_'
                f'{int(args.max_angle)}deg{variant_suffix}.pkl')
            out = {
                'chart': CHART,
                'direction_family': args.family,
                'seed_anchors': fw.get('seed', 42),
                'eps_thresholds': eps_thresholds(),
                'model': args.target,
                'model_full': cfg['model'],
                'perturb_layer': args.layer,
                'measure_layer': measure_layer,
                'measure_offset': args.measure_offset,
                'perturb_pos': -1,
                'mode': 'geodesic',
                'metrics': ['l2'],
                'direction_labels': (a, b),
                'source_name': 'fineweb',
                'anchor_source': 'fineweb',
                'angles_deg': angles_deg,
                'n_anchors': len(activations),
                'anchor_norm_mean': float(activations.norm(dim=-1).mean()),
                'direction_signs': {a: scaf_signs.get(name, 'ext'),
                                    b: 'learned->learned'},
                **{k: r[k] for k in r},
            }
            tmp = out_path + '.tmp'
            with open(tmp, 'wb') as f:
                pickle.dump(out, f)
            os.replace(tmp, out_path)
            print(f'    saved {out_path}', flush=True)


if __name__ == '__main__':
    main()
