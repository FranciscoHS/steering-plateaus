#!/usr/bin/env python3
"""Run 2D L^2 sweeps for arbitrary pairs of feature directions.

Loads features [K, d] from a results.pkl (e.g. from decompose_plateau_breaking.py
or learn_superellipse_dict.py), picks specific pairs by index, and runs
run_2d_per_anchors for each. Output filenames follow fsec-anonymous's
sweep_2d.py convention so plot_fig3_local.py can consume them.

Run from fsec-anonymous's cwd (so imports resolve):
    cd /workspace/fsec-anonymous && \\
    python /workspace/steering-plateaus/scripts/superellipse/sweep_feature_pairs.py \\
        --target gemma --layer 15 \\
        --features_pkl /workspace/steering-plateaus/results/decomp_plateau/gemma-2-9b_L15_a30/results.pkl \\
        --features_key features \\
        --pairs '1,9;1,3;3,6;3,4;2,9'
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
from scripts.lib import registry, activations as actlib
from scripts.lib.sweep_core import run_2d_per_anchors
from scripts.lib.parametrize import eps_thresholds, CHART


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--features_pkl', required=True,
                    help='pkl with a [K, d] features tensor or dict-of-tensors')
    ap.add_argument('--features_key', default='features',
                    help='key in the pkl for the features tensor')
    ap.add_argument('--pairs', required=True,
                    help='semicolon-separated 1-indexed pairs, e.g. "1,9;1,3;3,6"')
    ap.add_argument('--family', default='fdcdecomp',
                    help='filename suffix tag (default _dirfdcdecomp)')
    ap.add_argument('--name_prefix', default='d',
                    help='feature name prefix, e.g. "d" -> d1, d2, ...')
    ap.add_argument('--n_steps', type=int, default=31)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['bfloat16', 'float16', 'float32'])
    ap.add_argument('--measure_offset', type=int, default=-2)
    ap.add_argument('--out_dir', default='results/sweeps_2d')
    args = ap.parse_args()

    pair_list = []
    for chunk in args.pairs.split(';'):
        i, j = chunk.split(',')
        pair_list.append((int(i), int(j)))
    print(f'pairs: {pair_list}')

    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
                 'float32': torch.float32}
    cfg = registry.TARGETS[args.target]
    print(f'loading {cfg["model"]} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(cfg['model'],
                                          dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers + args.measure_offset
    print(f'n_layers={n_layers} measure_layer={measure_layer}', flush=True)

    # Load features
    with open(args.features_pkl, 'rb') as f:
        blob = pickle.load(f)
    feats = blob[args.features_key]
    if not isinstance(feats, torch.Tensor):
        feats = torch.tensor(feats)
    feats = feats.float().to(device)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    K = feats.shape[0]
    print(f'features: shape={tuple(feats.shape)} (K={K})', flush=True)

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

    for (i, j) in pair_list:
        if not (1 <= i <= K and 1 <= j <= K):
            raise ValueError(f'pair index out of range: {(i,j)}, K={K}')
        d1 = feats[i - 1]
        d2 = feats[j - 1]
        # Filename names: alphabetic order? Use string sort like sweep_2d does.
        n1 = f'{args.name_prefix}{i}'
        n2 = f'{args.name_prefix}{j}'
        a, b = (n1, n2) if n1 < n2 else (n2, n1)
        d_a, d_b = (d1, d2) if n1 < n2 else (d2, d1)
        t0 = time.time()
        r = run_2d_per_anchors(model, contexts, activations, d_a, d_b,
                               args.layer, measure_layer, device,
                               angles_rad, mode='geodesic',
                               record_cos=False, record_kl=False,
                               single_batch_ref=False, pos=-1,
                               full_hidden_list=None)
        l2 = r['l2']
        print(f'  {a:>6s} x {b:<6s} {time.time()-t0:.1f}s '
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
            'direction_signs': {a: 'learned', b: 'learned'},
            **{k: r[k] for k in r},
        }
        tmp = out_path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(out, f)
        os.replace(tmp, out_path)
        print(f'    saved {out_path}', flush=True)


if __name__ == '__main__':
    main()
