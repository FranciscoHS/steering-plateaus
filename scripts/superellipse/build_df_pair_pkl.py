#!/usr/bin/env python3
"""Build a fsec-anonymous direction cache pkl containing:
  - d_f from results/feature_dir_vs_contrastive/.../results.pkl
  - K randomly-picked contrastive scaffold directions from the same run

Output is in the format expected by fsec-anonymous's sweep_2d.py
`--directions_pkl` mode:
  {'directions': {name: tensor[d]}, 'signs': {name: str},
   'layer': int, 'model': str, 'family': str}

Usage:
    python scripts/superellipse/build_df_pair_pkl.py \\
        --results_pkl results/feature_dir_vs_contrastive/gemma-2-9b_L15_a30/results.pkl \\
        --scaffold_pkl /workspace/fsec-anonymous/results/directions/dirs_gemma_L15.pkl \\
        --n_pick 2 --seed 42 \\
        --out /workspace/fsec-anonymous/results/directions/dirs_gemma_L15_fdc.pkl
"""
import argparse
import os
import pickle
import random

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_pkl', required=True,
                    help='learn_feature_dir_vs_contrastive results.pkl')
    ap.add_argument('--scaffold_pkl', required=True,
                    help='fsec-anonymous dirs_<target>_L<layer>.pkl')
    ap.add_argument('--n_pick', type=int, default=2,
                    help='number of scaffold directions to pair with d_f')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--family', default='fdc',
                    help='family tag stored in the pkl (used in filenames)')
    ap.add_argument('--df_name', default='df',
                    help='name for the learned d_f in the output directions dict')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.results_pkl, 'rb') as f:
        res = pickle.load(f)
    d_f = res['d_f'].float()
    d_f = d_f / d_f.norm()
    scaffold_names_used = res['scaffold_names']
    layer = res['args']['layer']
    model = res['args']['model']
    print(f"d_f: shape={tuple(d_f.shape)}, norm={d_f.norm().item():.4f}")
    print(f"used scaffold (in training): {scaffold_names_used}")
    print(f"layer={layer} model={model}")

    with open(args.scaffold_pkl, 'rb') as f:
        scaf = pickle.load(f)
    scaf_dirs = scaf['directions']
    scaf_signs = scaf.get('signs', {})

    candidates = [n for n in scaffold_names_used if n in scaf_dirs]
    missing = [n for n in scaffold_names_used if n not in scaf_dirs]
    if missing:
        print(f"WARN: {missing} not present in scaffold_pkl; skipping")

    rng = random.Random(args.seed)
    picked = rng.sample(candidates, k=args.n_pick)
    print(f"picked (seed={args.seed}): {picked}")

    out_dirs = {args.df_name: d_f}
    out_signs = {args.df_name: 'learned->learned'}
    for n in picked:
        v = scaf_dirs[n].float()
        v = v / v.norm()
        out_dirs[n] = v
        out_signs[n] = scaf_signs.get(n, 'ext')

    out_blob = {
        'directions': out_dirs,
        'signs': out_signs,
        'layer': layer,
        'model': model,
        'family': args.family,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(out_blob, f)
    print(f"Saved: {args.out}")
    print(f"directions in pkl: {list(out_dirs.keys())}")


if __name__ == '__main__':
    main()
