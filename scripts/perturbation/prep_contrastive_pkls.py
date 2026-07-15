#!/usr/bin/env python3
"""Wrap selected contrastive (diff-of-means) feature directions into per-direction
pkls for run_feature_vs_random_sweep.py (same {'directions': [vec]} format)."""
import sys, os, argparse, pickle
sys.path.insert(0, '.')
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contrastive_pkl', required=True,
                    help="pkl with a 'features' dict of named direction vectors")
    ap.add_argument('--features', required=True, help='comma-separated feature names')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--negate', action='store_true',
                    help='store -v (perturb toward English) with a _neg suffix')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d = pickle.load(open(args.contrastive_pkl, 'rb'))
    print(f"layer={d.get('layer')} model={d.get('model')} "
          f"available={list(d['features'].keys())}")
    sign = -1.0 if args.negate else 1.0
    suffix = '_neg' if args.negate else ''
    for name in args.features.split(','):
        v = d['features'][name].float()
        v = sign * v / v.norm()
        path = os.path.join(args.out_dir, f'contrastive_{name}{suffix}.pkl')
        with open(path, 'wb') as f:
            pickle.dump({'directions': [v], 'feature': f'{name}{suffix}'}, f)
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
