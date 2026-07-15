#!/usr/bin/env python3
"""Make one direction-pkl per atom for run_feature_vs_random_sweep.py.

Each output pkl has {'directions': [unit_vec], 'atom_index': idx} so the sweep
script can load it via --direction_pkl <file> (it indexes directions[0]).
"""
import sys, os, argparse, pickle
sys.path.insert(0, '.')
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--atoms', required=True, help='comma-separated atom indices')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    blob = torch.load(args.dict_path, map_location='cpu', weights_only=False)
    D = blob['dictionary'].float()
    for idx in [int(x) for x in args.atoms.split(',')]:
        v = D[idx]
        v = v / v.norm()
        path = os.path.join(args.out_dir, f'atom{idx}.pkl')
        with open(path, 'wb') as f:
            pickle.dump({'directions': [v], 'atom_index': idx}, f)
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
