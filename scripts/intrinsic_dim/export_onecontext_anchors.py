#!/usr/bin/env python3
"""Export chosen endpoints from the one-context run as single-anchor .pt files.

eval_real_vs_random_acts.py pairs extra-anchor row i with its own context i, so
each anchor we want swept inside the shared carrier must live in its own file
with exactly one row. This writes one file per requested index, key 'X_opt'.

Run the sweep with --n_act 1 --seq_len 5 --seed 42, whose single FineWeb context
is the carrier of the one-context run ('a combination of one or'), so extra
anchor row 0 is swept in exactly the context the activation belongs to.
"""

import argparse
import os
import pickle

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--indir', default='results/intrinsic_dim/onecontext_gpt2xl')
    p.add_argument('--real', default='', help='comma-separated real indices')
    p.add_argument('--jac', default='', help='comma-separated jac-opt indices')
    p.add_argument('--outdir',
                   default='results/intrinsic_dim/onecontext_gpt2xl/anchors')
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(os.path.join(args.indir, 'results.pkl'), 'rb') as f:
        res = pickle.load(f)

    for kind, key, idxs in [('real', 'X_real', args.real),
                            ('jacopt', 'X_jac', args.jac)]:
        for k in [int(x) for x in idxs.split(',') if x.strip()]:
            vec = res[key][k].float().reshape(1, -1)
            path = os.path.join(args.outdir, f'{kind}{k}.pt')
            torch.save({'X_opt': vec}, path)
            print(f'{path}  shape {tuple(vec.shape)}  '
                  f'||x||={vec.norm():.2f}')


if __name__ == '__main__':
    main()
