#!/usr/bin/env python3
"""Build a 'combo dictionary': each entry is the normalized sum of combo_n
random distinct atoms from a source dictionary. Lets the existing pair-fit
(fit_pairs_dictionary.py) measure p for combination-vs-combination pairs.
"""
import argparse
import sys
sys.path.insert(0, '.')
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--dict_path', required=True)
ap.add_argument('--M', type=int, default=200, help='# of combos to build')
ap.add_argument('--combo_n', type=int, default=2, help='atoms summed per combo')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
args = ap.parse_args()

blob = torch.load(args.dict_path, map_location='cpu')
D = blob['dictionary'].float()
D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
N = D.shape[0]
rng = np.random.default_rng(args.seed)

combos, members = [], []
for _ in range(args.M):
    idx = rng.choice(N, size=args.combo_n, replace=False)
    v = D[torch.tensor(idx)].sum(0)
    combos.append(v / v.norm())
    members.append(idx.tolist())
C = torch.stack(combos)
torch.save({'dictionary': C, 'n_done': C.shape[0],
            'combo_n': args.combo_n, 'members': members,
            'source_dict': args.dict_path}, args.out)
print(f'Saved {args.out}  shape={tuple(C.shape)}  '
      f'(combo_n={args.combo_n}, from N={N})')
