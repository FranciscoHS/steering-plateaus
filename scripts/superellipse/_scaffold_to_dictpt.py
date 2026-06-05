#!/usr/bin/env python3
"""Convert an fsec contrastive-directions pkl into a dictionary.pt so the
existing dict_* diagnostics can run on the contrastive scaffold directly."""
import argparse
import pickle
import sys
sys.path.insert(0, '.')
import torch
from scripts.superellipse.build_dictionary import SEMANTIC_NAMES

ap = argparse.ArgumentParser()
ap.add_argument('--dirs_pkl', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()

blob = pickle.load(open(args.dirs_pkl, 'rb'))
dd = blob['directions'] if 'directions' in blob else blob
rows, names = [], []
for n in SEMANTIC_NAMES:
    v = dd[n].float()
    rows.append(v / v.norm())
    names.append(n)
D = torch.stack(rows)
torch.save({'dictionary': D, 'n_done': D.shape[0], 'names': names}, args.out)
print(f'Saved {args.out}  shape={tuple(D.shape)}  names={names}')
