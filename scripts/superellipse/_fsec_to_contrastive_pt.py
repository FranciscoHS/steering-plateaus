#!/usr/bin/env python3
"""Convert an fsec-anonymous directions cache (dirs_<target>_L<layer>.pkl, a
{'directions': {name: vec}} blob) into the {'dictionary': [K,d], 'names': [...]}
format expected by dict_vs_contrastive_align.py."""
import argparse
import pickle
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True, help='fsec dirs_*.pkl')
ap.add_argument('--out', required=True, help='output .pt')
args = ap.parse_args()

b = pickle.load(open(args.src, 'rb'))
dirs = b['directions'] if isinstance(b, dict) and 'directions' in b else b
names = sorted(dirs.keys())
rows = []
for n in names:
    v = dirs[n].float()
    rows.append(v / v.norm())
D = torch.stack(rows)
torch.save({'dictionary': D, 'names': names}, args.out)
print(f'wrote {args.out}: {tuple(D.shape)}, {len(names)} names')
