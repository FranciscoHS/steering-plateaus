#!/usr/bin/env python3
"""Cosine similarity between two atoms of a dictionary.pt."""
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--dict_path', required=True)
ap.add_argument('--atoms', required=True, help='comma list, e.g. 413,577')
args = ap.parse_args()

D = torch.load(args.dict_path, map_location='cpu')['dictionary'].float()
D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
idx = [int(a) for a in args.atoms.split(',')]
for i in range(len(idx)):
    for j in range(i + 1, len(idx)):
        c = (D[idx[i]] @ D[idx[j]]).item()
        print(f'cos(atom {idx[i]}, atom {idx[j]}) = {c:+.4f}')
