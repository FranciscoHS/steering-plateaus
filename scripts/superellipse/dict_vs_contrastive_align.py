#!/usr/bin/env python3
"""Cosine alignment between found dictionary directions and the contrastive
directions (semantic scaffold + language + code).

For each found atom: best-matching contrastive direction (by |cos|). Reports
the biggest hits overall, the best found-atom match per contrastive direction,
and a random-direction null (max |cos| over the 36 contrastive dirs).

Usage:
    python scripts/superellipse/dict_vs_contrastive_align.py \\
        --dict_path .../dictionary_snap3500.pt \\
        --contrastive_pts A.pt B.pt --n_random 500 --out .../dict_vs_contrastive
"""
import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--dict_path', required=True)
ap.add_argument('--contrastive_pts', nargs='+', required=True,
                help='one or more dict.pt files with "dictionary" + "names"')
ap.add_argument('--n_random', type=int, default=500)
ap.add_argument('--topn', type=int, default=25, help='# of top hits to list')
ap.add_argument('--rand_seed', type=int, default=0)
ap.add_argument('--out', required=True)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Contrastive directions (concatenate all provided files, keep names).
C_rows, names = [], []
for p in args.contrastive_pts:
    b = torch.load(p, map_location='cpu')
    C_rows.append(b['dictionary'].float())
    names += list(b['names'])
C = torch.cat(C_rows, 0)
C = (C / C.norm(dim=-1, keepdim=True)).to(device)
nC, d_model = C.shape
print(f'Contrastive dirs: {nC}  ({len(names)} names)', flush=True)

blob = torch.load(args.dict_path, map_location='cpu')
D = blob['dictionary'].float()
D = (D / D.norm(dim=-1, keepdim=True)).to(device)
K = D.shape[0]
print(f'Found dict: {K} atoms', flush=True)

cos = D @ C.T                                    # [K, nC] signed
acos = cos.abs()

# Random null: max |cos| of random unit dirs over the contrastive set.
g = torch.Generator(device=device).manual_seed(args.rand_seed)
Rnd = torch.randn(args.n_random, d_model, generator=g, device=device)
Rnd = Rnd / Rnd.norm(dim=-1, keepdim=True)
rand_max = (Rnd @ C.T).abs().max(dim=1).values.cpu().numpy()
r95 = float(np.quantile(rand_max, 0.95))

# Per found-atom best contrastive match.
found_best = acos.max(dim=1)
found_best_cos = found_best.values.cpu().numpy()
found_best_j = found_best.indices.cpu().numpy()

# Top hits overall (atom, contrastive) by |cos|.
flat = acos.reshape(-1)
topv, topi = torch.topk(flat, args.topn)
print(f'\n=== top {args.topn} (found atom, contrastive) by |cos| ===')
print(f'(random null max|cos|: mean={rand_max.mean():.3f} '
      f'95pct={r95:.3f} max={rand_max.max():.3f}, n={args.n_random})')
hits = []
for v, idx in zip(topv.cpu().numpy(), topi.cpu().numpy()):
    k = int(idx) // nC; j = int(idx) % nC
    signed = float(cos[k, j].item())
    hits.append({'atom': k, 'contrastive': names[j], 'cos': signed})
    print(f'  atom {k:5d}  ~  {names[j]:12s}  cos={signed:+.3f}', flush=True)

# Best found atom per contrastive direction.
print('\n=== best found-atom match per contrastive direction ===')
best_per_c = acos.max(dim=0)
bpc_cos = best_per_c.values.cpu().numpy()
bpc_k = best_per_c.indices.cpu().numpy()
order = np.argsort(-bpc_cos)
per_c = []
for j in order:
    signed = float(cos[int(bpc_k[j]), j].item())
    per_c.append({'contrastive': names[j], 'best_atom': int(bpc_k[j]),
                  'cos': signed})
    flag = ' *' if abs(signed) > r95 else ''
    print(f'  {names[j]:12s}  best atom {int(bpc_k[j]):5d}  '
          f'cos={signed:+.3f}{flag}', flush=True)

n_above = int((found_best_cos > r95).sum())
print(f'\nfound atoms with best|cos| > random95 ({r95:.3f}): '
      f'{n_above}/{K} ({100*n_above/K:.1f}%)')
print(f'found best|cos|: mean={found_best_cos.mean():.3f} '
      f'median={np.median(found_best_cos):.3f} max={found_best_cos.max():.3f}')

with open(os.path.join(args.out, 'dict_vs_contrastive.pkl'), 'wb') as f:
    pickle.dump({'args': vars(args), 'names': names, 'cos': cos.cpu().numpy(),
                 'found_best_cos': found_best_cos, 'found_best_j': found_best_j,
                 'rand_max': rand_max, 'top_hits': hits, 'per_c': per_c}, f)
print(f"\nSaved {os.path.join(args.out, 'dict_vs_contrastive.pkl')}")
