#!/usr/bin/env python3
"""Per-atom best |cos| against the SAE decoder dictionary; sample atoms from
the bottom decile (lowest SAE alignment) for dashboard inspection.

    python scripts/superellipse/sae_align_bottom_sample.py \\
        --dict_path .../dictionary.pt \\
        --sae_pt results/sae_qwen_L14_L0_50/dictionary.pt \\
        --n_sample 10 --out .../sae_align_scores.pkl
"""
import argparse
import pickle

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--dict_path', required=True)
ap.add_argument('--sae_pt', required=True)
ap.add_argument('--n_sample', type=int, default=10)
ap.add_argument('--bottom_frac', type=float, default=0.10)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
args = ap.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
D = torch.load(args.dict_path, map_location=dev)['dictionary'].float()
D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
S = torch.load(args.sae_pt, map_location=dev)['dictionary'].float()
S = S / (S.norm(dim=-1, keepdim=True) + 1e-12)
print(f'atoms: {D.shape[0]}  sae latents: {S.shape[0]}')

best = torch.empty(D.shape[0])
best_idx = torch.empty(D.shape[0], dtype=torch.long)
with torch.no_grad():
    for i in range(0, D.shape[0], 2048):
        j = min(D.shape[0], i + 2048)
        c = (D[i:j] @ S.T).abs()
        v, ix = c.max(dim=1)
        best[i:j], best_idx[i:j] = v.cpu(), ix.cpu()

best_np = best.numpy()
q = np.quantile(best_np, args.bottom_frac)
bottom = np.where(best_np <= q)[0]
rng = np.random.default_rng(args.seed)
sample = sorted(rng.choice(bottom, size=min(args.n_sample, len(bottom)),
                           replace=False).tolist())
print(f'best|cos|: mean={best_np.mean():.3f} med={np.median(best_np):.3f} '
      f'p10={q:.3f} min={best_np.min():.3f} max={best_np.max():.3f}')
print(f'bottom-{args.bottom_frac:.0%} pool: {len(bottom)} atoms (<= {q:.3f})')
print('sampled atoms (atom -> best SAE latent : cos):')
for a in sample:
    print(f'  atom {a:5d} -> #{best_idx[a].item()} : {best_np[a]:.3f}')

with open(args.out, 'wb') as f:
    pickle.dump({'best_cos': best_np, 'best_idx': best_idx.numpy(),
                 'sample': sample, 'args': vars(args)}, f)
print(f'Saved {args.out}')
