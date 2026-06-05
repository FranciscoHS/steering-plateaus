#!/usr/bin/env python3
"""Counterfactual check for sharding the dictionary build.

Compares pairwise |cos| of dictionary atoms:
  - WITHIN coupled shard A: first 10 atoms of the K100 build (seed 42), which
    were optimized WITH the lam_prev diversity coupling among themselves.
  - WITHIN coupled shard B: the seed-202 10-atom shard (also internally coupled).
  - CROSS A x B: no lam_prev coupling between the two shards -> a direct sample
    of what sharding produces.

If CROSS ~ WITHIN, seed noise alone diversifies and sharding is safe.
If CROSS >> WITHIN, the coupling is load-bearing and sharding re-explores the
same directions.
"""

import sys
import numpy as np
import torch

A_PATH = ('results/dictionary/qwen3-1.7b_L14_K100_ratio30_actresample/'
          'dictionary.pt')
B_PATH = 'results/dictionary/_shardtest/seed202/dictionary.pt'
N = 10


def load_dict(path, n):
    blob = torch.load(path, map_location='cpu')
    d = blob['dictionary'] if isinstance(blob, dict) else blob
    d = d[:n].float()
    return d / d.norm(dim=-1, keepdim=True)


def offdiag_abscos(X, Y=None):
    """|cos| values. If Y is None: upper-triangle of X x X. Else all X x Y."""
    if Y is None:
        G = (X @ X.T).abs().numpy()
        iu = np.triu_indices(G.shape[0], k=1)
        return G[iu]
    return (X @ Y.T).abs().numpy().reshape(-1)


def stats(label, v):
    print(f'{label:<22}  n={len(v):3d}  mean={v.mean():.4f}  '
          f'median={np.median(v):.4f}  p90={np.quantile(v,0.9):.4f}  '
          f'max={v.max():.4f}', flush=True)
    return v


def main():
    A = load_dict(A_PATH, N)
    B = load_dict(B_PATH, N)
    d_model = A.shape[1]
    rand_floor = (1.0 / d_model) ** 0.5  # E[|cos|]~sqrt(2/(pi*d)); std cos~1/sqrt(d)
    print(f'd_model={d_model}  random-dir cos std ~ 1/sqrt(d) = '
          f'{rand_floor:.4f}\n', flush=True)

    within_A = stats('within A (seed42)', offdiag_abscos(A))
    within_B = stats('within B (seed202)', offdiag_abscos(B))
    cross = stats('CROSS A x B (no coup)', offdiag_abscos(A, B))

    within_all = np.concatenate([within_A, within_B])
    print(f'\nwithin (pooled) mean={within_all.mean():.4f}  '
          f'cross mean={cross.mean():.4f}  '
          f'ratio cross/within={cross.mean()/within_all.mean():.2f}', flush=True)

    # random baseline for the same shapes, for reference
    g = torch.Generator().manual_seed(0)
    R1 = torch.randn(N, d_model, generator=g); R1 /= R1.norm(dim=-1, keepdim=True)
    R2 = torch.randn(N, d_model, generator=g); R2 /= R2.norm(dim=-1, keepdim=True)
    rnd = offdiag_abscos(R1, R2)
    print(f'random-vs-random cross mean={rnd.mean():.4f} '
          f'median={np.median(rnd):.4f} max={rnd.max():.4f}', flush=True)


if __name__ == '__main__':
    main()
