#!/usr/bin/env python3
"""Are the (top-kurtosis) dictionary directions anisotropically clustered?

If the same corpus tokens recur as top activating examples across many
directions, those directions must point into a common cone (under cosine, a
single token can only be top for many directions if they cluster). This is a
pure-linalg check on the directions themselves — no model, runs locally.

For each group (top-N by kurtosis / median band / bottom-N / random dict atoms),
report:
  - pairwise |cosine| stats vs the random baseline 1/sqrt(d) ~= 0.022
  - centroid alignment: mean (d_i . chat) where chat is the unit group centroid
    (high => the directions share a common component)
  - SVD energy: fraction of total energy in the top-1 / top-5 singular
    directions, and participation ratio (effective # of axes the group spans)
"""

import argparse
import pickle
import sys

sys.path.insert(0, '.')

import numpy as np
import torch


def pairwise_cos_stats(V):
    """V: [n, d] unit rows. Returns mean/95p/max of off-diagonal |cosine|."""
    C = (V @ V.T).numpy()
    n = C.shape[0]
    iu = np.triu_indices(n, k=1)
    off = np.abs(C[iu])
    return off.mean(), np.percentile(off, 95), off.max()


def svd_energy(V):
    """Fraction of energy in top-1 and top-5 PCs; participation ratio."""
    s = torch.linalg.svdvals(V).numpy()
    e = s ** 2
    e = e / e.sum()
    pr = (e.sum() ** 2) / (e ** 2).sum()      # participation ratio (>=1)
    top5 = e[:5].sum()
    return e[0], top5, pr


def centroid_align(V):
    c = V.mean(0)
    cn = c.norm()
    if cn < 1e-9:
        return 0.0, 0.0
    chat = c / cn
    align = (V @ chat).mean().item()
    return align, cn.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', default='/tmp/dict_25k_snapshot.pt')
    ap.add_argument('--ranking_pkl', required=True)
    ap.add_argument('--n', type=int, default=50)
    args = ap.parse_args()

    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    K, d = D.shape
    rk = pickle.load(open(args.ranking_pkl, 'rb'))
    order = np.asarray(rk['order'])
    print(f'dict {K} atoms, d={d}; random-pair cosine baseline ~ '
          f'1/sqrt(d) = {1/np.sqrt(d):.4f}\n')

    mid = K // 2
    rng = np.random.default_rng(0)
    groups = {
        f'top{args.n}': order[:args.n],
        f'median{args.n}': order[mid - args.n // 2: mid - args.n // 2 + args.n],
        f'bottom{args.n}': order[-args.n:],
        f'randomdict{args.n}': rng.choice(K, size=args.n, replace=False),
    }

    print(f'{"group":>14} {"pw|cos|mean":>12} {"95p":>7} {"max":>7} '
          f'{"centAlign":>9} {"E_pc1":>7} {"E_top5":>7} {"partRatio":>9}')
    for name, idxs in groups.items():
        V = D[torch.as_tensor(np.asarray(idxs))]
        m, p95, mx = pairwise_cos_stats(V)
        align, cn = centroid_align(V)
        e1, e5, pr = svd_energy(V)
        print(f'{name:>14} {m:>12.4f} {p95:>7.3f} {mx:>7.3f} '
              f'{align:>9.3f} {e1:>7.3f} {e5:>7.3f} {pr:>9.1f}')

    # Top-10 atoms: explicit pairwise cosine matrix (look for near-duplicates).
    print('\nTop-10 atoms pairwise cosine:')
    top10 = order[:10]
    V = D[torch.as_tensor(np.asarray(top10))]
    C = (V @ V.T).numpy()
    print('       ' + ' '.join(f'{int(a):>6}' for a in top10))
    for i, a in enumerate(top10):
        row = ' '.join(f'{C[i, j]:>6.2f}' for j in range(len(top10)))
        print(f'{int(a):>6} {row}')


if __name__ == '__main__':
    main()
