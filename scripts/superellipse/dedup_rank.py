#!/usr/bin/env python3
"""Greedy de-duplication of the feature-ness ranking.

The top-by-kurtosis directions turned out to be a tight cluster (mean pairwise
|cos| ~0.4, ~5 effective axes for 50 atoms), so the "top features" are mostly
one direction found many times. This walks the kurtosis-sorted order keeping an
atom only if its |cosine| with every already-kept atom is <= --thr, yielding
distinct representatives. De-dup is by AXIS (|cos|), so d and -d count as one.

Writes a new ranking_dedup.pkl (same fields as ranking.pkl, with `order`
replaced by the de-duplicated order + `n_kept`) that the dashboard can consume.
Pure linalg, no model — runs locally.
"""

import argparse
import pickle
import sys

sys.path.insert(0, '.')

import numpy as np
import torch


def pairwise_stats(V):
    C = (V @ V.T).numpy()
    iu = np.triu_indices(C.shape[0], k=1)
    off = np.abs(C[iu])
    return off.mean(), np.percentile(off, 95), off.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', default='/tmp/dict_25k_snapshot.pt')
    ap.add_argument('--ranking_pkl', required=True)
    ap.add_argument('--thr', type=float, default=0.3,
                    help='keep an atom only if |cos| <= thr with all kept atoms')
    ap.add_argument('--out_pkl', required=True)
    args = ap.parse_args()

    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    K, d = D.shape
    rk = pickle.load(open(args.ranking_pkl, 'rb'))
    order = np.asarray(rk['order'])
    kurt = rk['kurtosis']
    print(f'{K} atoms, d={d}; de-dup threshold |cos| <= {args.thr}', flush=True)

    kept = []                                   # atom indices, in kurtosis order
    kept_mat = torch.empty(0, d)
    for a in order:
        v = D[int(a)]
        if kept_mat.shape[0] > 0:
            if (kept_mat @ v).abs().max().item() > args.thr:
                continue
        kept.append(int(a))
        kept_mat = torch.cat([kept_mat, v.unsqueeze(0)], 0)
    kept = np.asarray(kept)
    print(f'kept {len(kept)} / {K} distinct directions '
          f'({100*len(kept)/K:.1f}%)', flush=True)

    out = dict(rk)
    out['order'] = kept
    out['n_kept'] = len(kept)
    out['dedup_thr'] = args.thr
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(out, f)
    print(f'wrote {args.out_pkl}', flush=True)

    # Confirm de-clustering: pairwise stats of the new top-50.
    V = D[torch.as_tensor(kept[:50])]
    m, p95, mx = pairwise_stats(V)
    print(f'\nde-duped top-50 pairwise |cos|: mean={m:.4f} 95p={p95:.3f} '
          f'max={mx:.3f}  (was mean=0.385 95p=0.71 max=0.81 before)')
    print('\nde-duped top-15 (atom, kurtosis):')
    for r, a in enumerate(kept[:15]):
        print(f'  {r:>3} atom {a:>6}  kurt={kurt[a]:.2f}')


if __name__ == '__main__':
    main()
