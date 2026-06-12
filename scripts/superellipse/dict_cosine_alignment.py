#!/usr/bin/env python3
"""Pairwise cosine alignment among the first K atoms of one or more dictionaries.

Reports, per dictionary:
  - mean / median |cos| over all atom pairs
  - mean SIGNED cos (the 'shared component' indicator: >0 means atoms share a
    common positive direction / sit in a cone)
  - ||mean unit atom||  (0 = isotropic, 1 = all identical) — a direct shared-
    component magnitude
Overlays the pairwise-cosine distributions for all dictionaries in one plot.

Pure linear algebra; runs anywhere.

Usage:
    python scripts/superellipse/dict_cosine_alignment.py \\
        --dict_paths A.pt B.pt C.pt --labels lam0 lam1 lam10 \\
        --limit 100 --out .../cos_align
"""
import argparse
import os
import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_paths', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--limit', type=int, default=100,
                    help='use the first `limit` atoms of each dict')
    ap.add_argument('--atoms', default=None,
                    help="explicit atom subset (e.g. '1802,8994,18354'); "
                         'overrides --limit, applied to every dict')
    ap.add_argument('--remove_mean', action='store_true',
                    help='project the shared mean-atom direction out of each '
                         'dict and renormalize before measuring alignment')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    assert len(args.dict_paths) == len(args.labels)
    os.makedirs(args.out, exist_ok=True)

    stats = {}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = [f'C{i}' for i in range(len(args.dict_paths))]
    for path, lab, col in zip(args.dict_paths, args.labels, colors):
        blob = torch.load(path, map_location='cpu')
        D = blob['dictionary'].float()
        D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
        if args.atoms:
            idx = [int(a) for a in args.atoms.split(',')]
            D = D[idx]
            K = D.shape[0]
        else:
            K = min(args.limit, D.shape[0])
            D = D[:K]
        if args.remove_mean:
            mhat = D.mean(0)
            mhat = mhat / (mhat.norm() + 1e-12)
            D = D - (D @ mhat).unsqueeze(-1) * mhat.unsqueeze(0)
            D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
        G = D @ D.T                                          # [K, K]
        iu = torch.triu_indices(K, K, offset=1)
        cos = G[iu[0], iu[1]].numpy()                        # pairwise cosines
        mean_unit = D.mean(0)
        shared = float(mean_unit.norm())                     # ||mean atom||
        s = {'n_atoms': K, 'mean_abs_cos': float(np.abs(cos).mean()),
             'median_abs_cos': float(np.median(np.abs(cos))),
             'mean_signed_cos': float(cos.mean()),
             'shared_norm': shared,
             'p95_abs_cos': float(np.quantile(np.abs(cos), 0.95))}
        stats[lab] = s
        print(f'[{lab}]  n={K}  mean|cos|={s["mean_abs_cos"]:.4f}  '
              f'median|cos|={s["median_abs_cos"]:.4f}  '
              f'mean_signed={s["mean_signed_cos"]:.4f}  '
              f'||mean atom||={shared:.4f}  '
              f'p95|cos|={s["p95_abs_cos"]:.4f}', flush=True)
        ax.hist(cos, bins=60, density=True, histtype='step', lw=2,
                color=col,
                label=f'{lab}: mean|cos|={s["mean_abs_cos"]:.3f}, '
                      f'signed={s["mean_signed_cos"]:+.3f}, '
                      f'||mean||={shared:.3f}')

    ax.axvline(0.0, color='k', lw=0.6, alpha=0.4)
    ax.set_xlabel('pairwise cosine between atoms')
    ax.set_ylabel('density')
    ax.set_title(f'Dictionary atom alignment (first {args.limit} atoms)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'cos_alignment.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    with open(os.path.join(args.out, 'cos_alignment.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'stats': stats}, f)
    print(f'\nSaved {plot_path}')


if __name__ == '__main__':
    main()
