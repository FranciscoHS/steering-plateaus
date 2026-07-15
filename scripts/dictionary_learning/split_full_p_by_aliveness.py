#!/usr/bin/env python3
"""Split eval_sae_full_p.py's per-pair p values by atom aliveness.

eval_sae_full_p.py samples its (d, contrast) atom pairs with a seeded CPU
generator, so the exact pair indices can be replayed from the saved args
(dir_seed, n_dir, n_contrast, d_sae). Given firing counts (from
sae_vs_sae_alignment.py's alignment.pkl), split the saved p distribution into
alive-alive / alive-dead / dead-dead pairs — to test whether high p lives in
the dead (never-firing, unconstrained-by-reconstruction) part of the dictionary.

Usage:
    python scripts/dictionary_learning/split_full_p_by_aliveness.py \\
        --p_pkl results/pcull_eval/popt_p.pkl \\
        --align_pkl ~/align_popt_vs_baseline/alignment.pkl --which a
"""
import argparse
import os
import pickle

import numpy as np
import torch


def replay_pairs(d_sae, n_dir, n_contrast, dir_seed):
    """Reproduce eval_sae_full_p.py's sampling: same generator, same call order."""
    g = torch.Generator().manual_seed(dir_seed)
    d_idx = torch.randperm(d_sae, generator=g)[:n_dir]
    pairs = []
    for di in d_idx.tolist():
        c_idx = torch.randperm(d_sae, generator=g)[:n_contrast + 1]
        c_idx = c_idx[c_idx != di][:n_contrast]
        for ci in c_idx.tolist():
            pairs.append((di, ci))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p_pkl', required=True)
    ap.add_argument('--align_pkl', required=True)
    ap.add_argument('--which', choices=['a', 'b'], default='a',
                    help='use fires_a (popt) or fires_b (baseline) from align_pkl')
    args = ap.parse_args()

    blob = pickle.load(open(os.path.expanduser(args.p_pkl), 'rb'))
    a = blob['args']
    p = np.asarray(blob['p'])
    pairs = replay_pairs(32768, a['n_dir'], a['n_contrast'], a['dir_seed'])
    assert len(pairs) == p.shape[0], (len(pairs), p.shape)

    al = pickle.load(open(os.path.expanduser(args.align_pkl), 'rb'))
    fires = np.asarray(al['fires_a' if args.which == 'a' else 'fires_b'])
    alive = fires > 0

    d_alive = np.array([alive[d] for d, _ in pairs])
    c_alive = np.array([alive[c] for _, c in pairs])
    groups = {'alive-alive': d_alive & c_alive,
              'alive-dead': d_alive ^ c_alive,
              'dead-dead': ~d_alive & ~c_alive}

    print(f'{args.p_pkl}  (n={p.size} pairs, overall median p={np.median(p):.3f})')
    for name, m in groups.items():
        if m.sum() == 0:
            print(f'  {name:>11s}: n=0')
            continue
        q = np.quantile(p[m], [0.25, 0.5, 0.75])
        print(f'  {name:>11s}: n={m.sum():4d}  median={q[1]:.3f}  '
              f'IQR=({q[0]:.3f},{q[2]:.3f})  frac p>2.5: {(p[m] > 2.5).mean():.2f}')


if __name__ == '__main__':
    main()
