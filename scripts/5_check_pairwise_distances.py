#!/usr/bin/env python3
"""Sanity check: did optimization collapse the points onto each other?

Pairwise distances within each activation set: jac-opt vs real vs Gaussian.

How close do the Jacobian-optimization endpoints land to each other,
compared with the spread of real activations and of the cov-matched
Gaussian? For each set, compute all pairwise L2 distances (n=200 -> 19900
pairs) and plot the three distributions.

Sets (qwen3-1.7b L2, saved tensors from the April intrinsic-dim runs):
  - real     200 random rows of X_real from jac_filtered_data.pt
  - gauss    200 random rows of X_gauss (cov-matched Gaussian pool)
  - jac-opt  the 200 optimized points from jac_optimized_data.pt (raw)

With --rescale, the jac-opt set is first multiplied by one global factor so
its mean norm matches the real set, removing the ~1.9x norm gap from the
comparison.

Needs the outputs of steps 1 and 2 (jac_filtered_data.pt, jac_optimized_data.pt).

Usage (from the repo root):
    python scripts/5_check_pairwise_distances.py            # -> pairwise_dist/
    python scripts/5_check_pairwise_distances.py --rescale  # -> pairwise_dist_rescaled/

Output: results/intrinsic_dim/pairwise_dist[_rescaled]/{results.pkl,
        pairwise_dist.png}
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, '.')  # run from the repo root

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = 'results/intrinsic_dim'
N = 200
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rescale', action='store_true',
                    help='rescale jac-opt to the real mean norm')
    args = ap.parse_args()
    OUT_DIR = os.path.join(DATA_DIR, 'pairwise_dist_rescaled' if args.rescale
                           else 'pairwise_dist')
    os.makedirs(OUT_DIR, exist_ok=True)
    filt = torch.load(os.path.join(DATA_DIR, 'jac_filtered_data.pt'),
                      map_location='cpu', weights_only=False)
    opt = torch.load(os.path.join(DATA_DIR, 'jac_optimized_data.pt'),
                     map_location='cpu', weights_only=False)

    g = torch.Generator().manual_seed(SEED)
    X_real = filt['X_real'].float()
    X_gauss = filt['X_gauss'].float()
    X_real = X_real[torch.randperm(X_real.shape[0], generator=g)[:N]]
    X_gauss = X_gauss[torch.randperm(X_gauss.shape[0], generator=g)[:N]]
    X_opt = opt['X_opt'].float()[:N]
    opt_name = 'jac-opt'
    if args.rescale:
        X_opt = X_opt * (X_real.norm(dim=-1).mean()
                         / X_opt.norm(dim=-1).mean())
        opt_name = 'jac-opt (rescaled)'

    sets = {'real': X_real, 'gauss (cov-matched)': X_gauss,
            opt_name: X_opt}

    results = {'n': N, 'seed': SEED, 'dists': {}, 'norms': {}}
    for name, X in sets.items():
        d = torch.pdist(X).numpy()
        results['dists'][name] = d
        results['norms'][name] = X.norm(dim=-1).numpy()
        q25, q75 = np.percentile(d, [25, 75])
        print(f'{name:22s} n={X.shape[0]}  mean norm={X.norm(dim=-1).mean():.1f}  '
              f'pairwise L2: median={np.median(d):.1f}  IQR={q25:.1f}-{q75:.1f}  '
              f'min={d.min():.1f}  max={d.max():.1f}', flush=True)

    with open(os.path.join(OUT_DIR, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'real': 'C0', 'gauss (cov-matched)': 'C1', opt_name: 'C4'}
    for name, d in results['dists'].items():
        ax.hist(d, bins=100, density=True, alpha=0.5, color=colors[name],
                label=f'{name} (median {np.median(d):.1f})')
    ax.set_xlabel('pairwise L2 distance (within-set)')
    ax.set_ylabel('density')
    ax.set_title('qwen3-1.7b L2: pairwise distances, n=200 per set'
                 + (', jac-opt rescaled to real mean norm' if args.rescale
                    else ''))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'pairwise_dist.png'), dpi=150,
                bbox_inches='tight')
    print(f"Saved {os.path.join(OUT_DIR, 'results.pkl')}")
    print(f"Saved {os.path.join(OUT_DIR, 'pairwise_dist.png')}")


if __name__ == '__main__':
    main()
