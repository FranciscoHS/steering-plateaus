#!/usr/bin/env python3
"""Pairwise distances within sets, with jac-opt rescaled to real mean norm.

Same as pairwise_dist_conditions.py (same seed, so identical real/gauss
draws), but the jac-opt set is rescaled by a single global factor so its
mean norm matches the real set — the same convention as the C2ST scripts
(rescale_to). Removes the ~1.9x norm gap from the distance comparison.

Output: results/intrinsic_dim/pairwise_dist_rescaled/{results.pkl,
        pairwise_dist.png}
"""

import os
import pickle
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/steering-plateaus')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = 'results/intrinsic_dim'
OUT_DIR = os.path.join(DATA_DIR, 'pairwise_dist_rescaled')
N = 200
SEED = 0


def main():
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
    X_opt_resc = X_opt * (X_real.norm(dim=-1).mean()
                          / X_opt.norm(dim=-1).mean())

    sets = {'real': X_real, 'gauss (cov-matched)': X_gauss,
            'jac-opt (rescaled)': X_opt_resc}

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
    colors = {'real': 'C0', 'gauss (cov-matched)': 'C1',
              'jac-opt (rescaled)': 'C4'}
    for name, d in results['dists'].items():
        ax.hist(d, bins=100, density=True, alpha=0.5, color=colors[name],
                label=f'{name} (median {np.median(d):.1f})')
    ax.set_xlabel('pairwise L2 distance (within-set)')
    ax.set_ylabel('density')
    ax.set_title('qwen3-1.7b L2: pairwise distances, n=200 per set, '
                 'jac-opt rescaled to real mean norm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'pairwise_dist.png'), dpi=150,
                bbox_inches='tight')
    print(f"Saved {os.path.join(OUT_DIR, 'results.pkl')}")
    print(f"Saved {os.path.join(OUT_DIR, 'pairwise_dist.png')}")


if __name__ == '__main__':
    main()
