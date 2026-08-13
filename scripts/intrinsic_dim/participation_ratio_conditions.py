#!/usr/bin/env python3
"""Per-vector participation ratio in the standard basis, three sets.

For each activation x, PR(x) = (sum_i x_i^2)^2 / sum_i x_i^4 — the
effective number of coordinates the vector is spread over (1 = one-hot,
d = perfectly uniform). Scale-invariant, so the jac-opt norm gap is
irrelevant. Same sets and draws as pairwise_dist_conditions.py (seed 0):
real / cov-matched Gaussian / jac-opt, n=200 each, qwen3-1.7b L2.

Output: results/intrinsic_dim/participation_ratio/{results.pkl, pr_hist.png}
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
OUT_DIR = os.path.join(DATA_DIR, 'participation_ratio')
N = 200
SEED = 0


def participation_ratio(X):
    """PR over standard-basis coordinates, one value per row."""
    x2 = X ** 2
    return (x2.sum(dim=-1) ** 2) / (x2 ** 2).sum(dim=-1)


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

    d_model = X_real.shape[1]
    sets = {'real': X_real, 'gauss (cov-matched)': X_gauss,
            'jac-opt': X_opt}

    results = {'n': N, 'seed': SEED, 'd_model': d_model, 'pr': {}}
    for name, X in sets.items():
        pr = participation_ratio(X).numpy()
        results['pr'][name] = pr
        q25, q75 = np.percentile(pr, [25, 75])
        print(f'{name:22s} PR: median={np.median(pr):.1f}  '
              f'IQR={q25:.1f}-{q75:.1f}  min={pr.min():.1f}  '
              f'max={pr.max():.1f}  (d={d_model})', flush=True)

    with open(os.path.join(OUT_DIR, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'real': 'C0', 'gauss (cov-matched)': 'C1', 'jac-opt': 'C4'}
    for name, pr in results['pr'].items():
        ax.hist(pr, bins=50, density=True, alpha=0.5, color=colors[name],
                label=f'{name} (median {np.median(pr):.0f})')
    ax.set_xlabel(f'participation ratio over standard-basis coords '
                  f'(d={d_model})')
    ax.set_ylabel('density')
    ax.set_title('qwen3-1.7b L2: per-vector participation ratio, '
                 'n=200 per set')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'pr_hist.png'), dpi=150,
                bbox_inches='tight')
    print(f"Saved {os.path.join(OUT_DIR, 'results.pkl')}")
    print(f"Saved {os.path.join(OUT_DIR, 'pr_hist.png')}")


if __name__ == '__main__':
    main()
