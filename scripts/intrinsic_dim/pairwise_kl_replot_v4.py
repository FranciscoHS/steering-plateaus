#!/usr/bin/env python3
"""Replot the pairwise output-KL histograms without 'real (own ctx)'.

That condition contains a ~zero KL pair, which drags the log axis down to
the 1e-8 floor and compresses everything else. Pure replot of
results_v3.pkl minus that condition; no new computation.

Output: results/intrinsic_dim/pairwise_kl/kl_hist_v4.png
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = 'results/intrinsic_dim/pairwise_kl'
SKIP = 'real (own ctx, fresh draw)'


def main():
    with open(os.path.join(OUT_DIR, 'results_v3.pkl'), 'rb') as f:
        results = pickle.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'real': 'C0', 'gauss (fresh)': 'C1', 'gauss rot 25deg': 'C4',
              'jac-opt': 'C2'}
    for name, mat in results['kl'].items():
        if name == SKIP:
            continue
        o = mat[~np.eye(mat.shape[0], dtype=bool)]
        ax.hist(np.log10(np.maximum(o, 1e-8)), bins=80, density=True,
                alpha=0.5, color=colors.get(name, 'C7'),
                label=f'{name} (median {np.median(o):.2f} nats)')
    ax.set_xlabel(r'$\log_{10}$ pairwise KL$(p_i \| p_j)$  (nats)')
    ax.set_ylabel('density')
    ax.set_title("qwen3-1.7b L2: pairwise output-KL within sets "
                 "(shared carrier ctx)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'kl_hist_v4.png'), dpi=150,
                bbox_inches='tight')
    print(f"Saved {os.path.join(OUT_DIR, 'kl_hist_v4.png')}")


if __name__ == '__main__':
    main()
