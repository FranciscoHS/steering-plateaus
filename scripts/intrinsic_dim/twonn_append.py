"""Generalized append-one-point script for the TwoNN-vs-N sweep.

Loads a cache, runs TwoNN at --n (single seed; 5-seed protocol is degenerate
when N equals cache size), appends the result to twonn_vs_n.json, regenerates
the plot. Used to add N=200k and beyond without re-running the earlier sweep.
"""

import argparse
import json
import os
import numpy as np
import torch
import skdim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = 'results/jacobian/twonn_vs_n'
JSON_PATH = os.path.join(OUT_DIR, 'twonn_vs_n.json')
PLOT_PATH = os.path.join(OUT_DIR, 'twonn_vs_n.png')


def run_twonn(X):
    X = X + np.random.randn(*X.shape) * 1e-7 * X.std()
    return skdim.id.TwoNN().fit(X).dimension_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=str, required=True,
                        help='Path to .pt activation cache of shape (>=N, d).')
    parser.add_argument('--n', type=int, required=True,
                        help='Sample size for TwoNN (must be <= cache size).')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {args.cache}")
    X = torch.load(args.cache, weights_only=True).float().numpy()
    print(f"  shape = {X.shape}")
    assert len(X) >= args.n, f"cache has {len(X)} < n={args.n}"

    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(X), size=args.n, replace=False)
    d = run_twonn(X[idx])
    print(f"N={args.n} seed={args.seed}: d = {d:.4f}")

    with open(JSON_PATH) as f:
        results = json.load(f)
    results[str(args.n)] = {
        'mean': float(d),
        'sem': 0.0,
        'runs': [float(d)],
        'note': f'single seed={args.seed}; 5-seed protocol is degenerate at N=N_total',
    }
    with open(JSON_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Appended to {JSON_PATH}")

    ns = sorted(int(k) for k in results.keys())
    means = [results[str(n)]['mean'] for n in ns]
    sems = [results[str(n)]['sem'] for n in ns]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(ns, means, yerr=sems, fmt='o-', capsize=4)
    ax.set_xscale('log')
    ax.set_xlabel('N (sample size)')
    ax.set_ylabel('TwoNN intrinsic dimension')
    ax.set_title('Global TwoNN ID vs sample size (Qwen3-1.7B, layer 10, last token, seq_len=10)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"Plot saved to {PLOT_PATH}")


if __name__ == '__main__':
    main()
