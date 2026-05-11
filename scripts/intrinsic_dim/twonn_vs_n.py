"""Diagnose why global TwoNN estimate differs between the layer sweep (N=2000 x 10 runs)
and the cluster sweep (N=50000 single run). Run TwoNN at increasing N on the cached
layer-10 activations."""

import json
import os
import numpy as np
import torch
import skdim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE = 'results/jacobian/l10_last_token_seq10.pt'
OUT_DIR = 'results/jacobian/twonn_vs_n'
N_LIST = [1000, 2000, 5000, 10000, 20000, 50000]
N_RUNS = 5


def run_twonn(X):
    X = X + np.random.randn(*X.shape) * 1e-7 * X.std()
    return skdim.id.TwoNN().fit(X).dimension_


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X = torch.load(CACHE, weights_only=True).float().numpy()
    print(f"Loaded {X.shape}")

    results = {}
    for n in N_LIST:
        runs = []
        for r in range(N_RUNS):
            rng = np.random.RandomState(r)
            idx = rng.choice(len(X), size=n, replace=False)
            runs.append(run_twonn(X[idx]))
        mean = float(np.mean(runs))
        sem = float(np.std(runs, ddof=1) / np.sqrt(len(runs)))
        results[str(n)] = {'mean': mean, 'sem': sem, 'runs': runs}
        print(f"  N={n:6d}: ID = {mean:.2f} ± {sem:.2f}")

    with open(os.path.join(OUT_DIR, 'twonn_vs_n.json'), 'w') as f:
        json.dump(results, f, indent=2)

    ns = N_LIST
    means = [results[str(n)]['mean'] for n in ns]
    sems = [results[str(n)]['sem'] for n in ns]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(ns, means, yerr=sems, fmt='o-', capsize=4)
    ax.set_xscale('log')
    ax.set_xlabel('N (sample size)')
    ax.set_ylabel('TwoNN intrinsic dimension')
    ax.set_title('Global TwoNN ID vs sample size (layer 10, last token, seq_len=10)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'twonn_vs_n.png'), dpi=150)
    print(f"Plot saved.")


if __name__ == '__main__':
    main()
