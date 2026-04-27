"""Append the N=100k point to the TwoNN-vs-N sweep.

Concats the two 50k caches (seed=42 prefix + seed=42 offset=50k) into a 100k
cache, runs TwoNN at N=100k with 5 seeds (same protocol as twonn_vs_n.py), and
appends the result to results/jacobian/twonn_vs_n/twonn_vs_n.json. Regenerates
the plot with the 100k point added.
"""

import json
import os
import numpy as np
import torch
import skdim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE_A = 'results/jacobian/l10_last_token_seq10.pt'
CACHE_B = 'results/jacobian/l10_last_token_seq10_part2.pt'
CACHE_100K = 'results/jacobian/l10_last_token_seq10_100k.pt'
OUT_DIR = 'results/jacobian/twonn_vs_n'
JSON_PATH = os.path.join(OUT_DIR, 'twonn_vs_n.json')
PLOT_PATH = os.path.join(OUT_DIR, 'twonn_vs_n.png')
N = 100000
N_RUNS = 5


def run_twonn(X):
    X = X + np.random.randn(*X.shape) * 1e-7 * X.std()
    return skdim.id.TwoNN().fit(X).dimension_


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(CACHE_100K):
        print(f"Loading existing 100k cache from {CACHE_100K}")
        X = torch.load(CACHE_100K, weights_only=True).float().numpy()
    else:
        print(f"Building 100k cache from {CACHE_A} + {CACHE_B}")
        a = torch.load(CACHE_A, weights_only=True)
        b = torch.load(CACHE_B, weights_only=True)
        print(f"  part A: {tuple(a.shape)}, part B: {tuple(b.shape)}")
        assert a.shape[1] == b.shape[1], "shape mismatch"
        full = torch.cat([a, b], dim=0)
        print(f"  concatenated: {tuple(full.shape)}")
        torch.save(full, CACHE_100K)
        print(f"  saved to {CACHE_100K}")
        X = full.float().numpy()

    assert len(X) >= N, f"cache has {len(X)} < {N}"
    print(f"Loaded {X.shape}")

    runs = []
    for r in range(N_RUNS):
        rng = np.random.RandomState(r)
        idx = rng.choice(len(X), size=N, replace=False)
        d = run_twonn(X[idx])
        runs.append(d)
        print(f"  seed={r}: d = {d:.4f}")
    mean = float(np.mean(runs))
    sem = float(np.std(runs, ddof=1) / np.sqrt(len(runs)))
    print(f"N={N}: ID = {mean:.2f} ± {sem:.2f}")

    with open(JSON_PATH) as f:
        results = json.load(f)
    results[str(N)] = {'mean': mean, 'sem': sem, 'runs': runs}
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
    ax.set_title('Global TwoNN ID vs sample size (layer 10, last token, seq_len=10)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"Plot saved to {PLOT_PATH}")


if __name__ == '__main__':
    main()
