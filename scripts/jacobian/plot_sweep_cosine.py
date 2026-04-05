#!/usr/bin/env python3
"""
Plot cosine distance from a feature_vs_random_sweep pkl.

Shows each found direction as an individual line vs random baseline band.

Usage:
    python scripts/jacobian/plot_sweep_cosine.py results/foo/feature_vs_random_sweep.pkl
    python scripts/jacobian/plot_sweep_cosine.py sweep.pkl --dir_keys d1 d2 d3 --title "First 3"
"""

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl_path', help='Path to sweep pkl file')
    parser.add_argument('--dir_keys', nargs='+', default=None,
                        help='Which direction keys to plot (default: all non-random)')
    parser.add_argument('--title', default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)

    angles_deg = data['angles_deg']
    cos_data = data['cos']

    # Random baseline
    rand = cos_data['random']  # [n_act, n_dirs, n_angles]
    rand_per_act = np.median(rand, axis=1)
    rand_median = np.median(rand_per_act, axis=0)
    rand_q25 = np.percentile(rand_per_act, 25, axis=0)
    rand_q75 = np.percentile(rand_per_act, 75, axis=0)

    # Direction keys
    if args.dir_keys:
        dir_keys = args.dir_keys
    else:
        dir_keys = [k for k in cos_data.keys() if k != 'random']

    fig, ax = plt.subplots(figsize=(8, 5))

    # Random band
    ax.fill_between(angles_deg, rand_q25, rand_q75, alpha=0.15, color='gray')
    ax.plot(angles_deg, rand_median, '--', color='gray', linewidth=2, label='Random')

    # Individual direction curves (median over activations)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(dir_keys)))
    for i, key in enumerate(dir_keys):
        curve = cos_data[key]  # [n_act, n_angles]
        median = np.median(curve, axis=0)
        ax.plot(angles_deg, median, '-', color=cmap[i], linewidth=1.5, alpha=0.7, label=key)

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel('Cosine distance (penultimate layer)', fontsize=12)
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, angles_deg[-1])

    if args.title:
        ax.set_title(args.title, fontsize=12)

    plt.tight_layout()
    if args.output:
        out = args.output
    else:
        out = os.path.join(os.path.dirname(args.pkl_path), 'cosine_sweep.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {out}")


if __name__ == '__main__':
    main()
