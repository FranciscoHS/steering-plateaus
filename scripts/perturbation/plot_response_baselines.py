#!/usr/bin/env python3
"""Re-plot the response-direction baseline sweep: one figure per baseline anchor,
with contrastive and random overlaid on the same axes (median + IQR over
activations x directions). Reads results.pkl from run_response_direction_sweep.py.

Usage:
    python scripts/perturbation/plot_response_baselines.py \
        --results_dir results/response_direction_sweep/qwen3-1.7b_L14_rotate
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', required=True)
    args = p.parse_args()

    out = pickle.load(open(os.path.join(args.results_dir, 'results.pkl'), 'rb'))
    xvals = out['xvals']
    xlabel = out['xlabel']
    xscale = 'log' if out['mode'] == 'additive' else 'linear'
    baseline_x = out['baseline_x']
    results = out['results']
    model = out['args']['model']
    layer = out['args']['layer']
    cap = out['args'].get('capture_layer')
    cap_str = 'final' if cap is None else f'L{cap}'
    mode = out['mode']

    conds = [('contrastive', 'C0'), ('random', 'C3')]

    for b, ba in enumerate(baseline_x):
        fig, ax = plt.subplots(figsize=(8, 6))
        for cond, color in conds:
            cos = results[cond]['cos_sim'][:, b]  # [n_act, n_sweep, n_dirs]
            flat = cos.transpose(1, 0, 2).reshape(cos.shape[1], -1)  # [n_sweep, n_act*n_dirs]
            med = np.median(flat, axis=1)
            q25, q75 = np.percentile(flat, 25, axis=1), np.percentile(flat, 75, axis=1)
            ax.plot(xvals, med, color=color, label=cond, linewidth=2)
            ax.fill_between(xvals, q25, q75, color=color, alpha=0.15)
        ax.axvline(ba, color='gray', linestyle=':', linewidth=1.5, label=f'baseline @ {ba:g}°')
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('cos(response(s), response@baseline)')
        ax.set_title(f'{model}, {mode} L{layer} -> {cap_str}: baseline anchor {ba:g}°')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(args.results_dir, f'response_baseline_{ba:g}deg.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {path}")


if __name__ == '__main__':
    main()
