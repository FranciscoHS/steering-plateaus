#!/usr/bin/env python3
"""
Plot raw (un-normalized) L2 and local slope dL2/dm per direction group, from the
decompose_diff_directions.py results.pkl. The inflection point ("where it starts
getting steeper") is the magnitude where dL2/dm peaks; "no plateau" = peak near m=0.

Usage:
    python scripts/perturbation/plot_decompose_raw_slope.py \
        --results results/diff_of_activations/gpt2-xl_L2_decompose
"""

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=str,
                    default='results/diff_of_activations/gpt2-xl_L2_decompose')
    args = ap.parse_args()

    with open(os.path.join(args.results, 'results.pkl'), 'rb') as f:
        out = pickle.load(f)
    mags = out['mags']
    order = ['real_diff', 'single', 'single-mean', 'mean', 'neg_base',
             'mixture', 'diff_orth', 'random']
    order = [g for g in order if g in out['groups']]
    cmap = plt.get_cmap('tab10')
    colors = {g: cmap(i) for i, g in enumerate(order)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    print("Inflection (magnitude where median dL2/dm peaks):")
    for g in order:
        l2 = out['groups'][g]['l2']                      # [n_dirs, n_steps]
        med = np.median(l2, axis=0)
        slope = np.median(np.gradient(l2, mags, axis=1), axis=0)
        infl_m = mags[int(np.argmax(slope))]
        print(f"  {g:12s} inflection @ m={infl_m:6.1f}  "
              f"(peak slope {slope.max():.3f}, final L2 {med[-1]:.1f})")

        axes[0].plot(mags, med, color=colors[g], lw=2, label=g)
        if l2.shape[0] > 1:
            axes[0].fill_between(mags, np.percentile(l2, 25, axis=0),
                                 np.percentile(l2, 75, axis=0), color=colors[g], alpha=0.10)
        axes[1].plot(mags, slope, color=colors[g], lw=2, label=g)
        axes[1].scatter([infl_m], [slope.max()], color=colors[g], s=40, zorder=5)

    axes[0].set_xlabel('perturbation magnitude m')
    axes[0].set_ylabel(f"L2 @ layer {out['measure_layer']} (raw)")
    axes[0].set_title('Raw L2 response')
    axes[1].set_xlabel('perturbation magnitude m')
    axes[1].set_ylabel('median dL2/dm')
    axes[1].set_title('Local slope (dots = inflection / peak slope)')
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Raw L2 & local slope by direction group  (base='{out['base_prompt']}')",
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.results, 'raw_slope.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
