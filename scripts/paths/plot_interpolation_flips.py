#!/usr/bin/env python3
"""
Re-plot interpolation neuron-flip results from saved .pt files.

Reads the .pt produced by interpolation_neuron_flips.py and draws, per threshold:
  - per-step ABSOLUTE flip count (summed over captured layers) on the left axis,
    with a secondary left axis showing the equivalent per-step flip FRACTION,
  - the logit relative-distance S-curve on the right axis.

No GPU / model needed — works purely from the saved arrays.

Usage:
    python scripts/paths/plot_interpolation_flips.py results/interpolation_flips/*.pt
"""

import argparse
import os

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_pair(res, save_path):
    taus = res['thresholds']
    t = res['t_values']
    d_mlp = res['d_mlp']
    n_layers = len(res['capture_layers'])
    # fraction is averaged within a layer (mean over d_mlp), summed over layers.
    # absolute count = fraction-per-layer * d_mlp, summed over layers = sum_frac * d_mlp.
    frac_to_count = d_mlp                      # since all layers share d_mlp
    max_count = d_mlp * n_layers

    fig, axes = plt.subplots(1, len(taus), figsize=(7.5 * len(taus), 5), squeeze=False)
    for k, tau in enumerate(taus):
        ax = axes[0][k]
        perstep_frac = res['perstep'][tau].sum(axis=1)      # summed over layers
        perstep_count = perstep_frac * frac_to_count

        ax.plot(t, perstep_count, color='C0', lw=2)
        ax.set_xlabel('Interpolation t')
        ax.set_ylabel(f'Per-step flips (count, summed over {n_layers} layers)', color='C0')
        ax.tick_params(axis='y', labelcolor='C0')
        ax.set_ylim(0, max(perstep_count.max() * 1.1, 1))
        ax.grid(True, alpha=0.3)

        # Secondary left axis: same data as a fraction (count <-> fraction rescale)
        sec = ax.secondary_yaxis(
            -0.16, functions=(lambda c: c / frac_to_count,
                              lambda f: f * frac_to_count))
        sec.set_ylabel('Per-step flip fraction (summed over layers)', color='C0')
        sec.tick_params(axis='y', labelcolor='C0')

        # Right axis: logit relative distance
        ax2 = ax.twinx()
        ax2.plot(t, res['rel_distances'], color='C3', lw=2, ls='--',
                 label='logit rel-distance')
        ax2.plot([0, 1], [0, 1], color='gray', lw=1, ls=':', alpha=0.6)
        ax2.set_ylabel('Logit relative distance', color='C3')
        ax2.tick_params(axis='y', labelcolor='C3')
        ax2.set_ylim(-0.02, 1.02)

        ax.set_title(f"tau={tau}  (max possible = {max_count} flips)")

    fig.suptitle(f"{res['model']}  L{res['layer']}->{res['measure_layer']}  "
                 f"{res['prompt_a']!r} -> {res['prompt_b']!r}", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pt_files', nargs='+', help='saved .pt result files')
    p.add_argument('--suffix', default='_count',
                   help='filename suffix for the new png (default: _count)')
    args = p.parse_args()

    for pt in args.pt_files:
        res = torch.load(pt, weights_only=False)
        out = pt.replace('.pt', f'{args.suffix}.png')
        plot_pair(res, out)


if __name__ == '__main__':
    main()
