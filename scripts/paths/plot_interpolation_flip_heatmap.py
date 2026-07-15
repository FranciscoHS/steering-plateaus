#!/usr/bin/env python3
"""
Heatmap of per-step neuron flips across (interpolation t) x (layer), from the
.pt produced by interpolation_neuron_flips.py.

Main panel: per-step flip COUNT at each captured layer as t goes 0 -> 1.
  - a vertical band  => boundary is synchronized across layers
  - a tilted band    => flip front propagates layer-by-layer
  - bright rows      => flips concentrated in particular layers
Top marginal:  per-step flips summed over layers vs t, with the logit
               rel-distance S-curve overlaid (right axis).
Right marginal: total flips per layer summed over the whole path.

No GPU / model needed. One figure per (file, threshold).

Usage:
    python scripts/paths/plot_interpolation_flip_heatmap.py results/interpolation_flips/*.pt
"""

import argparse
import os

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_heatmap(res, tau, save_path):
    t = res['t_values']
    layers = res['capture_layers']
    d_mlp = res['d_mlp']
    # [n_steps, n_layers] -> absolute count per layer per step
    counts = res['perstep'][tau] * d_mlp
    n_steps, n_layers = counts.shape

    top_marg = counts.sum(axis=1)          # per-step flips summed over layers
    right_marg = counts.sum(axis=0)        # total flips per layer over the path

    fig = plt.figure(figsize=(11, 7))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                  hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main heatmap: x=t, y=layer
    im = ax_main.imshow(
        counts.T, aspect='auto', origin='lower', cmap='magma',
        extent=[t[0], t[-1], layers[0] - 0.5, layers[-1] + 0.5])
    ax_main.set_xlabel('Interpolation t')
    ax_main.set_ylabel('Layer (MLP)')
    ax_main.set_yticks(layers)

    # Top marginal: total per-step flips + rel-distance overlay
    ax_top.plot(t, top_marg, color='C0', lw=1.8)
    ax_top.set_ylabel('flips/step\n(all layers)', fontsize=9, color='C0')
    ax_top.tick_params(axis='y', labelcolor='C0', labelsize=8)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.grid(True, alpha=0.3)
    ax_topr = ax_top.twinx()
    ax_topr.plot(t, res['rel_distances'], color='C3', lw=1.5, ls='--')
    ax_topr.plot([0, 1], [0, 1], color='gray', lw=1, ls=':', alpha=0.6)
    ax_topr.set_ylabel('rel-dist', fontsize=9, color='C3')
    ax_topr.tick_params(axis='y', labelcolor='C3', labelsize=8)
    ax_topr.set_ylim(-0.02, 1.02)

    # Right marginal: total flips per layer
    ax_right.plot(right_marg, layers, color='C0', lw=1.8)
    ax_right.fill_betweenx(layers, 0, right_marg, color='C0', alpha=0.25)
    ax_right.set_xlabel('total flips\n(over path)', fontsize=9)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.tick_params(axis='x', labelsize=8)
    ax_right.grid(True, alpha=0.3)

    cbar = fig.colorbar(im, ax=ax_right, fraction=0.15, pad=0.35)
    cbar.set_label('flips per step (count)', fontsize=9)

    ax_top.set_title(
        f"{res['model']}  L{res['layer']}->{res['measure_layer']}  "
        f"{res['prompt_a']!r} -> {res['prompt_b']!r}   (tau={tau})",
        fontsize=11)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pt_files', nargs='+')
    p.add_argument('--thresholds', type=float, nargs='+', default=None,
                   help='which thresholds to plot (default: all in the file)')
    args = p.parse_args()

    for pt in args.pt_files:
        res = torch.load(pt, weights_only=False)
        taus = args.thresholds if args.thresholds is not None else res['thresholds']
        for tau in taus:
            out = pt.replace('.pt', f'_heatmap_tau{tau}.png')
            plot_heatmap(res, tau, out)


if __name__ == '__main__':
    main()
