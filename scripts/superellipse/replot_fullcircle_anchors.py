#!/usr/bin/env python3
"""Replot per-anchor full-circle heatmaps from a fullcircle_grids.pkl.

plot_fullcircle_pair_heatmap.py stores the grids for ALL anchors in its
output pkl but only renders the first --plot_anchors of them. This script
renders every anchor (or a chosen pair only) without touching the model.

Usage:
    python scripts/superellipse/replot_fullcircle_anchors.py \\
        --grids_pkl results/fullcircle_heatmaps/contrastive_fsec/fullcircle_grids.pkl \\
        --pair 0,1 --out results/fullcircle_heatmaps/contrastive_fsec/anchors_0_1
"""
import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

from scripts.superellipse.plot_fullcircle_pair_heatmap import plot_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids_pkl', required=True)
    ap.add_argument('--pair', default=None,
                    help="'i,j' — replot only this pair (default: all pairs)")
    ap.add_argument('--measure_layer', type=int, default=26,
                    help='for the colorbar label when the run args stored '
                         'None (qwen3-1.7b default: 26)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.grids_pkl, 'rb') as f:
        saved = pickle.load(f)
    angles_deg = saved['angles_deg']
    run_args = saved['args']
    measure_layer = run_args.get('measure_layer') or args.measure_layer

    want = (tuple(int(x) for x in args.pair.split(',')) if args.pair else None)
    for pair, entry in saved['per_pair'].items():
        if want is not None and tuple(pair) != want:
            continue
        label1, label2 = entry['labels']
        fit = entry.get('fit')
        cos = entry['cos']
        ptxt = f", p={fit['p']:.2f}" if fit else ''
        tag = f'{pair[0]}_{pair[1]}'
        for k in range(entry['grids'].shape[0]):
            plot_grid(entry['grids'][k], angles_deg, label1, label2,
                      measure_layer,
                      f'{label1} x {label2} — anchor {k} (cos={cos:.2f}{ptxt})',
                      os.path.join(args.out, f'heatmap_{tag}_anchor{k}.png'),
                      fit=fit)


if __name__ == '__main__':
    main()
