#!/usr/bin/env python3
"""Bar plot of ID vs OOD probe accuracy at the best C per condition.

'Best' = C that maximises OOD accuracy in the diagnostics sweep.
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostics_pkl', required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--title', default=None)
    parser.add_argument('--criterion', default='ood',
                        choices=['ood', 'id'],
                        help="Pick C per condition by best OOD or ID accuracy.")
    args = parser.parse_args()

    with open(args.diagnostics_pkl, 'rb') as f:
        d = pickle.load(f)
    res = d['results']  # res[name][C] = {train_acc, id_acc, ood_acc, gap, w_norm}

    conditions = ['raw', 'sae', 'ae']
    labels = ['Raw', 'SAE recon', 'AE recon']
    key = f'{args.criterion}_acc'

    chosen = {}  # name -> (C, id_acc, ood_acc)
    for name in conditions:
        best_C = max(res[name].keys(), key=lambda c: res[name][c][key])
        chosen[name] = (best_C, res[name][best_C]['id_acc'],
                        res[name][best_C]['ood_acc'])

    id_accs = [chosen[c][1] for c in conditions]
    ood_accs = [chosen[c][2] for c in conditions]
    cs = [chosen[c][0] for c in conditions]

    x = np.arange(len(conditions))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 5))
    b1 = ax.bar(x - width / 2, id_accs, width, label='ID (SST-2)', color='#2196F3')
    b2 = ax.bar(x + width / 2, ood_accs, width, label='OOD (Amazon)', color='#E91E63')

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9)

    xt_labels = [f'{lbl}\n(C={c:g})' for lbl, c in zip(labels, cs)]
    ax.set_xticks(x)
    ax.set_xticklabels(xt_labels)
    ax.set_ylabel('Probe accuracy')
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color='grey', linestyle=':', linewidth=0.8, label='Chance')

    layer = d['args'].get('activations', '').split('/')[-2]
    title = args.title or (f"Sentiment probe — best-{args.criterion.upper()} C per condition")
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)

    out = args.out or os.path.join(os.path.dirname(args.diagnostics_pkl),
                                   f'probe_bars_best_{args.criterion}.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    print(f"Chosen C per condition (by best {args.criterion}):")
    for name in conditions:
        c, ida, ooda = chosen[name]
        print(f"  {name:4s} C={c:g}  ID={ida:.4f}  OOD={ooda:.4f}")


if __name__ == '__main__':
    main()
