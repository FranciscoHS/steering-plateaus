#!/usr/bin/env python3
"""Bar plot of ID vs OOD probe accuracy for raw / SAE / AE conditions."""

import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_pkl', required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--title', default=None)
    args = parser.parse_args()

    with open(args.results_pkl, 'rb') as f:
        r = pickle.load(f)

    conditions = ['raw', 'sae', 'ae']
    labels = ['Raw', 'SAE recon', 'AE recon']
    id_accs = [r[c]['id_acc'] for c in conditions]
    ood_accs = [r[c]['ood_acc'] for c in conditions]

    x = np.arange(len(conditions))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - width / 2, id_accs, width, label='ID (SST-2)', color='#2196F3')
    b2 = ax.bar(x + width / 2, ood_accs, width, label='OOD (Amazon)', color='#E91E63')

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Probe accuracy')
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color='grey', linestyle=':', linewidth=0.8, label='Chance')
    title = args.title or f"Sentiment probe — layer {r['args']['layer']}"
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)

    out = args.out or os.path.join(os.path.dirname(args.results_pkl), 'probe_bars.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
