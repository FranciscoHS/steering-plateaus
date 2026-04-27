#!/usr/bin/env python3
"""Plot individual-direction and combination sweeps from ten_direction_fit.pkl.

Shows response (cos/L2/KL) vs angle for each of the K individual random
directions, with a horizontal line at the calibrated threshold. Also shows
the combination sweeps in a separate figure.
"""
import os, sys, pickle, argparse
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_sweeps(sweeps, angles, thresholds, alphas, title_prefix, save_path):
    """sweeps: list of K dicts with keys cos/l2/kl. angles: [N]. thresholds: dict."""
    n = len(sweeps)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    cmap = plt.get_cmap('tab10' if n <= 10 else 'tab20')
    for ax, metric, ylabel in zip(
            axes, ('cos', 'l2', 'kl'),
            ('Cosine distance', 'L2 distance', 'KL divergence (nats)')):
        for i, s in enumerate(sweeps):
            ax.plot(angles, s[metric], color=cmap(i % cmap.N),
                    lw=1.3, alpha=0.9, label=f'{i}')
            a_i = alphas[metric][i] if alphas is not None else None
            if a_i is not None and np.isfinite(a_i):
                ax.plot(a_i, thresholds[metric], 'o',
                        color=cmap(i % cmap.N), markersize=5)
        ax.axhline(thresholds[metric], color='k', ls='--', lw=1.2,
                   label=f'threshold = {thresholds[metric]:.3g}')
        ax.set_xlabel('Perturbation angle (deg)')
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2, loc='upper left')
    fig.suptitle(title_prefix, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', default='results/ten_direction_fit/qwen3-1.7b_L7_random/ten_direction_fit.pkl')
    parser.add_argument('--outdir', default=None)
    args = parser.parse_args()

    with open(args.pkl, 'rb') as f:
        r = pickle.load(f)
    outdir = args.outdir or os.path.dirname(args.pkl)

    thresholds = r['thresholds']
    angles = r['angles_deg']
    K = r['K']
    N_COMBO = r['N_COMBO']

    plot_sweeps(
        r['indiv_sweeps'], angles, thresholds, r['alpha_indiv'],
        title_prefix=f'qwen3-1.7b L7: individual-direction sweeps '
                     f'(K={K} random orthonormal directions)',
        save_path=os.path.join(outdir, 'indiv_sweeps.png'))

    plot_sweeps(
        r['combo_sweeps'], angles, thresholds, r['alpha_combo'],
        title_prefix=f'qwen3-1.7b L7: combination sweeps '
                     f'(N={N_COMBO} random dense unit combinations in K=10 span)',
        save_path=os.path.join(outdir, 'combo_sweeps.png'))


if __name__ == '__main__':
    main()
