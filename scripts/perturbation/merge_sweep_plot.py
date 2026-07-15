#!/usr/bin/env python3
"""Overlay named-direction curves from one or more feature_vs_random_sweep pkls
onto a single 3-panel (KL / L2 / cosine) plot, with the random-orthogonal band
as the baseline. Lets us add e.g. contrastive directions (run separately on the
SAME activations) onto an existing atom sweep without recomputing.
"""
import sys, os, argparse, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DASH = {'contrastive'}  # label-prefix -> dashed style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkls', nargs='+', required=True,
                    help='feature_vs_random_sweep.pkl files; named curves from all are overlaid')
    ap.add_argument('--random_from', type=int, default=0,
                    help='index into --pkls whose random band to draw (default first)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    data = [pickle.load(open(p, 'rb')) for p in args.pkls]
    angles = data[0]['angles_deg']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    metrics = [('kl', 'KL divergence (nats)'), ('l2', 'L2 distance'),
               ('cos', 'cosine distance')]

    # collect named curves across pkls (everything except 'random'), in order
    named = []  # (label, pkl_idx)
    for i, d in enumerate(data):
        for name in d['kl']:
            if name != 'random':
                named.append((name, i))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(10, len(named))))

    for ax, (m, ylabel) in zip(axes, metrics):
        # random baseline band (median + IQR over act x random_dirs)
        rnd = data[args.random_from][m]['random']           # [n_act, n_rand, n_ang]
        rflat = rnd.reshape(-1, rnd.shape[-1])
        ax.fill_between(angles, np.percentile(rflat, 25, 0), np.percentile(rflat, 75, 0),
                        color='gray', alpha=0.25, label='random (IQR)')
        ax.plot(angles, np.median(rflat, 0), color='gray', lw=1.5, ls=':')

        for k, (label, pi) in enumerate(named):
            curve = data[pi][m][label]                      # [n_act, n_ang]
            med = np.median(curve, 0)
            ls = '--' if any(label.startswith(p) for p in DASH) else '-'
            lw = 2.4 if ls == '--' else 1.6
            ax.plot(angles, med, ls=ls, lw=lw, color=cmap[k], label=label)

        ax.set_xlabel('angle (deg)'); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=1)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
