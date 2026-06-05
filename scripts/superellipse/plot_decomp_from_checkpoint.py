#!/usr/bin/env python3
"""Build the summary plot + a final results.pkl from a decomp checkpoint.

Used when the decomposition run was killed mid-flight. The checkpoint
contains features/alphas/residual_norm_frac/args; we lack the per-step
loss histories (those were only in memory), so the loss-curves panel is
omitted vs the in-script summary.
"""
import argparse
import os
import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.dirname(args.checkpoint)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    feats = ckpt['features']         # [K, d]
    alphas = ckpt['alphas']
    rnf = ckpt['residual_norm_frac']
    ckpt_args = ckpt['args']
    K = feats.shape[0]
    print(f'K={K}')
    print('alphas:', np.round(alphas, 3))
    print('residual_norm_frac:', np.round(rnf, 4))

    # Feature-feature gram
    feat_gram = (feats @ feats.T).numpy()

    # Save results.pkl with what we have
    results = {
        'args': ckpt_args,
        'features': feats,
        'alphas': alphas,
        'residual_norm_frac': rnf,
        'feat_gram': feat_gram,
        'K_used': K,
        'note': 'built from checkpoint; loss histories and scaffold grams not preserved',
    }
    out_pkl = os.path.join(out_dir, 'results.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f'saved {out_pkl}')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Residual trajectory
    ax = axes[0]
    ax.plot(np.arange(len(rnf)), rnf, 'o-')
    stop_frac = ckpt_args.get('stop_frac', 0.05)
    ax.axhline(stop_frac, color='gray', ls=':', label=f'stop={stop_frac}')
    ax.set_xlabel('k (features used)')
    ax.set_ylabel(r'$||r_k|| / ||a||$')
    ax.set_title('Residual norm fraction')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # |alpha_k|
    ax = axes[1]
    x = np.arange(1, K + 1)
    ax.bar(x, np.abs(alphas), color='C0')
    ax.set_xlabel('k'); ax.set_ylabel(r'$|\alpha_k|$')
    ax.set_title(r'Per-feature scalar projection (abs); $\alpha_k = d_k \cdot r_{k-1}$')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)

    # Feature-feature gram
    ax = axes[2]
    im = ax.imshow(feat_gram, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f'd{k+1}' for k in range(K)])
    ax.set_yticklabels([f'd{k+1}' for k in range(K)])
    ax.set_title(r'Feature-feature gram $d_i \cdot d_j$')
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"{ckpt_args.get('model')} L{ckpt_args.get('layer')}, "
                 f"K={K}, alpha={ckpt_args.get('angles')} deg, "
                 f"lam_overlap={ckpt_args.get('lam_overlap')} "
                 f"(killed early)", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {plot_path}')


if __name__ == '__main__':
    main()
