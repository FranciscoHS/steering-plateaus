#!/usr/bin/env python3
"""Principal (canonical) angles between feature subspaces from multiple
decomposition runs.

For each pair of runs, builds orthonormal bases via QR over the feature
directions (rows), then computes the singular values of Q_A^T Q_B — these are
cos(theta_i) for the principal angles theta_1 <= ... <= theta_min(Ka, Kb).

Saves per-pair cos lists, the pairwise mean cos matrix, and a plot.

Usage:
    python scripts/superellipse/principal_angles.py \\
        --runs dir1 dir2 ... --labels seedX seedY ... --out OUTDIR
"""

import argparse
import os
import pickle
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(d):
    r = pickle.load(open(os.path.join(d, 'results.pkl'), 'rb'))
    F = r['features']
    F = F.cpu().numpy() if hasattr(F, 'cpu') else np.asarray(F)
    F = F / (np.linalg.norm(F, axis=-1, keepdims=True) + 1e-12)
    return F.astype(np.float64), int(r['K_used'])


def principal_cos(A, B):
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(B.T)
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(sv, -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = [(lab, *load(d)) for lab, d in zip(args.labels, args.runs)]
    N = len(data)
    labels = [d[0] for d in data]

    lines = []
    lines.append(f'{"pair":<22}  Kmin  principal-angle cosines (desc)')
    lines.append('-' * 100)
    pair_cos = {}
    for (la, A, Ka), (lb, B, Kb) in combinations(data, 2):
        cos = principal_cos(A, B)
        pair_cos[(la, lb)] = cos
        lines.append(f'{la}<->{lb:<10}  {min(Ka, Kb):>3}    ' +
                     ' '.join(f'{c:+.3f}' for c in cos))

    mean_mat = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            cos = principal_cos(data[i][1], data[j][1])
            mean_mat[i, j] = mean_mat[j, i] = float(cos.mean())

    lines.append('\nPairwise mean cos of principal angles:')
    lines.append('              ' + ' '.join(f'{l:>10}' for l in labels))
    for i in range(N):
        lines.append(f'  {labels[i]:<10}  ' +
                     ' '.join(f'{mean_mat[i, j]:>+10.4f}' for j in range(N)))
    offdiag = mean_mat[~np.eye(N, dtype=bool)]
    lines.append(f'\nMean off-diag = {offdiag.mean():+.4f}  '
                 f'min={offdiag.min():+.4f}  max={offdiag.max():+.4f}')

    txt = os.path.join(args.out, 'principal_angles.txt')
    with open(txt, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(args.out, 'principal_angles.pkl'), 'wb') as f:
        pickle.dump({'labels': labels, 'pair_cos': pair_cos,
                     'mean_mat': mean_mat}, f)

    # Plot: per-rank cosine across pairs + heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    pairs_sorted = list(pair_cos.items())
    Kmax = max(len(v) for _, v in pairs_sorted)
    # For each rank i, gather cos values across pairs that have rank i
    ranks = list(range(Kmax))
    means, mins, maxs = [], [], []
    for k in ranks:
        vals = [v[k] for _, v in pairs_sorted if len(v) > k]
        means.append(np.mean(vals))
        mins.append(np.min(vals))
        maxs.append(np.max(vals))
    ax.plot([k + 1 for k in ranks], means, 'o-', label='mean over pairs')
    ax.fill_between([k + 1 for k in ranks], mins, maxs, alpha=0.2,
                    label='min/max')
    ax.set_xticks([k + 1 for k in ranks])
    ax.set_xlabel('principal-angle index i')
    ax.set_ylabel(r'$\cos(\theta_i)$')
    ax.set_ylim(-0.05, 1.05); ax.axhline(0, color='gray', lw=0.5)
    ax.set_title('Principal-angle cosines across pairs (sorted desc)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    im = ax.imshow(mean_mat, vmin=0, vmax=1, cmap='viridis')
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f'{mean_mat[i, j]:.3f}', ha='center', va='center',
                    fontsize=9,
                    color=('white' if mean_mat[i, j] < 0.5 else 'black'))
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_title('Pairwise mean cos of principal angles')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'principal_angles.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(open(txt).read())
    print(f'wrote {txt}')
    print(f'wrote {os.path.join(args.out, "principal_angles.png")}')


if __name__ == '__main__':
    main()
