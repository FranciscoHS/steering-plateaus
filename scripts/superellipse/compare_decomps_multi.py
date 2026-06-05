#!/usr/bin/env python3
"""Compare N greedy decompositions of the same target.

Loads N results.pkl files and reports:
  - per-run K, alphas, residual frac
  - rank-wise cross-run cosine tables (cos between d_k from each pair of runs,
    for each rank k where the run had >= k features)
  - reconstruction cosine across pairs (sanity)
  - full pairwise feature-cos matrices (saved to .pkl + heatmap PNGs)

Usage:
    python scripts/superellipse/compare_decomps_multi.py \\
        --runs dir1 dir2 dir3 ... \\
        --labels seed42 seed7 seed13 ... \\
        --out results/decomp_plateau/compare_paris_5seeds
"""

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_run(d):
    with open(os.path.join(d, 'results.pkl'), 'rb') as f:
        r = pickle.load(f)
    F = r['features']
    F = F.cpu().numpy() if hasattr(F, 'cpu') else np.asarray(F)
    F = F / (np.linalg.norm(F, axis=-1, keepdims=True) + 1e-12)
    a = r['a_target']
    a = a.cpu().numpy() if hasattr(a, 'cpu') else np.asarray(a)
    return {
        'features': F.astype(np.float64),
        'alphas': np.asarray(r['alphas'], dtype=np.float64),
        'a_target': a.astype(np.float64),
        'K': int(r['K_used']),
        'residual_frac': np.asarray(r['residual_norm_frac'], dtype=np.float64),
        'target_prompt': r['args'].get('target_prompt'),
        'seed': r['args'].get('seed'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True,
                    help='dirs with results.pkl')
    ap.add_argument('--labels', nargs='+', default=None,
                    help='labels per run (default: derived from --runs)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs = [load_run(d) for d in args.runs]
    labels = (args.labels if args.labels is not None
              else [os.path.basename(d) for d in args.runs])
    N = len(runs)
    assert len(labels) == N

    # Per-run summary
    lines = []
    lines.append('Per-run summary:')
    for i, r in enumerate(runs):
        lines.append(
            f'  {labels[i]:<24}  seed={r["seed"]}  K={r["K"]}  '
            f'alphas=[{", ".join(f"{a:+.2f}" for a in r["alphas"])}]  '
            f'final ||r||/||a||={r["residual_frac"][-1]:.4f}'
        )

    # Reconstructions (sanity)
    recons = [(r['alphas'][:, None] * r['features']).sum(axis=0) for r in runs]

    # Pairwise reconstruction cosines
    lines.append('\nPairwise reconstruction cosine cos(recon_i, recon_j):')
    rec_cos = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ui, uj = recons[i], recons[j]
            rec_cos[i, j] = float(ui @ uj /
                                  (np.linalg.norm(ui) * np.linalg.norm(uj)
                                   + 1e-12))
    header = '              ' + ' '.join(f'{labels[j]:>10}' for j in range(N))
    lines.append(header)
    for i in range(N):
        lines.append(f'  {labels[i]:<10}  ' +
                     ' '.join(f'{rec_cos[i,j]:>+10.4f}' for j in range(N)))

    # Pairwise feature-cos matrices (full Ka x Kb each)
    pairwise_feat_cos = {}
    for i in range(N):
        for j in range(i + 1, N):
            M = runs[i]['features'] @ runs[j]['features'].T
            pairwise_feat_cos[(i, j)] = M

    # Rank-wise comparison: for each k, build NxN cosine of d_k across runs
    Kmax = max(r['K'] for r in runs)
    lines.append('\nRank-wise cosine cos(d_k^i, d_k^j) — only runs with >= k features:')
    rank_tables = {}
    for k in range(Kmax):
        present = [i for i in range(N) if runs[i]['K'] > k]
        if len(present) < 2:
            continue
        M = np.full((len(present), len(present)), np.nan)
        for a, i in enumerate(present):
            for b, j in enumerate(present):
                M[a, b] = float(runs[i]['features'][k] @ runs[j]['features'][k])
        rank_tables[k] = (present, M)
        # off-diagonal stats
        offdiag = M[~np.eye(len(present), dtype=bool)]
        lines.append(
            f'  rank d{k+1}: present={[labels[i] for i in present]}  '
            f'mean(off-diag)={offdiag.mean():+.3f}  '
            f'median={np.median(offdiag):+.3f}  '
            f'min={offdiag.min():+.3f}  max={offdiag.max():+.3f}'
        )

    # Hungarian matching pairwise (best 1:1 feature pairing by |cos|)
    try:
        from scipy.optimize import linear_sum_assignment
        has_scipy = True
    except ImportError:
        has_scipy = False

    if has_scipy:
        lines.append('\nPairwise Hungarian-matched mean |cos| (best 1:1 pairing):')
        match_table = np.full((N, N), np.nan)
        for i in range(N):
            for j in range(N):
                if i == j:
                    match_table[i, j] = 1.0
                    continue
                M = (runs[i]['features'] @ runs[j]['features'].T)
                K_min = min(runs[i]['K'], runs[j]['K'])
                # maximize sum of |cos|: minimize negative
                row, col = linear_sum_assignment(-np.abs(M))
                # take top K_min by |cos| (keep all matches up to min size)
                matched_cos = np.abs(M[row, col])
                match_table[i, j] = float(matched_cos.mean())
        header = '              ' + ' '.join(f'{labels[j]:>10}' for j in range(N))
        lines.append(header)
        for i in range(N):
            lines.append(f'  {labels[i]:<10}  ' +
                         ' '.join(f'{match_table[i,j]:>+10.4f}'
                                  for j in range(N)))

    # Save
    summary_path = os.path.join(args.out, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(args.out, 'compare.pkl'), 'wb') as f:
        pickle.dump({
            'labels': labels,
            'rec_cos': rec_cos,
            'pairwise_feat_cos': pairwise_feat_cos,
            'rank_tables': rank_tables,
            'match_table': match_table if has_scipy else None,
            'runs_K': [r['K'] for r in runs],
            'runs_alphas': [r['alphas'] for r in runs],
        }, f)

    # Plot: rank-wise mean off-diagonal cosine vs k
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    means, ks, mins, maxs = [], [], [], []
    for k, (present, M) in rank_tables.items():
        offdiag = M[~np.eye(len(present), dtype=bool)]
        ks.append(k + 1); means.append(offdiag.mean())
        mins.append(offdiag.min()); maxs.append(offdiag.max())
    ax.plot(ks, means, 'o-', label='mean off-diag')
    ax.fill_between(ks, mins, maxs, alpha=0.2, label='min/max range')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('feature rank k'); ax.set_ylabel('cos(d_k^i, d_k^j) across runs')
    ax.set_ylim(-1.05, 1.05); ax.set_xticks(ks)
    ax.set_title('Rank-wise cross-seed cosine')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Hungarian-matched table heatmap
    if has_scipy:
        ax = axes[1]
        im = ax.imshow(match_table, vmin=0, vmax=1, cmap='viridis')
        for i in range(N):
            for j in range(N):
                ax.text(j, i, f'{match_table[i,j]:.3f}', ha='center',
                        va='center', fontsize=9,
                        color=('white' if match_table[i, j] < 0.5 else 'black'))
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        ax.set_title('Pairwise Hungarian-matched mean |cos|')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'summary.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    print(open(summary_path).read())
    print(f'wrote {summary_path}')
    print(f'wrote {os.path.join(args.out, "summary.png")}')


if __name__ == '__main__':
    main()
