#!/usr/bin/env python3
"""Is p a property of each direction, of the pair, or a bit of both?

Takes fit_pairs.pkl caches (from fit_pairs_dictionary.py) where each direction
appears in many pairs, and decomposes the pairwise p's:

    log p_ij = mu + a_i + a_j + eps_ij      (additive direction effects)

Reports, per cache:
  - R^2 of the additive fit (share of variance carried by direction effects),
  - a permutation null for that R^2 (shuffle p across pairs, refit): with
    n_dirs params on n_pairs points the null R^2 is well above 0, so the
    excess over the null is the honest signal,
  - the correlation between p's of pairs that share one direction
    (all dyads {(i,j),(i,k)}, symmetrized), with the same permutation null.

Pure post-processing of cached fits — no model forwards, safe to run locally.

Usage:
    python scripts/superellipse/p_additivity_analysis.py \
        --caches results/dictionary/fit_pairs_ratio_K20slice/fit_pairs.pkl \
                 results/dictionary/qwen3-1.7b_L14_K8000/fit_pairs_contrastive_vs_contrastive/fit_pairs.pkl \
        --labels atoms_K20 contrastive \
        --out results/dictionary/p_additivity
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_pairs(path, max_abs_cos=None):
    """Return (pairs [n,2] int re-indexed 0..n_dirs-1, p [n], cos [n]).

    Accepts two cache formats:
      - fit_pairs_dictionary.py: {'per_pair': [{'pair', 'p_fit', 'cos'}, ...]}
      - eval_sae_full_p.py:      {'p': [n], 'pair_d': [n], 'pair_c': [n]}
        (no cos stored; max_abs_cos is ignored for this format)
    """
    with open(path, 'rb') as f:
        d = pickle.load(f)
    pairs, ps, coss = [], [], []
    if 'per_pair' in d:
        for e in d['per_pair']:
            p = e.get('p_fit')
            if p is None or not np.isfinite(p):
                continue
            if max_abs_cos is not None and abs(e.get('cos', 0.0)) > max_abs_cos:
                continue
            pairs.append(e['pair'])
            ps.append(float(p))
            coss.append(float(e.get('cos', np.nan)))
    else:
        for di, ci, p in zip(np.asarray(d['pair_d']).ravel(),
                             np.asarray(d['pair_c']).ravel(),
                             np.asarray(d['p'], dtype=float).ravel()):
            if not np.isfinite(p) or di == ci:
                continue
            pairs.append((int(di), int(ci)))
            ps.append(float(p))
            coss.append(np.nan)
    idx = sorted({i for pr in pairs for i in pr})
    remap = {orig: k for k, orig in enumerate(idx)}
    pairs = np.array([[remap[i], remap[j]] for i, j in pairs], dtype=int)
    return pairs, np.array(ps), np.array(coss), len(idx)


def additive_r2(pairs, y, n_dirs):
    """R^2 of y_ij ~ mu + a_i + a_j (least squares). Returns (r2, y_pred)."""
    n = len(y)
    X = np.zeros((n, n_dirs + 1))
    X[:, 0] = 1.0
    X[np.arange(n), 1 + pairs[:, 0]] = 1.0
    X[np.arange(n), 1 + pairs[:, 1]] = 1.0
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot, y_pred


def shared_dyads(pairs, y):
    """All symmetrized dyads of pairs sharing exactly one direction.

    Returns (u, v): y-values of the two pairs in each dyad, both orders
    included so the correlation is symmetric.
    """
    by_dir = {}
    for row, (i, j) in enumerate(pairs):
        by_dir.setdefault(i, []).append(row)
        by_dir.setdefault(j, []).append(row)
    u, v = [], []
    for rows in by_dir.values():
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                u.append(y[rows[a]]); v.append(y[rows[b]])
                u.append(y[rows[b]]); v.append(y[rows[a]])
    return np.array(u), np.array(v)


def analyze(pairs, y, n_dirs, n_perm, rng):
    r2, y_pred = additive_r2(pairs, y, n_dirs)
    u, v = shared_dyads(pairs, y)
    r_shared = np.corrcoef(u, v)[0, 1]

    r2_null = np.empty(n_perm)
    r_null = np.empty(n_perm)
    for t in range(n_perm):
        y_sh = rng.permutation(y)
        r2_null[t], _ = additive_r2(pairs, y_sh, n_dirs)
        us, vs = shared_dyads(pairs, y_sh)
        r_null[t] = np.corrcoef(us, vs)[0, 1]

    return {
        'r2': r2, 'y_pred': y_pred,
        'r2_null_mean': r2_null.mean(), 'r2_null_sd': r2_null.std(),
        'r2_pval': float(np.mean(r2_null >= r2)),
        'r_shared': r_shared,
        'r_null_mean': r_null.mean(), 'r_null_sd': r_null.std(),
        'r_pval': float(np.mean(r_null >= r_shared)),
        'n_dyads': len(u) // 2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--caches', nargs='+', required=True,
                    help='fit_pairs.pkl paths')
    ap.add_argument('--labels', nargs='+', default=None,
                    help='one label per cache (default: parent dir name)')
    ap.add_argument('--max_abs_cos', type=float, default=None,
                    help='drop pairs with |cos| above this (default: keep all)')
    ap.add_argument('--use_log', action='store_true', default=True)
    ap.add_argument('--no_log', dest='use_log', action='store_false',
                    help='decompose raw p instead of log p')
    ap.add_argument('--n_perm', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/dictionary/p_additivity')
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(os.path.dirname(c))
                             for c in args.caches]
    assert len(labels) == len(args.caches)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summary = {'args': vars(args), 'caches': {}}
    lines = []
    fig, axes = plt.subplots(1, len(args.caches),
                             figsize=(5 * len(args.caches), 4.5), squeeze=False)

    for k, (cache, label) in enumerate(zip(args.caches, labels)):
        pairs, p, cos, n_dirs = load_pairs(cache, args.max_abs_cos)
        y = np.log(p) if args.use_log else p.copy()
        res = analyze(pairs, y, n_dirs, args.n_perm, rng)

        deg = np.bincount(pairs.ravel(), minlength=n_dirs)
        lines += [
            f"== {label}  ({cache})",
            f"   n_dirs={n_dirs}  n_pairs={len(p)}  "
            f"pairs/dir: min={deg.min()} max={deg.max()}",
            f"   p: median={np.median(p):.3f}  IQR=({np.percentile(p,25):.3f}, "
            f"{np.percentile(p,75):.3f})  scale={'log' if args.use_log else 'raw'}",
            f"   additive R^2 = {res['r2']:.3f}   "
            f"null = {res['r2_null_mean']:.3f} +- {res['r2_null_sd']:.3f}   "
            f"excess = {res['r2'] - res['r2_null_mean']:.3f}   "
            f"perm-pval = {res['r2_pval']:.4f}",
            f"   shared-dir corr r = {res['r_shared']:.3f}   "
            f"null = {res['r_null_mean']:.3f} +- {res['r_null_sd']:.3f}   "
            f"perm-pval = {res['r_pval']:.4f}   (n_dyads={res['n_dyads']})",
        ]
        summary['caches'][label] = {
            'cache': cache, 'n_dirs': n_dirs, 'n_pairs': len(p),
            'p': p, 'pairs': pairs, 'cos': cos, **res,
        }

        ax = axes[0, k]
        ax.scatter(res['y_pred'], y, s=18, alpha=0.7)
        lo = min(res['y_pred'].min(), y.min())
        hi = max(res['y_pred'].max(), y.max())
        ax.plot([lo, hi], [lo, hi], 'k:', alpha=0.5)
        ax.set_xlabel('additive prediction (mu + a_i + a_j)')
        ax.set_ylabel('log p' if args.use_log else 'p')
        ax.set_title(f"{label}\nR^2={res['r2']:.2f} "
                     f"(null {res['r2_null_mean']:.2f})")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.out, 'additive_fit.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(args.out, 'summary.txt'), 'w') as f:
        f.write(text + '\n')
    with open(os.path.join(args.out, 'summary.pkl'), 'wb') as f:
        pickle.dump(summary, f)
    print(f"\nSaved {args.out}/summary.txt, summary.pkl, additive_fit.png")


if __name__ == '__main__':
    main()
