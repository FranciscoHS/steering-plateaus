#!/usr/bin/env python3
"""Overlay per-direction mean-KL distributions across families, pooling one or more
ablate_directions_kl summary.pkl files (so a family run separately can be merged in)."""
import sys, os, argparse, pickle
import numpy as np

ORDER = ['random', 'sae', 'contrastive', 'found_noalign', 'found']
COLORS = {'random': 'gray', 'sae': 'tab:green', 'contrastive': 'tab:orange',
          'found_noalign': 'tab:red', 'found': 'tab:blue'}
LABELS = {'found': 'found (40k, +align)', 'found_noalign': 'found (no align)',
          'contrastive': 'contrastive (diff-of-means)', 'sae': 'SAE decoder',
          'random': 'random unit'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summaries', nargs='+', required=True, help='summary.pkl paths')
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='ablation KL by direction family')
    args = ap.parse_args()

    fam_vals = {}  # family -> list of per-direction mean KL
    for p in args.summaries:
        s = pickle.load(open(p, 'rb'))
        fam = np.array(s['family'])
        mk = np.asarray(s['mean_kl'])
        for f in np.unique(fam):
            fam_vals.setdefault(f, []).append(mk[fam == f])
    fam_vals = {f: np.concatenate(v) for f, v in fam_vals.items()}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    allv = np.concatenate(list(fam_vals.values()))
    eps = 1e-12
    lo = max(eps, float(allv[allv > 0].min()))
    hi = float(allv.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 50)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for f in ORDER + [f for f in fam_vals if f not in ORDER]:
        if f not in fam_vals:
            continue
        v = fam_vals[f]
        ax.hist(v.clip(lo, hi), bins=bins, density=True, alpha=0.55,
                color=COLORS.get(f), label=f'{LABELS.get(f, f)} (n={len(v)}, med={np.median(v):.2e})')
    ax.set_xscale('log')
    ax.set_xlabel('per-direction mean KL(base || ablated)  [nats/token]')
    ax.set_ylabel('density')
    ax.set_title(args.title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f'wrote {args.out}')
    for f in ORDER:
        if f in fam_vals:
            v = fam_vals[f]
            print(f'  {f:14s} n={len(v):4d} median={np.median(v):.4e} '
                  f'mean={v.mean():.4e} p90={np.quantile(v,0.9):.4e}')


if __name__ == '__main__':
    main()
