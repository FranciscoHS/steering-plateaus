#!/usr/bin/env python3
"""Activation-conditioned ablation-KL histograms: for each direction, mean KL over
its own top-p% most-active tokens (ranked by |h.d|). One panel per percentile.

Reads kl_per_token.pt + act_per_token.pt (+ summary.pkl for family labels) from a run
dir, so the percentile grid can be swept without re-running the GPU job."""
import sys, os, argparse, pickle
import numpy as np
import torch

ORDER = ['random', 'sae', 'contrastive', 'found_noalign', 'found']
COLORS = {'random': 'gray', 'sae': 'tab:green', 'contrastive': 'tab:orange',
          'found_noalign': 'tab:red', 'found': 'tab:blue'}
LABELS = {'found': 'found (40k, +align)', 'found_noalign': 'found (no align)',
          'contrastive': 'contrastive (diff-of-means)', 'sae': 'SAE decoder',
          'random': 'random unit'}


def cond_mean_kl(kl, act, p):
    """Per-direction mean KL over the top-p% tokens by activation |h.d|. -> [N]"""
    k = max(1, round(kl.shape[1] * p / 100))
    idx = act.topk(k, dim=1).indices
    return kl.gather(1, idx).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True, help='dir with kl_per_token.pt, act_per_token.pt, summary.pkl')
    ap.add_argument('--pcts', default='5,10,25')
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='qwen3-1.7b L14: ablation KL conditioned on activation')
    args = ap.parse_args()

    pcts = [int(x) for x in args.pcts.split(',')]
    s = pickle.load(open(os.path.join(args.run_dir, 'summary.pkl'), 'rb'))
    fam = np.array(s['family'])
    kl = torch.load(os.path.join(args.run_dir, 'kl_per_token.pt'))
    act = torch.load(os.path.join(args.run_dir, 'act_per_token.pt'))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(pcts), figsize=(5.5 * len(pcts), 4.6), sharey=True)
    if len(pcts) == 1:
        axes = [axes]

    for ax, p in zip(axes, pcts):
        vals = cond_mean_kl(kl, act, p).numpy()
        present = [f for f in ORDER if f in set(fam)]
        allv = vals[np.isin(fam, present)]
        lo = max(1e-12, float(allv[allv > 0].min())); hi = float(allv.max())
        bins = np.logspace(np.log10(lo), np.log10(hi), 45)
        for f in present:
            v = vals[fam == f]
            ax.hist(v.clip(lo, hi), bins=bins, density=True, alpha=0.55, color=COLORS.get(f),
                    label=f'{LABELS.get(f, f)} (med={np.median(v):.2e})')
        ax.set_xscale('log')
        ax.set_xlabel('mean KL over top-%d%% active tokens [nats/tok]' % p)
        ax.set_title(f'top {p}% by |h·d|')
        ax.legend(fontsize=7)
    axes[0].set_ylabel('density')
    fig.suptitle(args.title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f'wrote {args.out}')
    for p in pcts:
        vals = cond_mean_kl(kl, act, p).numpy()
        print(f'--- top {p}% by activation ---')
        for f in ORDER:
            if f in set(fam):
                v = vals[fam == f]
                print(f'  {f:14s} n={len(v):4d} median={np.median(v):.4e} mean={v.mean():.4e}')


if __name__ == '__main__':
    main()
