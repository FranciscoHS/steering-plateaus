#!/usr/bin/env python3
"""Combined beeswarm for the Qwen3.6-27B run: Contrastive vs J-Lens families.
One dot per pair (Contrastive: median_p over robustness cells; J-Lens: median
p over L2 thresholds). Reference line at p=2. Pure post-processing.
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.exp_map.plotting.plot_robustness_beeswarm import median_p_for_pair

FITS = 'results/exp_map/data/fits/fits_qwen36_L36.pkl'
JDIR = 'results/jlens/qwen3.6-27b'
OUT = os.path.join(JDIR, 'qwen36_p_beeswarm.png')


def jlens_medians(summary_pkl, category='ZH'):
    """median p over L2 thresholds per pair, for one J-Lens summary pkl."""
    with open(summary_pkl, 'rb') as f:
        rows = pickle.load(f)
    by = {}
    for r in rows:
        if r['category'] == category:
            by.setdefault(r['pair_idx'], []).append(r['p'])
    return [float(np.median(v)) for v in by.values()]


def main():
    with open(FITS, 'rb') as f:
        full = pickle.load(f)
    con = [median_p_for_pair(fit) for fit in full.values()]
    con = [p for p in con if np.isfinite(p)]

    zh100 = jlens_medians(os.path.join(JDIR, 'jlens_p_summary_L36_zh100.pkl'), 'ZH')
    en = jlens_medians(os.path.join(JDIR, 'jlens_p_summary_L36_en_concrete.pkl'), 'EN')

    cols = [('Contrastive', con, '#e41a1c'),
            ('J-Lens ZH', zh100, '#377eb8'),
            ('J-Lens EN\n(concrete)', en, '#984ea3')]
    fig, ax = plt.subplots(figsize=(7, 5))
    rng = np.random.default_rng(0)
    for i, (label, ps, c) in enumerate(cols):
        ps = np.asarray(ps)
        x = i + (rng.random(len(ps)) - 0.5) * 0.25
        ax.scatter(x, ps, s=34, color=c, alpha=0.75, edgecolor='k',
                   linewidth=0.4, zorder=3)
        med = np.median(ps)
        # bootstrap 99% CI on the median (uncertainty on the central value,
        # not the spread of the dots)
        boot = np.median(rng.choice(ps, size=(200_000, len(ps))), axis=1)
        lo, hi = np.percentile(boot, [0.5, 99.5])
        # CI drawn as a whiskered bar offset to the right of the swarm
        xc = i + 0.34
        ax.plot([xc, xc], [lo, hi], color='k', lw=1.6, zorder=4)
        for y in (lo, hi):
            ax.plot([xc - 0.05, xc + 0.05], [y, y], color='k', lw=1.6, zorder=4)
        ax.plot(xc, med, 'o', color='k', ms=5, zorder=5)
        ax.plot([i - 0.28, i + 0.28], [med, med], color='k', lw=2, zorder=4)
        ax.annotate(f"med {med:.2f}\n[{lo:.2f}, {hi:.2f}]\nn={len(ps)}",
                    (xc, med), textcoords='offset points', xytext=(12, 0),
                    fontsize=8.5, va='center')
    ax.axhline(2.0, color='gray', ls='--', lw=1, zorder=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols])
    ax.set_ylabel('superellipse exponent p')
    ax.set_title('Qwen3.6-27B  L36→L62\n'
                 'bars: bootstrap 99% CI on the median', fontsize=11)
    ax.set_xlim(-0.5, len(cols) - 1 + 0.95)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"Saved: {OUT}")


if __name__ == '__main__':
    main()
