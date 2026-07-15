#!/usr/bin/env python3
"""Disentangle single-leg rotation rate from baseline p.

The raw rate dp/dtheta may scale mechanically with how far p0 sits above the
p=2 floor (if curves decay proportionally toward 2). We test this by:
  (1) correlating per-pair p0 with the per-(pair,target) |raw slope|,
  (2) normalizing each slope by (p0 - 2) -> fractional rate of losing the
      above-2 excess per degree, and comparing the normalized distributions.

Reads the p_single_leg_rates.pkl files (need 'p_curves' [pairs,targets,nrot],
'rot_degs', 'p0' [pairs]).

    python scripts/superellipse/analyze_single_leg_normalized.py \\
        --groups "label=A.pkl" "label2=B.pkl,C.pkl" --out hist.png
"""
import argparse
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--groups', nargs='+', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()

COLORS = ['#3a7d3a', '#c1440e', '#1f77b4', '#7d3a7d']

groups = []
for spec in args.groups:
    label, paths = spec.split('=', 1)
    raw, norm, p0col = [], [], []
    for p in paths.split(','):
        b = pickle.load(open(p, 'rb'))
        rot = np.asarray(b['rot_degs'])
        pc = np.asarray(b['p_curves'])         # [pairs, targets, nrot]
        p0 = np.asarray(b['p0'])               # [pairs]
        for pi in range(pc.shape[0]):
            if not np.isfinite(p0[pi]):
                continue
            for di in range(pc.shape[1]):
                ps = pc[pi, di]
                if not np.isfinite(ps).all():
                    continue
                s = np.polyfit(rot, ps, 1)[0]  # raw dp/dtheta
                raw.append(s)
                p0col.append(p0[pi])
                denom = p0[pi] - 2.0
                norm.append(s / denom if abs(denom) > 1e-6 else np.nan)
    raw = np.array(raw); norm = np.array(norm); p0col = np.array(p0col)
    m = np.isfinite(norm)
    # correlation of |raw slope| with p0
    r = np.corrcoef(p0col, np.abs(raw))[0, 1]
    groups.append({'label': label, 'raw': raw, 'norm': norm[m]})
    print(f"{label}: n={raw.size}  |raw| median={np.median(np.abs(raw)):.4f}  "
          f"|norm| median={np.median(np.abs(norm[m])):.4f}  "
          f"corr(p0,|raw|)={r:+.2f}  "
          f"raw IQR=({np.quantile(raw,.25):+.4f},{np.quantile(raw,.75):+.4f})  "
          f"norm IQR=({np.quantile(norm[m],.25):+.4f},{np.quantile(norm[m],.75):+.4f})")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
lo = min(np.quantile(g['raw'], .01) for g in groups)
hi = max(np.quantile(g['raw'], .99) for g in groups)
b1 = np.linspace(lo, hi, 50)
for g, c in zip(groups, COLORS):
    ax[0].hist(g['raw'], bins=b1, density=True, alpha=0.5, color=c,
               label=f"{g['label']} (med|·|={np.median(np.abs(g['raw'])):.4f})")
ax[0].axvline(0, color='k', ls='--', alpha=0.5); ax[0].set_title('raw dp/dθ (per deg)')
ax[0].set_xlabel('dp/dθ'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

lo = min(np.quantile(g['norm'], .01) for g in groups)
hi = max(np.quantile(g['norm'], .99) for g in groups)
b2 = np.linspace(lo, hi, 50)
for g, c in zip(groups, COLORS):
    ax[1].hist(g['norm'], bins=b2, density=True, alpha=0.5, color=c,
               label=f"{g['label']} (med|·|={np.median(np.abs(g['norm'])):.4f})")
ax[1].axvline(0, color='k', ls='--', alpha=0.5)
ax[1].set_title('normalized: (dp/dθ)/(p0−2)  [fractional rate]')
ax[1].set_xlabel('(dp/dθ)/(p0−2) per deg'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(args.out, dpi=160, bbox_inches='tight', facecolor='white')
print(f'Saved {args.out}')
