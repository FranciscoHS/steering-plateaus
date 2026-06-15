#!/usr/bin/env python3
"""Plot p vs build-order bucket using the per-pair median (and IQR band)
instead of the pooled fit.

    python scripts/superellipse/plot_p_vs_buildorder_median.py \\
        --pkl .../p_vs_buildorder/p_vs_buildorder.pkl \\
        --out .../p_vs_buildorder/p_vs_buildorder_median.png
"""
import argparse
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--pkl', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()

with open(args.pkl, 'rb') as f:
    blob = pickle.load(f)
print('pkl keys:', sorted(blob.keys()))
buckets = blob['buckets']
print('bucket keys:', sorted(buckets[0].keys()))

centers, meds, q1s, q3s = [], [], [], []
for b in buckets:
    ps = np.asarray([p for p in b['p_fits'] if np.isfinite(p)])
    lo, hi = b['lo'], b['hi']
    centers.append(0.5 * (lo + hi))
    meds.append(np.median(ps))
    q1s.append(np.percentile(ps, 25))
    q3s.append(np.percentile(ps, 75))
    print(f'bucket [{lo},{hi}): n={len(ps)} med={meds[-1]:.3f} '
          f'IQR=({q1s[-1]:.3f},{q3s[-1]:.3f})')

centers, meds = np.array(centers), np.array(meds)
q1s, q3s = np.array(q1s), np.array(q3s)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar(centers, meds, yerr=[meds - q1s, q3s - meds],
            fmt='o-', capsize=4, color='tab:blue', label='per-pair median (IQR)')
ax.set_xlabel('build-order index (bucket center)')
ax.set_ylabel('superellipse exponent p')
ax.set_title('Per-pair median p vs build order')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f'Saved {args.out}')
