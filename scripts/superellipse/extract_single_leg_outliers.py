#!/usr/bin/env python3
"""List the most-negative normalized single-leg rates (fast p-loss outliers)
and the atoms involved, for dashboarding.

    python scripts/superellipse/extract_single_leg_outliers.py \\
        --pkl .../p_single_leg_rates.pkl --topn 12
"""
import argparse
import pickle

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--pkl', required=True)
ap.add_argument('--topn', type=int, default=12)
ap.add_argument('--min_p0', type=float, default=0.0,
                help='only consider pairs with p0 above this (avoids the '
                     '(p0-2)->0 normalization blow-up)')
args = ap.parse_args()

b = pickle.load(open(args.pkl, 'rb'))
rot = np.asarray(b['rot_degs'])
pc = np.asarray(b['p_curves'])      # [pairs, targets, nrot]
p0 = np.asarray(b['p0'])
pair_idx = np.asarray(b['pair_idx'])  # [pairs, 2] -> (a rotated, b fixed)

rows = []
for pi in range(pc.shape[0]):
    if not np.isfinite(p0[pi]) or p0[pi] < args.min_p0:
        continue
    denom = p0[pi] - 2.0
    for di in range(pc.shape[1]):
        ps = pc[pi, di]
        if not np.isfinite(ps).all():
            continue
        s = np.polyfit(rot, ps, 1)[0]
        ns = s / denom if abs(denom) > 1e-6 else np.nan
        rows.append((ns, s, pi, di, int(pair_idx[pi, 0]), int(pair_idx[pi, 1]),
                     float(p0[pi])))
rows = [r for r in rows if np.isfinite(r[0])]
rows.sort(key=lambda r: r[0])   # most negative first

print(f"{'norm_slope':>11} {'raw_slope':>10} {'pair':>5} {'tgt':>4} "
      f"{'a(rot)':>7} {'b(fix)':>7} {'p0':>6}")
for ns, s, pi, di, a, bb, pp in rows[:args.topn]:
    print(f"{ns:11.4f} {s:10.4f} {pi:5d} {di:4d} {a:7d} {bb:7d} {pp:6.2f}")

# distinct rotated-leg atoms among the outliers (for dashboarding)
a_atoms = []
for r in rows[:args.topn]:
    if r[4] not in a_atoms:
        a_atoms.append(r[4])
b_atoms = []
for r in rows[:args.topn]:
    if r[5] not in b_atoms:
        b_atoms.append(r[5])
print('\nrotated-leg atoms (a):', ','.join(map(str, a_atoms)))
print('fixed-leg atoms  (b):', ','.join(map(str, b_atoms)))
# how concentrated? count pairs in the bottom topn
from collections import Counter
c = Counter(r[2] for r in rows[:args.topn])
print('pair-index counts in tail:', dict(c))
