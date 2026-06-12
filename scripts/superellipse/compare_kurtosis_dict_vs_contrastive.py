#!/usr/bin/env python3
"""Compare featureness (excess kurtosis of centered-cosine activation
distribution) between dictionary atoms and the labeled contrastive directions.

Reads ranking.pkl files produced by rank_directions_featureness.py (same
corpus/settings so values are comparable) and writes a compact comparison:
dict kurtosis quantiles + top atoms vs every contrastive direction's kurtosis
and its would-be percentile rank inside the dict distribution.

    python scripts/superellipse/compare_kurtosis_dict_vs_contrastive.py \\
        --dict_ranking .../featureness/ranking.pkl \\
        --contrastive_rankings .../featureness_contrastive_scaffold/ranking.pkl \\
                               .../featureness_contrastive_langcode/ranking.pkl \\
        --out .../featureness/compare_contrastive.md
"""
import argparse
import pickle

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--dict_ranking', required=True)
ap.add_argument('--contrastive_rankings', nargs='+', required=True)
ap.add_argument('--contrastive_dicts', nargs='+', required=True,
                help='the dictionary.pt files with "names", same order')
ap.add_argument('--out', required=True)
args = ap.parse_args()

dk = pickle.load(open(args.dict_ranking, 'rb'))
kurt = np.asarray(dk['kurtosis'])
order = np.argsort(-kurt)

lines = ['# Kurtosis: dictionary atoms vs contrastive directions', '',
         f'- dict: n={len(kurt)} atoms',
         f'- dict kurtosis quantiles: '
         f'p50={np.median(kurt):.2f}  p90={np.quantile(kurt, .9):.2f}  '
         f'p99={np.quantile(kurt, .99):.2f}  '
         f'p99.9={np.quantile(kurt, .999):.2f}  max={kurt.max():.2f}', '',
         '## Top 20 dict atoms by kurtosis']
for r, a in enumerate(order[:20]):
    lines.append(f'- #{r + 1}: atom {int(a)}  kurt={kurt[a]:.2f}')
lines += ['', '## Contrastive directions',
          '(pct = percentile within the dict kurtosis distribution; '
          'rank = how many dict atoms beat it)']

rows = []
for rk_path, dict_path in zip(args.contrastive_rankings,
                              args.contrastive_dicts):
    rk = pickle.load(open(rk_path, 'rb'))
    ck = np.asarray(rk['kurtosis'])
    names = torch.load(dict_path, map_location='cpu')['names']
    for n, v in zip(names, ck):
        rows.append((n, float(v)))
rows.sort(key=lambda t: -t[1])
for n, v in rows:
    pct = float((kurt < v).mean()) * 100
    beat = int((kurt > v).sum())
    lines.append(f'- {n:14s} kurt={v:8.2f}   pct={pct:5.1f}   '
                 f'dict atoms above: {beat}')

with open(args.out, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'\nSaved {args.out}')
