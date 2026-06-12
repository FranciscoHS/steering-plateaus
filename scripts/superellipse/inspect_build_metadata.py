#!/usr/bin/env python3
"""Quick stats on the per-atom records stored in a build's metadata.pkl
(final_loss, final_p_hat, ...): distributions, extremes, and the
correlation between build loss and the build's own p_hat estimate.

    python scripts/superellipse/inspect_build_metadata.py \\
        --meta results/dictionary/.../metadata.pkl
"""
import argparse
import pickle

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--meta', required=True)
ap.add_argument('--topn', type=int, default=15)
args = ap.parse_args()

with open(args.meta, 'rb') as f:
    meta = pickle.load(f)
atoms = meta['atoms']
print(f'{len(atoms)} atom records; keys: {sorted(atoms[0].keys())}')

for key in ['final_loss', 'final_resp', 'final_p_hat', 'final_align',
            'final_scaffold_overlap', 'final_prev_overlap']:
    if key not in atoms[0]:
        continue
    v = np.array([a[key] for a in atoms], dtype=np.float64)
    q = np.percentile(v, [1, 25, 50, 75, 99])
    print(f'{key}: mean={v.mean():.4f} std={v.std():.4f}  '
          f'p1={q[0]:.4f} q25={q[1]:.4f} med={q[2]:.4f} '
          f'q75={q[3]:.4f} p99={q[4]:.4f}  '
          f'min={v.min():.4f} max={v.max():.4f}')

loss = np.array([a['final_loss'] for a in atoms])
p_hat = np.array([a['final_p_hat'] for a in atoms])
ks = np.array([a['k'] for a in atoms])

def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])

def spearman(a, b):
    return pearson(np.argsort(np.argsort(a)).astype(float),
                   np.argsort(np.argsort(b)).astype(float))

print(f'\ncorr(final_loss, final_p_hat): pearson={pearson(loss, p_hat):.3f} '
      f'spearman={spearman(loss, p_hat):.3f}')
print(f'corr(k, final_p_hat):          pearson={pearson(ks.astype(float), p_hat):.3f} '
      f'spearman={spearman(ks.astype(float), p_hat):.3f}')
print(f'corr(k, final_loss):           pearson={pearson(ks.astype(float), loss):.3f} '
      f'spearman={spearman(ks.astype(float), loss):.3f}')

if not np.isnan(p_hat).all():
    order = np.argsort(-p_hat)
    print(f'\ntop {args.topn} atoms by final_p_hat:')
    for i in order[:args.topn]:
        print(f'  atom {atoms[i]["k"]:5d}  p_hat={p_hat[i]:.3f}  '
              f'loss={loss[i]:.4f}')

order = np.argsort(loss)
print(f'\nbest (lowest final_loss) {args.topn} atoms:')
for i in order[:args.topn]:
    print(f'  atom {atoms[i]["k"]:5d}  loss={loss[i]:.4f}  '
          f'resp={atoms[i]["final_resp"]:.4f}')
print(f'\nworst (highest final_loss) {args.topn} atoms:')
for i in order[::-1][:args.topn]:
    print(f'  atom {atoms[i]["k"]:5d}  loss={loss[i]:.4f}  '
          f'resp={atoms[i]["final_resp"]:.4f}')
