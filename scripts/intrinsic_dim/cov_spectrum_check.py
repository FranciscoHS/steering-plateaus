#!/usr/bin/env python3
"""Compare the real-activation covariance spectra used for the Gaussian
conditions: qwen3-1.7b L2 (jac_filtered_data.pt) vs gpt2-xl L2
(gpt2xl_shift/real_data.pt).

Reports, per model: d, n, participation ratio (sum(l))^2/sum(l^2),
effective ranks at 90/99/99.9% variance, top-1 variance share, and the
eigenvalue at a few depth quantiles relative to the largest.

Output: results/intrinsic_dim/cov_spectrum_check.pkl (+ printed summary)
"""

import os
import pickle

import numpy as np
import torch

OUT = 'results/intrinsic_dim/cov_spectrum_check.pkl'

SOURCES = {
    'qwen3-1.7b': ('results/intrinsic_dim/jac_filtered_data.pt', 'X_real'),
    'gpt2-xl': ('results/intrinsic_dim/gpt2xl_shift/real_data.pt', 'X_real'),
}

results = {}
for name, (path, key) in SOURCES.items():
    X = torch.load(path, map_location='cpu',
                   weights_only=False)[key].float().numpy()
    n, d = X.shape
    Xc = X - X.mean(0)
    # eigenvalues of the sample covariance via SVD of the centered data
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2 / (n - 1)
    tot = lam.sum()
    frac = np.cumsum(lam) / tot
    pr = tot ** 2 / (lam ** 2).sum()
    res = {
        'n': n, 'd': d,
        'participation_ratio': float(pr),
        'rank_90': int(np.searchsorted(frac, 0.90) + 1),
        'rank_99': int(np.searchsorted(frac, 0.99) + 1),
        'rank_999': int(np.searchsorted(frac, 0.999) + 1),
        'top1_share': float(lam[0] / tot),
        'lam_rel': {q: float(lam[int(q * (len(lam) - 1))] / lam[0])
                    for q in (0.1, 0.25, 0.5, 0.9)},
        'eigenvalues': lam,
    }
    results[name] = res
    print(f'{name}: n={n} d={d}  PR={pr:.1f}  '
          f'rank90={res["rank_90"]}  rank99={res["rank_99"]}  '
          f'rank99.9={res["rank_999"]}  top1={res["top1_share"]:.3f}')
    print(f'  lambda/lambda_max at depth 10/25/50/90%: '
          + '  '.join(f'{res["lam_rel"][q]:.2e}'
                      for q in (0.1, 0.25, 0.5, 0.9)), flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'wb') as f:
    pickle.dump(results, f)
print(f'Saved {OUT}')
