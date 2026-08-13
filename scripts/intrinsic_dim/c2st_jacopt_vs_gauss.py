#!/usr/bin/env python3
"""C2ST: jac-optimized points vs the cov-matched Gaussian they started from.

Companion to c2st_conditions.py (which tested everything against real).
Here the question is whether the jac-opt set is separable from its own source
distribution:

  - jac-opt (rescaled) vs gauss pool     (fresh cov-matched Gaussian draws)
  - jac-opt (rescaled) vs gauss start    (the exact 200 starting points)
  - jac-opt (raw)      vs gauss start    (reference, norm gap included)

Same protocol: n=200/class, 5-fold CV, 10 draws (the jac-opt side is fixed
at its 200 points; the Gaussian pool side is redrawn per draw).

Output: results/intrinsic_dim/c2st/jacopt_vs_gauss.pkl (+ printed summary)
"""

import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2st_conditions import cv_auc, rescale_to, DATA_DIR, OUT_DIR, \
    SEED, N_PER_CLASS, R_DRAWS


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    filt = torch.load(os.path.join(DATA_DIR, 'jac_filtered_data.pt'),
                      map_location='cpu', weights_only=False)
    opt = torch.load(os.path.join(DATA_DIR, 'jac_optimized_data.pt'),
                     map_location='cpu', weights_only=False)
    X_real = filt['X_real'].float()
    X_gauss = filt['X_gauss'].float()
    X_opt = opt['X_opt'].float()
    X_gstart = opt['X_gauss_subset'].float()
    mean_norm_real = X_real.norm(dim=-1).mean()
    X_opt_resc = rescale_to(X_opt, mean_norm_real)

    conditions = {
        'jac-opt (resc) vs gauss pool': (X_opt_resc, X_gauss),
        'jac-opt (resc) vs gauss start': (X_opt_resc, X_gstart),
        'jac-opt (raw) vs gauss start': (X_opt, X_gstart),
    }

    results = {'seed': SEED, 'n_per_class': N_PER_CLASS,
               'r_draws': R_DRAWS, 'auc': {}}
    print(f'C2ST jac-opt vs Gaussian: n={N_PER_CLASS}/class, '
          f'{R_DRAWS} draws', flush=True)
    for cname, (A, B) in conditions.items():
        per_clf = {k: [] for k in ('logistic', 'knn', 'mlp')}
        for r in range(R_DRAWS):
            rs = SEED + 1000 * (r + 1)
            gr = torch.Generator().manual_seed(rs)
            if B.shape[0] > N_PER_CLASS:
                ib = torch.randperm(B.shape[0], generator=gr)[:N_PER_CLASS]
                Xb = B[ib].numpy()
            else:
                Xb = B.numpy()
            aucs, _ = cv_auc(A.numpy(), Xb, seed=rs)
            for k in per_clf:
                per_clf[k].append(aucs[k])
        results['auc'][cname] = {
            k: {'mean': float(np.mean(v)), 'sd': float(np.std(v)),
                'draws': v} for k, v in per_clf.items()}
        line = '  '.join(f'{k}={np.mean(v):.3f}±{np.std(v):.3f}'
                         for k, v in per_clf.items())
        print(f'  {cname:30s} {line}', flush=True)

    out_pkl = os.path.join(OUT_DIR, 'jacopt_vs_gauss.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f'\nSaved {out_pkl}')


if __name__ == '__main__':
    main()
