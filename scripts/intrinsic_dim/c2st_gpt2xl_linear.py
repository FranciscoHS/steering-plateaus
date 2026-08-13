#!/usr/bin/env python3
"""C2ST on gpt2-xl L2 activations — linear (logistic) probe only.

Replicates the qwen3-1.7b C2ST comparisons on the gpt2-xl tensors from
gpt2xl_shift_replication.py, to check whether the linear-separability
pattern (real ~ gauss ~ gauss-start inseparable; jac-opt separable from
all) transfers. Logistic regression only, per Francisco's request.

Same protocol as c2st_conditions.py: n=200/class, 5-fold stratified CV,
out-of-fold AUC, 10 independent draws. Gaussian pool draws use rows
400:1000 (rows 0:200 = jac-opt start, 200:400 = the shifted-anchor fresh
set).

Output: results/intrinsic_dim/gpt2xl_shift/c2st_linear.pkl
"""

import os
import pickle
import sys

sys.path.insert(0, "/workspace/steering-plateaus")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA_DIR = 'results/intrinsic_dim/gpt2xl_shift'
SEED = 0
N_PER_CLASS = 200
N_FOLDS = 5
R_DRAWS = 10


def cv_auc_logistic(X0, X1, seed):
    X = np.concatenate([X0, X1])
    y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))])
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000, random_state=seed))
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def rescale_to(X, target_mean_norm):
    return X * (target_mean_norm / X.norm(dim=-1).mean())


def main():
    real = torch.load(os.path.join(DATA_DIR, 'real_data.pt'),
                      map_location='cpu', weights_only=False)
    gauss = torch.load(os.path.join(DATA_DIR, 'gauss_data.pt'),
                       map_location='cpu', weights_only=False)
    optb = torch.load(os.path.join(DATA_DIR, 'jac_optimized.pt'),
                      map_location='cpu', weights_only=False)
    X_real = real['X_real'].float()
    X_gpool = gauss['X_gauss'].float()[2 * N_PER_CLASS:]   # rows 400:1000
    X_opt = optb['X_opt'].float()
    X_gstart = optb['X_gauss_subset'].float()
    mean_norm_real = X_real.norm(dim=-1).mean()
    X_opt_resc = rescale_to(X_opt, mean_norm_real)

    # (name, side A, side B); 'real' side A means a fresh real draw each rep
    conditions = [
        ('real vs real (floor)', X_real, None),
        ('real vs gauss pool', X_real, X_gpool),
        ('real vs gauss start', X_real, X_gstart),
        ('real vs jac-opt (resc)', X_real, X_opt_resc),
        ('real vs jac-opt (raw)', X_real, X_opt),
        ('jac-opt (resc) vs gauss pool', X_opt_resc, X_gpool),
        ('jac-opt (resc) vs gauss start', X_opt_resc, X_gstart),
        ('jac-opt (raw) vs gauss start', X_opt, X_gstart),
    ]

    results = {'seed': SEED, 'n_per_class': N_PER_CLASS,
               'n_folds': N_FOLDS, 'r_draws': R_DRAWS, 'auc': {}}
    print(f'gpt2-xl linear C2ST: n={N_PER_CLASS}/class, {N_FOLDS}-fold CV, '
          f'{R_DRAWS} draws  d={X_real.shape[1]}', flush=True)
    for name, A, B in conditions:
        aucs = []
        for r in range(R_DRAWS):
            rs = SEED + 1000 * (r + 1)
            gr = torch.Generator().manual_seed(rs)
            if A is X_real:
                ia = torch.randperm(X_real.shape[0], generator=gr)
                Xa = X_real[ia[:N_PER_CLASS]].numpy()
                if B is None:                       # floor: disjoint real draw
                    Xb = X_real[ia[N_PER_CLASS:2 * N_PER_CLASS]].numpy()
                elif B.shape[0] > N_PER_CLASS:
                    ib = torch.randperm(B.shape[0],
                                        generator=gr)[:N_PER_CLASS]
                    Xb = B[ib].numpy()
                else:
                    Xb = B.numpy()
            else:                                   # jac-opt side is fixed
                Xa = A.numpy()
                if B.shape[0] > N_PER_CLASS:
                    ib = torch.randperm(B.shape[0],
                                        generator=gr)[:N_PER_CLASS]
                    Xb = B[ib].numpy()
                else:
                    Xb = B.numpy()
            aucs.append(cv_auc_logistic(Xa, Xb, seed=rs))
        results['auc'][name] = {'mean': float(np.mean(aucs)),
                                'sd': float(np.std(aucs)), 'draws': aucs}
        print(f'  {name:30s} auc={np.mean(aucs):.3f}±{np.std(aucs):.3f}',
              flush=True)

    out = os.path.join(DATA_DIR, 'c2st_linear.pkl')
    with open(out, 'wb') as f:
        pickle.dump(results, f)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
