#!/usr/bin/env python3
"""Classifier two-sample test (C2ST): real activations vs synthetic sets.

Follow-up to energy_distance_conditions.py, which turned out to be blind to
anything beyond the first two moments (cov-matched Gaussian scored at the
floor). Here a classifier is trained to tell real from synthetic; held-out
AUC ~ 0.5 means indistinguishable, ~1.0 means trivially separable.

Conditions (same saved tensors as before, qwen3-1.7b L2):
  - real vs real            (disjoint draws; protocol floor, expect AUC 0.5)
  - real vs gauss           (mean+cov matched Gaussian)
  - real vs isotropic       (random dirs x real-norm empirical dist)
  - real vs gauss start     (the 200 Gaussian points jac-opt started from)
  - real vs jac-opt (resc.) (the 200 optimized points, rescaled to real
                             mean norm so the norm gap alone can't separate)
  - real vs jac-opt (raw)   (reference: includes the norm mismatch)

Classifiers (each probes a different kind of difference):
  - logistic  L2 logistic regression: linear / mean-direction differences
  - knn       k=5 nearest neighbours: local density / manifold structure
  - mlp       1x64 MLP, weight decay: flexible nonlinear differences

Protocol: n=200 per class, 5-fold stratified CV, repeated over R=10
independent real draws. Score = out-of-fold AUC, reported mean +/- sd over
draws. Per-point "P(real)" out-of-fold scores are saved for the jac-opt set.

Output: results/intrinsic_dim/c2st/{results.pkl, c2st.png}
"""

import os
import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA_DIR = 'results/intrinsic_dim'
OUT_DIR = os.path.join(DATA_DIR, 'c2st')
SEED = 0
N_PER_CLASS = 200
N_FOLDS = 5
R_DRAWS = 10


def make_classifiers(seed):
    return {
        'logistic': make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2000, random_state=seed)),
        'knn': make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5)),
        'mlp': make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2,
                          max_iter=500, random_state=seed)),
    }


def cv_auc(X0, X1, seed):
    """5-fold out-of-fold AUC per classifier. Returns {clf: auc},
    {clf: oof_scores_for_X1}."""
    X = np.concatenate([X0, X1])
    y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))])
    aucs, oof1 = {}, {}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for name, clf in make_classifiers(seed).items():
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        aucs[name] = roc_auc_score(y, oof)
        oof1[name] = oof[len(X0):]          # scores of the synthetic class
    return aucs, oof1


def rescale_to(X, target_mean_norm):
    return X * (target_mean_norm / X.norm(dim=-1).mean())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    g = torch.Generator().manual_seed(SEED)

    filt = torch.load(os.path.join(DATA_DIR, 'jac_filtered_data.pt'),
                      map_location='cpu', weights_only=False)
    opt = torch.load(os.path.join(DATA_DIR, 'jac_optimized_data.pt'),
                     map_location='cpu', weights_only=False)
    X_real = filt['X_real'].float()
    X_gauss = filt['X_gauss'].float()
    X_opt = opt['X_opt'].float()
    X_gstart = opt['X_gauss_subset'].float()
    d_model = X_real.shape[1]
    mean_norm_real = X_real.norm(dim=-1).mean()

    real_norms = X_real.norm(dim=-1)
    iso = torch.randn(5000, d_model, generator=g)
    iso = iso / iso.norm(dim=-1, keepdim=True)
    idx = torch.randint(0, len(real_norms), (5000,), generator=g)
    X_iso = iso * real_norms[idx].unsqueeze(-1)

    conditions = {
        'real (disjoint draw)': X_real,     # handled specially below
        'gauss (cov-matched)': X_gauss,
        'isotropic (norm-matched)': X_iso,
        'gauss start (n=200)': X_gstart,
        'jac-opt (rescaled)': rescale_to(X_opt, mean_norm_real),
        'jac-opt (raw)': X_opt,
    }

    results = {'seed': SEED, 'n_per_class': N_PER_CLASS, 'n_folds': N_FOLDS,
               'r_draws': R_DRAWS, 'auc': {}, 'oof_jacopt': {}}

    print(f'C2ST: n={N_PER_CLASS}/class, {N_FOLDS}-fold CV, '
          f'{R_DRAWS} draws  d={d_model}', flush=True)
    for cname, Y in conditions.items():
        per_clf = {k: [] for k in ('logistic', 'knn', 'mlp')}
        oof_acc = {k: [] for k in ('logistic', 'knn', 'mlp')}
        for r in range(R_DRAWS):
            rs = SEED + 1000 * (r + 1)
            gr = torch.Generator().manual_seed(rs)
            ir = torch.randperm(X_real.shape[0], generator=gr)
            X0 = X_real[ir[:N_PER_CLASS]].numpy()
            if cname.startswith('real'):
                X1 = X_real[ir[N_PER_CLASS:2 * N_PER_CLASS]].numpy()
            elif Y.shape[0] > N_PER_CLASS:
                iy = torch.randperm(Y.shape[0], generator=gr)[:N_PER_CLASS]
                X1 = Y[iy].numpy()
            else:
                X1 = Y.numpy()
            aucs, oof1 = cv_auc(X0, X1, seed=rs)
            for k in per_clf:
                per_clf[k].append(aucs[k])
                if cname.startswith('jac-opt'):
                    oof_acc[k].append(oof1[k])
        results['auc'][cname] = {
            k: {'mean': float(np.mean(v)), 'sd': float(np.std(v)),
                'draws': v} for k, v in per_clf.items()}
        if cname.startswith('jac-opt'):
            # per-point mean "P(synthetic)" over draws -> save 1 - that
            results['oof_jacopt'][cname] = {
                k: 1.0 - np.stack(v).mean(0) for k, v in oof_acc.items()}
        line = '  '.join(f'{k}={np.mean(v):.3f}±{np.std(v):.3f}'
                         for k, v in per_clf.items())
        print(f'  {cname:26s} {line}', flush=True)

    with open(os.path.join(OUT_DIR, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    names = list(conditions.keys())
    clfs = ['logistic', 'knn', 'mlp']
    colors = {'logistic': '#4477aa', 'knn': '#ee7733', 'mlp': '#228833'}
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, k in enumerate(clfs):
        m = [results['auc'][n][k]['mean'] for n in names]
        s = [results['auc'][n][k]['sd'] for n in names]
        ax.bar(x + (j - 1) * w, m, w, yerr=s, capsize=3,
               color=colors[k], label=k)
    ax.axhline(0.5, color='gray', ls='--', lw=1, label='chance (AUC 0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('out-of-fold AUC (real vs condition)')
    ax.set_ylim(0.3, 1.02)
    ax.set_title(f'C2ST vs real activations — qwen3-1.7b L2\n'
                 f'(n={N_PER_CLASS}/class, {N_FOLDS}-fold CV, '
                 f'mean±sd over {R_DRAWS} draws)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'c2st.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved {os.path.join(OUT_DIR, "results.pkl")}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
