#!/usr/bin/env python3
"""Recompute the raw-activation probe ceiling with a properly-calibrated probe.

The native-probe suite used unstandardized RidgeClassifier for raw, which is
miscalibrated on residual-stream scales. Here we compare probes on the SAME raw
activations so the ceiling is fair.
"""
import argparse
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score


def probe(kind, Xtr, ytr, Xte, yte):
    if kind == 'ridge_raw':
        clf = RidgeClassifier(alpha=1.0); clf.fit(Xtr, ytr)
    elif kind == 'logreg_raw':
        clf = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000); clf.fit(Xtr, ytr)
    elif kind == 'logreg_std':
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
        clf.fit(sc.transform(Xtr), ytr)
        return accuracy_score(yte, clf.predict(sc.transform(Xte)))
    return accuracy_score(yte, clf.predict(Xte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acts', default='results/probe_suite_highp/suite_acts.pt')
    ap.add_argument('--out', default='results/probe_native/raw_ceiling.pkl')
    args = ap.parse_args()

    acts = torch.load(args.acts, weights_only=False)
    kinds = ['ridge_raw', 'logreg_raw', 'logreg_std']
    per = {k: [] for k in kinds}
    for name, a in acts.items():
        Xtr, ytr = a['train'].numpy(), a['train_y']
        Xte, yte = a['test'].numpy(), a['test_y']
        for k in kinds:
            per[k].append(probe(k, Xtr, ytr, Xte, yte))
        print(name, {k: round(per[k][-1], 3) for k in kinds}, flush=True)
    print('\n=== MEAN over %d concepts ===' % len(acts))
    means = {k: float(np.mean(per[k])) for k in kinds}
    for k in kinds:
        print(f'  {k:12s} {means[k]:.3f}')
    with open(args.out, 'wb') as f:
        pickle.dump({'per_concept': per, 'means': means}, f)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
