#!/usr/bin/env python3
"""
Native top-k SAE sparse probing: probe on the SAE's own sparse CODES.

For each SAE, encode the cached concept activations with the real encoder
(relu(W_enc (x - b_dec))), keep the top-k active latents per token, and probe a
linear classifier on that d_sae-dim SPARSE code vector. This is the canonical
SAE feature-quality test — it asks "are the SAE's active features linearly
predictive of the concept" — and avoids the reconstruction/variance confound of
the OMP/top-k-projection suite (where random directions win by capturing
variance).

Reuses the activations cached by run_sparse_probe_suite.py (suite_acts.pt).

Usage:
    python -u scripts/probes/eval_probe_native.py \
        --acts results/probe_suite_highp/suite_acts.pt \
        --saes baseline:results/trained_sae/l10_doubled_k32_2M/sae_weights.pt,popt:results/trained_sae_popt/l10_lam1/sae_weights.pt,cullmore:results/trained_sae_pcull/l10_cull_more/sae_weights.pt \
        --top_k_values 1,2,5,10,20,32,50 \
        --out results/probe_native
"""
import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import scipy.sparse as sp
import torch

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score

from src.autoencoders import load_sae


def sparse_probe(Xtr, ytr, Xte, yte, seed=42):
    """liblinear logistic on sparse high-dim codes (fast, exploits sparsity)."""
    clf = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=seed)
    clf.fit(Xtr, ytr)
    return accuracy_score(yte, clf.predict(Xte))


def dense_probe(Xtr, ytr, Xte, yte, seed=42):
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(Xtr, ytr)
    return accuracy_score(yte, clf.predict(Xte))


@torch.no_grad()
def encode_codes(sae, X, k, device, batch=4096):
    """Encode X [N,d_in] -> csr sparse top-k code matrix [N, d_sae]."""
    vals_all, idx_all = [], []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(device).float()
        pre = sae.encode_full(xb)                 # [b, d_sae] post-relu
        v, ix = pre.topk(k, dim=-1)               # [b, k]
        vals_all.append(v.cpu()); idx_all.append(ix.cpu())
    vals = torch.cat(vals_all).numpy().ravel()
    idx = torch.cat(idx_all).numpy().ravel()
    n = len(X)
    indptr = np.arange(0, n * k + 1, k)
    return sp.csr_matrix((vals, idx, indptr), shape=(n, sae.d_sae))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acts', required=True)
    ap.add_argument('--saes', required=True, help='label:path,label:path,...')
    ap.add_argument('--top_k_values', default='1,2,5,10,20,32,50')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}', flush=True)
    k_values = [int(k) for k in args.top_k_values.split(',')]
    sae_specs = [(p.split(':', 1)[0].strip(), p.split(':', 1)[1].strip())
                 for p in args.saes.split(',')]

    acts = torch.load(args.acts, weights_only=False)
    saes = [(label, load_sae(path, device)) for label, path in sae_specs]
    for label, s in saes:
        print(f'  sae {label}: d_sae={s.d_sae}, k={s.k}', flush=True)

    method_keys = ['raw'] + [f'sae:{l}' for l, _ in saes]
    results = {'args': vars(args), 'k_values': k_values,
               'sae_labels': [l for l, _ in saes], 'concepts': {}}

    for name, a in acts.items():
        t0 = time.time()
        tr_a, tr_y = a['train'], a['train_y']
        te_a, te_y = a['test'], a['test_y']
        res = {m: {} for m in method_keys}
        res['raw'] = dense_probe(tr_a.numpy(), tr_y, te_a.numpy(), te_y, args.seed)
        for label, sae in saes:
            for k in k_values:
                Xtr = encode_codes(sae, tr_a, k, device)
                Xte = encode_codes(sae, te_a, k, device)
                res[f'sae:{label}'][k] = sparse_probe(Xtr, tr_y, Xte, te_y, args.seed)
        results['concepts'][name] = res
        sline = ' '.join(f'{l}:{res["sae:"+l][k_values[-1]]:.3f}' for l, _ in saes)
        print(f'[{name}] raw={res["raw"]:.3f} @k{k_values[-1]}=({sline}) [{time.time()-t0:.0f}s]',
              flush=True)
        with open(os.path.join(args.out, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)

    print('\n=== MEAN over %d concepts (native top-k sparse codes) ===' % len(results['concepts']))
    print(f'raw ceiling: {np.mean([c["raw"] for c in results["concepts"].values()]):.3f}')
    for m in [f'sae:{l}' for l, _ in saes]:
        row = ' '.join(f'k{k}={np.mean([c[m][k] for c in results["concepts"].values()]):.3f}'
                       for k in k_values)
        print(f'  {m:16s} {row}')
    print(f'\nSaved: {os.path.join(args.out, "results.pkl")}')


if __name__ == '__main__':
    main()
