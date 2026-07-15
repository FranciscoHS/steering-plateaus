#!/usr/bin/env python3
"""
Max-selectivity single-latent scan: does any SAE latent alone capture a concept?

For each concept and each SAE, encode the cached activations to full post-relu
latent activations, score every latent by class separation (|t-stat|) on TRAIN,
pick the most selective latent, and measure that single latent's TEST accuracy
(1-D logistic). This is the "is there a dedicated latent for concept X" test, and
it hands back the latent index to open in the dashboard.

Reuses suite_acts.pt from run_sparse_probe_suite.py.

Usage:
    python -u scripts/probes/max_selectivity.py \
        --acts results/probe_suite_highp/suite_acts.pt \
        --saes baseline:.../l10_doubled_k32_2M/sae_weights.pt,popt:.../l10_lam1/sae_weights.pt \
        --out results/probe_native/max_selectivity.pkl
"""
import argparse
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.autoencoders import load_sae


@torch.no_grad()
def encode_full(sae, X, device, batch=4096):
    """Full post-relu latent activations [N, d_sae] (pre-topk)."""
    out = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(device).float()
        out.append(sae.encode_full(xb).cpu())
    return torch.cat(out)


def tstat(A, y):
    """Per-latent |t-statistic| between the two classes. A [N,d], y {0,1}."""
    y = np.asarray(y)
    a1, a0 = A[y == 1], A[y == 0]
    m1, m0 = a1.mean(0), a0.mean(0)
    v1, v0 = a1.var(0) + 1e-8, a0.var(0) + 1e-8
    return np.abs(m1 - m0) / np.sqrt(v1 / len(a1) + v0 / len(a0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acts', required=True)
    ap.add_argument('--saes', required=True)
    ap.add_argument('--topn', type=int, default=10, help='top latents to record per concept')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}', flush=True)
    acts = torch.load(args.acts, weights_only=False)
    sae_specs = [(p.split(':', 1)[0].strip(), p.split(':', 1)[1].strip())
                 for p in args.saes.split(',')]
    saes = [(label, load_sae(path, device)) for label, path in sae_specs]

    results = {}
    for name, a in acts.items():
        tr, tr_y = a['train'], a['train_y']
        te, te_y = a['test'], a['test_y']
        results[name] = {}
        line = []
        for label, sae in saes:
            A_tr = encode_full(sae, tr, device).numpy()
            A_te = encode_full(sae, te, device).numpy()
            t = tstat(A_tr, tr_y)
            order = np.argsort(-t)
            best = int(order[0])
            clf = LogisticRegression(solver='liblinear', C=1.0)
            clf.fit(A_tr[:, [best]], tr_y)
            acc = accuracy_score(te_y, clf.predict(A_te[:, [best]]))
            results[name][label] = {
                'best_latent': best, 'best_tstat': float(t[best]),
                'test_acc_1latent': float(acc),
                'top_latents': order[:args.topn].tolist(),
                'top_tstats': t[order[:args.topn]].round(3).tolist(),
            }
            line.append(f'{label}: L{best} acc={acc:.3f}')
        print(f'[{name}] ' + ' | '.join(line), flush=True)
        with open(args.out, 'wb') as f:
            pickle.dump(results, f)

    print('\n=== best single-latent TEST acc (mean over concepts) ===')
    for label, _ in saes:
        accs = [results[n][label]['test_acc_1latent'] for n in results]
        print(f'  {label:10s} {np.mean(accs):.3f}')
    print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
