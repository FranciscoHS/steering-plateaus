#!/usr/bin/env python3
"""
Fast sparse-probe suite using top-k PROJECTION (not greedy OMP).

Reuses the activations already cached by run_sparse_probe_suite.py
(suite_acts.pt) and, for each dictionary, reconstructs each concept's
activations from the k atoms with the largest |projection| (one matmul to score
all atoms), then a single least-squares refit on those k atoms — i.e. the
one-shot approximation to OMP. ~k× faster than greedy OMP and closer to how an
SAE actually encodes (top-k of the projections).

Same probe (liblinear logreg) and mean-as-free-bias (abar) as the OMP suite, so
results are directly comparable across dictionaries.

Usage:
    python -u scripts/probes/eval_probe_topk.py \
        --acts results/probe_suite_highp/suite_acts.pt \
        --dicts baseline:results/probe_dicts/baseline.pt,popt:results/probe_dicts/popt.pt,cullmore:results/probe_dicts/cullmore.pt \
        --top_k_values 1,2,5,10,20,32,50 \
        --out results/probe_suite_highp_topk
"""
import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import torch

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

from scripts.probes.run_probe_dict_vs_sae import load_dict


def run_probe(train_X, train_y, test_X, test_y, seed=42):
    """Closed-form ridge linear probe.

    The k-atom reconstructions are rank-k (highly collinear); iterative logistic
    solvers crawl on them. RidgeClassifier is a single closed-form least-squares
    solve with L2 (alpha) handling the collinearity — fast and robust. Applied
    identically to every method.
    """
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(train_X, train_y)
    return accuracy_score(test_y, clf.predict(test_X))


def topk_proj_gpu(dictionary, activations, k, device, batch=2000):
    """Top-k-by-projection + least-squares refit. Returns recon [N, d] (CPU).

    dictionary: [M, d] unit-norm rows. activations: [N, d].
    One-shot approximation to omp_gpu: pick the k atoms with largest |corr| in a
    single step, then solve the k-atom least-squares (no greedy residual loop).
    """
    D = dictionary.to(device).float()
    out = torch.empty_like(activations)
    eye = 1e-6 * torch.eye(k, device=device).unsqueeze(0)
    for i in range(0, len(activations), batch):
        X = activations[i:i + batch].to(device).float()       # [n, d]
        corr = (X @ D.T).abs()                                 # [n, M]
        idx = corr.topk(k, dim=1).indices                      # [n, k]
        D_sel = D[idx]                                         # [n, k, d]
        gram = D_sel @ D_sel.transpose(1, 2) + eye             # [n, k, k]
        rhs = (D_sel @ X.unsqueeze(2)).squeeze(2)              # [n, k]
        c = torch.linalg.solve(gram, rhs)                      # [n, k]
        recon = (c.unsqueeze(2) * D_sel).sum(dim=1)            # [n, d]
        out[i:i + len(X)] = recon.cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acts', required=True, help='suite_acts.pt from the OMP suite')
    ap.add_argument('--dicts', required=True, help='label:path,label:path,...')
    ap.add_argument('--top_k_values', default='1,2,5,10,20,32,50')
    ap.add_argument('--n_random', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch', type=int, default=2000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device} (cuda_available={torch.cuda.is_available()})', flush=True)
    torch.set_num_threads(4)  # avoid CPU BLAS oversubscription
    k_values = [int(k) for k in args.top_k_values.split(',')]
    dict_specs = [(p.split(':', 1)[0].strip(), p.split(':', 1)[1].strip())
                  for p in args.dicts.split(',')]

    acts = torch.load(args.acts, weights_only=False)
    dicts = [(label, load_dict(path)) for label, path in dict_specs]
    for label, D in dicts:
        print(f'  dict {label}: {D.shape[0]} atoms')
    d_model = acts[next(iter(acts))]['train'].shape[1]
    n_rand = args.n_random if args.n_random is not None else max(k_values) * 4
    rng = np.random.RandomState(args.seed + 1000)
    rand = torch.from_numpy(rng.randn(n_rand, d_model).astype(np.float32))
    rand = rand / rand.norm(dim=-1, keepdim=True)

    method_keys = ['raw'] + [f'dict:{l}' for l, _ in dicts] + ['random']
    results = {'args': vars(args), 'k_values': k_values,
               'dict_labels': [l for l, _ in dicts], 'concepts': {}}

    for name, a in acts.items():
        t0 = time.time()
        tr_a, tr_y = a['train'], a['train_y']
        te_a, te_y = a['test'], a['test_y']
        abar = tr_a.mean(dim=0, keepdim=True)
        res = {m: {} for m in method_keys}
        res['raw'] = run_probe(tr_a.numpy(), tr_y, te_a.numpy(), te_y, args.seed)
        t_raw = time.time() - t0

        def acc(D, k):
            trr = abar + topk_proj_gpu(D, tr_a - abar, k, device, args.batch)
            ter = abar + topk_proj_gpu(D, te_a - abar, k, device, args.batch)
            return run_probe(trr.numpy(), tr_y, ter.numpy(), te_y, args.seed)

        for k in k_values:
            for label, D in dicts:
                res[f'dict:{label}'][k] = acc(D, k)
            res['random'][k] = acc(rand, k)
        results['concepts'][name] = res
        sline = ' '.join(f'{l}:{res["dict:"+l][k_values[-1]]:.3f}' for l, _ in dicts)
        print(f'[{name}] raw={res["raw"]:.3f} @k{k_values[-1]}=({sline}) '
              f'[{time.time()-t0:.0f}s, raw_probe={t_raw:.0f}s]', flush=True)
        with open(os.path.join(args.out, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)

    # ---- aggregate: mean over concepts per method per k ----
    print('\n=== MEAN over %d concepts ===' % len(results['concepts']))
    raw_mean = np.mean([c['raw'] for c in results['concepts'].values()])
    print(f'raw ceiling: {raw_mean:.3f}')
    for m in [f'dict:{l}' for l, _ in dicts] + ['random']:
        row = ' '.join(f'k{k}={np.mean([c[m][k] for c in results["concepts"].values()]):.3f}'
                       for k in k_values)
        print(f'  {m:16s} {row}')
    print(f'\nSaved: {os.path.join(args.out, "results.pkl")}')


if __name__ == '__main__':
    main()
