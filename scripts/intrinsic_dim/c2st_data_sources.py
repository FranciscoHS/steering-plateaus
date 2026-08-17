#!/usr/bin/env python3
"""C2ST: are jac-optimized activations separable from real ones of ANY source?

The jac-opt points (results/intrinsic_dim/jac_optimized_data.pt) were made by
minimizing ||J||_F starting from cov-matched Gaussian points, with FineWeb
contexts, on Qwen3-1.7B layer 2. A logistic probe separates them from FineWeb
real activations at AUC ~1.0 (results/intrinsic_dim/c2st/results.pkl).

Open question: is that separation "jac-opt vs real activations", or only
"jac-opt vs FineWeb"? Here we collect real activations from several other
sources (Wikipedia en/fr/zh/ar, Python code, arXiv) with the *same* protocol
(layer 2, seq_len 10, last token) and run the same C2ST of jac-opt against
each one.

Per source S, two comparisons:
  S vs jacopt_raw   the optimized points as they are
  S vs jacopt_resc  optimized points rescaled to S's own mean norm

Stage 1 (GPU) caches activations per source, so re-running the C2ST is cheap.

Outputs in --outdir (default results/intrinsic_dim/c2st_data_sources/):
  acts/<source>.pt, results.pkl, auc_by_source.png, summary.txt
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/steering-plateaus')

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_sources import SOURCE_LOADERS

JAC_DIR = 'results/intrinsic_dim'


# ── Activation collection ───────────────────────────────────────────────

def collect_layer(model, tokenizer, device, texts, seq_len, layer, batch_size=64):
    """Last-token activations at `layer`, same protocol as
    intrinsic_dim_jac_filtered.py (right padding to max_length=seq_len)."""
    chunks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(
            batch, return_tensors='pt', padding='max_length',
            truncation=True, max_length=seq_len,
        ).to(device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        h = out.hidden_states[layer + 1]          # (b, seq_len, d)
        chunks.append(h[:, -1, :].float().cpu())
    return torch.cat(chunks, dim=0)


def load_model_lazy(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16,
                 'float32': torch.float32}
    print(f'Loading {args.model} ({args.dtype}) ...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype_map[args.dtype], device_map='auto')
    model.eval()
    device = next(model.parameters()).device
    return model, tokenizer, device


def collect_sources(args, acts_dir):
    """Return {source: tensor}; caches each source to acts_dir/<name>.pt."""
    os.makedirs(acts_dir, exist_ok=True)
    acts, missing = {}, []
    for name in args.sources:
        path = os.path.join(acts_dir, f'{name}.pt')
        if os.path.exists(path):
            acts[name] = torch.load(path, map_location='cpu',
                                    weights_only=False).float()
            print(f'  cached {name}: {tuple(acts[name].shape)}', flush=True)
        else:
            missing.append(name)
    if not missing:
        return acts

    model, tokenizer, device = load_model_lazy(args)
    min_chars = args.seq_len * 3          # same filter as the original run
    for name in missing:
        print(f'\n=== {name}: loading {args.n_pool} texts '
              f'(min {min_chars} chars) ===', flush=True)
        try:
            texts = SOURCE_LOADERS[name](args.n_pool, min_chars, args.seed)
        except Exception as e:                        # gated / missing dataset
            print(f'  FAILED to load {name}: {e!r}', flush=True)
            continue
        print(f'  got {len(texts)} texts', flush=True)
        X = collect_layer(model, tokenizer, device, texts,
                          args.seq_len, args.layer, args.batch_size)
        torch.save(X, os.path.join(acts_dir, f'{name}.pt'))
        acts[name] = X
        print(f'  activations {tuple(X.shape)}  mean||x||='
              f'{X.norm(dim=-1).mean():.2f}  -> cached', flush=True)

    del model
    torch.cuda.empty_cache()
    return acts


# ── C2ST ────────────────────────────────────────────────────────────────

def cv_auc(X0, X1, seed, n_folds):
    """Out-of-fold AUC of an L2 logistic probe separating X0 from X1."""
    X = np.concatenate([X0, X1])
    y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))])
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.1, max_iter=2000,
                                           random_state=seed))
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def draw_pair(A, B, n, rs):
    """Sample n rows from each side. A is B means the disjoint-draw floor, in
    which case the two halves come from one permutation."""
    g = torch.Generator().manual_seed(rs)
    if A is B:
        n = min(n, A.shape[0] // 2)
        idx = torch.randperm(A.shape[0], generator=g)
        return A[idx[:n]].numpy(), A[idx[n:2 * n]].numpy()
    ia = torch.randperm(A.shape[0], generator=g)[:min(n, A.shape[0])]
    ib = torch.randperm(B.shape[0], generator=g)[:min(n, B.shape[0])]
    return A[ia].numpy(), B[ib].numpy()


def auc_over_draws(A, B, args, norm_only=False):
    aucs = []
    for r in range(args.r_draws):
        rs = args.seed + 1000 * (r + 1)
        Xa, Xb = draw_pair(A, B, args.n_per_class, rs)
        if norm_only:
            Xa = np.linalg.norm(Xa, axis=-1, keepdims=True)
            Xb = np.linalg.norm(Xb, axis=-1, keepdims=True)
        aucs.append(cv_auc(Xa, Xb, rs, args.n_folds))
    return float(np.mean(aucs)), float(np.std(aucs)), aucs


def rescale_to(X, target_mean_norm):
    return X * (target_mean_norm / X.norm(dim=-1).mean())


# ── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen3-1.7B')
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--seq_len', type=int, default=10)
    p.add_argument('--n_pool', type=int, default=2000,
                   help='texts collected per source')
    p.add_argument('--sources', nargs='+', default=list(SOURCE_LOADERS.keys()),
                   choices=list(SOURCE_LOADERS.keys()))
    p.add_argument('--n_per_class', type=int, default=200)
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--r_draws', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--dtype', default='float16')
    p.add_argument('--outdir', default='results/intrinsic_dim/c2st_data_sources')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    acts = collect_sources(args, os.path.join(args.outdir, 'acts'))

    # The jac-opt points and the FineWeb real set they were compared against.
    filt = torch.load(os.path.join(JAC_DIR, 'jac_filtered_data.pt'),
                      map_location='cpu', weights_only=False)
    opt = torch.load(os.path.join(JAC_DIR, 'jac_optimized_data.pt'),
                     map_location='cpu', weights_only=False)
    X_opt = opt['X_opt'].float()
    X_fw_orig = filt['X_real'].float()

    # fineweb_orig is the original real set: continuity check against the
    # already-published AUC, and a check that re-collection matches it.
    reals = dict(acts)
    reals['fineweb_orig'] = X_fw_orig
    names = list(reals.keys())

    res = {'args': vars(args), 'n_rows': {}, 'mean_norms': {}, 'auc': {}}
    print(f'\njac-opt: n={X_opt.shape[0]}, d={X_opt.shape[1]}, '
          f'mean||x||={X_opt.norm(dim=-1).mean():.2f}')
    print('Real populations:')
    for n in names:
        res['n_rows'][n] = int(reals[n].shape[0])
        res['mean_norms'][n] = float(reals[n].norm(dim=-1).mean())
        print(f'  {n:14s} n={reals[n].shape[0]:6d}  d={reals[n].shape[1]}  '
              f'mean||x||={res["mean_norms"][n]:.2f}')

    print(f'\nC2ST: logistic, n={args.n_per_class}/class, {args.n_folds}-fold '
          f'CV, {args.r_draws} draws', flush=True)
    for n in names:
        S = reals[n]
        entry = {}
        for tag, B in [('jacopt_raw', X_opt),
                       ('jacopt_resc', rescale_to(X_opt, S.norm(dim=-1).mean()))]:
            mean, sd, draws = auc_over_draws(S, B, args)
            entry[tag] = {'mean': mean, 'sd': sd, 'draws': draws}
        res['auc'][n] = entry
        print(f'  {n:14s} raw={entry["jacopt_raw"]["mean"]:.3f}±'
              f'{entry["jacopt_raw"]["sd"]:.3f}  '
              f'resc={entry["jacopt_resc"]["mean"]:.3f}±'
              f'{entry["jacopt_resc"]["sd"]:.3f}', flush=True)
        with open(os.path.join(args.outdir, 'results.pkl'), 'wb') as f:
            pickle.dump(res, f)

    out = os.path.join(args.outdir, 'results.pkl')
    with open(out, 'wb') as f:
        pickle.dump(res, f)
    print(f'\nSaved {out}')

    # Bar chart: one group per source.
    done = [n for n in names if n in res['auc']]
    series = [('jacopt_raw', 'C0'), ('jacopt_resc', 'C1')]
    x = np.arange(len(done))
    w = 0.35
    fig, ax = plt.subplots(figsize=(1.4 * len(done) + 3, 4.5))
    for k, (tag, color) in enumerate(series):
        vals = [res['auc'][n][tag]['mean'] for n in done]
        errs = [res['auc'][n][tag]['sd'] for n in done]
        ax.bar(x + (k - 0.5) * w, vals, w, yerr=errs, capsize=2,
               label=tag, color=color, alpha=0.9)
    ax.axhline(0.5, color='k', lw=0.8, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels(done, rotation=20, ha='right')
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel('out-of-fold AUC')
    ax.set_title(f'jac-opt vs real activations by source — {args.model} '
                 f'L{args.layer}, seq_len={args.seq_len}', fontsize=10)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plot = os.path.join(args.outdir, 'auc_by_source.png')
    plt.savefig(plot, dpi=150)
    print(f'Saved {plot}')

    with open(os.path.join(args.outdir, 'summary.txt'), 'w') as f:
        f.write(f'{args.model} L{args.layer} seq_len={args.seq_len} '
                f'n={args.n_per_class}/class r={args.r_draws}\n')
        f.write(f'jac-opt n={X_opt.shape[0]} '
                f'mean||x||={X_opt.norm(dim=-1).mean():.2f}\n\n')
        f.write(f'{"source":14s} {"n":>6s} {"mean|x|":>8s} {"raw":>7s} '
                f'{"resc":>7s}\n')
        for n in done:
            e = res['auc'][n]
            f.write(f'{n:14s} {res["n_rows"][n]:6d} {res["mean_norms"][n]:8.2f} '
                    f'{e["jacopt_raw"]["mean"]:7.3f} '
                    f'{e["jacopt_resc"]["mean"]:7.3f}\n')
    print(f'Saved {os.path.join(args.outdir, "summary.txt")}')


if __name__ == '__main__':
    main()
