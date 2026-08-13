#!/usr/bin/env python3
"""Does moving Gaussian points along the separating normal reduce ||J||_F?

The jac-optimized points are linearly separable from the cov-matched Gaussian
points they started from (C2ST logistic AUC ~1.0). Francisco's hypothesis:
the normal w of that separating plane is itself a "Jacobian-reducing"
direction — applying a simple linear map along w to fresh Gaussian points
should lower their Jacobian norm, even if the endpoints differ from the
gradient-descent ones.

Procedure:
  1. Fit logistic regression (raw activation space) on the 200 paired sets
     gauss-start (0) vs jac-opt (1) -> unit normal w_hat, intercept.
  2. Take 200 FRESH Gaussian points from the saved cov-matched pool
     (rows 200:400; rows 0:200 are the jac-opt starting points).
  3. Conditions, all measured with finite-difference Hutchinson ||J||_F
     (same estimator/settings as intrinsic_dim_jac_filtered.py):
       gauss fresh              baseline
       gauss + c*w_hat          constant shift; c = mean displacement of the
                                jac-opt pairs along w_hat
       gauss mirrored           reflection across the separating plane
       jac-opt                  reference (recomputed, same contexts)
  4. Contexts: first 200 FineWeb contexts, re-extracted with the exact
     recipe of intrinsic_dim_jac_filtered.py (seed 42, seq_len 10) — the
     same contexts the jac-opt points were optimized with.

Output: results/intrinsic_dim/jac_along_normal/{results.pkl, jac_hist.png}
"""

import os
import pickle
import sys

sys.path.insert(0, "/workspace/steering-plateaus")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model
from intrinsic_dim_jac_filtered import estimate_jac_norms_batched

DATA_DIR = 'results/intrinsic_dim'
OUT_DIR = os.path.join(DATA_DIR, 'jac_along_normal')
SOURCE_LAYER = 2
SEQ_LEN = 10
N_POINTS = 200
N_PROBES = 10


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    filt = torch.load(os.path.join(DATA_DIR, 'jac_filtered_data.pt'),
                      map_location='cpu', weights_only=False)
    opt = torch.load(os.path.join(DATA_DIR, 'jac_optimized_data.pt'),
                     map_location='cpu', weights_only=False)
    X_gauss = filt['X_gauss'].float()
    X_opt = opt['X_opt'].float()
    X_gstart = opt['X_gauss_subset'].float()

    # 1. Separating plane from the paired before/after sets (raw space).
    X = np.concatenate([X_gstart.numpy(), X_opt.numpy()])
    y = np.concatenate([np.zeros(N_POINTS), np.ones(N_POINTS)])
    clf = LogisticRegression(C=0.1, max_iter=5000)
    clf.fit(X, y)
    acc = clf.score(X, y)
    w = torch.from_numpy(clf.coef_[0]).float()
    b = float(clf.intercept_[0])
    w_hat = w / w.norm()
    print(f'Logistic on gauss-start vs jac-opt: train acc={acc:.3f}',
          flush=True)

    # Mean displacement of the optimized pairs along w_hat -> shift size c.
    c = float(((X_opt - X_gstart) @ w_hat).mean())
    print(f'Mean pair displacement along w_hat: c={c:.2f}  '
          f'(mean total displacement '
          f'{(X_opt - X_gstart).norm(dim=-1).mean():.2f})', flush=True)

    # 2. Fresh Gaussian points + the two linear maps.
    G = X_gauss[N_POINTS:2 * N_POINTS]
    G_shift = G + c * w_hat
    signed = (G @ w) + b                       # logit; plane is signed = 0
    G_mirror = G - 2 * (signed / (w.norm() ** 2)).unsqueeze(-1) * w
    print(f'Fraction of shifted points past the plane: '
          f'{((G_shift @ w) + b > 0).float().mean():.2f};  '
          f'mirrored: {((G_mirror @ w) + b > 0).float().mean():.2f}',
          flush=True)

    # 3. Contexts: exact recipe of intrinsic_dim_jac_filtered.py.
    print('Loading model...', flush=True)
    model, tokenizer, device = load_model('Qwen/Qwen3-1.7B',
                                          dtype=torch.float16)
    print('Extracting contexts...', flush=True)
    ds = load_dataset('HuggingFaceFW/FineWeb-Edu', 'sample-10BT',
                      split='train', streaming=True)
    ds = ds.shuffle(seed=42)
    texts = []
    for ex in ds:
        if len(ex['text']) >= SEQ_LEN * 3:
            texts.append(ex['text'])
        if len(texts) >= N_POINTS:
            break
    all_ctx = []
    for i in range(0, len(texts), 32):
        tokens = tokenizer(texts[i:i + 32], return_tensors='pt',
                           padding='max_length', truncation=True,
                           max_length=SEQ_LEN).to(device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        all_ctx.append(out.hidden_states[SOURCE_LAYER + 1][:, :-1, :].cpu())
    contexts = torch.cat(all_ctx, dim=0)
    print(f'Contexts: {tuple(contexts.shape)}', flush=True)

    # 4. Hutchinson ||J||_F per condition.
    conditions = {
        'gauss fresh': G,
        'gauss + c*w': G_shift,
        'gauss mirrored': G_mirror,
        'jac-opt (reference)': X_opt,
    }
    jac = {}
    for name, pts in conditions.items():
        print(f'Estimating ||J||_F: {name}', flush=True)
        jac[name] = estimate_jac_norms_batched(
            model, contexts, pts, SOURCE_LAYER, n_probes=N_PROBES)
        print(f'  {name:22s} mean={jac[name].mean():.1f}  '
              f'median={jac[name].median():.1f}', flush=True)

    saved_real = filt['real_jac_norms']
    saved_gauss = filt['gauss_jac_norms']
    print('\n=== ||J||_F summary ===', flush=True)
    print(f'  saved real   (n={len(saved_real)}): '
          f'mean={saved_real.mean():.1f} median={saved_real.median():.1f}')
    print(f'  saved gauss  (n={len(saved_gauss)}): '
          f'mean={saved_gauss.mean():.1f} median={saved_gauss.median():.1f}')
    for name, v in jac.items():
        print(f'  {name:22s} mean={v.mean():.1f}  median={v.median():.1f}  '
              f'mean||a||={conditions[name].norm(dim=-1).mean():.1f}')

    with open(os.path.join(OUT_DIR, 'results.pkl'), 'wb') as f:
        pickle.dump({
            'jac': {k: v.numpy() for k, v in jac.items()},
            'w': w.numpy(), 'b': b, 'c': c, 'train_acc': acc,
            'norms': {k: v.norm(dim=-1).numpy()
                      for k, v in conditions.items()},
            'saved_real_jac': saved_real.numpy(),
            'saved_gauss_jac': saved_gauss.numpy(),
            'n_probes': N_PROBES, 'source_layer': SOURCE_LAYER,
        }, f)

    colors = {'gauss fresh': '#4477aa', 'gauss + c*w': '#ee7733',
              'gauss mirrored': '#228833', 'jac-opt (reference)': '#aa3377'}
    hi = max(float(v.max()) for v in jac.values())
    bins = np.linspace(0, hi * 1.02, 50)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for name, v in jac.items():
        ax.hist(v.numpy(), bins=bins, density=True, histtype='step', lw=1.8,
                color=colors[name], label=name)
    ax.axvline(float(saved_real.mean()), color='gray', ls='--', lw=1.2,
               label='real (saved mean)')
    ax.set_xlabel(r'$\|J\|_F$ (Hutchinson, L2$\to$final)')
    ax.set_ylabel('density')
    ax.set_title('Jacobian norm after linear maps along the separating '
                 'normal — qwen3-1.7b L2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'jac_hist.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved {os.path.join(OUT_DIR, "results.pkl")}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
