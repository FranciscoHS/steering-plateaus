#!/usr/bin/env python3
"""Energy distance between the real activation distribution and synthetic ones.

Question: how close does Jacobian-norm minimization get random points to the
real activation distribution?  Ladder of comparisons, all against real
activations (qwen3-1.7b L2, last token of seq_len-10 FineWeb):

  - real vs real          (disjoint halves; finite-sample floor)
  - real vs gauss         (mean+cov matched Gaussian, the jac-opt start dist)
  - real vs isotropic     (random unit dirs x real-norm empirical dist; ceiling)
  - real vs gauss_start   (the exact 200 Gaussian points jac-opt started from)
  - real vs jac_opt       (the 200 points after ||J||_F minimization)

All samples come from saved tensors (jac_filtered_data.pt,
jac_optimized_data.pt) — no model forward needed.

Two norm treatments (the norm mismatch alone can dominate energy distance):
  raw       — sets as-is
  rescaled  — each synthetic set scaled so its mean norm equals real's

Two estimates per condition:
  full      — all available samples (n varies per condition)
  matched   — n=200 vs n=200 subsamples, repeated; mean +/- sd (fair
              comparison with the 200-point jac-opt set)

Output: results/intrinsic_dim/energy_distance/{results.pkl, energy_distance.png}
"""

import os
import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = 'results/intrinsic_dim'
OUT_DIR = os.path.join(DATA_DIR, 'energy_distance')
SEED = 0
N_FULL = 5000       # cap per set for the full-sample estimate
N_MATCH = 200       # matched subsample size (= size of the jac-opt set)
R_MATCH = 50        # matched-estimate repeats


def pairwise_mean(X, Y=None, chunk=1024):
    """Mean pairwise L2 distance. If Y is None: mean over off-diagonal pairs
    within X (the unbiased within-set term)."""
    if Y is None:
        n = X.shape[0]
        tot = 0.0
        for st in range(0, n, chunk):
            d = torch.cdist(X[st:st + chunk], X)
            tot += d.sum().item()
        return tot / (n * (n - 1))          # diagonal is 0, drop it from count
    tot = 0.0
    for st in range(0, X.shape[0], chunk):
        tot += torch.cdist(X[st:st + chunk], Y).sum().item()
    return tot / (X.shape[0] * Y.shape[0])


def energy_distance(X, Y):
    """Unbiased sample energy distance: 2 E|X-Y| - E|X-X'| - E|Y-Y'|."""
    return 2 * pairwise_mean(X, Y) - pairwise_mean(X) - pairwise_mean(Y)


def rescale_to(X, target_mean_norm):
    return X * (target_mean_norm / X.norm(dim=-1).mean())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    g = torch.Generator().manual_seed(SEED)

    filt = torch.load(os.path.join(DATA_DIR, 'jac_filtered_data.pt'),
                      map_location='cpu', weights_only=False)
    opt = torch.load(os.path.join(DATA_DIR, 'jac_optimized_data.pt'),
                     map_location='cpu', weights_only=False)
    X_real = filt['X_real'].float()                       # [5000, d]
    X_gauss = filt['X_gauss'].float()                     # [20000, d]
    X_opt = opt['X_opt'].float()                          # [200, d]
    X_gstart = opt['X_gauss_subset'].float()              # [200, d]
    d_model = X_real.shape[1]

    # Isotropic ceiling: random unit directions, norms resampled from the real
    # empirical norm distribution (matches the norm profile, kills all
    # covariance structure).
    real_norms = X_real.norm(dim=-1)
    iso = torch.randn(N_FULL, d_model, generator=g)
    iso = iso / iso.norm(dim=-1, keepdim=True)
    idx = torch.randint(0, len(real_norms), (N_FULL,), generator=g)
    X_iso = iso * real_norms[idx].unsqueeze(-1)

    mean_norm_real = real_norms.mean()
    print(f'd={d_model}  mean||a||: real={mean_norm_real:.2f}  '
          f'gauss={X_gauss.norm(dim=-1).mean():.2f}  '
          f'iso={X_iso.norm(dim=-1).mean():.2f}  '
          f'gauss_start={X_gstart.norm(dim=-1).mean():.2f}  '
          f'jac_opt={X_opt.norm(dim=-1).mean():.2f}', flush=True)

    # Shuffle real once; halves give the floor condition.
    perm = torch.randperm(X_real.shape[0], generator=g)
    X_real = X_real[perm]
    half = X_real.shape[0] // 2
    real_A, real_B = X_real[:half], X_real[half:2 * half]

    conditions = {
        'real (split half)': real_B,
        'gauss (cov-matched)': X_gauss[torch.randperm(X_gauss.shape[0],
                                                      generator=g)[:N_FULL]],
        'isotropic (norm-matched)': X_iso,
        'gauss start (n=200)': X_gstart,
        'jac-optimized (n=200)': X_opt,
    }

    results = {'mean_norm_real': float(mean_norm_real), 'full': {},
               'matched': {}, 'seed': SEED, 'n_match': N_MATCH,
               'r_match': R_MATCH}

    # --- full-sample estimates -------------------------------------------
    print('\n=== full-sample energy distance vs real ===', flush=True)
    for name, Y in conditions.items():
        base = real_A if name.startswith('real') else X_real
        raw = energy_distance(base, Y)
        Ys = Y if name.startswith('real') else rescale_to(Y, mean_norm_real)
        res = energy_distance(base, Ys)
        results['full'][name] = {'raw': raw, 'rescaled': res,
                                 'n': Y.shape[0]}
        print(f'  {name:28s} n={Y.shape[0]:5d}  raw={raw:9.2f}  '
              f'rescaled={res:9.2f}', flush=True)

    # --- matched-n estimates (200 vs 200, repeated) ----------------------
    print(f'\n=== matched-n estimates ({N_MATCH} vs {N_MATCH}, '
          f'{R_MATCH} repeats) ===', flush=True)
    for name, Y in conditions.items():
        raws, ress = [], []
        Yr = Y if name.startswith('real') else rescale_to(Y, mean_norm_real)
        for _ in range(R_MATCH):
            ia = torch.randperm(X_real.shape[0], generator=g)[:2 * N_MATCH]
            A = X_real[ia[:N_MATCH]]
            if name.startswith('real'):
                # disjoint real draw as the "other" set
                B_raw = B_res = X_real[ia[N_MATCH:]]
            else:
                ib = torch.randperm(Y.shape[0], generator=g)[:N_MATCH]
                B_raw, B_res = Y[ib], Yr[ib]
            raws.append(energy_distance(A, B_raw))
            ress.append(energy_distance(A, B_res))
        results['matched'][name] = {
            'raw_mean': float(np.mean(raws)), 'raw_sd': float(np.std(raws)),
            'res_mean': float(np.mean(ress)), 'res_sd': float(np.std(ress))}
        m = results['matched'][name]
        print(f'  {name:28s} raw={m["raw_mean"]:9.2f}±{m["raw_sd"]:6.2f}  '
              f'rescaled={m["res_mean"]:9.2f}±{m["res_sd"]:6.2f}', flush=True)

    with open(os.path.join(OUT_DIR, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # --- plot: matched-n estimates (fair across n) -----------------------
    names = list(conditions.keys())
    x = np.arange(len(names))
    w = 0.38
    raw_m = [results['matched'][n]['raw_mean'] for n in names]
    raw_s = [results['matched'][n]['raw_sd'] for n in names]
    res_m = [results['matched'][n]['res_mean'] for n in names]
    res_s = [results['matched'][n]['res_sd'] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, raw_m, w, yerr=raw_s, capsize=3, color='#4477aa',
           label='raw')
    ax.bar(x + w / 2, res_m, w, yerr=res_s, capsize=3, color='#ee7733',
           label='rescaled to real mean norm')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('energy distance to real activations')
    ax.set_title(f'qwen3-1.7b L2 activations — energy distance vs real\n'
                 f'(matched n={N_MATCH} vs {N_MATCH}, mean±sd over '
                 f'{R_MATCH} draws)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'energy_distance.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved {os.path.join(OUT_DIR, "results.pkl")}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
