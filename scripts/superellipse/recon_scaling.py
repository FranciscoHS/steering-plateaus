"""Does OMP reconstruction improve FASTER for our dictionary than for a random
dictionary as N (dict size) grows? The feature hypothesis predicts the gap
(random_err - ours_err) widens with N (more of each activation's true features
get covered); structured-random predicts a ~flat gap (both improve only via
best-of-N adaptivity).

Truncates the dictionary to N = [..] atoms, runs OMP vs a matched random dict of
the same N, at several K_eval. Pure linear algebra on cached held activations.

Usage:
    python scripts/superellipse/recon_scaling.py \\
        --dict_path /tmp/snap_latest.pt --held_acts_path .../held_acts.pt \\
        --N 250 500 1000 1500 2100 --K_eval 16 32 64 128 --out .../recon_scaling
"""
import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.superellipse.eval_dictionary_omp import omp_batched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--held_acts_path', required=True)
    ap.add_argument('--N', type=int, nargs='+',
                    default=[250, 500, 1000, 1500, 2100])
    ap.add_argument('--K_eval', type=int, nargs='+', default=[16, 32, 64, 128])
    ap.add_argument('--baseline_seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    D_all = torch.load(args.dict_path, map_location='cpu')['dictionary'].float()
    D_all = D_all / D_all.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    n_have = D_all.shape[0]
    A = torch.load(args.held_acts_path, map_location='cpu')['activations'].float()
    A = A.to(device)
    d = D_all.shape[1]
    print(f'dict has {n_have} atoms; held acts {tuple(A.shape)}', flush=True)

    Ns = [N for N in args.N if N <= n_have]
    Ke = sorted(args.K_eval)
    # results[method][N] = array over K_eval of mean rec_err^2
    res = {'ours': {}, 'random': {}}
    g = torch.Generator().manual_seed(args.baseline_seed)
    for N in Ns:
        Kev = [k for k in Ke if k <= N]
        Do = D_all[:N].to(device)
        rec = omp_batched(Do, A, max(Kev), Kev, device)
        res['ours'][N] = {k: float(rec[i].mean()) for i, k in enumerate(Kev)}
        Rn = torch.randn(N, d, generator=g)
        Rn = (Rn / Rn.norm(dim=-1, keepdim=True)).to(device)
        recr = omp_batched(Rn, A, max(Kev), Kev, device)
        res['random'][N] = {k: float(recr[i].mean()) for i, k in enumerate(Kev)}
        msg = '  '.join(
            f'K{k}: ours={res["ours"][N][k]:.3f} rand={res["random"][N][k]:.3f} '
            f'gap={res["random"][N][k]-res["ours"][N][k]:+.3f}' for k in Kev)
        print(f'N={N:5d}  {msg}', flush=True)

    with open(os.path.join(args.out, 'recon_scaling.pkl'), 'wb') as f:
        pickle.dump({'res': res, 'Ns': Ns, 'K_eval': Ke}, f)

    # plot: gap (random - ours) vs N, one line per K
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for k in Ke:
        xs = [N for N in Ns if k in res['ours'][N]]
        gap = [res['random'][N][k] - res['ours'][N][k] for N in xs]
        ax1.plot(xs, gap, 'o-', label=f'K={k}')
        ax2.plot(xs, [res['ours'][N][k] for N in xs], 'o-', label=f'ours K={k}')
        ax2.plot(xs, [res['random'][N][k] for N in xs], 'o--', alpha=0.5)
    ax1.set_xlabel('N (dict size)'); ax1.set_ylabel('gap = rand_err - ours_err')
    ax1.set_title('reconstruction advantage vs N (feature hyp: should widen)')
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.axhline(0, color='gray', lw=0.5)
    ax2.set_xlabel('N'); ax2.set_ylabel('mean rec_err^2')
    ax2.set_title('solid=ours, dashed=random'); ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'recon_scaling.png'), dpi=150,
                bbox_inches='tight')
    print(f'\nSaved {os.path.join(args.out, "recon_scaling.png")}', flush=True)


if __name__ == '__main__':
    main()
