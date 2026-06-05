#!/usr/bin/env python3
"""K-sparse reconstruction eval for a dictionary built by build_dictionary.py.

For each held-out FineWeb activation a:
  - OMP up to K_max atoms on our dictionary; record ||a-â_K||²/||a||² at K_eval.

Baselines (each evaluated at the same K_eval where applicable):
  - random unit dictionary, same N — OMP
  - random unit dictionary at SAE width (e.g. 32k) — OMP (harder baseline)
  - PCA top-K projection (fixed K-dim subspace, no per-activation atom choice)
  - Qwen-Scope-style TopK SAE: load W_enc, b_enc, W_dec, b_dec; for each a,
        pre = a @ W_enc.T + b_enc
        keep top-K (=fixed K, here 50/100), then â = pre_sparse @ W_dec.T + b_dec

If --held_acts_path is given, skip model loading — useful for CPU-only runs.

Usage:
    python scripts/superellipse/eval_dictionary_omp.py \\
        --dict_path .../dictionary_K8000.pt \\
        --held_acts_path .../held_acts.pt \\
        --sae_ckpt .../layer14.sae.pt --sae_K 50 100 \\
        --K_eval 8 16 32 64 100 256 \\
        --pca_pool_path .../act_pool.pt \\
        --out .../eval_omp_K8000
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def omp_batched(D, A, K_max, K_eval, device, chunk=64):
    """Batched OMP across activations.

    D: [N, d]  unit dictionary atoms (torch on `device`, float32).
    A: [B, d]  held-out activations (torch on `device`, float32).
    K_max:     max number of atoms to select.
    K_eval:    sorted list of K values at which to record rec error.

    Returns rec_err [len(K_eval), B] (||a-â||² / ||a||²).
    """
    B, d = A.shape
    N = D.shape[0]
    K_eval = sorted(K_eval)
    K_max = max(K_max, K_eval[-1])
    out = torch.zeros(len(K_eval), B, dtype=torch.float32)
    a_norm_sq = (A ** 2).sum(dim=-1).cpu()
    K_eval_set = {k: i for i, k in enumerate(K_eval)}

    for b0 in range(0, B, chunk):
        b1 = min(B, b0 + chunk)
        Ab = A[b0:b1]
        Rb = Ab.clone()
        sel = torch.full((Ab.shape[0], K_max), -1,
                         dtype=torch.long, device=device)
        for k in range(K_max):
            proj = Rb @ D.T
            if k > 0:
                proj.scatter_(1, sel[:, :k], 0.0)
            j = proj.abs().argmax(dim=-1)
            sel[:, k] = j

            D_sel = D[sel[:, :k + 1]]                    # [b, k+1, d]
            G = D_sel @ D_sel.transpose(-1, -2)
            rhs = (D_sel * Ab.unsqueeze(1)).sum(dim=-1)
            G = G + 1e-7 * torch.eye(k + 1, device=device)
            coeffs = torch.linalg.solve(G, rhs)
            recon = (coeffs.unsqueeze(-1) * D_sel).sum(dim=1)
            Rb = Ab - recon

            if (k + 1) in K_eval_set:
                e = (Rb ** 2).sum(dim=-1).cpu()
                out[K_eval_set[k + 1], b0:b1] = e / a_norm_sq[b0:b1]

    return out


def pca_topk_eval(A_pool, A_held, K_list, device):
    """Fit PCA on A_pool, project A_held onto top-K PCs for each K in K_list.

    Returns rec_err [len(K_list), B_held].
    """
    K_list = sorted(K_list)
    A_pool = A_pool.to(device=device, dtype=torch.float32)
    A_held = A_held.to(device=device, dtype=torch.float32)
    mu = A_pool.mean(dim=0, keepdim=True)
    Xc = A_pool - mu
    # SVD of mean-centered: components in V[:k]
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    V = Vh                                                # [d, d]
    K_max = max(K_list)
    if K_max > V.shape[0]:
        raise ValueError(f'K_max={K_max} > d_model={V.shape[0]}')
    out = torch.zeros(len(K_list), A_held.shape[0])
    Hc = A_held - mu                                      # mean-centered held-out
    norm_sq = (A_held ** 2).sum(dim=-1).cpu()             # eval against raw a, not Hc
    for i, K in enumerate(K_list):
        Vk = V[:K]                                        # [K, d]
        # Project Hc onto Vk subspace, reconstruct in raw coords by + mu.
        coefs = Hc @ Vk.T                                 # [B, K]
        recon = coefs @ Vk + mu                           # [B, d]
        out[i] = ((A_held - recon) ** 2).sum(dim=-1).cpu() / norm_sq
    return out


def sae_topk_eval(W_enc, b_enc, W_dec, b_dec, A_held, K_list, device):
    """For each K in K_list, encode A_held → top-K → decode.

    W_enc: [d_sae, d]
    b_enc: [d_sae]
    W_dec: [d, d_sae]   or  [d_sae, d]   (try both; SAE conventions vary)
    b_dec: [d]
    """
    K_list = sorted(K_list)
    A_held = A_held.to(device=device, dtype=torch.float32)
    W_enc = W_enc.to(device=device, dtype=torch.float32)
    b_enc = b_enc.to(device=device, dtype=torch.float32)
    W_dec = W_dec.to(device=device, dtype=torch.float32)
    b_dec = b_dec.to(device=device, dtype=torch.float32)
    # Resolve W_dec shape
    d = A_held.shape[1]
    if W_dec.shape == (d, W_enc.shape[0]):
        W_dec_use = W_dec                                 # [d, d_sae]
    elif W_dec.shape == (W_enc.shape[0], d):
        W_dec_use = W_dec.T
    else:
        raise ValueError(f'Unexpected W_dec shape {tuple(W_dec.shape)} '
                         f'for d={d}, d_sae={W_enc.shape[0]}')

    pre = A_held @ W_enc.T + b_enc                        # [B, d_sae]
    norm_sq = (A_held ** 2).sum(dim=-1).cpu()
    out = torch.zeros(len(K_list), A_held.shape[0])
    for i, K in enumerate(K_list):
        topv, topi = pre.topk(K, dim=-1)                  # [B, K]
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topi, topv)
        recon = acts @ W_dec_use.T + b_dec
        out[i] = ((A_held - recon) ** 2).sum(dim=-1).cpu() / norm_sq
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True,
                    help='dictionary.pt (or _K8000.pt snapshot)')
    ap.add_argument('--held_acts_path', required=True,
                    help='pre-extracted held-out activations .pt file')
    ap.add_argument('--K_eval', type=int, nargs='+',
                    default=[8, 16, 32, 64, 100, 256])
    ap.add_argument('--K_max', type=int, default=None,
                    help='OMP iters; default = max(K_eval)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--baseline_seed', type=int, default=0,
                    help='seed for random-direction baseline dict')
    ap.add_argument('--rand_wide_N', type=int, default=32000,
                    help='size for wider random baseline (match SAE width)')
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    ap.add_argument('--sae_ckpt', default=None,
                    help='Qwen-Scope SAE checkpoint .sae.pt (e.g. layer14.sae.pt)')
    ap.add_argument('--sae_K', type=int, nargs='+', default=[50, 100],
                    help='TopK values to eval the SAE at')
    ap.add_argument('--pca_pool_path', default=None,
                    help='activation pool .pt used to fit PCA (different from held)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.K_max is None:
        args.K_max = max(args.K_eval)
    K_eval = sorted(args.K_eval)

    device = (torch.device('cuda') if args.device == 'cuda'
              else torch.device('cpu') if args.device == 'cpu'
              else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}', flush=True)

    # Load dictionary
    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(device=device, dtype=torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    N = D.shape[0]
    d_model = D.shape[1]
    print(f'Dictionary: {N} atoms, d={d_model}', flush=True)

    # Held-out activations (pre-extracted)
    held_blob = torch.load(args.held_acts_path, map_location='cpu')
    A = held_blob['activations'].to(device=device, dtype=torch.float32)
    print(f'Held-out activations: {tuple(A.shape)}  '
          f'mean ||a||={A.norm(dim=-1).mean().item():.2f}', flush=True)

    results = {'args': vars(args), 'K_eval': K_eval, 'N': N,
               'd_model': d_model, 'n_held': A.shape[0]}

    # --- OMP on our dictionary ---
    print(f'\n[OMP] dict ({N} atoms)  K_max={args.K_max}', flush=True)
    t0 = time.time()
    rec_dict = omp_batched(D, A, args.K_max, K_eval, device, chunk=args.chunk)
    results['rec_err_dict'] = rec_dict.numpy()
    print(f'  done in {time.time()-t0:.1f}s', flush=True)

    # --- Random baseline same N ---
    print(f'\n[OMP] random_{N}  (same width as dict)', flush=True)
    g = torch.Generator(device=device).manual_seed(args.baseline_seed)
    D_rand = torch.randn(N, d_model, generator=g, device=device,
                         dtype=torch.float32)
    D_rand = D_rand / D_rand.norm(dim=-1, keepdim=True)
    t0 = time.time()
    rec_rand = omp_batched(D_rand, A, args.K_max, K_eval, device,
                           chunk=args.chunk)
    results['rec_err_rand_sameN'] = rec_rand.numpy()
    print(f'  done in {time.time()-t0:.1f}s', flush=True)

    # --- Random baseline at SAE width ---
    if args.rand_wide_N and args.rand_wide_N != N:
        print(f'\n[OMP] random_{args.rand_wide_N}  (SAE-width baseline)',
              flush=True)
        g2 = torch.Generator(device=device).manual_seed(args.baseline_seed + 1)
        D_wide = torch.randn(args.rand_wide_N, d_model, generator=g2,
                             device=device, dtype=torch.float32)
        D_wide = D_wide / D_wide.norm(dim=-1, keepdim=True)
        t0 = time.time()
        rec_wide = omp_batched(D_wide, A, args.K_max, K_eval, device,
                               chunk=args.chunk)
        results['rec_err_rand_wide'] = rec_wide.numpy()
        results['rand_wide_N'] = args.rand_wide_N
        print(f'  done in {time.time()-t0:.1f}s', flush=True)

    # --- PCA projection ---
    if args.pca_pool_path:
        print(f'\n[PCA] fitting on pool: {args.pca_pool_path}', flush=True)
        pool_blob = torch.load(args.pca_pool_path, map_location='cpu')
        A_pool = pool_blob['activations']
        print(f'  pool shape={tuple(A_pool.shape)}', flush=True)
        t0 = time.time()
        rec_pca = pca_topk_eval(A_pool, A, K_eval, device)
        results['rec_err_pca'] = rec_pca.numpy()
        print(f'  done in {time.time()-t0:.1f}s', flush=True)

    # --- Qwen-Scope SAE at fixed K ---
    if args.sae_ckpt:
        print(f'\n[SAE] loading {args.sae_ckpt}', flush=True)
        sae = torch.load(args.sae_ckpt, map_location='cpu')
        keys = list(sae.keys())
        # Try common conventions
        def find(*names):
            for n in names:
                if n in sae: return sae[n]
            raise KeyError(f'None of {names} in SAE; keys={keys}')
        W_enc = find('W_enc', 'encoder.weight')
        b_enc = find('b_enc', 'encoder.bias')
        W_dec = find('W_dec', 'decoder.weight')
        b_dec = find('b_dec', 'decoder.bias', 'b_dec_bias')
        print(f'  W_enc={tuple(W_enc.shape)} b_enc={tuple(b_enc.shape)} '
              f'W_dec={tuple(W_dec.shape)} b_dec={tuple(b_dec.shape)}',
              flush=True)
        t0 = time.time()
        rec_sae = sae_topk_eval(W_enc, b_enc, W_dec, b_dec, A,
                                args.sae_K, device)
        results['rec_err_sae'] = rec_sae.numpy()
        results['sae_K_list'] = sorted(args.sae_K)
        results['sae_d_sae'] = int(W_enc.shape[0])
        print(f'  done in {time.time()-t0:.1f}s', flush=True)

    out_pkl = os.path.join(args.out, 'omp_eval.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(results, f)

    # --- Summary table ---
    print('\n=== Summary: mean rec error vs K (lower is better) ===')
    headers = ['K', f'dict (N={N})', f'rand (N={N})']
    if 'rec_err_rand_wide' in results:
        headers.append(f'rand (N={args.rand_wide_N})')
    if 'rec_err_pca' in results:
        headers.append('PCA top-K')
    print('  '.join(f'{h:>16}' for h in headers))
    for i, k in enumerate(K_eval):
        row = [f'{k}',
               f'{rec_dict[i].mean():.4f}',
               f'{rec_rand[i].mean():.4f}']
        if 'rec_err_rand_wide' in results:
            row.append(f'{torch.from_numpy(results["rec_err_rand_wide"])[i].mean():.4f}')
        if 'rec_err_pca' in results:
            row.append(f'{torch.from_numpy(results["rec_err_pca"])[i].mean():.4f}')
        print('  '.join(f'{c:>16}' for c in row))
    if 'rec_err_sae' in results:
        print('\nSAE TopK (fixed K from training):')
        for i, k in enumerate(results['sae_K_list']):
            v = torch.from_numpy(results['rec_err_sae'])[i].mean()
            print(f'  K={k:>3}  mean rec err = {v:.4f}')

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_eval, rec_dict.mean(dim=1).numpy(), 'o-',
            label=f'dict (N={N}, OMP)', color='C0')
    ax.plot(K_eval, rec_rand.mean(dim=1).numpy(), 's--',
            label=f'random (N={N}, OMP)', color='C1')
    if 'rec_err_rand_wide' in results:
        v = torch.from_numpy(results['rec_err_rand_wide']).mean(dim=1).numpy()
        ax.plot(K_eval, v, '^--', label=f'random (N={args.rand_wide_N}, OMP)',
                color='C2')
    if 'rec_err_pca' in results:
        v = torch.from_numpy(results['rec_err_pca']).mean(dim=1).numpy()
        ax.plot(K_eval, v, 'd-.', label='PCA top-K', color='C3')
    if 'rec_err_sae' in results:
        sae_x = results['sae_K_list']
        sae_y = torch.from_numpy(results['rec_err_sae']).mean(dim=1).numpy()
        ax.plot(sae_x, sae_y, '*', markersize=14,
                label=f'Qwen-Scope SAE (N={results["sae_d_sae"]}, TopK)',
                color='C4')
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xlabel('K (atoms / active features)')
    ax.set_ylabel(r'$\|a-\hat a\|^2/\|a\|^2$')
    ax.set_title(f'K-sparse reconstruction  (held-out FineWeb, d={d_model})')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')
    plot_path = os.path.join(args.out, 'rec_err.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nSaved {out_pkl}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
