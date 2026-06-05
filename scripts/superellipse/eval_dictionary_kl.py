#!/usr/bin/env python3
"""KL-divergence reconstruction eval for a dictionary built by build_dictionary.py.

For each held-out FineWeb (ctx, act) pair:
  reference_logits = model.forward_from_layer(ctx, act, layer)
  for each (source, K):
      recon = reconstruct(act, source, K)
      recon_logits = model.forward_from_layer(ctx, recon, layer)
      KL_per_example = KL(softmax(ref_logits) || softmax(recon_logits))

Sources:
  - our dictionary (OMP at each K in K_eval)
  - random unit dictionary, same width (OMP)
  - random unit dictionary, SAE width (OMP)
  - PCA top-K projection
  - Qwen-Scope SAE (fixed K from training)

Sanity baselines:
  - "identity" (recon = act)  → KL ≈ 0
  - "mean" (recon = mean activation from PCA pool)  → "naive" upper bound

Needs the model loaded → must run on GPU pod.

Usage:
    python scripts/superellipse/eval_dictionary_kl.py \\
        --dict_path .../dictionary_K8000.pt \\
        --model qwen3-1.7b --layer 14 --n_held 500 --held_seed 2024 \\
        --pca_pool_path .../pca_pool_acts.pt \\
        --sae_ckpt .../layer14.sae.pt --sae_K 50 100 \\
        --K_eval 8 16 32 64 100 256 --rand_wide_N 32000 \\
        --out .../eval_kl_K8000
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)
from scripts.superellipse.eval_dictionary_omp import omp_batched


def kl_pq(logits_p, logits_q):
    """KL(softmax(p) || softmax(q)) per row. logits_p, logits_q: [B, V]."""
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1)  # [B]


def forward_batched(model, ctx_batch, points_batch, layer, chunk=32):
    """Forward [B, T-1, d] contexts paired with [B, 1, d] points.

    `forward_from_layer` takes a SINGLE ctx + batched points; loop over batch.
    Returns logits [B, V].
    """
    B = ctx_batch.shape[0]
    model_dtype = next(model.parameters()).dtype
    out = None
    for i in range(B):
        ctx_i = ctx_batch[i:i + 1].to(dtype=model_dtype)        # [1, T-1, d]
        pt_i = points_batch[i:i + 1].unsqueeze(0).to(dtype=model_dtype)  # [1, 1, d] -> [1,1,d]
        # Note: forward_from_layer expects point of shape [n_points, 1, d]
        pt_i = points_batch[i:i + 1].unsqueeze(1).to(dtype=model_dtype)  # [1, 1, d]
        logits = forward_from_layer(model, ctx_i, pt_i, layer)  # [1, V]
        if out is None:
            out = torch.empty(B, logits.shape[-1], device=logits.device,
                              dtype=torch.float32)
        out[i] = logits[0].float()
    return out


def omp_reconstruct_at_K(D, A, K, device, chunk=64):
    """Like omp_batched but returns the reconstruction at exactly one K (not error)."""
    B, d = A.shape
    A_dev = A.to(device=device, dtype=torch.float32)
    D_dev = D.to(device=device, dtype=torch.float32)
    recon_all = torch.zeros_like(A_dev)
    for b0 in range(0, B, chunk):
        b1 = min(B, b0 + chunk)
        Ab = A_dev[b0:b1]
        Rb = Ab.clone()
        sel = torch.full((Ab.shape[0], K), -1, dtype=torch.long, device=device)
        for k in range(K):
            proj = Rb @ D_dev.T
            if k > 0:
                proj.scatter_(1, sel[:, :k], 0.0)
            j = proj.abs().argmax(dim=-1)
            sel[:, k] = j
            D_sel = D_dev[sel[:, :k + 1]]
            G = D_sel @ D_sel.transpose(-1, -2)
            rhs = (D_sel * Ab.unsqueeze(1)).sum(dim=-1)
            G = G + 1e-7 * torch.eye(k + 1, device=device)
            coeffs = torch.linalg.solve(G, rhs)
            recon = (coeffs.unsqueeze(-1) * D_sel).sum(dim=1)
            Rb = Ab - recon
        recon_all[b0:b1] = Ab - Rb
    return recon_all


def pca_reconstruct_at_K(A_pool, A_held, K, device):
    A_pool = A_pool.to(device=device, dtype=torch.float32)
    A_held = A_held.to(device=device, dtype=torch.float32)
    mu = A_pool.mean(dim=0, keepdim=True)
    Xc = A_pool - mu
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    Vk = Vh[:K]                                                # [K, d]
    Hc = A_held - mu
    coefs = Hc @ Vk.T
    return coefs @ Vk + mu                                     # [B, d]


def sae_reconstruct_at_K(W_enc, b_enc, W_dec, b_dec, A_held, K, device):
    A_held = A_held.to(device=device, dtype=torch.float32)
    W_enc = W_enc.to(device=device, dtype=torch.float32)
    b_enc = b_enc.to(device=device, dtype=torch.float32)
    W_dec = W_dec.to(device=device, dtype=torch.float32)
    b_dec = b_dec.to(device=device, dtype=torch.float32)
    d_sae = W_enc.shape[0]
    d = A_held.shape[1]
    if W_dec.shape == (d, d_sae):
        W_dec_use = W_dec
    elif W_dec.shape == (d_sae, d):
        W_dec_use = W_dec.T
    else:
        raise ValueError(f'Unexpected W_dec shape {tuple(W_dec.shape)}')
    pre = A_held @ W_enc.T + b_enc
    topv, topi = pre.topk(K, dim=-1)
    acts = torch.zeros_like(pre)
    acts.scatter_(-1, topi, topv)
    return acts @ W_dec_use.T + b_dec                          # [B, d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_held', type=int, default=500)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--K_eval', type=int, nargs='+',
                    default=[8, 16, 32, 64, 100, 256])
    ap.add_argument('--rand_wide_N', type=int, default=32000)
    ap.add_argument('--baseline_seed', type=int, default=0)
    ap.add_argument('--omp_chunk', type=int, default=64)
    ap.add_argument('--fwd_chunk', type=int, default=32)
    ap.add_argument('--pca_pool_path', default=None,
                    help='activations .pt for PCA fitting + mean-baseline')
    ap.add_argument('--sae_ckpt', default=None)
    ap.add_argument('--sae_K', type=int, nargs='+', default=[50, 100])
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    K_eval = sorted(args.K_eval)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    d_model = model._plateau_info['d_model']
    print(f'Device: {device} d_model={d_model} n_layers={n_layers} '
          f'layer={args.layer}', flush=True)

    # Load dictionary
    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(device=device, dtype=torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    N = D.shape[0]
    print(f'Dictionary: {N} atoms, d={d_model}', flush=True)

    # Held-out: extract both ctx and act at the layer
    print(f'Extracting {args.n_held} held-out (ctx, act) at L{args.layer} '
          f'(seed={args.held_seed})', flush=True)
    tok = load_fineweb_fixed_length(args.n_held, tokenizer,
                                    seq_len=args.seq_len,
                                    seed=args.held_seed)
    ctx, act = extract_context_and_activation(model, tok, args.layer, device)
    # ctx: [B, T-1, d]    act: [B, d]
    ctx = ctx.to(device=device, dtype=torch.float32)
    act = act.to(device=device, dtype=torch.float32)
    print(f'  ctx={tuple(ctx.shape)}  act={tuple(act.shape)}  '
          f'mean ||a||={act.norm(dim=-1).mean().item():.2f}', flush=True)

    # --- Reference logits ---
    print('Computing reference logits...', flush=True)
    t0 = time.time()
    ref_logits = forward_batched(model, ctx, act, args.layer,
                                 chunk=args.fwd_chunk)
    print(f'  done in {time.time()-t0:.1f}s   ref_logits={tuple(ref_logits.shape)}',
          flush=True)

    results = {'args': vars(args), 'K_eval': K_eval, 'N': N,
               'd_model': d_model, 'n_held': act.shape[0]}

    def eval_source(name, recons_dict):
        """recons_dict: {K: tensor[B, d]} → mean KL per K."""
        out = {}
        for K, recon in recons_dict.items():
            logits = forward_batched(model, ctx, recon, args.layer,
                                     chunk=args.fwd_chunk)
            kl = kl_pq(ref_logits, logits).cpu().numpy()
            out[K] = kl
            print(f'  [{name} K={K}]  mean KL={kl.mean():.4f}  '
                  f'med={np.median(kl):.4f}  '
                  f'p90={np.quantile(kl, 0.9):.4f}', flush=True)
        return out

    # --- Identity / mean baselines ---
    print('\n[sanity] identity (recon = act):', flush=True)
    eval_source('identity', {0: act})

    # Pre-load PCA pool for both PCA recon and mean baseline
    A_pool = None
    if args.pca_pool_path:
        pool_blob = torch.load(args.pca_pool_path, map_location='cpu')
        A_pool = pool_blob['activations'].to(device=device,
                                              dtype=torch.float32)
        mean_act = A_pool.mean(dim=0, keepdim=True).expand(act.shape[0], -1)
        print('\n[sanity] mean (recon = pool mean):', flush=True)
        eval_source('mean', {0: mean_act})

    # --- Our dictionary (OMP) ---
    print(f'\n[OMP] dict ({N} atoms)', flush=True)
    recs = {K: omp_reconstruct_at_K(D, act, K, device, chunk=args.omp_chunk)
            for K in K_eval}
    results['kl_dict'] = eval_source('dict', recs)

    # --- Random same N ---
    print(f'\n[OMP] random_{N}', flush=True)
    g = torch.Generator(device=device).manual_seed(args.baseline_seed)
    D_rand = torch.randn(N, d_model, generator=g, device=device,
                         dtype=torch.float32)
    D_rand = D_rand / D_rand.norm(dim=-1, keepdim=True)
    recs = {K: omp_reconstruct_at_K(D_rand, act, K, device,
                                    chunk=args.omp_chunk) for K in K_eval}
    results['kl_rand_sameN'] = eval_source(f'rand_{N}', recs)

    # --- Random wide ---
    if args.rand_wide_N and args.rand_wide_N != N:
        print(f'\n[OMP] random_{args.rand_wide_N}', flush=True)
        g2 = torch.Generator(device=device).manual_seed(args.baseline_seed + 1)
        D_wide = torch.randn(args.rand_wide_N, d_model, generator=g2,
                             device=device, dtype=torch.float32)
        D_wide = D_wide / D_wide.norm(dim=-1, keepdim=True)
        recs = {K: omp_reconstruct_at_K(D_wide, act, K, device,
                                        chunk=args.omp_chunk) for K in K_eval}
        results['kl_rand_wide'] = eval_source(f'rand_{args.rand_wide_N}', recs)
        results['rand_wide_N'] = args.rand_wide_N

    # --- PCA top-K ---
    if A_pool is not None:
        print('\n[PCA] top-K projection', flush=True)
        recs = {K: pca_reconstruct_at_K(A_pool, act, K, device)
                for K in K_eval}
        results['kl_pca'] = eval_source('pca', recs)

    # --- Qwen-Scope SAE ---
    if args.sae_ckpt:
        print(f'\n[SAE] loading {args.sae_ckpt}', flush=True)
        sae = torch.load(args.sae_ckpt, map_location='cpu')
        W_enc = sae['W_enc']; b_enc = sae['b_enc']
        W_dec = sae['W_dec']; b_dec = sae['b_dec']
        print(f'  W_enc={tuple(W_enc.shape)} W_dec={tuple(W_dec.shape)}',
              flush=True)
        recs = {K: sae_reconstruct_at_K(W_enc, b_enc, W_dec, b_dec, act,
                                        K, device) for K in args.sae_K}
        results['kl_sae'] = eval_source('sae', recs)
        results['sae_K_list'] = sorted(args.sae_K)
        results['sae_d_sae'] = int(W_enc.shape[0])

    # --- Save ---
    out_pkl = os.path.join(args.out, 'kl_eval.pkl')
    # Convert tensor logits to nothing — just keep numpy arrays in `results`
    with open(out_pkl, 'wb') as f:
        pickle.dump(results, f)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    def line(d, label, marker, color):
        ks = sorted(d.keys())
        ys = [d[k].mean() for k in ks]
        ax.plot(ks, ys, marker, label=label, color=color)

    line(results['kl_dict'], f'dict (N={N}, OMP)', 'o-', 'C0')
    line(results['kl_rand_sameN'], f'random (N={N}, OMP)', 's--', 'C1')
    if 'kl_rand_wide' in results:
        line(results['kl_rand_wide'],
             f'random (N={args.rand_wide_N}, OMP)', '^--', 'C2')
    if 'kl_pca' in results:
        line(results['kl_pca'], 'PCA top-K', 'd-.', 'C3')
    if 'kl_sae' in results:
        ks = results['sae_K_list']
        ys = [results['kl_sae'][k].mean() for k in ks]
        ax.plot(ks, ys, '*', markersize=14,
                label=f'Qwen-Scope SAE (N={results["sae_d_sae"]}, TopK)',
                color='C4')

    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xlabel('K (atoms / active features)')
    ax.set_ylabel(r'mean $KL(p_{\mathrm{ref}}\,\|\,p_{\mathrm{recon}})$')
    ax.set_title(f'KL reconstruction  ({args.model} L{args.layer}->end)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')
    plot_path = os.path.join(args.out, 'kl_rec.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nSaved {out_pkl}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
