#!/usr/bin/env python3
"""Loss-recovered (CE-recovered) reconstruction eval for a dictionary built by
build_dictionary.py.

For each held-out FineWeb example we take the layer-`L` residual activation x at
a single position, replace it with a reconstruction x̂, run the rest of the
model, and measure the cross-entropy of the genuine next-token prediction. The
fidelity score is

    loss_recovered = (H* - H0) / (Horig - H0)

  Horig : CE with the true activation x        (clean; best, lowest loss)
  H*    : CE with the reconstruction x̂          (substituted)
  H0    : CE with x zero-ablated (x := 0)       (worst, highest loss)

loss_recovered = 1 → reconstruction is as good as the true activation; 0 → no
better than zeroing the activation. Aggregated as a ratio of mean losses over
the held set (the standard convention).

To have a real next-token target, the substituted activation is the
SECOND-TO-LAST position of a length-T sequence, so its prediction target is the
in-sequence last token. This is the same per-activation granularity the dict was
built/evaluated at; we substitute one position and score that position's
next-token CE (not an all-positions patch).

Sources: our dict (OMP), random same-N (OMP), random SAE-width (OMP), PCA top-K,
Qwen-Scope SAE (fixed TopK). Needs the model → run on a GPU pod.

Usage:
    python scripts/superellipse/eval_dictionary_lossrecovered.py \\
        --dict_path .../dictionary.pt --model qwen3-1.7b --layer 14 \\
        --n_held 1000 --seq_len 16 --held_seed 2024 \\
        --pca_pool_path .../pca_pool_acts.pt \\
        --sae_ckpt .../sae_L0_50_layer14.pt --sae_K 50 100 \\
        --K_eval 8 16 32 64 100 256 --rand_wide_N 32000 \\
        --out .../lossrec_n25k
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
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

from src.model import load_model, forward_from_layer, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.eval_dictionary_kl import (
    forward_batched, omp_reconstruct_at_K, pca_reconstruct_at_K,
    sae_reconstruct_at_K,
)


def extract_ctx_act_target(model, token_ids_list, layer, device, batch_size=64):
    """Layer-`layer` residual at position T-2 (act) + prefix (ctx) + the real
    next token (target = the T-1 token, which act's logits predict).

    Returns ctx [B, T-2, d], act [B, d], target [B] (long).
    """
    all_ctx, all_act, all_tgt = [], [], []
    for i in range(0, len(token_ids_list), batch_size):
        batch = torch.stack(token_ids_list[i:i + batch_size]).to(device)  # [b, T]
        with torch.no_grad():
            out = model(batch, output_hidden_states=True)
            hidden = out.hidden_states[layer + 1]                          # [b, T, d]
        all_ctx.append(hidden[:, :-2, :].float().cpu())
        all_act.append(hidden[:, -2, :].float().cpu())
        all_tgt.append(batch[:, -1].cpu())
    return (torch.cat(all_ctx, 0), torch.cat(all_act, 0),
            torch.cat(all_tgt, 0))


def ce_loss(logits, target):
    """Per-example cross-entropy. logits [B, V], target [B] long → [B]."""
    return F.cross_entropy(logits.float(), target.to(logits.device),
                           reduction='none').cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--seq_len', type=int, default=16)
    ap.add_argument('--n_held', type=int, default=1000)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--K_eval', type=int, nargs='+',
                    default=[8, 16, 32, 64, 100, 256])
    ap.add_argument('--rand_wide_N', type=int, default=32000)
    ap.add_argument('--baseline_seed', type=int, default=0)
    ap.add_argument('--omp_chunk', type=int, default=64)
    ap.add_argument('--fwd_chunk', type=int, default=32)
    ap.add_argument('--pca_pool_path', default=None)
    ap.add_argument('--sae_ckpt', default=None)
    ap.add_argument('--sae_K', type=int, nargs='+', default=[50, 100])
    ap.add_argument('--no_center_omp', action='store_false', dest='center_omp',
                    default=True,
                    help='by default OMP (dict/random) reconstructs the '
                         'mean-subtracted activation and adds the mean back (a '
                         "free bias, like PCA's mu / the SAE's b_dec), so all "
                         'methods are compared apples-to-apples; this disables '
                         'it (raw reconstruction).')
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

    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(device=device, dtype=torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    N = D.shape[0]
    print(f'Dictionary: {N} atoms', flush=True)

    print(f'Extracting {args.n_held} held (ctx, act, target) at L{args.layer} '
          f'seq_len={args.seq_len} seed={args.held_seed}', flush=True)
    tok = load_fineweb_fixed_length(args.n_held, tokenizer,
                                    seq_len=args.seq_len, seed=args.held_seed)
    ctx, act, target = extract_ctx_act_target(model, tok, args.layer, device)
    ctx = ctx.to(device=device, dtype=torch.float32)
    act = act.to(device=device, dtype=torch.float32)
    print(f'  ctx={tuple(ctx.shape)} act={tuple(act.shape)} '
          f'target={tuple(target.shape)} mean||a||={act.norm(dim=-1).mean():.2f}',
          flush=True)

    # --- Clean (Horig) and zero-ablation (H0) endpoints ---
    print('Computing Horig (clean) and H0 (zero-ablation)...', flush=True)
    ref_logits = forward_batched(model, ctx, act, args.layer,
                                 chunk=args.fwd_chunk)
    ce_orig = ce_loss(ref_logits, target)
    zero_logits = forward_batched(model, ctx, torch.zeros_like(act),
                                  args.layer, chunk=args.fwd_chunk)
    ce_zero = ce_loss(zero_logits, target)
    Horig, H0 = float(ce_orig.mean()), float(ce_zero.mean())
    print(f'  Horig={Horig:.4f}  H0(zero)={H0:.4f}  gap={H0-Horig:.4f}',
          flush=True)

    results = {'args': vars(args), 'K_eval': K_eval, 'N': N, 'd_model': d_model,
               'n_held': act.shape[0], 'Horig': Horig, 'H0': H0,
               'ce_orig': ce_orig, 'ce_zero': ce_zero, 'lr': {}}

    denom = Horig - H0  # negative

    # Free mean for the OMP methods (parity with PCA's mu and the SAE's b_dec):
    # reconstruct (a - mu) and add mu back. mu is the pool mean if available
    # (no held leakage), else the held-set mean.
    if args.center_omp:
        if args.pca_pool_path:
            mu_omp = torch.load(args.pca_pool_path, map_location='cpu')[
                'activations'].to(device=device,
                                  dtype=torch.float32).mean(0, keepdim=True)
            mu_src = 'pool'
        else:
            mu_omp = act.mean(0, keepdim=True)
            mu_src = 'held'
        print(f'OMP free-mean: {mu_src} mean (de-mean before OMP)', flush=True)
    else:
        mu_omp = torch.zeros(1, d_model, device=device)
        print('OMP free-mean: OFF (raw reconstruction)', flush=True)

    def omp_recon(Dmat, K):
        return mu_omp + omp_reconstruct_at_K(Dmat, act - mu_omp, K, device,
                                             chunk=args.omp_chunk)

    def eval_source(name, recons_dict):
        """recons_dict {K: tensor[B,d]} → {K: loss_recovered}, also store H*."""
        lr_out, hstar_out = {}, {}
        for K, recon in recons_dict.items():
            logits = forward_batched(model, ctx, recon, args.layer,
                                     chunk=args.fwd_chunk)
            ce = ce_loss(logits, target)
            Hstar = float(ce.mean())
            lr = (Hstar - H0) / denom
            lr_out[K] = lr
            hstar_out[K] = Hstar
            print(f'  [{name} K={K}]  H*={Hstar:.4f}  loss_recovered={lr:.4f}',
                  flush=True)
        return lr_out, hstar_out

    # --- dict (OMP) ---
    print(f'\n[OMP] dict ({N} atoms)', flush=True)
    recs = {K: omp_recon(D, K) for K in K_eval}
    results['lr']['dict'], _ = eval_source('dict', recs)

    # --- random same N ---
    print(f'\n[OMP] random_{N}', flush=True)
    g = torch.Generator(device=device).manual_seed(args.baseline_seed)
    D_rand = torch.randn(N, d_model, generator=g, device=device,
                         dtype=torch.float32)
    D_rand = D_rand / D_rand.norm(dim=-1, keepdim=True)
    recs = {K: omp_recon(D_rand, K) for K in K_eval}
    results['lr']['rand_sameN'], _ = eval_source(f'rand_{N}', recs)

    # --- random wide ---
    if args.rand_wide_N and args.rand_wide_N != N:
        print(f'\n[OMP] random_{args.rand_wide_N}', flush=True)
        g2 = torch.Generator(device=device).manual_seed(args.baseline_seed + 1)
        D_wide = torch.randn(args.rand_wide_N, d_model, generator=g2,
                             device=device, dtype=torch.float32)
        D_wide = D_wide / D_wide.norm(dim=-1, keepdim=True)
        recs = {K: omp_recon(D_wide, K) for K in K_eval}
        results['lr']['rand_wide'], _ = eval_source(f'rand_{args.rand_wide_N}',
                                                    recs)
        results['rand_wide_N'] = args.rand_wide_N

    # --- PCA top-K ---
    A_pool = None
    if args.pca_pool_path:
        A_pool = torch.load(args.pca_pool_path, map_location='cpu')[
            'activations'].to(device=device, dtype=torch.float32)
        print('\n[PCA] top-K projection', flush=True)
        recs = {K: pca_reconstruct_at_K(A_pool, act, K, device) for K in K_eval}
        results['lr']['pca'], _ = eval_source('pca', recs)

    # --- Qwen-Scope SAE ---
    if args.sae_ckpt:
        print(f'\n[SAE] loading {args.sae_ckpt}', flush=True)
        sae = torch.load(args.sae_ckpt, map_location='cpu')
        recs = {K: sae_reconstruct_at_K(sae['W_enc'], sae['b_enc'],
                                        sae['W_dec'], sae['b_dec'], act, K,
                                        device) for K in args.sae_K}
        results['lr']['sae'], _ = eval_source('sae', recs)
        results['sae_K_list'] = sorted(args.sae_K)
        results['sae_d_sae'] = int(sae['W_enc'].shape[0])

    out_pkl = os.path.join(args.out, 'lossrec_eval.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(results, f)

    # --- Summary ---
    print('\n=== loss recovered vs K (1=clean, 0=zero-ablation) ===')
    print(f'  Horig={Horig:.4f}  H0={H0:.4f}')
    for name in ['dict', 'rand_sameN', 'rand_wide', 'pca']:
        if name in results['lr']:
            row = '  '.join(f'K{k}:{results["lr"][name][k]:.3f}'
                            for k in K_eval)
            print(f'  {name:12s} {row}')
    if 'sae' in results['lr']:
        row = '  '.join(f'K{k}:{results["lr"]["sae"][k]:.3f}'
                        for k in results['sae_K_list'])
        print(f'  {"sae":12s} {row}')

    # --- Plot (integer K tick labels) ---
    lr = results['lr']
    fig, ax = plt.subplots(figsize=(8, 5))
    style = {'dict': ('o-', 'C0', f'dict (N={N}, OMP)'),
             'rand_sameN': ('s--', 'C1', f'random (N={N}, OMP)'),
             'rand_wide': ('^--', 'C2', f'random (N={args.rand_wide_N}, OMP)'),
             'pca': ('d-.', 'C3', 'PCA top-K')}
    for name, (mk, col, lab) in style.items():
        if name in lr:
            ax.plot(K_eval, [lr[name][k] for k in K_eval], mk, color=col,
                    label=lab)
    if 'sae' in lr:
        ax.plot(results['sae_K_list'],
                [lr['sae'][k] for k in results['sae_K_list']], '*',
                markersize=14, color='C4',
                label=f'Qwen-Scope SAE (N={results["sae_d_sae"]}, TopK)')
    ax.axhline(1.0, color='k', ls=':', lw=1, alpha=0.6, label='clean (=1)')
    ax.set_xscale('log', base=2)
    xticks = sorted(set(list(K_eval) + list(results.get('sae_K_list', []))))
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(k) for k in xticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('K (atoms / active features)')
    ax.set_ylabel('loss recovered  (H*-H0)/(Horig-H0)')
    _fair = 'free-mean OMP' if args.center_omp else 'raw OMP'
    ax.set_title(f'CE loss recovered  ({args.model} L{args.layer}->end, '
                 f'single-pos; {_fair})')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')
    plot_path = os.path.join(args.out, 'loss_recovered.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nSaved {out_pkl}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
