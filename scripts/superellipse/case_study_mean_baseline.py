#!/usr/bin/env python3
"""Mean-only baseline for the loss-recovered case studies.

How good is the loss if we substitute JUST the pool mean (recon = mu, zero
atoms)? If lr_mean is already high for an example, the OMP atoms aren't doing
much there — the activation is simply close to the mean.

Loads ce_orig / ce_zero / lr (the K-atom OMP one) from an existing
case_studies.pkl (same held set: deterministic FineWeb cache by seed), runs
one forward pass with recon = mu for all held examples, and reports
per-example lr_mean overall + for the stored cases.

    HF_HOME=/workspace/hf_cache python -u \\
        scripts/superellipse/case_study_mean_baseline.py \\
        --case_pkl .../case_studies_K16/case_studies.pkl \\
        --pca_pool_path .../pca_pool_acts.pt \\
        --out .../case_studies_K16
"""
import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch

from src.model import load_model
from src.data import load_fineweb_fixed_length
from scripts.superellipse.eval_dictionary_kl import forward_batched
from scripts.superellipse.eval_dictionary_lossrecovered import (
    extract_ctx_act_target, ce_loss,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case_pkl', required=True)
    ap.add_argument('--pca_pool_path', required=True)
    ap.add_argument('--fwd_chunk', type=int, default=32)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    cs = pickle.load(open(args.case_pkl, 'rb'))
    cargs = cs['args']
    ce_orig, ce_zero = cs['ce_orig'], cs['ce_zero']
    lr_omp, gap = cs['lr'], cs['gap']

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {cargs["model"]} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(cargs['model'],
                                          dtype=dtype_map[args.dtype])

    print(f'Extracting held set (n={cargs["n_held"]} seq_len={cargs["seq_len"]} '
          f'seed={cargs["held_seed"]})', flush=True)
    tok = load_fineweb_fixed_length(cargs['n_held'], tokenizer,
                                    seq_len=cargs['seq_len'],
                                    seed=cargs['held_seed'])
    ctx, act, target = extract_ctx_act_target(model, tok, cargs['layer'],
                                              device)
    ctx = ctx.to(device=device, dtype=torch.float32)
    act = act.to(device=device, dtype=torch.float32)
    assert act.shape[0] == len(ce_orig), 'held set mismatch with case pkl'

    mu = torch.load(args.pca_pool_path, map_location='cpu')[
        'activations'].to(device=device, dtype=torch.float32).mean(
        0, keepdim=True)
    recon = mu.expand(act.shape[0], -1)

    # Sanity: recomputed clean CE should match the stored one.
    ce_orig_chk = ce_loss(forward_batched(model, ctx, act, cargs['layer'],
                                          chunk=args.fwd_chunk), target)
    print(f'ce_orig check: stored mean={ce_orig.mean():.4f}  '
          f'recomputed={ce_orig_chk.mean():.4f}', flush=True)

    print('Forward with recon = mu (mean only, zero atoms)...', flush=True)
    ce_mean = ce_loss(forward_batched(model, ctx, recon, cargs['layer'],
                                      chunk=args.fwd_chunk), target)

    lr_mean = (ce_mean - ce_zero) / (ce_orig - ce_zero + 1e-12)
    eligible = gap > cargs['min_gap']

    print('\n=== mean-only vs OMP-K{} loss recovered ==='.format(cargs['K']))
    print(f'  aggregate (ratio of means, eligible n={int(eligible.sum())}):')
    Ho, Hz = ce_orig[eligible].mean(), ce_zero[eligible].mean()
    Hm = ce_mean[eligible].mean()
    print(f'    Horig={Ho:.4f}  H0={Hz:.4f}  Hmean={Hm:.4f}  '
          f'lr_mean={(Hm - Hz) / (Ho - Hz):.4f}', flush=True)
    print(f'  per-example lr_mean over eligible: '
          f'mean={lr_mean[eligible].mean():.4f}  '
          f'median={np.median(lr_mean[eligible]):.4f}', flush=True)
    print(f'  per-example lr_omp  over eligible: '
          f'mean={lr_omp[eligible].mean():.4f}  '
          f'median={np.median(lr_omp[eligible]):.4f}', flush=True)

    print('\n  stored cases:')
    for c in cs['cases']:
        i = c['idx']
        print(f'    example {i}:  lr_omp={c["lr"]:.4f}  '
              f'lr_mean={lr_mean[i]:.4f}  ce_orig={ce_orig[i]:.3f}  '
              f'ce_mean={ce_mean[i]:.3f}  ce_recon={c["ce_recon"]:.3f}  '
              f'ce_zero={ce_zero[i]:.3f}', flush=True)

    out_pkl = os.path.join(args.out, 'mean_baseline.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({'args': vars(args), 'ce_mean': ce_mean,
                     'lr_mean': lr_mean}, f)
    print(f'\nSaved {out_pkl}', flush=True)


if __name__ == '__main__':
    main()
