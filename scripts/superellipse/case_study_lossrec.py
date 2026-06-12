#!/usr/bin/env python3
"""Case studies: held-out activations best reconstructed by the dictionary.

Reuses the loss-recovered eval setup (same held set: FineWeb seq_len-16,
seed 2024, activation at position T-2 of layer L). For every held example we
compute the PER-EXAMPLE loss recovered at a single sparsity K:

    lr_i = (ce_recon_i - ce_zero_i) / (ce_orig_i - ce_zero_i)

with the same free-mean OMP convention as eval_dictionary_lossrecovered.py
(reconstruct a - mu_pool, add mu_pool back). Examples where zero-ablation
barely hurts (ce_zero - ce_orig < min_gap) are excluded — lr is meaningless
when the denominator is tiny.

For the top --n_cases examples we record:
  - the decoded context, the substituted position and the true next token
  - ce_orig / ce_zero / ce_recon and lr
  - the K OMP-selected atoms in selection order, their coefficients, and each
    atom's marginal residual drop (how much ||r||^2 fell when it was added)

Outputs: case_studies.pkl + a human-readable report.md, plus the union of
atom indices across cases (printed + stored) for feeding straight into
build_neuronpedia_dashboard.py --atoms.

Run on a GPU pod:
    HF_HOME=/workspace/hf_cache python -u \\
        scripts/superellipse/case_study_lossrec.py \\
        --dict_path .../dictionary.pt --model qwen3-1.7b --layer 14 \\
        --K 16 --n_cases 3 --pca_pool_path .../pca_pool_acts.pt \\
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


def omp_with_trace(D, A, K, device, chunk=64):
    """OMP recording the selected atoms + final coefficients.

    Returns recon [B, d], sel [B, K] (selection order), coeffs [B, K]
    (least-squares coefficients of the FINAL K-atom fit, in selection order),
    res_sq [B, K+1] (residual ||r||^2 after 0..K atoms).
    """
    B, d = A.shape
    A = A.to(device=device, dtype=torch.float32)
    D = D.to(device=device, dtype=torch.float32)
    recon_all = torch.zeros_like(A)
    sel_all = torch.zeros(B, K, dtype=torch.long)
    coef_all = torch.zeros(B, K)
    res_all = torch.zeros(B, K + 1)
    for b0 in range(0, B, chunk):
        b1 = min(B, b0 + chunk)
        Ab = A[b0:b1]
        Rb = Ab.clone()
        res_all[b0:b1, 0] = (Rb ** 2).sum(-1).cpu()
        sel = torch.full((Ab.shape[0], K), -1, dtype=torch.long, device=device)
        coeffs = None
        for k in range(K):
            proj = Rb @ D.T
            if k > 0:
                proj.scatter_(1, sel[:, :k], 0.0)
            j = proj.abs().argmax(dim=-1)
            sel[:, k] = j
            D_sel = D[sel[:, :k + 1]]
            G = D_sel @ D_sel.transpose(-1, -2)
            rhs = (D_sel * Ab.unsqueeze(1)).sum(dim=-1)
            G = G + 1e-7 * torch.eye(k + 1, device=device)
            coeffs = torch.linalg.solve(G, rhs)
            recon = (coeffs.unsqueeze(-1) * D_sel).sum(dim=1)
            Rb = Ab - recon
            res_all[b0:b1, k + 1] = (Rb ** 2).sum(-1).cpu()
        recon_all[b0:b1] = Ab - Rb
        sel_all[b0:b1] = sel.cpu()
        coef_all[b0:b1] = coeffs.cpu()
    return recon_all, sel_all, coef_all, res_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--seq_len', type=int, default=16)
    ap.add_argument('--n_held', type=int, default=1000)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--n_cases', type=int, default=3)
    ap.add_argument('--min_gap', type=float, default=1.0,
                    help='min (ce_zero - ce_orig) in nats for an example to '
                         'be eligible (lr denominator)')
    ap.add_argument('--pca_pool_path', default=None,
                    help='pool .pt for the free-mean mu (else held mean)')
    ap.add_argument('--omp_chunk', type=int, default=64)
    ap.add_argument('--fwd_chunk', type=int, default=32)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])

    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(device=device, dtype=torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    N, d_model = D.shape
    print(f'Dictionary: {N} atoms, d={d_model}', flush=True)

    print(f'Extracting {args.n_held} held (ctx, act, target) at L{args.layer} '
          f'seq_len={args.seq_len} seed={args.held_seed}', flush=True)
    tok = load_fineweb_fixed_length(args.n_held, tokenizer,
                                    seq_len=args.seq_len, seed=args.held_seed)
    tok_ids = torch.stack(tok)                                  # [B, T]
    ctx, act, target = extract_ctx_act_target(model, tok, args.layer, device)
    ctx = ctx.to(device=device, dtype=torch.float32)
    act = act.to(device=device, dtype=torch.float32)
    B = act.shape[0]
    print(f'  ctx={tuple(ctx.shape)} act={tuple(act.shape)}', flush=True)

    # Endpoints
    print('Computing per-example Horig / H0...', flush=True)
    ce_orig = ce_loss(forward_batched(model, ctx, act, args.layer,
                                      chunk=args.fwd_chunk), target)
    ce_zero = ce_loss(forward_batched(model, ctx, torch.zeros_like(act),
                                      args.layer, chunk=args.fwd_chunk), target)

    # Free mean (same convention as the lossrec eval)
    if args.pca_pool_path:
        mu = torch.load(args.pca_pool_path, map_location='cpu')[
            'activations'].to(device=device, dtype=torch.float32).mean(
            0, keepdim=True)
        mu_src = 'pool'
    else:
        mu = act.mean(0, keepdim=True)
        mu_src = 'held'
    print(f'Free-mean: {mu_src}', flush=True)

    print(f'OMP (K={args.K}) with trace...', flush=True)
    recon, sel, coef, res_sq = omp_with_trace(D, act - mu, args.K, device,
                                              chunk=args.omp_chunk)
    recon = recon + mu

    ce_recon = ce_loss(forward_batched(model, ctx, recon, args.layer,
                                       chunk=args.fwd_chunk), target)

    gap = ce_zero - ce_orig
    lr = (ce_recon - ce_zero) / (ce_orig - ce_zero + 1e-12)
    eligible = gap > args.min_gap
    print(f'Eligible (gap > {args.min_gap}): {int(eligible.sum())}/{B}',
          flush=True)
    print(f'Mean lr over eligible: {lr[eligible].mean():.4f}  '
          f'(all: {lr.mean():.4f})', flush=True)

    rank = np.argsort(-np.where(eligible, lr, -np.inf))
    cases = []
    lines = [f'# Case studies — best loss-recovered examples (K={args.K})',
             '',
             f'- dict: `{args.dict_path}` ({N} atoms)',
             f'- held: n={B} seq_len={args.seq_len} seed={args.held_seed}, '
             f'free-mean={mu_src}, eligibility gap > {args.min_gap} nats',
             f'- mean lr over eligible: {lr[eligible].mean():.4f}', '']
    for rank_i, i in enumerate(rank[:args.n_cases]):
        i = int(i)
        toks = [tokenizer.decode([int(t)]) for t in tok_ids[i]]
        # substituted position is T-2; target is the in-sequence last token
        text_marked = (''.join(toks[:-2]) + '[[' + toks[-2] + ']]' + toks[-1])
        atoms = sel[i].tolist()
        coefs = coef[i].tolist()
        drops = (res_sq[i, :-1] - res_sq[i, 1:]).tolist()  # per-step ||r||² drop
        cases.append({'idx': i, 'lr': float(lr[i]), 'gap': float(gap[i]),
                      'ce_orig': float(ce_orig[i]), 'ce_zero': float(ce_zero[i]),
                      'ce_recon': float(ce_recon[i]),
                      'tokens': toks, 'token_ids': tok_ids[i].tolist(),
                      'atoms': atoms, 'coeffs': coefs, 'res_drops': drops,
                      'res_sq': res_sq[i].tolist()})
        lines += [f'## Case {rank_i + 1}: held example {i}',
                  f'- lr = {lr[i]:.4f}   ce_orig={ce_orig[i]:.3f}  '
                  f'ce_recon={ce_recon[i]:.3f}  ce_zero={ce_zero[i]:.3f}',
                  f'- text (substituted token in [[...]], last token is the '
                  f'prediction target):',
                  f'  > {text_marked!r}',
                  f'- atoms (selection order; coeff = final LS coefficient, '
                  f'drop = residual energy removed at that step):']
        for k, (a, c, dr) in enumerate(zip(atoms, coefs, drops)):
            frac = dr / res_sq[i, 0].item()
            lines.append(f'  - step {k:2d}: atom {a:5d}  coeff={c:+9.2f}  '
                         f'drop={frac*100:5.1f}% of ||a-mu||^2')
        lines.append('')

    union = sorted({a for c in cases for a in c['atoms']})
    lines += ['## Atom union (for dashboard)',
              '- ' + ','.join(str(a) for a in union)]
    print('\nAtom union across cases: ' + ','.join(str(a) for a in union),
          flush=True)

    with open(os.path.join(args.out, 'case_studies.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'cases': cases, 'atom_union': union,
                     'ce_orig': ce_orig, 'ce_zero': ce_zero,
                     'ce_recon': ce_recon, 'lr': lr, 'gap': gap,
                     'sel': sel.numpy(), 'coef': coef.numpy()}, f)
    with open(os.path.join(args.out, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved {os.path.join(args.out, 'case_studies.pkl')}", flush=True)
    print(f"Saved {os.path.join(args.out, 'report.md')}", flush=True)


if __name__ == '__main__':
    main()
