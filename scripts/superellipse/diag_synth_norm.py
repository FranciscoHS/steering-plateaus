#!/usr/bin/env python3
"""Diagnostic: decompose the synthetic-activation feature-part norm per carrier
basis, to see whether the dict-vs-SAE norm gap is driven by carrier alignment.

feature_part = Σ_i m_i u_i   (u_i unit carriers, m_i shared magnitudes)
||feature_part||^2 = Σ m_i^2  +  Σ_{i≠j} m_i m_j cos(u_i,u_j)
                   = S1 (ortho baseline^2)  +  C (cross term)

weighted mean pairwise cos = C / [ (Σ m_i)^2 - Σ m_i^2 ]

Reuses the SAME anchors/seeds as eval_synth_basis_plateau.py.
"""
import argparse
import os
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks
from scripts.superellipse.eval_real_vs_random_acts import fineweb_ctx_acts


def decompose(mags, carriers_pool, sae_K, n_act, seed, device):
    """Per anchor: pick sae_K random unit carriers, decompose the norm.

    Returns dict of mean-over-anchor stats.
    """
    P = carriers_pool.shape[0]
    rng = np.random.default_rng(seed)
    ortho, actual, wcos, mean_abs_cos, mean_signed_cos = [], [], [], [], []
    for i in range(n_act):
        pick = rng.choice(P, size=sae_K, replace=(P < sae_K))
        u = carriers_pool[torch.as_tensor(pick, device=device)]      # [K, d] unit
        m = mags[i]                                                  # [K]
        S1 = (m ** 2).sum()                                          # ortho^2
        feat = (m.unsqueeze(-1) * u).sum(0)
        act_norm = feat.norm()
        cross = act_norm ** 2 - S1
        denom = m.sum() ** 2 - S1                                    # Σ_{i≠j} m_i m_j
        wcos.append(float(cross / denom.clamp_min(1e-9)))
        ortho.append(float(S1.sqrt()))
        actual.append(float(act_norm))
        # raw pairwise cosines among carriers (unweighted)
        G = u @ u.T                                                  # [K,K]
        off = ~torch.eye(sae_K, dtype=torch.bool, device=device)
        mean_abs_cos.append(float(G[off].abs().mean()))
        mean_signed_cos.append(float(G[off].mean()))
    return {'ortho_baseline': np.mean(ortho), 'actual_norm': np.mean(actual),
            'weighted_mean_cos': np.mean(wcos),
            'mean_abs_cos': np.mean(mean_abs_cos),
            'mean_signed_cos': np.mean(mean_signed_cos)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--sae_ckpt', required=True)
    ap.add_argument('--sae_K', type=int, default=50)
    ap.add_argument('--dedup_cache', required=True)
    ap.add_argument('--n_act', type=int, default=30)
    ap.add_argument('--seq_len', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--synth_seed', type=int, default=0)
    args = ap.parse_args()

    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)
    _ = len(_get_blocks(model))
    _, activations = fineweb_ctx_acts(model, tokenizer, device, args.layer,
                                      args.n_act, args.seq_len, args.seed)

    sae = torch.load(args.sae_ckpt, map_location='cpu')
    W_enc = sae['W_enc'].float(); b_enc = sae['b_enc'].float()
    W_dec = sae['W_dec'].float()
    dec_norm = W_dec.norm(dim=0)
    pre = activations.float() @ W_enc.T + b_enc
    topv, topi = pre.topk(args.sae_K, dim=-1)
    mags = (topv * dec_norm[topi]).to(device)

    sae_carriers = (W_dec / dec_norm.clamp_min(1e-12)).T.to(device)
    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].float()
    D = (D / (D.norm(dim=-1, keepdim=True) + 1e-12)).to(device)
    dedup_idx = torch.load(args.dedup_cache)['kept_idx'].to(device)
    D_dedup = D[dedup_idx]

    print(f'\nanchors={args.n_act}  sae_K={args.sae_K}  '
          f'(no b_dec; feature-part only)\n')
    print(f'{"case":12s} {"ortho√Σm²":>10s} {"actual":>8s} {"ratio":>6s} '
          f'{"w_cos":>8s} {"|cos|":>7s} {"signed_cos":>11s}')
    for name, pool, sd in [('sae', sae_carriers, args.synth_seed + 1),
                           ('dict', D, args.synth_seed + 2),
                           ('dict_dedup', D_dedup, args.synth_seed + 3)]:
        s = decompose(mags, pool, args.sae_K, args.n_act, sd, device)
        print(f'{name:12s} {s["ortho_baseline"]:10.1f} '
              f'{s["actual_norm"]:8.1f} '
              f'{s["actual_norm"]/s["ortho_baseline"]:6.3f} '
              f'{s["weighted_mean_cos"]:8.4f} {s["mean_abs_cos"]:7.4f} '
              f'{s["mean_signed_cos"]:11.4f}')


if __name__ == '__main__':
    main()
