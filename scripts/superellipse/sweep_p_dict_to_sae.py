#!/usr/bin/env python3
"""p as we rotate optimized directions toward their best-match SAE latents.

Setup: take the optimized atoms whose best-match SAE-decoder cosine exceeds
--match_thresh (the "close hits" from dict_sae_alignment.py). Form within-set
pairs (dᵢ, dⱼ), keeping only pairs that are nearly orthogonal to each other
(within-pair |cos| < --within_cos) and whose two best-match latents differ.
For each pair, slerp BOTH legs toward their own best-match SAE latent:
    u(t) = slerp(dᵢ, sᵢ, t),   v(t) = slerp(dⱼ, sⱼ, t)
and fit the superellipse exponent p of (u(t), v(t)) at each t with the standard
LLM estimator (pair_p). t=0 is our-direction × our-direction; t=1 is
SAE-latent × SAE-latent.

Prediction: p(t=0) (our pairs) >> p(t=1) (SAE pairs), with a monotonic decay.

Run on a GPU pod.

Usage:
    HF_HOME=/workspace/hf_cache python -u \\
      scripts/superellipse/sweep_p_dict_to_sae.py \\
      --dict_path .../dictionary_snapshot.pt \\
      --sae_ckpt  .../sae_L0_50_layer14.pt \\
      --align_pkl .../sae_align_L0_50/sae_alignment.pkl \\
      --match_thresh 0.5 --within_cos 0.1 --n_pairs 50 --n_t 7 \\
      --model qwen3-1.7b --layer 14 --n_anchors 20 \\
      --out .../p_dict_to_sae
"""
import argparse
import math
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

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)
from scripts.superellipse.fit_p_fixed_direction import pair_p
from scripts.superellipse.sweep_p_vs_rotation_llm import slerp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--sae_ckpt', required=True)
    ap.add_argument('--align_pkl', required=True,
                    help='sae_alignment.pkl (found_max_cos, found_max_idx)')
    ap.add_argument('--match_thresh', type=float, default=0.5,
                    help='keep atoms whose best-match SAE cos exceeds this')
    ap.add_argument('--within_cos', type=float, default=0.1,
                    help='keep pairs with within-pair |cos| below this')
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_pairs', type=int, default=50)
    ap.add_argument('--n_t', type=int, default=7)
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--n_steps_grid', type=int, default=31)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--pair_seed', type=int, default=0)
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
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    print(f'Device: {device}  L{args.layer}->{measure_layer}', flush=True)

    # --- optimized directions ---
    blob = torch.load(args.dict_path, map_location='cpu', weights_only=False)
    D = blob['dictionary'].to(torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    N, d_model = D.shape

    # --- SAE decoder directions (unit) ---
    sae = torch.load(args.sae_ckpt, map_location='cpu', weights_only=False)
    W_dec = sae['W_dec'].to(torch.float32)                 # [d, d_sae]
    L = (W_dec / W_dec.norm(dim=0, keepdim=True).clamp_min(1e-12)).T  # [d_sae, d]

    # --- best-match latent per atom (from alignment pkl) ---
    al = pickle.load(open(args.align_pkl, 'rb'))
    fmc = np.asarray(al['found_max_cos'])
    fmi = np.asarray(al['found_max_idx'])
    cand = np.where(fmc > args.match_thresh)[0]
    cand = cand[cand < N]                                  # safety: within dict
    print(f'Candidate atoms (best-match SAE cos > {args.match_thresh}): '
          f'{cand.size}', flush=True)

    # --- enumerate within-set pairs, filter ---
    rng = np.random.default_rng(args.pair_seed)
    all_pairs = []
    for a_idx in range(len(cand)):
        for b_idx in range(a_idx + 1, len(cand)):
            i, j = int(cand[a_idx]), int(cand[b_idx])
            if fmi[i] == fmi[j]:                           # same SAE latent -> skip
                continue
            c = float((D[i] * D[j]).sum())
            if abs(c) < args.within_cos:
                all_pairs.append((i, j, c))
    print(f'Pairs passing |cos|<{args.within_cos} & distinct latent: '
          f'{len(all_pairs)}', flush=True)
    rng.shuffle(all_pairs)
    pairs = all_pairs[:args.n_pairs]
    print(f'Using {len(pairs)} pairs', flush=True)

    # --- anchors ---
    print(f'Extracting {args.n_anchors} anchors (seed={args.held_seed})',
          flush=True)
    tok = load_fineweb_fixed_length(args.n_anchors, tokenizer,
                                    seq_len=args.seq_len, seed=args.held_seed)
    ctx, act = extract_context_and_activation(model, tok, args.layer, device)
    contexts = [ctx[k:k + 1].float().cpu() for k in range(ctx.shape[0])]
    activations = act.float().cpu()

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps_grid)
    angles_rad = np.deg2rad(angles_deg)
    ts = np.linspace(0.0, 1.0, args.n_t)

    P = len(pairs)
    ps = np.full((P, args.n_t), np.nan)
    meta = []   # per-pair: indices, latents, within-pair cos (ours & SAE), matches

    t0 = time.time()
    for pi, (i, j, c_ours) in enumerate(pairs):
        li, lj = int(fmi[i]), int(fmi[j])
        s_u, s_v = L[li], L[lj]
        u0, v0 = D[i], D[j]
        c_sae = float((s_u * s_v).sum())
        meta.append({'i': i, 'j': j, 'lat_i': li, 'lat_j': lj,
                     'cos_ours': c_ours, 'cos_sae': c_sae,
                     'match_i': float(fmc[i]), 'match_j': float(fmc[j])})
        for ti, t in enumerate(ts):
            u, _ = slerp(u0, s_u, float(t))
            v, _ = slerp(v0, s_v, float(t))
            p, _ = pair_p(model, contexts, activations, u, v,
                          args.layer, measure_layer, device,
                          angles_deg, angles_rad)
            ps[pi, ti] = p
        print(f'  pair {pi+1}/{P} (atoms {i},{j} -> lat {li},{lj})  '
              f'cos_ours={c_ours:+.3f} cos_sae={c_sae:+.3f}  '
              f'p[t=0]={ps[pi,0]:.2f} p[t=1]={ps[pi,-1]:.2f}  '
              f'elapsed={time.time()-t0:.0f}s', flush=True)
        with open(os.path.join(args.out, 'p_dict_to_sae.pkl'), 'wb') as f:
            pickle.dump({'args': vars(args), 'ts': ts, 'ps': ps, 'meta': meta,
                         'measure_layer': measure_layer}, f)

    # --- summary ---
    p0 = ps[:, 0][np.isfinite(ps[:, 0])]
    p1 = ps[:, -1][np.isfinite(ps[:, -1])]
    def q(x):
        return (f'median={np.median(x):.3f}  IQR('
                f'{np.quantile(x,.25):.3f},{np.quantile(x,.75):.3f})  '
                f'n={x.size}')
    print(f'\nour pairs   (t=0): {q(p0)}', flush=True)
    print(f'SAE pairs   (t=1): {q(p1)}', flush=True)

    # --- plot: median +- IQR vs t ---
    med = np.nanmedian(ps, axis=0)
    lo = np.nanquantile(ps, 0.25, axis=0)
    hi = np.nanquantile(ps, 0.75, axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for row in ps:
        ax.plot(ts, row, '-', color='#888888', lw=0.6, alpha=0.30)
    ax.fill_between(ts, lo, hi, color='#7a3a7d', alpha=0.22, label='IQR')
    ax.plot(ts, med, '-o', color='#7a3a7d', lw=2.2, ms=4, label='median p')
    ax.axhline(2.0, color='#1F77B4', ls='--', alpha=0.7, label='p=2 (Euclidean)')
    ax.set_xlabel(r'rotation fraction $t$  '
                  r'(0 = our$\times$our, 1 = SAE$\times$SAE best-match)',
                  fontsize=11)
    ax.set_ylabel('fitted superellipse p', fontsize=12)
    ax.set_title(f'p: optimized dirs -> best-match SAE latents\n'
                 f'{args.model} L{args.layer}->{measure_layer}  '
                 f'N={P} pairs (match>{args.match_thresh}, '
                 f'within|cos|<{args.within_cos})', fontsize=9.5)
    ax.grid(True, color='#cccccc', lw=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, loc='best')
    plt.tight_layout()
    png = os.path.join(args.out, 'p_dict_to_sae.png')
    fig.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\nSaved {png}', flush=True)
    print(f'Saved {os.path.join(args.out, "p_dict_to_sae.pkl")}', flush=True)


if __name__ == '__main__':
    main()
