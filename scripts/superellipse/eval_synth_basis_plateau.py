#!/usr/bin/env python3
"""Plateau sweep for synthetic activations built by reusing a real activation's
SAE magnitude spectrum on DIFFERENT carrier bases.

Recipe (per FineWeb anchor a):
  1. SAE-decompose a: top-K latents, effective magnitudes m_i = z_i·||W_dec[:,i]||.
  2. Reuse the SAME multiset {m_i} on a random selection of carrier directions:
       Case 1 'sae'        : random SAE decoder latents (unit W_dec columns)
       Case 2 'dict'       : random dictionary atoms
       Case 3 'dict_dedup' : random atoms from a greedily deduplicated dict
                             (keep atoms with pairwise |cos| < thr)
     synth = b_dec + Σ m_i · (unit carrier_i)
  3. Run the 1D spherical plateau sweep (downstream L2 response vs perturbation
     angle along random directions) on each synthetic activation, and compare to
     the real activation and a norm-matched random activation.

The question: do these scrambled-feature activations sit on a low-curvature
plateau like real activations, and does the carrier basis matter?

Run on a GPU pod.

Usage:
    python scripts/superellipse/eval_synth_basis_plateau.py \\
        --model qwen3-1.7b --layer 14 \\
        --dict_path /tmp/dict_25k_snapshot.pt \\
        --sae_ckpt .../sae_L0_50_layer14.pt --sae_K 50 \\
        --dedup_thr 0.25 --n_act 30 --n_dirs 10 --seq_len 5 \\
        --max_angle 60 --n_steps 31 --out .../synth_basis_plateau
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

from src.model import load_model, _get_blocks
from scripts.superellipse.eval_real_vs_random_acts import (
    fineweb_ctx_acts, sweep_1d,
)


def greedy_dedup(D_unit, thr, device, cache_path=None):
    """Greedy: keep an atom only if |cos| with every already-kept atom < thr."""
    if cache_path and os.path.exists(cache_path):
        idx = torch.load(cache_path)['kept_idx']
        print(f'  loaded dedup cache: {len(idx)} atoms ({cache_path})',
              flush=True)
        return idx
    Dn = D_unit.to(device)
    kept_idx = [0]
    kept = Dn[0:1].clone()                                   # [m, d]
    for i in range(1, Dn.shape[0]):
        if (kept @ Dn[i]).abs().max() < thr:
            kept = torch.cat([kept, Dn[i:i + 1]], 0)
            kept_idx.append(i)
    idx = torch.tensor(kept_idx)
    print(f'  greedy dedup (|cos|<{thr}): kept {len(idx)}/{Dn.shape[0]}',
          flush=True)
    if cache_path:
        torch.save({'kept_idx': idx, 'threshold': thr}, cache_path)
    return idx


def build_synth(mags, carriers_pool, b_dec, sae_K, n_act, seed, device):
    """For each anchor, pick sae_K random unit carriers and sum m_i·carrier_i.

    mags: [n_act, sae_K]  carriers_pool: [P, d] (unit)  b_dec: [d]
    Returns [n_act, d].
    """
    P = carriers_pool.shape[0]
    d = carriers_pool.shape[1]
    out = torch.zeros(n_act, d, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    for i in range(n_act):
        pick = rng.choice(P, size=sae_K, replace=(P < sae_K))
        carr = carriers_pool[torch.as_tensor(pick, device=device)]   # [K, d]
        out[i] = b_dec + (mags[i].unsqueeze(-1) * carr).sum(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--sae_ckpt', required=True)
    ap.add_argument('--sae_K', type=int, default=50)
    ap.add_argument('--dedup_thr', type=float, default=0.25)
    ap.add_argument('--n_act', type=int, default=30)
    ap.add_argument('--n_dirs', type=int, default=10)
    ap.add_argument('--seq_len', type=int, default=5)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--n_steps', type=int, default=31)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rand_seed', type=int, default=0)
    ap.add_argument('--synth_seed', type=int, default=0)
    ap.add_argument('--dedup_cache', default=None,
                    help='path to cache/load the dedup kept_idx')
    ap.add_argument('--no_bdec', action='store_true',
                    help='build synth = Σ m_i·carrier_i WITHOUT the SAE decoder '
                         'bias b_dec (default: include b_dec)')
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
    d_model = model._plateau_info['d_model']
    model_dtype = next(model.parameters()).dtype
    print(f'Device: {device} d_model={d_model} L{args.layer}->{measure_layer}',
          flush=True)

    # --- Anchors ---
    print(f'Extracting {args.n_act} FineWeb anchors (seq_len={args.seq_len}, '
          f'seed={args.seed})', flush=True)
    contexts, activations = fineweb_ctx_acts(model, tokenizer, device,
                                             args.layer, args.n_act,
                                             args.seq_len, args.seed)
    a_norms = activations.norm(dim=-1)
    print(f'  mean ||a||={a_norms.mean():.2f}', flush=True)

    # --- SAE decomposition → magnitudes m_i = z_i·||W_dec[:,i]|| ---
    sae = torch.load(args.sae_ckpt, map_location='cpu')
    W_enc = sae['W_enc'].float(); b_enc = sae['b_enc'].float()
    W_dec = sae['W_dec'].float()                             # [d, d_sae]
    b_dec = sae['b_dec'].float().to(device)                  # [d]
    if args.no_bdec:
        b_dec = torch.zeros_like(b_dec)
        print('b_dec: EXCLUDED (synth = sum of magnitude-weighted carriers)',
              flush=True)
    else:
        print(f'b_dec: included  ||b_dec||={b_dec.norm():.1f}', flush=True)
    dec_norm = W_dec.norm(dim=0)                              # [d_sae]
    pre = activations.float() @ W_enc.T + b_enc               # [n_act, d_sae]
    topv, topi = pre.topk(args.sae_K, dim=-1)                 # [n_act, K]
    mags = (topv * dec_norm[topi]).to(device)                # [n_act, K]
    print(f'SAE: {W_enc.shape[0]} latents, top-{args.sae_K} per anchor; '
          f'mean Σm_i={mags.sum(1).mean():.1f}', flush=True)

    # --- Carrier pools (all unit) ---
    # Case 1: SAE decoder latents (unit columns of W_dec)
    sae_carriers = (W_dec / dec_norm.clamp_min(1e-12)).T.to(device)  # [d_sae, d]
    # Case 2: dictionary atoms
    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].float()
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    print(f'Dict: {D.shape[0]} atoms', flush=True)
    # Case 3: greedily deduplicated dict
    dedup_idx = greedy_dedup(D, args.dedup_thr, device,
                             cache_path=args.dedup_cache)
    D_dev = D.to(device)
    D_dedup = D_dev[dedup_idx.to(device)]

    # --- Build the 3 synthetic conditions (same magnitudes, different bases) ---
    cases = {
        'sae': build_synth(mags, sae_carriers, b_dec, args.sae_K, args.n_act,
                           args.synth_seed + 1, device),
        'dict': build_synth(mags, D_dev, b_dec, args.sae_K, args.n_act,
                            args.synth_seed + 2, device),
        'dict_dedup': build_synth(mags, D_dedup, b_dec, args.sae_K, args.n_act,
                                 args.synth_seed + 3, device),
    }
    for name, s in cases.items():
        cos = float(F.cosine_similarity(s.cpu(), activations, dim=-1).abs().mean())
        print(f'  case {name:11s}: mean||v||={s.norm(dim=-1).mean():.1f}  '
              f'mean|cos| to real={cos:.3f}', flush=True)

    # --- Norm-matched random activation (floor) ---
    g_act = torch.Generator(device=device).manual_seed(args.rand_seed)
    rand_acts = torch.randn(args.n_act, d_model, generator=g_act, device=device,
                            dtype=torch.float32)
    rand_acts = (rand_acts / rand_acts.norm(dim=-1, keepdim=True)) * \
        a_norms.to(device).unsqueeze(-1)

    # --- Shared random perturbation directions ---
    g_d = torch.Generator(device=device).manual_seed(args.rand_seed + 1)
    rand_dirs = torch.randn(args.n_dirs, d_model, generator=g_d, device=device,
                            dtype=torch.float32)
    rand_dirs = rand_dirs / rand_dirs.norm(dim=-1, keepdim=True)

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)
    print(f'Sweep 0..{args.max_angle}° ({args.n_steps} steps), '
          f'n_dirs={args.n_dirs}', flush=True)

    series_names = ['real', 'rand_act', 'sae', 'dict', 'dict_dedup']
    out = {n: np.zeros((args.n_act, args.n_dirs, args.n_steps), dtype=np.float32)
           for n in series_names}
    sources = {  # per-anchor [d] activation tensors
        'real': activations.to(device),
        'rand_act': rand_acts,
        'sae': cases['sae'], 'dict': cases['dict'],
        'dict_dedup': cases['dict_dedup'],
    }

    t0 = time.time()
    for i in range(args.n_act):
        ctx = contexts[i].to(device=device, dtype=model_dtype)
        for r in range(args.n_dirs):
            d = rand_dirs[r]
            for name in series_names:
                a = sources[name][i].to(device=device, dtype=torch.float32)
                out[name][i, r] = sweep_1d(model, ctx, a, d, args.layer,
                                           measure_layer, angles_rad,
                                           model_dtype, device)
        if (i + 1) % 5 == 0 or i == args.n_act - 1:
            print(f'  anchor {i+1}/{args.n_act}  '
                  f'elapsed={time.time()-t0:.1f}s', flush=True)

    out_pkl = os.path.join(args.out, 'results.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({'args': vars(args), 'angles_deg': angles_deg,
                     'measure_layer': measure_layer,
                     'l2': out, 'a_norms': a_norms.numpy(),
                     'n_dedup': int(len(dedup_idx))}, f)

    # --- Plot: median + IQR per series ---
    style = {'real': ('C0', 'real activation'),
             'rand_act': ('C7', 'norm-matched random act'),
             'sae': ('C1', 'Case 1: random SAE latents'),
             'dict': ('C2', 'Case 2: random dict atoms'),
             'dict_dedup': ('C3', f'Case 3: dedup dict (|cos|<{args.dedup_thr}, '
                                  f'N={len(dedup_idx)})')}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name in series_names:
        arr = out[name].reshape(-1, args.n_steps)
        med = np.median(arr, axis=0)
        q1, q3 = np.quantile(arr, 0.25, axis=0), np.quantile(arr, 0.75, axis=0)
        col, lab = style[name]
        ax.plot(angles_deg, med, '-', color=col, label=lab, lw=2)
        ax.fill_between(angles_deg, q1, q3, color=col, alpha=0.15)
    ax.set_xlabel('perturbation angle (deg)')
    ax.set_ylabel(f'downstream L2 response  (L{args.layer}->{measure_layer})')
    ax.set_title(f'Plateau sweep: real vs magnitude-reused synthetic bases\n'
                 f'{args.model}  sae_K={args.sae_K}  n_act={args.n_act}  '
                 f'n_dirs={args.n_dirs}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'synth_basis_plateau.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nSaved {out_pkl}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
