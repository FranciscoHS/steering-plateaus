#!/usr/bin/env python3
"""p vs rotation from optimized atoms to best-match SAE latents — FULL-p fit.

Same experiment as sweep_p_dict_to_sae.py (slerp both legs of a pair toward
their best-match SAE latents, measure p at each rotation fraction t), but using
the SAME p estimator as eval_sae_full_p.py (soft_p_proxy.soft_p_estimate:
ray sweep -> threshold -> geodesic-normalized intercepts -> Gauss-Newton fit,
eval frame, anchor-medianed), so the endpoints are directly comparable to the
pcull_eval full-p numbers (baseline 2.13 / popt 2.83, alive-alive splits).

Pair selection is code-identical to sweep_p_dict_to_sae.py — with the same
inputs and --pair_seed it sweeps the same pairs.

Usage:
    python -u scripts/superellipse/sweep_p_dict_to_sae_fullfit.py \\
        --dict_path .../dict_popt_sub.pt --sae_ckpt .../sae_baseline_shim.pt \\
        --align_pkl .../align_sub.pkl --n_pairs 50 --n_t 7 \\
        --model Qwen/Qwen3-1.7B --layer 10 --out .../p_dict_to_sae_fullfit
"""
import argparse
import importlib.util
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
from scripts.superellipse.sweep_p_vs_rotation_llm import slerp


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

pcull = _load('scripts/dictionary_learning/train_sae_pcull.py', 'pcull')
spp = _load('scripts/superellipse/soft_p_proxy.py', 'soft_p_proxy')


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--sae_ckpt', required=True)
    ap.add_argument('--align_pkl', required=True)
    ap.add_argument('--match_thresh', type=float, default=0.5)
    ap.add_argument('--within_cos', type=float, default=0.1)
    ap.add_argument('--n_pairs', type=int, default=50)
    ap.add_argument('--n_t', type=int, default=7)
    ap.add_argument('--pair_seed', type=int, default=0)
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--layer', type=int, default=10)
    ap.add_argument('--measure_layer', type=int, default=None)
    # fit params: defaults identical to eval_sae_full_p.py
    ap.add_argument('--n_anchor', type=int, default=8)
    ap.add_argument('--n_rays', type=int, default=13)
    ap.add_argument('--n_radii', type=int, default=40)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--p_cap', type=float, default=10.0)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--anchor_seed', type=int, default=999)
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    print(f'model={args.model} L{args.layer}->{measure_layer}', flush=True)

    # --- optimized directions ---
    blob = torch.load(args.dict_path, map_location='cpu', weights_only=False)
    D = blob['dictionary'].to(torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    N, d_model = D.shape

    # --- SAE decoder directions (unit), [d, d_sae] convention ---
    sae = torch.load(args.sae_ckpt, map_location='cpu', weights_only=False)
    W_dec = sae['W_dec'].to(torch.float32)
    L = (W_dec / W_dec.norm(dim=0, keepdim=True).clamp_min(1e-12)).T

    # --- best-match latent per atom; pair selection (same as sweep_p_dict_to_sae) ---
    al = pickle.load(open(args.align_pkl, 'rb'))
    fmc = np.asarray(al['found_max_cos'])
    fmi = np.asarray(al['found_max_idx'])
    cand = np.where(fmc > args.match_thresh)[0]
    cand = cand[cand < N]
    print(f'Candidate atoms (cos > {args.match_thresh}): {cand.size}', flush=True)

    rng = np.random.default_rng(args.pair_seed)
    all_pairs = []
    for a_idx in range(len(cand)):
        for b_idx in range(a_idx + 1, len(cand)):
            i, j = int(cand[a_idx]), int(cand[b_idx])
            if fmi[i] == fmi[j]:
                continue
            c = float((D[i] * D[j]).sum())
            if abs(c) < args.within_cos:
                all_pairs.append((i, j, c))
    print(f'Pairs passing filters: {len(all_pairs)}', flush=True)
    rng.shuffle(all_pairs)
    pairs = all_pairs[:args.n_pairs]
    print(f'Using {len(pairs)} pairs', flush=True)

    # --- anchor pool + ray grid: identical to eval_sae_full_p.py ---
    acts, ctxs, f_refs = pcull.build_anchor_pool(
        model, tokenizer, args.layer, args.n_anchor, args.seq_len,
        args.anchor_seed, device, measure_layer)
    acts_stack = torch.stack(acts)
    ctx_stack = torch.cat(ctxs, 0)
    f_refs_stack = torch.stack(f_refs)
    phis_rad, radii_rad = spp.default_phis_radii(
        args.n_rays, args.n_radii, args.max_angle)

    ts = np.linspace(0.0, 1.0, args.n_t)
    P = len(pairs)
    ps = np.full((P, args.n_t), np.nan)
    meta = []
    t0 = time.time()
    for pi, (i, j, c_ours) in enumerate(pairs):
        li, lj = int(fmi[i]), int(fmi[j])
        s_u, s_v = L[li], L[lj]
        u0, v0 = D[i], D[j]
        meta.append({'i': i, 'j': j, 'lat_i': li, 'lat_j': lj,
                     'cos_ours': c_ours,
                     'cos_sae': float((s_u * s_v).sum()),
                     'match_i': float(fmc[i]), 'match_j': float(fmc[j])})
        for ti, t in enumerate(ts):
            u, _ = slerp(u0, s_u, float(t))
            v, _ = slerp(v0, s_v, float(t))
            u = (u / u.norm()).to(device)
            v = (v / v.norm()).to(device)
            _, diag = spp.soft_p_estimate(
                model, acts_stack, ctx_stack, f_refs_stack, u, v.unsqueeze(0),
                args.layer, measure_layer, phis_rad, radii_rad,
                p_cap=args.p_cap, anchor_reduce='median', chunk=args.chunk,
                return_diag=True)
            ps[pi, ti] = float(diag['p_k'][0].item())
        print(f'  pair {pi+1}/{P} (atoms {i},{j} -> lat {li},{lj})  '
              f'p[t=0]={ps[pi,0]:.2f} p[t=1]={ps[pi,-1]:.2f}  '
              f'elapsed={time.time()-t0:.0f}s', flush=True)
        with open(os.path.join(args.out, 'p_dict_to_sae_fullfit.pkl'), 'wb') as f:
            pickle.dump({'args': vars(args), 'ts': ts, 'ps': ps, 'meta': meta,
                         'measure_layer': measure_layer}, f)

    p0 = ps[:, 0][np.isfinite(ps[:, 0])]
    p1 = ps[:, -1][np.isfinite(ps[:, -1])]
    def q(x):
        return (f'median={np.median(x):.3f}  IQR({np.quantile(x,.25):.3f},'
                f'{np.quantile(x,.75):.3f})  n={x.size}')
    print(f'\nour pairs   (t=0): {q(p0)}', flush=True)
    print(f'SAE pairs   (t=1): {q(p1)}', flush=True)

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
    ax.set_ylabel('fitted superellipse p (soft_p_estimate, eval frame)',
                  fontsize=11)
    ax.set_title(f'p: optimized dirs -> best-match SAE latents (FULL-p fit)\n'
                 f'{args.model} L{args.layer}->{measure_layer}  N={P} pairs',
                 fontsize=9.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'p_dict_to_sae_fullfit.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {plot_path}', flush=True)


if __name__ == '__main__':
    main()
