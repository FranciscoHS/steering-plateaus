#!/usr/bin/env python3
"""Reconstruction quality (y) vs superellipse-p of the directions used (x).

For each direction *type* (learned dict / PCA / random / SAE) we make points
at sparsity levels K in {16, 32, 64}:
  - y = mean relative L2 reconstruction error ||a-a_hat||^2/||a||^2 over the
        held-out activations at that K.
  - x = pooled superellipse-p estimated from a sample of pairs drawn from the
        directions actually USED in the reconstruction at that K.

p is a pairwise property (the iso-L2-response boundary in the 2D plane spanned
by perturbing along d_i and d_j at the build layer), so for each (type, K) we
sample n_pairs pairs from the used-direction set and fit p per pair, then fit a
single pooled-p on the union of normalized boundary points (bootstrap CI over
pairs).  This mirrors scripts/superellipse/fit_pairs_dictionary.py.

Pair sampling per type:
  - dict / SAE : co-occurrence -- pick a held activation, then 2 of the K
                 atoms/latents it selected.  Weights pairs by how the
                 reconstruction actually uses them.
  - PCA        : top-K PCs form a fixed subspace; sample uniform pairs of PCs.
  - random     : pair-p distribution is K-independent (all random unit pairs are
                 statistically identical) -> estimate ONCE, reuse x for all K.

Usage (see Plans/p_vs_recon.md):
    python -u scripts/superellipse/p_vs_recon.py \
        --dict_path .../snap_dict_first20k.pt \
        --held_acts_path .../held_acts.pt \
        --pca_pool_path .../pca_pool_acts.pt \
        --sae_ckpt .../sae_L0_50_layer14.pt \
        --model qwen3-1.7b --layer 14 \
        --K_list 16 32 64 --sae_K 32 --n_pairs 100 --n_anchors 20 \
        --out .../p_vs_recon
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

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)
from scripts.superellipse.canonical.sweep_core import run_2d_per_anchors
from scripts.superellipse.canonical.superellipse import fit_superellipse
from scripts.superellipse.plot_boundary_pair import boundary_for_set
# Reuse the exact normalization + per-pair p collapse used for dict pair fits.
from scripts.superellipse.fit_pairs_dictionary import (
    normalized_points, summarize_fit,
)


# --------------------------------------------------------------------------- #
# Reconstruction (y-axis) + which directions get used                          #
# --------------------------------------------------------------------------- #
def omp_with_indices(D, A, K_list, device, chunk=64):
    """Batched OMP. Returns (rec_err[len(K_list), B], sel[B, K_max]).

    rec_err is ||a-a_hat||^2/||a||^2 at each K in K_list; sel are the global
    atom indices selected (in order), so sel[:, :K] = atoms used at K.
    """
    K_list = sorted(K_list)
    K_max = K_list[-1]
    B = A.shape[0]
    rec = torch.zeros(len(K_list), B, dtype=torch.float32)
    sel_all = torch.full((B, K_max), -1, dtype=torch.long)
    a_norm_sq = (A ** 2).sum(dim=-1).cpu()
    Kset = {k: i for i, k in enumerate(K_list)}
    for b0 in range(0, B, chunk):
        b1 = min(B, b0 + chunk)
        Ab = A[b0:b1]
        Rb = Ab.clone()
        sel = torch.full((Ab.shape[0], K_max), -1, dtype=torch.long, device=device)
        for k in range(K_max):
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
            if (k + 1) in Kset:
                e = (Rb ** 2).sum(dim=-1).cpu()
                rec[Kset[k + 1], b0:b1] = e / a_norm_sq[b0:b1]
        sel_all[b0:b1] = sel.cpu()
    return rec, sel_all


def pca_fit(A_pool, device):
    """Return (mu[1,d], V[d,d]) principal components (rows of V)."""
    A_pool = A_pool.to(device=device, dtype=torch.float32)
    mu = A_pool.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(A_pool - mu, full_matrices=False)
    return mu, Vh                                      # V[k] is k-th PC


def pca_recon_err(A_held, mu, V, K, device):
    A_held = A_held.to(device=device, dtype=torch.float32)
    Vk = V[:K]
    Hc = A_held - mu
    recon = (Hc @ Vk.T) @ Vk + mu
    norm_sq = (A_held ** 2).sum(dim=-1)
    return (((A_held - recon) ** 2).sum(dim=-1) / norm_sq).cpu()


def sae_load(sae_ckpt, d, device):
    """Load Qwen-Scope-style SAE -> (W_enc[d_sae,d], b_enc, W_dec_atoms[d_sae,d], b_dec).

    W_dec_atoms[i] is the (raw, un-normalized) decoder direction for latent i.
    """
    sae = torch.load(sae_ckpt, map_location='cpu')
    keys = list(sae.keys())

    def find(*names):
        for n in names:
            if n in sae:
                return sae[n]
        raise KeyError(f'none of {names} in SAE keys={keys}')
    W_enc = find('W_enc', 'encoder.weight').float()
    b_enc = find('b_enc', 'encoder.bias').float()
    W_dec = find('W_dec', 'decoder.weight').float()
    b_dec = find('b_dec', 'decoder.bias', 'b_dec_bias').float()
    d_sae = W_enc.shape[0]
    if W_dec.shape == (d, d_sae):
        dec_atoms = W_dec.T.contiguous()               # [d_sae, d]
    elif W_dec.shape == (d_sae, d):
        dec_atoms = W_dec.contiguous()
    else:
        raise ValueError(f'unexpected W_dec {tuple(W_dec.shape)}')
    return (W_enc.to(device), b_enc.to(device),
            dec_atoms.to(device), b_dec.to(device))


def sae_topk_recon_err_and_sel(W_enc, b_enc, dec_atoms, b_dec, A, K, device,
                               chunk=256):
    """TopK encode -> decode at fixed K. Returns (rec_err[B], sel[B, K])."""
    A = A.to(device=device, dtype=torch.float32)
    rec = torch.zeros(A.shape[0])
    sel = torch.full((A.shape[0], K), -1, dtype=torch.long)
    for b0 in range(0, A.shape[0], chunk):
        b1 = min(A.shape[0], b0 + chunk)
        Ab = A[b0:b1]
        pre = Ab @ W_enc.T + b_enc
        topv, topi = pre.topk(K, dim=-1)
        recon = (topv.unsqueeze(-1) * dec_atoms[topi]).sum(dim=1) + b_dec
        rec[b0:b1] = (((Ab - recon) ** 2).sum(-1)
                      / (Ab ** 2).sum(-1)).cpu()
        sel[b0:b1] = topi.cpu()
    return rec, sel


# --------------------------------------------------------------------------- #
# Pair sampling                                                                #
# --------------------------------------------------------------------------- #
def sample_cooccurrence_pairs(sel, n_pairs, rng):
    """Sample up to n_pairs unique unordered global-index pairs (i,j) by
    picking a row of `sel` then 2 distinct columns. Weighted by co-selection.
    sel: [B, K] global indices (>=0)."""
    B, K = sel.shape
    seen = set()
    tries = 0
    max_tries = n_pairs * 200
    while len(seen) < n_pairs and tries < max_tries:
        tries += 1
        b = int(rng.integers(B))
        c1, c2 = rng.integers(K), rng.integers(K)
        if c1 == c2:
            continue
        i, j = int(sel[b, c1]), int(sel[b, c2])
        if i == j:
            continue
        if i > j:
            i, j = j, i
        seen.add((i, j))
    return sorted(seen)


def sample_uniform_pairs(idx_pool, n_pairs, rng):
    """Sample up to n_pairs unique unordered pairs from a flat index pool."""
    idx_pool = list(idx_pool)
    M = len(idx_pool)
    seen = set()
    target = min(n_pairs, M * (M - 1) // 2)
    while len(seen) < target:
        a, b = int(rng.integers(M)), int(rng.integers(M))
        if a == b:
            continue
        i, j = idx_pool[a], idx_pool[b]
        if i > j:
            i, j = j, i
        seen.add((i, j))
    return sorted(seen)


# --------------------------------------------------------------------------- #
# p estimation for a list of direction pairs                                   #
# --------------------------------------------------------------------------- #
def estimate_pooled_p(pair_dirs, model, contexts, activations, layer,
                      measure_layer, device, angles_deg, n_pool_bootstrap,
                      boot_rng, label=''):
    """pair_dirs: list of (d1, d2) unit tensors on CPU.

    Returns dict with per-pair p_fit, pooled p, 95% CI over pairs, n_pairs.
    """
    angles_rad = np.deg2rad(angles_deg)
    p_fits, pool_xs, pool_ys = [], [], []
    t0 = time.time()
    for k, (d1, d2) in enumerate(pair_dirs):
        r = run_2d_per_anchors(model, contexts, activations, d1, d2,
                               layer, measure_layer, device, angles_rad,
                               mode='geodesic', single_batch_ref=True)
        median_grid = np.median(r['l2'], axis=0)
        b = boundary_for_set(angles_deg, median_grid, exact=True)
        p_fits.append(b['fit']['p'])
        xs, ys = normalized_points(angles_deg, median_grid)
        pool_xs.append(xs)
        pool_ys.append(ys)
        if (k + 1) % 25 == 0 or k == 0:
            rate = (time.time() - t0) / (k + 1)
            print(f'    [{label}] {k+1}/{len(pair_dirs)} '
                  f'rate={rate:.2f}s/pair '
                  f'ETA={rate*(len(pair_dirs)-k-1)/60:.1f}min', flush=True)

    p_fits = np.array(p_fits, dtype=float)
    valid = [k for k in range(len(pool_xs)) if pool_xs[k].size > 0]
    xs_all = (np.concatenate([pool_xs[k] for k in valid]) if valid
              else np.zeros(0))
    ys_all = (np.concatenate([pool_ys[k] for k in valid]) if valid
              else np.zeros(0))
    p_pool = float(fit_superellipse(xs_all, ys_all)['p']) if xs_all.size else float('nan')
    p_boot = []
    if valid:
        for _ in range(n_pool_bootstrap):
            idx = boot_rng.integers(0, len(valid), len(valid))
            xb = np.concatenate([pool_xs[valid[k]] for k in idx])
            yb = np.concatenate([pool_ys[valid[k]] for k in idx])
            if xb.size:
                p_boot.append(fit_superellipse(xb, yb)['p'])
    p_boot = np.array(p_boot, dtype=float)
    ci = ((float(np.quantile(p_boot, 0.025)), float(np.quantile(p_boot, 0.975)))
          if p_boot.size else (float('nan'), float('nan')))
    finite = np.isfinite(p_fits)
    return {
        'p_pool': p_pool, 'p_pool_ci95': ci,
        'p_fits': p_fits,
        'p_median': float(np.median(p_fits[finite])) if finite.any() else float('nan'),
        'p_iqr': ((float(np.quantile(p_fits[finite], 0.25)),
                   float(np.quantile(p_fits[finite], 0.75)))
                  if finite.any() else (float('nan'), float('nan'))),
        'n_pairs': len(pair_dirs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--held_acts_path', required=True)
    ap.add_argument('--pca_pool_path', required=True)
    ap.add_argument('--sae_ckpt', default=None)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--K_list', type=int, nargs='+', default=[16, 32, 64])
    ap.add_argument('--sae_K', type=int, default=32,
                    help='single K for the SAE point')
    ap.add_argument('--n_pairs', type=int, default=100)
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--pair_seed', type=int, default=0)
    ap.add_argument('--rand_seed', type=int, default=777)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--n_steps_grid', type=int, default=31)
    ap.add_argument('--n_pool_bootstrap', type=int, default=2000)
    ap.add_argument('--n_held', type=int, default=None,
                    help='cap held activations for rec err (default: all)')
    ap.add_argument('--types', nargs='+',
                    default=['dict', 'pca', 'random', 'sae'],
                    choices=['dict', 'pca', 'random', 'sae'])
    ap.add_argument('--smoke', action='store_true',
                    help='tiny run: 4 pairs, 200 held, to measure rate')
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.smoke:
        args.n_pairs = 4
        args.n_held = 200
        args.n_pool_bootstrap = 100
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}

    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    d_model = model._plateau_info['d_model']
    print(f'device={device} d={d_model} n_layers={n_layers} '
          f'L{args.layer}->{measure_layer}', flush=True)

    # Held-out activations for reconstruction (y).
    held = torch.load(args.held_acts_path, map_location='cpu')['activations']
    if args.n_held:
        held = held[:args.n_held]
    A = held.to(device=device, dtype=torch.float32)
    print(f'held acts for rec err: {tuple(A.shape)}', flush=True)

    # Anchors for p-fitting (x): n_anchors held-out FineWeb ctx+act.
    tok = load_fineweb_fixed_length(args.n_anchors, tokenizer,
                                    seq_len=args.seq_len, seed=args.held_seed)
    ctx, act = extract_context_and_activation(model, tok, args.layer, device)
    contexts = [ctx[i:i + 1].float().cpu() for i in range(ctx.shape[0])]
    activations = act.float().cpu()
    print(f'p-fit anchors: {len(contexts)}', flush=True)

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps_grid)
    pair_rng = np.random.default_rng(args.pair_seed)
    boot_rng = np.random.default_rng(args.pair_seed + 1)

    # ----- build direction matrices per type -----
    dirs = {}                # type -> [N, d] UNIT directions on device
    if 'dict' in args.types:
        Dd = torch.load(args.dict_path, map_location='cpu')['dictionary']
        Dd = Dd.to(device=device, dtype=torch.float32)
        dirs['dict'] = Dd / (Dd.norm(dim=-1, keepdim=True) + 1e-12)
        print(f"dict atoms: {dirs['dict'].shape[0]}", flush=True)
    if 'random' in args.types:
        N_rand = dirs['dict'].shape[0] if 'dict' in dirs else 20000
        g = torch.Generator(device=device).manual_seed(args.rand_seed)
        Dr = torch.randn(N_rand, d_model, generator=g, device=device,
                         dtype=torch.float32)
        dirs['random'] = Dr / Dr.norm(dim=-1, keepdim=True)
    pca_mu = pca_V = None
    if 'pca' in args.types:
        A_pool = torch.load(args.pca_pool_path, map_location='cpu')['activations']
        pca_mu, pca_V = pca_fit(A_pool, device)
        dirs['pca'] = pca_V / (pca_V.norm(dim=-1, keepdim=True) + 1e-12)
        print(f"pca comps: {dirs['pca'].shape[0]}", flush=True)
    sae_pack = None
    if 'sae' in args.types:
        if not args.sae_ckpt:
            raise SystemExit('sae type requested but --sae_ckpt not given')
        sae_pack = sae_load(args.sae_ckpt, d_model, device)
        dec_atoms = sae_pack[2]
        dirs['sae'] = dec_atoms / (dec_atoms.norm(dim=-1, keepdim=True) + 1e-12)
        print(f"sae decoder atoms: {dirs['sae'].shape[0]}", flush=True)

    def get_pair_dirs(type_name, pairs):
        Dall = dirs[type_name]
        return [(Dall[i].cpu(), Dall[j].cpu()) for (i, j) in pairs]

    results = {'args': vars(args), 'measure_layer': measure_layer,
               'angles_deg': angles_deg, 'points': []}

    # ===================================================================== #
    # dict / random : OMP rec err + co-occurrence pairs                      #
    # ===================================================================== #
    for tname in [t for t in ('dict', 'random') if t in args.types]:
        print(f'\n=== {tname}: OMP ===', flush=True)
        rec, sel = omp_with_indices(dirs[tname], A, args.K_list, device)
        rand_p_cache = None
        for ki, K in enumerate(sorted(args.K_list)):
            y = float(rec[ki].mean())
            if tname == 'random' and rand_p_cache is not None:
                # K-independent: reuse the single random-pair p estimate.
                pest = rand_p_cache
            else:
                pairs = sample_cooccurrence_pairs(sel[:, :K], args.n_pairs,
                                                  pair_rng)
                pest = estimate_pooled_p(
                    get_pair_dirs(tname, pairs), model, contexts, activations,
                    args.layer, measure_layer, device, angles_deg,
                    args.n_pool_bootstrap, boot_rng, label=f'{tname}K{K}')
                if tname == 'random':
                    rand_p_cache = pest
            print(f'  {tname} K={K}: recL2={y:.4f}  p_pool={pest["p_pool"]:.3f} '
                  f'CI={pest["p_pool_ci95"]}', flush=True)
            results['points'].append({'type': tname, 'K': K, 'rec_l2': y,
                                      **pest})

    # ===================================================================== #
    # pca : fixed top-K subspace                                             #
    # ===================================================================== #
    if 'pca' in args.types:
        print('\n=== pca ===', flush=True)
        for K in sorted(args.K_list):
            y = float(pca_recon_err(A, pca_mu, pca_V, K, device).mean())
            pairs = sample_uniform_pairs(list(range(K)), args.n_pairs, pair_rng)
            pest = estimate_pooled_p(
                get_pair_dirs('pca', pairs), model, contexts, activations,
                args.layer, measure_layer, device, angles_deg,
                args.n_pool_bootstrap, boot_rng, label=f'pcaK{K}')
            print(f'  pca K={K}: recL2={y:.4f}  p_pool={pest["p_pool"]:.3f} '
                  f'CI={pest["p_pool_ci95"]}', flush=True)
            results['points'].append({'type': 'pca', 'K': K, 'rec_l2': y,
                                      **pest})

    # ===================================================================== #
    # sae : single point at sae_K                                           #
    # ===================================================================== #
    if 'sae' in args.types:
        print('\n=== sae ===', flush=True)
        K = args.sae_K
        rec_s, sel_s = sae_topk_recon_err_and_sel(*sae_pack, A, K, device)
        y = float(rec_s.mean())
        pairs = sample_cooccurrence_pairs(sel_s, args.n_pairs, pair_rng)
        pest = estimate_pooled_p(
            get_pair_dirs('sae', pairs), model, contexts, activations,
            args.layer, measure_layer, device, angles_deg,
            args.n_pool_bootstrap, boot_rng, label=f'saeK{K}')
        print(f'  sae K={K}: recL2={y:.4f}  p_pool={pest["p_pool"]:.3f} '
              f'CI={pest["p_pool_ci95"]}', flush=True)
        results['points'].append({'type': 'sae', 'K': K, 'rec_l2': y, **pest})

    # ----- save -----
    out_pkl = os.path.join(args.out, 'p_vs_recon.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(results, f)

    # ----- plot -----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = {'dict': 'C0', 'pca': 'C3', 'random': 'C1', 'sae': 'C4'}
    markers = {'dict': 'o', 'pca': 'd', 'random': 's', 'sae': '*'}
    for tname in args.types:
        pts = [p for p in results['points'] if p['type'] == tname]
        if not pts:
            continue
        pts = sorted(pts, key=lambda p: p['K'])
        xs = [p['p_pool'] for p in pts]
        ys = [p['rec_l2'] for p in pts]
        xerr = np.array([[p['p_pool'] - p['p_pool_ci95'][0] for p in pts],
                         [p['p_pool_ci95'][1] - p['p_pool'] for p in pts]])
        ax.errorbar(xs, ys, xerr=xerr, fmt=markers[tname], color=colors[tname],
                    ms=11 if tname == 'sae' else 8, capsize=3, lw=1,
                    label=tname)
        for p in pts:
            ax.annotate(f'K={p["K"]}', (p['p_pool'], p['rec_l2']),
                        fontsize=7, xytext=(4, 4),
                        textcoords='offset points')
    ax.axvline(2.0, color='gray', ls=':', lw=1, label='p=2 (ellipse)')
    ax.set_xlabel('pooled superellipse-p of directions used')
    ax.set_ylabel(r'mean rel. $L^2$  $\|a-\hat a\|^2/\|a\|^2$')
    ax.set_title(f'{args.model} L{args.layer}->{measure_layer}: '
                 f'reconstruction vs p  (n_pairs={args.n_pairs})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    png = os.path.join(args.out, 'p_vs_recon.png')
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nSaved {out_pkl}')
    print(f'Saved {png}')


if __name__ == '__main__':
    main()
