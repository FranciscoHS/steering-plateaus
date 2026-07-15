#!/usr/bin/env python3
"""
Full superellipse-p evaluation of an SAE's decoder atoms (eval frame).

Unlike eval_sae_proxy_p.py (single-angle surrogate L), this samples random PAIRS
of decoder atoms and fits the actual exponent p in |x|^p + |y|^p = 1 to the
anchor-medianed response contour, using the eval-frame estimator from
scripts/superellipse/soft_p_proxy.py (ray sweep -> threshold -> geodesic-
normalized intercepts -> Gauss-Newton p fit). p≈2 = no interference; p>2 = the
super-additive geometry we want.

Run on baseline and cull SAEs and compare the p distributions.

Usage:
    python -u scripts/dictionary_learning/eval_sae_full_p.py \
        --sae results/trained_sae/l10_doubled_k32_2M/sae_weights.pt \
        --model Qwen/Qwen3-1.7B --layer 10 \
        --n_dir 40 --n_contrast 8 --n_anchor 8 \
        --out results/pcull_eval/baseline_p.pkl
"""

import argparse
import importlib.util
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch

from src.model import load_model, _get_blocks
from src.autoencoders import load_sae


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

pcull = _load(os.path.join(os.path.dirname(__file__), 'train_sae_pcull.py'), 'pcull')
spp = _load('scripts/superellipse/soft_p_proxy.py', 'soft_p_proxy')


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sae', required=True)
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--layer', type=int, default=10)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--n_dir', type=int, default=40, help='number of d atoms sampled')
    ap.add_argument('--n_contrast', type=int, default=8, help='contrasts per d')
    ap.add_argument('--n_anchor', type=int, default=8)
    ap.add_argument('--n_rays', type=int, default=13, help='phi grid points [0,90]deg')
    ap.add_argument('--n_radii', type=int, default=40)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--p_cap', type=float, default=10.0)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--anchor_seed', type=int, default=999)
    ap.add_argument('--dir_seed', type=int, default=0)
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--pairs_independent', action='store_true',
                    help='sample n_dir*n_contrast pairs with BOTH atoms drawn '
                         'independently per pair (instead of n_dir atoms x '
                         'n_contrast shared partners)')
    ap.add_argument('--alive_fires', default=None,
                    help='path to per-latent fires tensor (.pt) from '
                         'eval_sae_dead_frac.py; restrict atom sampling to '
                         'latents with fires > 0')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = args.measure_layer if args.measure_layer is not None else n_layers - 2
    print(f"model={args.model} layer={args.layer} measure_layer={measure_layer}")

    sae = load_sae(args.sae, device)
    d_sae = sae.d_sae
    W = torch.nn.functional.normalize(sae.W_dec.detach(), dim=-1)
    print(f"SAE d_sae={d_sae}, k={sae.k}")

    acts, ctxs, f_refs = pcull.build_anchor_pool(
        model, tokenizer, args.layer, args.n_anchor, args.seq_len,
        args.anchor_seed, device, measure_layer)
    acts_stack = torch.stack(acts)                  # [A, d] f32
    ctx_stack = torch.cat(ctxs, 0)                  # [A, seq-1, d] model dtype
    f_refs_stack = torch.stack(f_refs)              # [A, d] f32

    phis_rad, radii_rad = spp.default_phis_radii(args.n_rays, args.n_radii, args.max_angle)

    g = torch.Generator().manual_seed(args.dir_seed)

    # Sampling pool: all latents, or only those alive on data
    pool = torch.arange(d_sae)
    if args.alive_fires:
        fires = torch.load(args.alive_fires, map_location='cpu', weights_only=True)
        assert fires.shape[0] == d_sae, f"fires len {fires.shape[0]} != d_sae {d_sae}"
        pool = (fires > 0).nonzero(as_tuple=True)[0]
        print(f"alive-only sampling: pool {len(pool)}/{d_sae}")

    all_p, all_t1, all_t2 = [], [], []
    if args.pairs_independent:
        # n_dir*n_contrast pairs, both atoms drawn independently per pair
        n_pairs = args.n_dir * args.n_contrast
        d_idx = pool[torch.randint(0, len(pool), (n_pairs,), generator=g)]
        c_idx = pool[torch.randint(0, len(pool), (n_pairs,), generator=g)]
        coll = c_idx == d_idx
        while coll.any():
            c_idx[coll] = pool[torch.randint(0, len(pool), (int(coll.sum()),), generator=g)]
            coll = c_idx == d_idx
        for j, (di, ci) in enumerate(zip(d_idx.tolist(), c_idx.tolist())):
            _, diag = spp.soft_p_estimate(
                model, acts_stack, ctx_stack, f_refs_stack, W[di],
                W[ci].unsqueeze(0), args.layer, measure_layer, phis_rad,
                radii_rad, p_cap=args.p_cap, anchor_reduce='median',
                chunk=args.chunk, return_diag=True)
            all_p.append(diag['p_k'].cpu().numpy())
            all_t1.append(diag['theta1_deg'].cpu().numpy())
            all_t2.append(diag['theta2_deg'].cpu().numpy())
            if (j + 1) % 40 == 0:
                print(f"  {j+1}/{n_pairs} pairs done")
        pairs_meta = {'d_idx': d_idx.numpy(), 'c_idx': c_idx.numpy()}
    else:
        d_idx = pool[torch.randperm(len(pool), generator=g)[:args.n_dir]]
        for j, di in enumerate(d_idx.tolist()):
            # contrasts: sample n_contrast distinct atoms != di
            c_idx = pool[torch.randperm(len(pool), generator=g)[:args.n_contrast + 1]]
            c_idx = c_idx[c_idx != di][:args.n_contrast]
            d_vec = W[di]
            contrast = W[c_idx.to(device)]
            _, diag = spp.soft_p_estimate(
                model, acts_stack, ctx_stack, f_refs_stack, d_vec, contrast,
                args.layer, measure_layer, phis_rad, radii_rad,
                p_cap=args.p_cap, anchor_reduce='median', chunk=args.chunk,
                return_diag=True)
            all_p.append(diag['p_k'].cpu().numpy())
            all_t1.append(diag['theta1_deg'].cpu().numpy())
            all_t2.append(diag['theta2_deg'].cpu().numpy())
            if (j + 1) % 10 == 0:
                print(f"  {j+1}/{args.n_dir} dirs done")
        pairs_meta = {'d_idx': d_idx.numpy()}

    p = np.concatenate(all_p)
    t1 = np.concatenate(all_t1); t2 = np.concatenate(all_t2)
    q = np.quantile(p, [0.1, 0.25, 0.5, 0.75, 0.9])
    summary = {
        'sae': args.sae, 'n_pairs': int(p.shape[0]), 'n_anchor': args.n_anchor,
        'layer': args.layer, 'measure_layer': measure_layer, 'max_angle': args.max_angle,
        'p_mean': float(p.mean()), 'p_median': float(q[2]),
        'p_p10': float(q[0]), 'p_p25': float(q[1]), 'p_p75': float(q[3]), 'p_p90': float(q[4]),
        'frac_p_gt_2_5': float((p > 2.5).mean()), 'frac_p_gt_3': float((p > 3.0).mean()),
        'frac_at_cap': float((p >= args.p_cap - 1e-3).mean()),
        'theta1_med': float(np.median(t1)), 'theta2_med': float(np.median(t2)),
    }
    with open(args.out, 'wb') as f:
        pickle.dump({'p': p, 'theta1': t1, 'theta2': t2, 'summary': summary,
                     'pairs': pairs_meta, 'args': vars(args)}, f)
    print("\n=== full-p summary ===")
    for kk, vv in summary.items():
        print(f"  {kk}: {vv}")
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
