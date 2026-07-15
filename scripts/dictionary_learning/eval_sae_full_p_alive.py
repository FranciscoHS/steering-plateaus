#!/usr/bin/env python3
"""Full superellipse-p eval (ray fit) restricted to ALIVE decoder atoms.

Same estimator and setup as eval_sae_full_p.py (soft_p_estimate, 8 anchors seed
999, 13 rays x 40 radii, max 60deg, eval frame), but both the sampled d atoms
and their contrasts are drawn from atoms that fire at least once on cached
activations — the original eval sampled from the full dictionary, which is
~75% dead atoms, and its alive-alive subset was tiny (n~26) and clustered.

Firing counts are computed here from --cache_path (TopKSAE encode, value>0 in
top-k), same criterion as sae_vs_sae_alignment.py.

Usage:
    python -u scripts/dictionary_learning/eval_sae_full_p_alive.py \\
        --sae results/trained_sae_popt/l10_lam1/sae_weights.pt \\
        --cache_path .../cached_activations.pt \\
        --n_dir 80 --n_contrast 8 --out results/pcull_eval_alive/popt_p.pkl
"""
import argparse
import importlib.util
import json
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch

from src.model import load_model, _get_blocks


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

pcull = _load('scripts/dictionary_learning/train_sae_pcull.py', 'pcull')
spp = _load('scripts/superellipse/soft_p_proxy.py', 'soft_p_proxy')
svsa = _load('scripts/dictionary_learning/sae_vs_sae_alignment.py', 'svsa')


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sae', required=True)
    ap.add_argument('--cache_path', required=True)
    ap.add_argument('--n_cache_rows', type=int, default=200000)
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--layer', type=int, default=10)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--n_dir', type=int, default=80)
    ap.add_argument('--n_contrast', type=int, default=8)
    ap.add_argument('--n_anchor', type=int, default=8)
    ap.add_argument('--n_rays', type=int, default=13)
    ap.add_argument('--n_radii', type=int, default=40)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--p_cap', type=float, default=10.0)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--anchor_seed', type=int, default=999)
    ap.add_argument('--dir_seed', type=int, default=0)
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # firing counts -> alive mask
    W, sd, cfg = svsa.load_decoder(args.sae)
    meta = json.load(open(args.cache_path + '.meta.json'))
    mm = np.memmap(args.cache_path, dtype=np.dtype(meta['dtype']), mode='r',
                   shape=tuple(meta['shape']))
    X = torch.from_numpy(np.array(mm[:args.n_cache_rows]))
    fires = svsa.firing_counts(sd, X, cfg.get('k', 32), device)
    alive = np.where(fires > 0)[0]
    del X
    print(f'{args.sae}: {W.shape[0]} atoms, {alive.size} alive '
          f'({alive.size / W.shape[0]:.1%})', flush=True)

    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    print(f'model={args.model} L{args.layer}->{measure_layer}', flush=True)

    W = W.to(device)
    acts, ctxs, f_refs = pcull.build_anchor_pool(
        model, tokenizer, args.layer, args.n_anchor, args.seq_len,
        args.anchor_seed, device, measure_layer)
    acts_stack = torch.stack(acts)
    ctx_stack = torch.cat(ctxs, 0)
    f_refs_stack = torch.stack(f_refs)
    phis_rad, radii_rad = spp.default_phis_radii(
        args.n_rays, args.n_radii, args.max_angle)

    rng = np.random.default_rng(args.dir_seed)
    d_idx = rng.choice(alive, size=args.n_dir, replace=False)

    all_p, pair_d, pair_c = [], [], []
    for j, di in enumerate(d_idx.tolist()):
        c_idx = rng.choice(alive[alive != di], size=args.n_contrast,
                           replace=False)
        _, diag = spp.soft_p_estimate(
            model, acts_stack, ctx_stack, f_refs_stack, W[di],
            W[torch.from_numpy(c_idx).to(device)],
            args.layer, measure_layer, phis_rad, radii_rad,
            p_cap=args.p_cap, anchor_reduce='median', chunk=args.chunk,
            return_diag=True)
        all_p.append(diag['p_k'].cpu().numpy())
        pair_d += [di] * args.n_contrast
        pair_c += c_idx.tolist()
        if (j + 1) % 10 == 0:
            print(f'  {j+1}/{args.n_dir} dirs done', flush=True)

    p = np.concatenate(all_p)
    q = np.quantile(p, [0.1, 0.25, 0.5, 0.75, 0.9])
    summary = {
        'sae': args.sae, 'n_pairs': int(p.shape[0]), 'n_alive': int(alive.size),
        'p_mean': float(p.mean()), 'p_median': float(q[2]),
        'p_p10': float(q[0]), 'p_p25': float(q[1]), 'p_p75': float(q[3]),
        'p_p90': float(q[4]),
        'frac_p_gt_2_5': float((p > 2.5).mean()),
        'frac_p_gt_3': float((p > 3.0).mean()),
        'frac_at_cap': float((p >= args.p_cap - 1e-3).mean()),
    }
    with open(args.out, 'wb') as f:
        pickle.dump({'p': p, 'pair_d': np.array(pair_d),
                     'pair_c': np.array(pair_c), 'fires': fires,
                     'summary': summary, 'args': vars(args)}, f)
    print('\n=== alive-only full-p summary ===')
    for kk, vv in summary.items():
        print(f'  {kk}: {vv}')
    print(f'\nSaved: {args.out}', flush=True)


if __name__ == '__main__':
    main()
