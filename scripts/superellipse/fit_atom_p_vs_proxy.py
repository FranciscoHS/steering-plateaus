#!/usr/bin/env python3
"""Does the proxy loss track p? For a sample of found atoms, measure a per-atom
p (atom vs random partner directions, averaged) and correlate it with that
atom's final proxy loss (final_resp from build metadata).

If the proxy is a p-maximizer, atoms with LOWER final_resp should have HIGHER
per-atom p -> negative correlation. No correlation -> proxy and p are decoupled.

Usage:
    python scripts/superellipse/fit_atom_p_vs_proxy.py \\
        --dict_dir results/dictionary/qwen3-1.7b_L14_K8000_randcontrast_stepresample \\
        --dict_file dictionary_snap_now.pt --n_atoms 60 --n_partners 3 \\
        --out .../p_vs_proxy
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
from scripts.superellipse.plot_boundary_pair import boundary_for_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_dir', required=True)
    ap.add_argument('--dict_file', default='dictionary.pt')
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_atoms', type=int, default=60)
    ap.add_argument('--n_partners', type=int, default=3)
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--atom_seed', type=int, default=1)
    ap.add_argument('--rand_seed', type=int, default=7)
    ap.add_argument('--n_steps_grid', type=int, default=31)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16}
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    d_model = model._plateau_info['d_model']

    blob = torch.load(os.path.join(args.dict_dir, args.dict_file),
                      map_location='cpu')
    D = blob['dictionary'].to(device=device, dtype=torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    N = D.shape[0]
    with open(os.path.join(args.dict_dir, 'metadata.pkl'), 'rb') as f:
        meta = pickle.load(f)
    resp = {a['k']: a['final_resp'] for a in meta['atoms']}
    print(f'Dict: {N} atoms, metadata has {len(resp)} final_resp', flush=True)

    # Sample atoms that have a recorded final_resp.
    rng = np.random.default_rng(args.atom_seed)
    cand = np.array([k for k in range(N) if k in resp])
    atoms = np.sort(rng.choice(cand, size=min(args.n_atoms, cand.size),
                               replace=False))

    # Random partner directions (shared pool across atoms).
    g = torch.Generator(device=device).manual_seed(args.rand_seed)
    P = torch.randn(args.n_partners, d_model, generator=g, device=device)
    P = P / P.norm(dim=-1, keepdim=True)

    # Anchors.
    tok = load_fineweb_fixed_length(args.n_anchors, tokenizer,
                                    seq_len=args.seq_len, seed=args.held_seed)
    ctx, act = extract_context_and_activation(model, tok, args.layer, device)
    contexts = [ctx[i:i + 1].float().cpu() for i in range(ctx.shape[0])]
    activations = act.float().cpu()

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps_grid)
    angles_rad = np.deg2rad(angles_deg)

    rows = []
    t0 = time.time()
    for n, k in enumerate(atoms):
        d1 = D[k].cpu()
        ps = []
        for m in range(args.n_partners):
            d2 = P[m].cpu()
            r = run_2d_per_anchors(model, contexts, activations, d1, d2,
                                   args.layer, measure_layer, device,
                                   angles_rad, mode='geodesic',
                                   single_batch_ref=True)
            median_grid = np.median(r['l2'], axis=0)
            b = boundary_for_set(angles_deg, median_grid, exact=True)
            ps.append(b['fit']['p'])
        ps = np.array(ps, float)
        rows.append({'atom': int(k), 'p_mean': float(np.nanmean(ps)),
                     'p_each': ps.tolist(), 'final_resp': float(resp[int(k)])})
        if (n + 1) % 10 == 0 or n == len(atoms) - 1:
            print(f'  {n+1}/{len(atoms)}  atom {k}  p={np.nanmean(ps):.3f}  '
                  f'resp={resp[int(k)]:.4f}  elapsed={time.time()-t0:.0f}s',
                  flush=True)

    p_mean = np.array([r['p_mean'] for r in rows])
    resp_arr = np.array([r['final_resp'] for r in rows])
    ok = np.isfinite(p_mean) & np.isfinite(resp_arr)
    pm, ra = p_mean[ok], resp_arr[ok]
    # Pearson + Spearman.
    pear = float(np.corrcoef(pm, ra)[0, 1]) if pm.size > 2 else float('nan')
    rp = np.argsort(np.argsort(pm)); rr = np.argsort(np.argsort(ra))
    spear = float(np.corrcoef(rp, rr)[0, 1]) if pm.size > 2 else float('nan')

    print(f'\n=== per-atom p vs final_resp  (n={pm.size}) ===')
    print(f'  p:    mean={pm.mean():.3f}  sd={pm.std():.3f}  '
          f'min={pm.min():.3f}  max={pm.max():.3f}')
    print(f'  resp: mean={ra.mean():.4f}  sd={ra.std():.4f}')
    print(f'  Pearson(p, resp) ={pear:+.3f}   Spearman={spear:+.3f}')
    print(f'  (proxy tracks p  =>  expect NEGATIVE correlation)')

    with open(os.path.join(args.out, 'p_vs_proxy.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'rows': rows, 'pearson': pear,
                     'spearman': spear}, f)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ra, pm, s=20, alpha=0.7)
    ax.axhline(2.0, color='gray', ls='--', lw=1, label='p=2')
    ax.set_xlabel('final proxy loss  (final_resp; lower = better-optimized)')
    ax.set_ylabel('per-atom p (vs random partners)')
    ax.set_title(f'{args.model} L{args.layer}: proxy vs p  '
                 f'Pearson={pear:+.2f} Spearman={spear:+.2f} (n={pm.size})')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'p_vs_proxy.png'), dpi=150,
                bbox_inches='tight')
    print(f"\nSaved {os.path.join(args.out, 'p_vs_proxy.pkl')}")
    print(f"Saved {os.path.join(args.out, 'p_vs_proxy.png')}")


if __name__ == '__main__':
    main()
