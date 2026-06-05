#!/usr/bin/env python3
"""1D spherical perturbation sweep: contrastive directions vs random.

Sanity check: are there *any* plateaus in this model/layer? If yes,
contrastive directions should exhibit a saturating L2 response while
random directions should grow ~linearly with angle.

Template: scripts/perturbation/extended_1d_sweeps_75deg.py.

Outputs:
  - results.pkl with L2(angle) per (direction, anchor)
  - sweep.png with median+IQR for contrastive vs random

Usage:
    python scripts/superellipse/eval_plateaus_1d.py \\
        --model qwen3-1.7b --layer 14 \\
        --n_act 30 --seq_len 5 --max_angle 60 --n_steps 31 --n_random 10 \\
        --out results/dictionary/qwen3-1.7b_L14_K8000/plateaus_1d
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

from src.model import load_model, _get_blocks, forward_from_layer_to_layer
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_feature_dir_vs_contrastive import (
    load_or_build_scaffold,
)
from scripts.superellipse.build_dictionary import SEMANTIC_NAMES


def fineweb_ctx_acts(model, tokenizer, device, layer, n_act, seq_len, seed):
    """Last-token L2 anchors + their prefix contexts."""
    cache_dir = '/workspace/fineweb_cache'
    os.makedirs(cache_dir, exist_ok=True)
    token_lists = load_fineweb_fixed_length(
        n_act, tokenizer, seq_len=seq_len, seed=seed, cache_dir=cache_dir)
    contexts, activations = [], []
    for tids in token_lists:
        tids = tids.unsqueeze(0).to(device)
        with torch.no_grad():
            outs = model(tids, output_hidden_states=True)
        h = outs.hidden_states[layer + 1]
        contexts.append(h[:, :-1, :].float().cpu())
        activations.append(h[:, -1, :].squeeze(0).float().cpu())
    return contexts, torch.stack(activations)


def sweep_1d(model, ctx_d, a_dev, d_dev, layer, measure_layer,
             angles_rad, model_dtype, device, batch=64):
    """Spherical 1D L2 sweep at one anchor along direction d. Returns
    L2 (n_steps,)."""
    R = a_dev.norm(); a_unit = a_dev / R
    d_o = F.normalize(d_dev - (d_dev @ a_unit) * a_unit, dim=0)
    n_steps = len(angles_rad)
    s = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)
    c = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    pts32 = R * (c[:, None] * a_unit[None, :] + s[:, None] * d_o[None, :])  # [n, d]
    pts = pts32.to(model_dtype)
    ref = forward_from_layer_to_layer(
        model, ctx_d, a_dev[None, None].to(model_dtype),
        layer, measure_layer).float()
    out = np.zeros(n_steps, dtype=np.float32)
    for st in range(0, n_steps, batch):
        ed = min(n_steps, st + batch)
        pf = forward_from_layer_to_layer(
            model, ctx_d, pts[st:ed].unsqueeze(1),
            layer, measure_layer).float()
        out[st:ed] = (pf - ref).norm(dim=-1).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--n_act', type=int, default=30)
    ap.add_argument('--seq_len', type=int, default=5)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--n_steps', type=int, default=31)
    ap.add_argument('--n_random', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rand_seed', type=int, default=0)
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

    # Scaffold
    scaffold = load_or_build_scaffold(model, tokenizer, device, args.model,
                                      args.layer, SEMANTIC_NAMES)
    scaffold = scaffold / (scaffold.norm(dim=-1, keepdim=True) + 1e-12)
    print(f'Scaffold: shape={tuple(scaffold.shape)} '
          f'names={SEMANTIC_NAMES}', flush=True)

    # Random unit dirs
    g = torch.Generator(device=device).manual_seed(args.rand_seed)
    rand_dirs = torch.randn(args.n_random, d_model, generator=g, device=device,
                            dtype=torch.float32)
    rand_dirs = rand_dirs / rand_dirs.norm(dim=-1, keepdim=True)

    # Anchors
    print(f'Extracting {args.n_act} FineWeb anchors '
          f'(seq_len={args.seq_len}, seed={args.seed})', flush=True)
    contexts, activations = fineweb_ctx_acts(model, tokenizer, device,
                                              args.layer, args.n_act,
                                              args.seq_len, args.seed)

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)
    print(f'Sweep angles 0..{args.max_angle}° in {args.n_steps} steps',
          flush=True)

    # Run sweeps: contrastive (K_scaf, n_act, n_steps), random (n_rand, n_act, n_steps)
    K_scaf = scaffold.shape[0]
    out_contrast = np.zeros((K_scaf, args.n_act, args.n_steps), dtype=np.float32)
    out_random = np.zeros((args.n_random, args.n_act, args.n_steps),
                          dtype=np.float32)

    t0 = time.time()
    for k in range(K_scaf):
        d = scaffold[k].to(device=device, dtype=torch.float32)
        for i in range(args.n_act):
            ctx = contexts[i].to(device=device, dtype=model_dtype)
            a = activations[i].to(device=device)
            out_contrast[k, i] = sweep_1d(model, ctx, a, d, args.layer,
                                          measure_layer, angles_rad,
                                          model_dtype, device)
        if (k + 1) % 5 == 0 or k == K_scaf - 1:
            print(f'  contrastive {k+1}/{K_scaf} ({SEMANTIC_NAMES[k]})  '
                  f'elapsed={time.time()-t0:.1f}s', flush=True)

    t1 = time.time()
    for r in range(args.n_random):
        d = rand_dirs[r]
        for i in range(args.n_act):
            ctx = contexts[i].to(device=device, dtype=model_dtype)
            a = activations[i].to(device=device)
            out_random[r, i] = sweep_1d(model, ctx, a, d, args.layer,
                                        measure_layer, angles_rad,
                                        model_dtype, device)
        if (r + 1) % 2 == 0 or r == args.n_random - 1:
            print(f'  random {r+1}/{args.n_random}  '
                  f'elapsed={time.time()-t1:.1f}s', flush=True)

    out_pkl = os.path.join(args.out, 'results.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'args': vars(args),
            'angles_deg': angles_deg,
            'measure_layer': measure_layer,
            'names': SEMANTIC_NAMES,
            'contrastive_l2': out_contrast,   # [K_scaf, n_act, n_steps]
            'random_l2': out_random,           # [n_rand, n_act, n_steps]
        }, f)

    # Median across (direction, anchor) for each family
    contrast_flat = out_contrast.reshape(-1, args.n_steps)
    random_flat = out_random.reshape(-1, args.n_steps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Aggregate medians + IQR
    ax = axes[0]
    c_med = np.median(contrast_flat, axis=0)
    c_q25 = np.quantile(contrast_flat, 0.25, axis=0)
    c_q75 = np.quantile(contrast_flat, 0.75, axis=0)
    r_med = np.median(random_flat, axis=0)
    r_q25 = np.quantile(random_flat, 0.25, axis=0)
    r_q75 = np.quantile(random_flat, 0.75, axis=0)
    ax.plot(angles_deg, c_med, '-', color='C0',
            label=f'contrastive (median over {K_scaf}×{args.n_act} curves)')
    ax.fill_between(angles_deg, c_q25, c_q75, alpha=0.2, color='C0')
    ax.plot(angles_deg, r_med, '-', color='C1',
            label=f'random (median over {args.n_random}×{args.n_act} curves)')
    ax.fill_between(angles_deg, r_q25, r_q75, alpha=0.2, color='C1')
    ax.set_xlabel('perturbation angle (deg)')
    ax.set_ylabel(r'L2 at measure_layer ($\|\Delta\|$)')
    ax.set_title(f'1D plateau check: {args.model} L{args.layer}->{measure_layer}\n'
                 f'(seq_len={args.seq_len}, n_act={args.n_act})')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Per-contrastive-name median across anchors
    ax = axes[1]
    for k, name in enumerate(SEMANTIC_NAMES):
        m = np.median(out_contrast[k], axis=0)
        ax.plot(angles_deg, m, alpha=0.7, label=name)
    r_med_each = np.median(out_random, axis=1)   # [n_rand, n_steps]
    for r in range(args.n_random):
        ax.plot(angles_deg, r_med_each[r], '--', color='gray', alpha=0.4)
    ax.plot([], [], '--', color='gray', label='random (each, faded)')
    ax.set_xlabel('perturbation angle (deg)')
    ax.set_ylabel('median L2 across anchors')
    ax.set_title('Per-direction curves')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.out, 'sweep.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nSaved {out_pkl}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
