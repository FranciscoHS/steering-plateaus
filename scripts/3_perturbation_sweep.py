#!/usr/bin/env python3
"""Step 3: 1D spherical perturbation sweeps on real, random and
Jacobian-optimized activations.

(Originally scripts/superellipse/eval_real_vs_random_acts.py, trimmed to the
conditions used here.)

Each anchor activation a (last token, at --layer) is rotated on the sphere of
radius ||a|| towards a random direction by 0..max_angle degrees, and we record
the L2 distance of the last-token residual at --measure_layer from its value
at the unperturbed anchor. Conditions:
  real        FineWeb last-token activations, each in its own context
  random      random activations (--rand_mode norm: isotropic, norm-matched
              per anchor; cov: Gaussian fitted to real mean + covariance)
  extra sets  any .pt passed with --extra_anchors_pt, e.g. the Jacobian-
              optimized points from 2_minimize_jacobian.py
All conditions use the same contexts and the same random directions.
CAVEAT: anchor i of every condition is evaluated in FineWeb context i of
THIS run, which is not the context the optimized point was optimized in.

Usage (from the repo root):
    python scripts/3_perturbation_sweep.py --model qwen3-1.7b --layer 2 \
        --rand_mode cov \
        --extra_anchors_pt results/intrinsic_dim/jac_optimized_data.pt \
        --extra_label "jac-optimized" \
        --out results/perturbation_sweep/qwen3-1.7b_L2

Output in --out:
  - results.pkl with L2(angle) per (anchor, direction) for each condition
  - sweep.png with median + IQR per condition
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
from src.data import load_fineweb_fixed_length, sample_random_activations_v2


def ctx_acts_from_token_lists(model, device, layer, token_lists):
    contexts, activations = [], []
    for tids in token_lists:
        tids = tids.unsqueeze(0).to(device)
        with torch.no_grad():
            outs = model(tids, output_hidden_states=True)
        h = outs.hidden_states[layer + 1]
        contexts.append(h[:, :-1, :].float().cpu())
        activations.append(h[:, -1, :].squeeze(0).float().cpu())
    return contexts, torch.stack(activations)


def fineweb_ctx_acts(model, tokenizer, device, layer, n_act, seq_len, seed):
    cache_dir = 'fineweb_cache'
    os.makedirs(cache_dir, exist_ok=True)
    token_lists = load_fineweb_fixed_length(
        n_act, tokenizer, seq_len=seq_len, seed=seed, cache_dir=cache_dir)
    return ctx_acts_from_token_lists(model, device, layer, token_lists)


def sweep_1d(model, ctx_d, a_dev, d_dev, layer, measure_layer,
             angles_rad, model_dtype, device, batch=64):
    """Spherical L2 sweep at one anchor along direction d. Returns L2[n_steps]."""
    R = a_dev.norm(); a_unit = a_dev / R
    d_o = F.normalize(d_dev - (d_dev @ a_unit) * a_unit, dim=0)
    n_steps = len(angles_rad)
    s = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)
    c = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    pts32 = R * (c[:, None] * a_unit[None, :] + s[:, None] * d_o[None, :])
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
    ap.add_argument('--model', required=True)
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--n_act', type=int, default=30)
    ap.add_argument('--seq_len', type=int, default=5)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--n_steps', type=int, default=31)
    ap.add_argument('--n_dirs', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rand_seed', type=int, default=0)
    ap.add_argument('--extra_anchors_pt', action='append', default=None,
                    help='.pt file with extra synthetic anchors to sweep as '
                         'an additional condition (e.g. '
                         'jac_optimized_data.pt); the first n_act rows are '
                         'used, paired with the same contexts and random '
                         'directions. Repeat the flag for several sets.')
    ap.add_argument('--extra_anchors_key', action='append', default=None,
                    help='key of the anchor tensor inside the .pt (ignored '
                         'if the file holds a raw tensor); one per '
                         '--extra_anchors_pt (default X_opt)')
    ap.add_argument('--extra_label', action='append', default=None,
                    help='legend label, one per --extra_anchors_pt')
    ap.add_argument('--rand_mode', default='norm', choices=['norm', 'cov'],
                    help="how the random-activation baseline is drawn. 'norm': "
                         "isotropic, matched to each anchor's norm (the "
                         "original behaviour, kept as default so earlier runs "
                         "reproduce). 'cov': Gaussian fitted to a pool of real "
                         "activations, matching mean and full covariance.")
    ap.add_argument('--rand_cov_pool', type=int, default=2000,
                    help='--rand_mode cov: real activations used for the fit')
    ap.add_argument('--rand_cov_seed', type=int, default=43,
                    help='--rand_mode cov: FineWeb seed for that pool, kept '
                         'distinct from --seed so the pool and the anchors are '
                         'different samples')
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

    print(f'Extracting {args.n_act} FineWeb anchors '
          f'(seq_len={args.seq_len}, seed={args.seed})', flush=True)
    contexts, activations = fineweb_ctx_acts(model, tokenizer, device,
                                              args.layer, args.n_act,
                                              args.seq_len, args.seed)
    a_norms = activations.norm(dim=-1)
    print(f'  mean ||a||={a_norms.mean().item():.2f}  '
          f'std={a_norms.std().item():.2f}', flush=True)

    # Random "activations". 'norm': same per-anchor norm, random unit direction
    # (isotropic). 'cov': drawn from a Gaussian fitted to a pool of real
    # activations, so the anisotropy of the residual stream is matched too —
    # a much stronger off-manifold baseline than the isotropic one.
    g_act = torch.Generator(device=device).manual_seed(args.rand_seed)
    if args.rand_mode == 'cov':
        print(f'Fitting cov-matched Gaussian on {args.rand_cov_pool} FineWeb '
              f'activations (seed={args.rand_cov_seed})', flush=True)
        _, pool_acts = fineweb_ctx_acts(model, tokenizer, device, args.layer,
                                        args.rand_cov_pool, args.seq_len,
                                        args.rand_cov_seed)
        rand_acts, _ = sample_random_activations_v2(
            pool_acts.float(), args.n_act, mode='full', seed=args.rand_seed,
            device=str(device))
        rand_acts = rand_acts.float().cpu()
    else:
        rand_acts = torch.randn(args.n_act, d_model, generator=g_act,
                                device=device, dtype=torch.float32)
        rand_acts = rand_acts / rand_acts.norm(dim=-1, keepdim=True)
        rand_acts = rand_acts * a_norms.to(device).unsqueeze(-1)
        rand_acts = rand_acts.cpu()
    print(f'  random activations ({args.rand_mode}-matched): '
          f'mean||a||={rand_acts.norm(dim=-1).mean().item():.2f}', flush=True)

    # Random unit directions (n_dirs of them, shared across all anchors)
    g_d = torch.Generator(device=device).manual_seed(args.rand_seed + 1)
    rand_dirs = torch.randn(args.n_dirs, d_model, generator=g_d, device=device,
                            dtype=torch.float32)
    rand_dirs = rand_dirs / rand_dirs.norm(dim=-1, keepdim=True)
    rand_dirs = rand_dirs.cpu()

    # Optional extra anchor sets from tensor files (e.g. Jacobian-optimized
    # points). NOTE: paired with THIS run's contexts, which need not be the
    # contexts the points were built/optimized with.
    extra_sets = []          # list of (label, [n_act, d] tensor)
    if args.extra_anchors_pt:
        keys = args.extra_anchors_key or []
        labels = args.extra_label or []
        for j, pt in enumerate(args.extra_anchors_pt):
            key = keys[j] if j < len(keys) else 'X_opt'
            label = (labels[j] if j < len(labels)
                     else os.path.basename(pt).rsplit('.', 1)[0])
            eb = torch.load(pt, map_location='cpu', weights_only=False)
            acts = (eb[key] if isinstance(eb, dict) else eb).float()
            if acts.shape[0] < args.n_act:
                raise SystemExit(f'extra anchors {pt}: only {acts.shape[0]} '
                                 f'rows for n_act={args.n_act}')
            acts = acts[:args.n_act]
            extra_sets.append((label, acts))
            print(f'Extra anchors ({label}): {tuple(acts.shape)} '
                  f'mean||a||={acts.norm(dim=-1).mean():.2f}', flush=True)

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)
    print(f'Sweep angles 0..{args.max_angle}° in {args.n_steps} steps  '
          f'n_dirs={args.n_dirs}', flush=True)

    out_real = np.zeros((args.n_act, args.n_dirs, args.n_steps),
                        dtype=np.float32)
    out_rand = np.zeros((args.n_act, args.n_dirs, args.n_steps),
                        dtype=np.float32)
    out_extras = [np.zeros((args.n_act, args.n_dirs, args.n_steps),
                           dtype=np.float32) for _ in extra_sets]

    t0 = time.time()
    for i in range(args.n_act):
        ctx = contexts[i].to(device=device, dtype=model_dtype)
        a_real = activations[i].to(device=device, dtype=torch.float32)
        a_rand = rand_acts[i].to(device=device, dtype=torch.float32)
        for r in range(args.n_dirs):
            d = rand_dirs[r].to(device=device, dtype=torch.float32)
            out_real[i, r] = sweep_1d(model, ctx, a_real, d, args.layer,
                                      measure_layer, angles_rad,
                                      model_dtype, device)
            out_rand[i, r] = sweep_1d(model, ctx, a_rand, d, args.layer,
                                      measure_layer, angles_rad,
                                      model_dtype, device)
            for j, (_, acts) in enumerate(extra_sets):
                a_extra = acts[i].to(device=device, dtype=torch.float32)
                out_extras[j][i, r] = sweep_1d(model, ctx, a_extra, d,
                                               args.layer, measure_layer,
                                               angles_rad, model_dtype,
                                               device)
        if (i + 1) % 5 == 0 or i == args.n_act - 1:
            print(f'  anchor {i+1}/{args.n_act}  elapsed={time.time()-t0:.1f}s',
                  flush=True)

    out_pkl = os.path.join(args.out, 'results.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'args': vars(args),
            'angles_deg': angles_deg,
            'measure_layer': measure_layer,
            'real_l2': out_real,    # [n_act, n_dirs, n_steps]
            'rand_l2': out_rand,
            'extra_l2': out_extras,  # list of [n_act, n_dirs, n_steps]
            'extra_labels': [lab for lab, _ in extra_sets],
            'a_norms': a_norms.numpy(),
        }, f)

    real_flat = out_real.reshape(-1, args.n_steps)
    rand_flat = out_rand.reshape(-1, args.n_steps)
    series = [(real_flat, 'C0', 'real activation'),
              (rand_flat, 'C1',
               f'{args.rand_mode}-matched random activation')]
    extra_colors = ['C2', 'C4', 'C5', 'C6', 'C7']
    for j, (lab, _) in enumerate(extra_sets):
        series.append((out_extras[j].reshape(-1, args.n_steps),
                       extra_colors[j % len(extra_colors)], lab))
    fig, ax = plt.subplots(figsize=(8, 5))
    for arr, col, lab in series:
        m = np.median(arr, axis=0)
        q25 = np.quantile(arr, 0.25, axis=0)
        q75 = np.quantile(arr, 0.75, axis=0)
        ax.plot(angles_deg, m, '-', color=col, label=lab)
        ax.fill_between(angles_deg, q25, q75, alpha=0.2, color=col)
    ax.set_xlabel('perturbation angle (deg)')
    ax.set_ylabel(r'L2 at L$_{measure}$  ($\|f(a_\theta)-f(a)\|$)')
    ax.set_title(f'{args.model}  L{args.layer}->{measure_layer}  '
                 f'(seq_len={args.seq_len}, n_act={args.n_act}×n_dirs={args.n_dirs})')
    ax.legend(); ax.grid(True, alpha=0.3)
    plot_path = os.path.join(args.out, 'sweep.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    # Console summary: median L2 per condition at sample angles.
    print('\n=== median L2 at sample angles ===')
    cols = [('real', real_flat), ('rand', rand_flat)]
    for j, (lab, _) in enumerate(extra_sets):
        cols.append((lab[:9], out_extras[j].reshape(-1, args.n_steps)))
    print('  angle  ' + '  '.join(f'{n:>9}' for n, _ in cols))
    for ang in [10, 20, 30, 45, 60]:
        if ang <= args.max_angle:
            j = int(np.argmin(np.abs(angles_deg - ang)))
            vals = '  '.join(f'{np.median(a[:, j]):9.2f}' for _, a in cols)
            print(f'  {angles_deg[j]:5.1f}  {vals}')

    print(f'\nSaved {out_pkl}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
