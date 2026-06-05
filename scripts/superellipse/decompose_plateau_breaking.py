#!/usr/bin/env python3
"""Greedy decomposition of an activation into plateau-breaking directions.

Same superellipse proxy as learn_feature_dir_vs_contrastive.py (R_mix/(R_f+R_c)
averaged over a contrastive scaffold on a set of opt-activations) + an overlap
term that pulls each direction toward the current residual of a separate
target activation:

    L(d_k) = mean_c [R_mix / (R_f + R_c)]
           + lam_scaffold * mean_c (d_k . d_c)^2
           + lam_prev     * mean_prev (d_k . d_prev)^2
           - lam_overlap  * (d_k . r_hat_{k-1})

Greedy outer loop:
    r <- a_target;  features <- []
    for k = 1, 2, ...:
        d_k <- argmin_d L(d) with current r
        alpha_k <- d_k . r
        features.append((d_k, alpha_k))
        r <- r - alpha_k * d_k
        if ||r||/||a_target|| < stop_frac: break

Stop condition default: residual <= 5% of ||a_target||. Safety cap: K_max.

Usage:
    python scripts/superellipse/decompose_plateau_breaking.py \\
        --model gemma-2-9b --layer 15 --n_act 10 --angles 30 \\
        --lam_overlap 0.1 --stop_frac 0.05 --k_max 20
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from src.data import load_fineweb_fixed_length

from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)
from scripts.superellipse.learn_feature_dir_vs_contrastive import (
    load_or_build_scaffold, DEFAULT_NAMES,
)


def batched_proxy_loss(model, acts_stack, ctx_stack, f_refs_stack, d,
                       scaffold, angle_rad, layer, measure_layer,
                       chunk_k=None):
    """Batched superellipse proxy: one forward over (K_scaffold, n_act, 3) points.

    acts_stack:    [n_act, d]         (float32, on device)
    ctx_stack:     [n_act, T-1, d]    (model_dtype, on device)
    f_refs_stack:  [n_act, d]         (float32, on device)
    d:             [d]                (float32, may require_grad)
    scaffold:      [K, d]             (float32, on device)
    chunk_k:       int or None        (chunk size across scaffold; None=all)

    Returns: (loss scalar, (mean R_f, mean R_c, mean R_mix))
    """
    K, d_model = scaffold.shape
    n_act, T_minus_1, _ = ctx_stack.shape
    model_dtype = next(model.parameters()).dtype

    a_norm = acts_stack.norm(dim=-1, keepdim=True)                  # [n_act, 1]
    a_hat = acts_stack / (a_norm + 1e-10)                           # [n_act, d]

    # d orthogonalized against each a_i (same across scaffold)
    d_exp = d.unsqueeze(0).expand(n_act, -1)                        # [n_act, d]
    d_p = d_exp - (d_exp * a_hat).sum(-1, keepdim=True) * a_hat
    d_p = d_p / (d_p.norm(dim=-1, keepdim=True) + 1e-10)            # [n_act, d]

    cos_t = float(np.cos(angle_rad))
    sin_t = float(np.sin(angle_rad))

    if chunk_k is None or chunk_k >= K:
        chunk_k = K
    accum_R_f_sq_sum = 0.0
    accum_R_c_sq_sum = 0.0
    accum_R_mix_sq_sum = 0.0
    ratio_chunks = []
    for k0 in range(0, K, chunk_k):
        k1 = min(K, k0 + chunk_k)
        scaf_chunk = scaffold[k0:k1]                                # [Kc, d]
        Kc = scaf_chunk.shape[0]

        # d_c orthogonalized against each a_i → [Kc, n_act, d]
        d_c_exp = scaf_chunk.unsqueeze(1).expand(Kc, n_act, d_model)
        a_hat_e = a_hat.unsqueeze(0).expand(Kc, n_act, d_model)
        d_cp = d_c_exp - (d_c_exp * a_hat_e).sum(-1, keepdim=True) * a_hat_e
        d_cp = d_cp / (d_cp.norm(dim=-1, keepdim=True) + 1e-10)

        # mix = normalize(d_p + d_cp)
        d_p_e = d_p.unsqueeze(0).expand(Kc, n_act, d_model)
        mix = d_p_e + d_cp
        mix = mix / (mix.norm(dim=-1, keepdim=True) + 1e-10)

        td = torch.stack([d_p_e, d_cp, mix], dim=2)                  # [Kc, n_act, 3, d]
        a_e = acts_stack.unsqueeze(0).unsqueeze(2).expand(Kc, n_act, 3, d_model)
        R_e = a_norm.unsqueeze(0).unsqueeze(2).expand(Kc, n_act, 3, 1)
        perturbed = cos_t * a_e + sin_t * R_e * td                   # [Kc, n_act, 3, d]

        N = Kc * n_act * 3
        points = perturbed.reshape(N, 1, d_model).to(model_dtype)
        ctxs_b = (ctx_stack.unsqueeze(0).unsqueeze(2)
                  .expand(Kc, n_act, 3, T_minus_1, d_model)
                  .reshape(N, T_minus_1, d_model))

        f_pert = forward_from_layer_to_layer(
            model, ctxs_b, points, layer, measure_layer).float()      # [N, d]
        f_pert = f_pert.reshape(Kc, n_act, 3, d_model)

        f_refs_e = f_refs_stack.unsqueeze(0).unsqueeze(2).expand(Kc, n_act, 3, d_model)
        responses = (f_pert - f_refs_e).norm(dim=-1)                  # [Kc, n_act, 3]

        R_f = responses[..., 0]
        R_c = responses[..., 1]
        R_mix = responses[..., 2]
        ratio_chunks.append((R_mix / (R_f + R_c + 1e-8)).reshape(-1))
        accum_R_f_sq_sum += R_f.detach().mean().item() * Kc
        accum_R_c_sq_sum += R_c.detach().mean().item() * Kc
        accum_R_mix_sq_sum += R_mix.detach().mean().item() * Kc

    loss = torch.cat(ratio_chunks).mean()
    mR_f = accum_R_f_sq_sum / K
    mR_c = accum_R_c_sq_sum / K
    mR_mix = accum_R_mix_sq_sum / K
    return loss, (mR_f, mR_c, mR_mix)


def batched_ratio_multiangle(model, acts_stack, ctx_stack, f_refs_stack, d,
                             scaffold, angles_rad, layer, measure_layer,
                             chunk_k=None):
    """Multi-magnitude ratio in ONE forward over all push magnitudes.

    Identical proxy to batched_proxy_loss (mean R_mix/(R_f+R_c)), but evaluates
    every angle in `angles_rad` in a single batched forward instead of one
    forward per angle — a big speedup since we're latency-bound. Numerically
    equivalent to averaging batched_proxy_loss over angles.

    R_f sharing: the d-alone response R_f does not depend on the contrast, so
    we evaluate it once per (angle, activation) instead of once per contrast.
    This cuts the forward-pass point count from n_ang*K*n_act*3 to
    n_ang*n_act*(1 + 2*K) — ~25% fewer at K=4. The d_p perturbation point is
    bit-identical to the old td[...,0] entry, so the loss is unchanged.

    Crucially the R_f points are *concatenated into the same forward* as the
    first contrast chunk (not a separate forward call): at the build's batch
    sizes we are forward-call-overhead-bound, so a second small forward would
    cost more than the points it saves. They ride along in one call.

    Returns (loss, (mean R_f, mean R_c, mean R_mix)).
    """
    K, d_model = scaffold.shape
    n_act, T_minus_1, _ = ctx_stack.shape
    model_dtype = next(model.parameters()).dtype
    device = acts_stack.device
    n_ang = len(angles_rad)

    a_norm = acts_stack.norm(dim=-1, keepdim=True)
    a_hat = acts_stack / (a_norm + 1e-10)

    d_exp = d.unsqueeze(0).expand(n_act, -1)
    d_p = d_exp - (d_exp * a_hat).sum(-1, keepdim=True) * a_hat
    d_p = d_p / (d_p.norm(dim=-1, keepdim=True) + 1e-10)

    cos_v = torch.tensor([float(np.cos(a)) for a in angles_rad], device=device)
    sin_v = torch.tensor([float(np.sin(a)) for a in angles_rad], device=device)

    # --- R_f points: d-alone response, independent of contrast. [n_ang,n_act,d]
    perturbed_f = (cos_v.view(n_ang, 1, 1) * acts_stack.unsqueeze(0)
                   + sin_v.view(n_ang, 1, 1)
                   * (a_norm.unsqueeze(0) * d_p.unsqueeze(0)))
    Nf = n_ang * n_act
    points_f = perturbed_f.reshape(Nf, 1, d_model).to(model_dtype)
    ctxs_f = (ctx_stack.view(1, n_act, T_minus_1, d_model)
              .expand(n_ang, n_act, T_minus_1, d_model)
              .reshape(Nf, T_minus_1, d_model))
    f_refs_f = f_refs_stack.view(1, n_act, d_model).expand(n_ang, n_act, d_model)

    # --- R_c and R_mix: per-contrast (2 perturbation dirs instead of 3). ---
    cos_t = cos_v.view(n_ang, 1, 1, 1, 1)
    sin_t = sin_v.view(n_ang, 1, 1, 1, 1)
    if chunk_k is None or chunk_k >= K:
        chunk_k = K
    ratio_chunks = []
    accum_R_c = accum_R_mix = 0.0
    R_f = None                       # computed in the first chunk's forward
    for k0 in range(0, K, chunk_k):
        k1 = min(K, k0 + chunk_k)
        scaf_chunk = scaffold[k0:k1]
        Kc = scaf_chunk.shape[0]

        d_c_exp = scaf_chunk.unsqueeze(1).expand(Kc, n_act, d_model)
        a_hat_e = a_hat.unsqueeze(0).expand(Kc, n_act, d_model)
        d_cp = d_c_exp - (d_c_exp * a_hat_e).sum(-1, keepdim=True) * a_hat_e
        d_cp = d_cp / (d_cp.norm(dim=-1, keepdim=True) + 1e-10)

        d_p_e = d_p.unsqueeze(0).expand(Kc, n_act, d_model)
        mix = d_p_e + d_cp
        mix = mix / (mix.norm(dim=-1, keepdim=True) + 1e-10)

        td = torch.stack([d_cp, mix], dim=2)                 # [Kc, n_act, 2, d]
        a_e = acts_stack.unsqueeze(0).unsqueeze(2).expand(Kc, n_act, 2, d_model)
        R_e = a_norm.unsqueeze(0).unsqueeze(2).expand(Kc, n_act, 2, 1)

        # broadcast over angles: [n_ang, Kc, n_act, 2, d]
        perturbed = (cos_t * a_e.unsqueeze(0)
                     + sin_t * (R_e * td).unsqueeze(0))
        N = n_ang * Kc * n_act * 2
        points = perturbed.reshape(N, 1, d_model).to(model_dtype)
        ctxs_b = (ctx_stack.view(1, 1, n_act, 1, T_minus_1, d_model)
                  .expand(n_ang, Kc, n_act, 2, T_minus_1, d_model)
                  .reshape(N, T_minus_1, d_model))

        # First chunk: prepend the R_f points so they share this forward call.
        if R_f is None:
            pts_all = torch.cat([points_f, points], dim=0)
            ctx_all = torch.cat([ctxs_f, ctxs_b], dim=0)
            out = forward_from_layer_to_layer(
                model, ctx_all, pts_all, layer, measure_layer).float()
            f_pert_f = out[:Nf].reshape(n_ang, n_act, d_model)
            R_f = (f_pert_f - f_refs_f).norm(dim=-1)         # [n_ang, n_act]
            f_pert = out[Nf:].reshape(n_ang, Kc, n_act, 2, d_model)
        else:
            f_pert = forward_from_layer_to_layer(
                model, ctxs_b, points, layer, measure_layer).float()
            f_pert = f_pert.reshape(n_ang, Kc, n_act, 2, d_model)

        f_refs_e = f_refs_stack.view(1, 1, n_act, 1, d_model).expand(
            n_ang, Kc, n_act, 2, d_model)
        responses = (f_pert - f_refs_e).norm(dim=-1)         # [n_ang,Kc,n_act,2]

        R_c = responses[..., 0]
        R_mix = responses[..., 1]
        R_f_b = R_f.unsqueeze(1)                             # [n_ang, 1, n_act]
        ratio_chunks.append((R_mix / (R_f_b + R_c + 1e-8)).reshape(-1))
        accum_R_c += R_c.detach().mean().item() * Kc
        accum_R_mix += R_mix.detach().mean().item() * Kc

    loss = torch.cat(ratio_chunks).mean()
    return loss, (R_f.detach().mean().item(), accum_R_c / K, accum_R_mix / K)


def optimize_one_feature(model, acts_stack, ctx_stack, f_refs_stack,
                         scaffold, prev_dirs, r_hat, angles_rad,
                         layer, measure_layer, device, d_model,
                         n_steps, lr, lam_scaffold, lam_prev, lam_overlap,
                         chunk_k, seed, log_prefix=""):
    """Optimize one direction d_k via the batched proxy + extras.

    scaffold:  [K, d] frozen unit contrastive directions.
    prev_dirs: list of frozen unit feature directions (previously found).
    r_hat:     [d] unit vector = current residual direction in a_target.

    Loss = proxy_batched (mean over scaffold, acts, angles)
           + lam_scaffold * mean_c (d . d_c)^2
           + lam_prev     * mean_prev (d . d_prev)^2
           - lam_overlap  * (d . r_hat)
    """
    g = torch.Generator(device=device).manual_seed(seed)
    d = torch.randn(d_model, generator=g, device=device)
    d = (d / d.norm()).requires_grad_(True)
    optimizer = torch.optim.Adam([d], lr=lr)

    r_hat = r_hat.to(device=device, dtype=torch.float32).detach()
    K = scaffold.shape[0]

    if prev_dirs:
        prev_stack = torch.stack(prev_dirs)                        # [P, d]
        prev_stack = prev_stack / (
            prev_stack.norm(dim=-1, keepdim=True) + 1e-10)
    else:
        prev_stack = None

    history = []
    for step in range(n_steps):
        optimizer.zero_grad()

        # Average proxy loss over angles
        n_a = len(angles_rad)
        resp_acc = 0.0
        mR_f = mR_c = mR_mix = 0.0
        for angle_rad in angles_rad:
            rl, (a_f, a_c, a_mix) = batched_proxy_loss(
                model, acts_stack, ctx_stack, f_refs_stack, d,
                scaffold, angle_rad, layer, measure_layer,
                chunk_k=chunk_k)
            resp_acc = resp_acc + rl
            mR_f += a_f; mR_c += a_c; mR_mix += a_mix
        resp_loss = resp_acc / n_a
        mR_f /= n_a; mR_c /= n_a; mR_mix /= n_a

        du = d / (d.norm() + 1e-10)
        scaffold_overlap = ((du.unsqueeze(0) * scaffold).sum(-1) ** 2).mean()
        if prev_stack is not None:
            prev_overlap = ((du.unsqueeze(0) * prev_stack).sum(-1) ** 2).mean()
        else:
            prev_overlap = torch.zeros((), device=device)
        align = (du @ r_hat)

        loss = (resp_loss
                + lam_scaffold * scaffold_overlap
                + lam_prev * prev_overlap
                - lam_overlap * align)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            d.data.div_(d.data.norm())

        history.append({
            'step': step, 'loss': loss.item(),
            'resp_loss': resp_loss.item(),
            'scaffold_overlap': scaffold_overlap.item(),
            'prev_overlap': prev_overlap.item(),
            'align': align.item(),
            'R_f': mR_f, 'R_c': mR_c, 'R_mix': mR_mix,
        })
        if step % 20 == 0 or step == n_steps - 1:
            print(f"  {log_prefix}step {step:4d}  L={loss.item():.4f}  "
                  f"ratio={resp_loss.item():.4f}  scaf_ov={scaffold_overlap.item():.4f}  "
                  f"prev_ov={prev_overlap.item():.4f}  align={align.item():+.4f}",
                  flush=True)

    return d.detach(), history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gemma-2-9b')
    ap.add_argument('--layer', type=int, default=15)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_act', type=int, default=10,
                    help='opt-set activations (for R_f/R_c/R_mix)')
    ap.add_argument('--scaffold_names', nargs='+', default=DEFAULT_NAMES)
    ap.add_argument('--scaffold_pkl', default=None)
    ap.add_argument('--angles', type=float, nargs='+', default=[30.0])
    ap.add_argument('--n_steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.003)
    ap.add_argument('--lam_scaffold', type=float, default=1.0)
    ap.add_argument('--lam_prev', type=float, default=1.0)
    ap.add_argument('--lam_overlap', type=float, default=0.1)
    ap.add_argument('--k_max', type=int, default=20,
                    help='safety cap on number of features')
    ap.add_argument('--stop_frac', type=float, default=0.05,
                    help='stop when ||r||/||a_target|| < this')
    ap.add_argument('--seed', type=int, default=42,
                    help='seed for opt-set fineweb + init')
    ap.add_argument('--target_seed', type=int, default=999,
                    help='disjoint seed for the held-out target activation')
    ap.add_argument('--target_idx', type=int, default=0,
                    help='which sample from the target_seed batch to use')
    ap.add_argument('--target_prompt', default=None,
                    help='if set, decompose the last-token activation of this '
                         'prompt instead of a FineWeb sequence')
    ap.add_argument('--batch_chunk_k', type=int, default=None,
                    help='chunk size across scaffold for batched forward '
                         '(default: full = K). Lower if OOM.')
    ap.add_argument('--results_dir', default=None)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    angle_tag = '-'.join(f'{a:g}' for a in args.angles)
    if args.results_dir is None:
        args.results_dir = (f'results/decomp_plateau/'
                            f'{args.model}_L{args.layer}_a{angle_tag}')
    os.makedirs(args.results_dir, exist_ok=True)

    angles_rad = [np.deg2rad(a) for a in args.angles]

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    print(f'Device: {device} d_model={d_model} n_layers={n_layers} '
          f'perturb_layer={args.layer} measure_layer={measure_layer}',
          flush=True)
    print(f'Config: n_act={args.n_act} scaffold={args.scaffold_names} '
          f'angles={args.angles} stop_frac={args.stop_frac} '
          f'k_max={args.k_max} lam_scaffold={args.lam_scaffold} '
          f'lam_prev={args.lam_prev} lam_overlap={args.lam_overlap}',
          flush=True)

    # Opt-set
    print(f'Loading opt-set ({args.n_act} FineWeb, seed={args.seed})', flush=True)
    tok_opt = load_fineweb_fixed_length(args.n_act, tokenizer,
                                        seq_len=args.seq_len, seed=args.seed)
    ctx_opt, act_opt = extract_context_and_activation(model, tok_opt,
                                                     args.layer, device)
    model_dtype = next(model.parameters()).dtype
    acts_stack = act_opt.to(device=device, dtype=torch.float32)        # [n_act, d]
    ctx_stack = ctx_opt.to(device=device, dtype=model_dtype)           # [n_act, T-1, d]
    print(f'  mean ||a_opt||={acts_stack.norm(dim=-1).mean().item():.2f}',
          flush=True)

    # Target (held-out) — either a FineWeb sequence or a custom prompt.
    if args.target_prompt is not None:
        print(f'Loading target (custom prompt: {args.target_prompt!r})',
              flush=True)
        tgt_ids = tokenizer(args.target_prompt,
                            return_tensors='pt').input_ids[0]
        print(f'  tokens: {tgt_ids.tolist()} '
              f'({tokenizer.decode(tgt_ids)!r})', flush=True)
        ctx_tgt, act_tgt = extract_context_and_activation(
            model, [tgt_ids], args.layer, device)
    else:
        print(f'Loading target (1 FineWeb, seed={args.target_seed}, '
              f'idx={args.target_idx})', flush=True)
        tok_tgt_batch = load_fineweb_fixed_length(
            args.target_idx + 1, tokenizer,
            seq_len=args.seq_len, seed=args.target_seed)
        ctx_tgt, act_tgt = extract_context_and_activation(
            model, [tok_tgt_batch[args.target_idx]], args.layer, device)
    a_target = act_tgt[0].to(device=device, dtype=torch.float32)
    a_target_norm = a_target.norm().item()
    print(f'  ||a_target||={a_target_norm:.2f}', flush=True)

    # Precompute f_refs for opt-set in a single batched forward.
    with torch.no_grad():
        pts = acts_stack.unsqueeze(1).to(model_dtype)                  # [n_act, 1, d]
        f_refs_stack = forward_from_layer_to_layer(
            model, ctx_stack, pts, args.layer, measure_layer).float()  # [n_act, d]

    # Scaffold
    if args.scaffold_pkl is not None:
        with open(args.scaffold_pkl, 'rb') as f:
            blob = pickle.load(f)
        scaffold = torch.stack([(blob['directions'][n].to(device).float() /
                                 blob['directions'][n].to(device).float().norm())
                                for n in args.scaffold_names])
    else:
        scaffold = load_or_build_scaffold(model, tokenizer, device,
                                          args.model, args.layer,
                                          args.scaffold_names)
    print(f'Scaffold shape: {tuple(scaffold.shape)}', flush=True)

    # Greedy decomposition
    r = a_target.clone()
    features = []
    alphas = []
    histories = []
    residual_norm_frac = [1.0]
    print('\n=== Greedy decomposition ===', flush=True)
    for k in range(args.k_max):
        r_norm = r.norm().item()
        frac = r_norm / a_target_norm
        print(f'\n--- feature {k+1}/{args.k_max}  '
              f'||r||/||a||={frac:.4f} ---', flush=True)
        r_hat = r / (r_norm + 1e-10)
        t0 = time.time()
        d_k, hist = optimize_one_feature(
            model, acts_stack, ctx_stack, f_refs_stack, scaffold, features,
            r_hat, angles_rad, args.layer, measure_layer, device,
            d_model, args.n_steps, args.lr,
            args.lam_scaffold, args.lam_prev, args.lam_overlap,
            args.batch_chunk_k, args.seed + k, log_prefix=f'd{k+1} ')
        alpha_k = (d_k @ r).item()
        print(f'  done in {time.time()-t0:.1f}s  '
              f'alpha_k={alpha_k:+.3f}  '
              f'cos(d,r)={(d_k @ r_hat).item():+.4f}', flush=True)
        features.append(d_k)
        alphas.append(alpha_k)
        histories.append(hist)
        r = r - alpha_k * d_k
        new_frac = r.norm().item() / a_target_norm
        residual_norm_frac.append(new_frac)
        print(f'  new ||r||/||a||={new_frac:.4f}', flush=True)

        # Checkpoint after each feature
        torch.save({'features': torch.stack(features).cpu(),
                    'alphas': np.array(alphas),
                    'residual_norm_frac': np.array(residual_norm_frac),
                    'args': vars(args)},
                   os.path.join(args.results_dir, 'checkpoint.pkl'))

        if new_frac < args.stop_frac:
            print(f'  STOP: residual {new_frac:.4f} < {args.stop_frac}',
                  flush=True)
            break

    K_used = len(features)
    print(f'\n=== Done: K_used = {K_used}, final ||r||/||a|| = '
          f'{residual_norm_frac[-1]:.4f} ===', flush=True)

    feats = torch.stack(features)  # [K, d]
    alphas = np.array(alphas)
    residual_norm_frac = np.array(residual_norm_frac)

    # Grams
    feat_gram = (feats @ feats.T).cpu().numpy()
    feat_scaf_gram = (feats @ scaffold.T).cpu().numpy()
    scaf_gram = (scaffold @ scaffold.T).cpu().numpy()

    results = {
        'args': vars(args),
        'features': feats.cpu(),
        'alphas': alphas,
        'residual_norm_frac': residual_norm_frac,
        'a_target': a_target.cpu(),
        'a_target_norm': a_target_norm,
        'scaffold': scaffold.cpu(),
        'scaffold_names': args.scaffold_names,
        'measure_layer': measure_layer,
        'angles_deg': args.angles,
        'loss_histories': histories,
        'feat_gram': feat_gram,
        'feat_scaffold_gram': feat_scaf_gram,
        'scaffold_gram': scaf_gram,
        'K_used': K_used,
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Residual trajectory
    ax = axes[0, 0]
    ax.plot(np.arange(len(residual_norm_frac)), residual_norm_frac, 'o-')
    ax.axhline(args.stop_frac, color='gray', ls=':', label=f'stop={args.stop_frac}')
    ax.set_xlabel('k (features used)'); ax.set_ylabel(r'$||r_k|| / ||a||$')
    ax.set_title('Residual norm fraction')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # |alpha_k|
    ax = axes[0, 1]
    x = np.arange(1, K_used + 1)
    ax.bar(x, np.abs(alphas), color='C0')
    ax.set_xlabel('k'); ax.set_ylabel(r'$|\alpha_k|$')
    ax.set_title('Per-feature scalar projection (abs)')
    ax.grid(True, alpha=0.3)

    # Loss curves
    ax = axes[1, 0]
    for k, hist in enumerate(histories):
        ax.plot([h['step'] for h in hist], [h['loss'] for h in hist],
                label=f'd{k+1}', alpha=0.7)
    ax.set_xlabel('step'); ax.set_ylabel('loss')
    ax.set_title('Per-feature optimisation loss')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # Feature x scaffold gram heatmap
    ax = axes[1, 1]
    im = ax.imshow(feat_scaf_gram, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(len(args.scaffold_names)))
    ax.set_xticklabels(args.scaffold_names, rotation=45, ha='right')
    ax.set_yticks(range(K_used))
    ax.set_yticklabels([f'd{k+1}' for k in range(K_used)])
    ax.set_title(r'$d_k \cdot d_{scaffold}$')
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f'{args.model} L{args.layer}->{measure_layer}, '
                 f'K_used={K_used}, alpha={args.angles} deg, '
                 f'lam_overlap={args.lam_overlap}', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved {os.path.join(args.results_dir, "results.pkl")}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
