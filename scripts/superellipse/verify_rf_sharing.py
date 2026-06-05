#!/usr/bin/env python3
"""Verify the R_f-sharing optimization in batched_ratio_multiangle.

Step 0 of Plans/single30_ratio_dictionary_build.md. The d-alone response R_f is
now computed once per (angle, activation) instead of once per contrast. The
perturbation point for R_f is bit-identical to the old td[...,0] entry, so the
loss, the R_f/R_c/R_mix monitors, AND the gradient w.r.t. d should all match the
old implementation (up to bf16 kernel-batching noise).

This script inlines a reference copy of the PRE-optimization function and
compares it against the live (optimized) one on a small Qwen3-1.7B setup, for a
few random directions. It also times both so we can sanity-check the speedup.

Run on a GPU pod:
    python -u scripts/superellipse/verify_rf_sharing.py
"""

import sys
import time
sys.path.insert(0, '.')

import numpy as np
import torch

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)
from scripts.superellipse.decompose_plateau_breaking import (
    batched_ratio_multiangle,
)


def reference_ratio_multiangle(model, acts_stack, ctx_stack, f_refs_stack, d,
                               scaffold, angles_rad, layer, measure_layer,
                               chunk_k=None):
    """Pre-optimization implementation (computes R_f once per contrast).

    Verbatim copy of the old batched_ratio_multiangle, kept here only as the
    reference oracle for this check.
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

    cos_t = torch.tensor([float(np.cos(a)) for a in angles_rad],
                         device=device).view(n_ang, 1, 1, 1, 1)
    sin_t = torch.tensor([float(np.sin(a)) for a in angles_rad],
                         device=device).view(n_ang, 1, 1, 1, 1)

    if chunk_k is None or chunk_k >= K:
        chunk_k = K
    ratio_chunks = []
    accum_R_f = accum_R_c = accum_R_mix = 0.0
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

        td = torch.stack([d_p_e, d_cp, mix], dim=2)
        a_e = acts_stack.unsqueeze(0).unsqueeze(2).expand(Kc, n_act, 3, d_model)
        R_e = a_norm.unsqueeze(0).unsqueeze(2).expand(Kc, n_act, 3, 1)

        perturbed = (cos_t * a_e.unsqueeze(0)
                     + sin_t * (R_e * td).unsqueeze(0))
        N = n_ang * Kc * n_act * 3
        points = perturbed.reshape(N, 1, d_model).to(model_dtype)
        ctxs_b = (ctx_stack.view(1, 1, n_act, 1, T_minus_1, d_model)
                  .expand(n_ang, Kc, n_act, 3, T_minus_1, d_model)
                  .reshape(N, T_minus_1, d_model))

        f_pert = forward_from_layer_to_layer(
            model, ctxs_b, points, layer, measure_layer).float()
        f_pert = f_pert.reshape(n_ang, Kc, n_act, 3, d_model)
        f_refs_e = f_refs_stack.view(1, 1, n_act, 1, d_model).expand(
            n_ang, Kc, n_act, 3, d_model)
        responses = (f_pert - f_refs_e).norm(dim=-1)

        R_f = responses[..., 0]
        R_c = responses[..., 1]
        R_mix = responses[..., 2]
        ratio_chunks.append((R_mix / (R_f + R_c + 1e-8)).reshape(-1))
        accum_R_f += R_f.detach().mean().item() * Kc
        accum_R_c += R_c.detach().mean().item() * Kc
        accum_R_mix += R_mix.detach().mean().item() * Kc

    loss = torch.cat(ratio_chunks).mean()
    return loss, (accum_R_f / K, accum_R_c / K, accum_R_mix / K)


def main():
    model_name = 'qwen3-1.7b'
    layer = 14
    n_act = 5            # matches the recipe's M=5 activation minibatch
    n_contrast = 4       # matches the recipe's 4 random contrasts
    angles_rad = [np.deg2rad(30.0)]
    n_trials = 3

    print(f'Loading {model_name} (bf16)', flush=True)
    model, tokenizer, device = load_model(model_name, dtype=torch.bfloat16)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    model_dtype = next(model.parameters()).dtype
    print(f'device={device} d_model={d_model} measure_layer={measure_layer}',
          flush=True)

    tok = load_fineweb_fixed_length(n_act, tokenizer, seq_len=10, seed=42)
    ctx_all, act_all = extract_context_and_activation(model, tok, layer, device)
    acts_stack = act_all.to(device=device, dtype=torch.float32)
    ctx_stack = ctx_all.to(device=device, dtype=model_dtype)
    with torch.no_grad():
        pts = acts_stack.unsqueeze(1).to(model_dtype)
        f_refs_stack = forward_from_layer_to_layer(
            model, ctx_stack, pts, layer, measure_layer).float()

    max_loss_diff = 0.0
    max_grad_cos_err = 0.0
    for t in range(n_trials):
        g = torch.Generator(device=device).manual_seed(100 + t)
        gc = torch.Generator(device=device).manual_seed(500 + t)
        contrast = torch.randn(n_contrast, d_model, generator=gc, device=device)
        contrast = (contrast / contrast.norm(dim=-1, keepdim=True)).float()

        # --- reference ---
        d_ref = torch.randn(d_model, generator=g, device=device)
        d_ref = (d_ref / d_ref.norm()).clone().requires_grad_(True)
        loss_ref, (rf_ref, rc_ref, rm_ref) = reference_ratio_multiangle(
            model, acts_stack, ctx_stack, f_refs_stack, d_ref, contrast,
            angles_rad, layer, measure_layer)
        loss_ref.backward()
        grad_ref = d_ref.grad.detach().clone()

        # --- optimized (same init d) ---
        d_new = d_ref.detach().clone().requires_grad_(True)
        loss_new, (rf_new, rc_new, rm_new) = batched_ratio_multiangle(
            model, acts_stack, ctx_stack, f_refs_stack, d_new, contrast,
            angles_rad, layer, measure_layer)
        loss_new.backward()
        grad_new = d_new.grad.detach().clone()

        ldiff = abs(loss_ref.item() - loss_new.item())
        gcos = torch.nn.functional.cosine_similarity(
            grad_ref, grad_new, dim=0).item()
        gnorm_ratio = (grad_new.norm() / (grad_ref.norm() + 1e-12)).item()
        max_loss_diff = max(max_loss_diff, ldiff)
        max_grad_cos_err = max(max_grad_cos_err, 1.0 - gcos)
        print(f'trial {t}: loss ref={loss_ref.item():.6f} '
              f'new={loss_new.item():.6f} |d|={ldiff:.2e}  '
              f'grad_cos={gcos:.6f} grad_norm_ratio={gnorm_ratio:.4f}  '
              f'R_f ref={rf_ref:.4f} new={rf_new:.4f}  '
              f'R_c ref={rc_ref:.4f} new={rc_new:.4f}  '
              f'R_mix ref={rm_ref:.4f} new={rm_new:.4f}', flush=True)

    print(f'\nmax |loss diff| over {n_trials} trials = {max_loss_diff:.2e}',
          flush=True)
    print(f'max (1 - grad cos) = {max_grad_cos_err:.2e}', flush=True)

    # --- timing: averaged over a handful of forward+backward calls ---
    def timeit(fn, reps=10):
        d = torch.randn(d_model, device=device)
        d = (d / d.norm()).requires_grad_(True)
        torch.cuda.synchronize() if device == 'cuda' else None
        t0 = time.time()
        for _ in range(reps):
            if d.grad is not None:
                d.grad = None
            loss, _ = fn(d)
            loss.backward()
        torch.cuda.synchronize() if device == 'cuda' else None
        return (time.time() - t0) / reps

    gc = torch.Generator(device=device).manual_seed(0)
    contrast = torch.randn(n_contrast, d_model, generator=gc, device=device)
    contrast = (contrast / contrast.norm(dim=-1, keepdim=True)).float()
    t_ref = timeit(lambda d: reference_ratio_multiangle(
        model, acts_stack, ctx_stack, f_refs_stack, d, contrast,
        angles_rad, layer, measure_layer))
    t_new = timeit(lambda d: batched_ratio_multiangle(
        model, acts_stack, ctx_stack, f_refs_stack, d, contrast,
        angles_rad, layer, measure_layer))
    print(f'\ntiming/step (fwd+bwd): ref={t_ref*1000:.1f}ms '
          f'new={t_new*1000:.1f}ms  speedup={t_ref/t_new:.2f}x', flush=True)

    ok = max_loss_diff < 1e-3 and max_grad_cos_err < 1e-3
    print(f'\n{"PASS" if ok else "FAIL"}: '
          f'loss & gradient match within bf16 tolerance.', flush=True)


if __name__ == '__main__':
    main()
