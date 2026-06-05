#!/usr/bin/env python3
"""Why is the new ratio recipe slower per-atom than the old one?

Times a full 150-step single-atom optimization under combinations of:
  - activation: fixed-per-atom (old) vs per-step resample M=5 (new)
  - contrast:   fixed-per-atom (old) vs per-step resample (new)
isolating whether the cost is the extra forward points or the per-step
resampling overhead (CPU RNG + tensor creation serializing with the GPU).

Run on a GPU pod:
    python -u scripts/superellipse/bench_atom_recipe.py
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


def main():
    model_name, layer, seq_len, n_steps = 'qwen3-1.7b', 14, 10, 150
    n_pool = 200
    angles_rad = [np.deg2rad(30.0)]
    print(f'Loading {model_name} (bf16)', flush=True)
    model, tokenizer, device = load_model(model_name, dtype=torch.bfloat16)
    d_model = model._plateau_info['d_model']
    measure_layer = len(_get_blocks(model)) - 2
    model_dtype = next(model.parameters()).dtype

    tok = load_fineweb_fixed_length(n_pool, tokenizer, seq_len=seq_len, seed=42)
    ctx_all, act_all = extract_context_and_activation(model, tok, layer, device)
    act_all = act_all.to(device=device, dtype=torch.float32)
    ctx_all = ctx_all.to(device=device, dtype=model_dtype)
    with torch.no_grad():
        f_refs_all = torch.empty_like(act_all)
        for i in range(0, n_pool, 64):
            j = min(n_pool, i + 64)
            f_refs_all[i:j] = forward_from_layer_to_layer(
                model, ctx_all[i:j], act_all[i:j].unsqueeze(1).to(model_dtype),
                layer, measure_layer).float()

    def run(label, resample_act, resample_contrast, M, K):
        # fixed picks (used when not resampling)
        idx0 = torch.arange(M, device=device)
        g0 = torch.Generator(device=device).manual_seed(0)
        c0 = torch.randn(K, d_model, generator=g0, device=device)
        c0 = (c0 / c0.norm(dim=-1, keepdim=True)).float()

        d = torch.randn(d_model, device=device)
        d = (d / d.norm()).requires_grad_(True)
        opt = torch.optim.Adam([d], lr=0.003)
        torch.cuda.synchronize()
        t0 = time.time()
        for step in range(n_steps):
            if resample_act:
                rng = np.random.default_rng(1000 + step)
                idx = torch.as_tensor(
                    rng.choice(n_pool, size=M, replace=False), device=device)
            else:
                idx = idx0
            a, c, fr = act_all[idx], ctx_all[idx], f_refs_all[idx]
            if resample_contrast:
                g = torch.Generator(device=device).manual_seed(step)
                con = torch.randn(K, d_model, generator=g, device=device)
                con = (con / con.norm(dim=-1, keepdim=True)).float()
            else:
                con = c0
            opt.zero_grad()
            loss, _ = batched_ratio_multiangle(
                model, a, ctx_all[idx], fr, d, con, angles_rad,
                layer, measure_layer)
            loss.backward()
            opt.step()
            with torch.no_grad():
                d.data.div_(d.data.norm())
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f'{label:<42}  {dt:6.2f}s/atom  ({dt/n_steps*1000:5.1f} ms/step)  '
              f'pts/fwd={1*K*M*3}->{1*M*(1+2*K)}', flush=True)

    print(f'\n--- {n_steps}-step single-atom timing (pts shown old->Rf-shared) ---',
          flush=True)
    run('OLD  fixed act M=1, fixed contrast K=10', False, False, 1, 10)
    run('NEW  resample act M=5, resample con K=4', True, True, 5, 4)
    # isolate each knob
    run('  resample act M=5 only (fixed con K=4)', True, False, 5, 4)
    run('  resample con K=4 only (fixed act M=1)', False, True, 1, 4)
    run('  fixed act M=5, fixed con K=4', False, False, 5, 4)
    run('  fixed act M=1, fixed con K=4', False, False, 1, 4)


if __name__ == '__main__':
    main()
