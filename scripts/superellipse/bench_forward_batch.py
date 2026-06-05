#!/usr/bin/env python3
"""How many points fit / saturate one forward through layers L->measure_layer?

Times forward_from_layer_to_layer for a sweep of point counts N (each point is
a [1,d] last-token activation with a shared [T-1,d] context), fwd+bwd, on
Qwen3-1.7B L14->26. Tells us where we stop being latency-bound (flat time) and
become compute-bound (time ~ N), and the OOM ceiling.

Run on a GPU pod:
    python -u scripts/superellipse/bench_forward_batch.py
"""

import sys
import time
sys.path.insert(0, '.')

import torch

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)


def main():
    model_name, layer, seq_len = 'qwen3-1.7b', 14, 10
    print(f'Loading {model_name} (bf16)', flush=True)
    model, tokenizer, device = load_model(model_name, dtype=torch.bfloat16)
    d_model = model._plateau_info['d_model']
    measure_layer = len(_get_blocks(model)) - 2
    model_dtype = next(model.parameters()).dtype
    print(f'device={device} d_model={d_model} L{layer}->{measure_layer} '
          f'(forwarding {measure_layer-layer} blocks)', flush=True)

    # One real context + activation to clone for all points.
    tok = load_fineweb_fixed_length(1, tokenizer, seq_len=seq_len, seed=42)
    ctx1, act1 = extract_context_and_activation(model, tok, layer, device)
    ctx1 = ctx1.to(device=device, dtype=model_dtype)           # [1, T-1, d]
    act1 = act1.to(device=device, dtype=torch.float32)         # [1, d]
    T_minus_1 = ctx1.shape[1]

    def bench(N, reps=10):
        d = act1.expand(N, d_model).clone().requires_grad_(True)
        ctxs = ctx1.expand(N, T_minus_1, d_model)
        # warmup
        for _ in range(3):
            if d.grad is not None:
                d.grad = None
            out = forward_from_layer_to_layer(
                model, ctxs, d.unsqueeze(1).to(model_dtype),
                layer, measure_layer).float()
            out.sum().backward()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            d.grad = None
            out = forward_from_layer_to_layer(
                model, ctxs, d.unsqueeze(1).to(model_dtype),
                layer, measure_layer).float()
            out.sum().backward()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / reps
        peak = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        return dt, peak

    Ns = [1, 5, 10, 20, 45, 60, 100, 200, 400, 800, 1600, 3200, 6400, 12800,
          25600]
    print(f'\n{"N":>7}  {"ms/fwbw":>9}  {"us/point":>9}  {"peak_GB":>8}',
          flush=True)
    for N in Ns:
        try:
            dt, peak = bench(N)
            print(f'{N:>7}  {dt*1000:>9.2f}  {dt/N*1e6:>9.2f}  {peak:>8.2f}',
                  flush=True)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'{N:>7}  OOM', flush=True)
                torch.cuda.empty_cache()
                break
            raise


if __name__ == '__main__':
    main()
