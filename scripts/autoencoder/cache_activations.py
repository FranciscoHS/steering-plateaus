"""Cache last-token resid_post activations to a raw memmap for AE training.

Mirrors the caching path that produced l10_cache_10M/cached_activations.pt for
Qwen3-1.7B, but model-agnostic: probes the model structure to find layers.

Saves a raw numpy memmap + sidecar meta.json that hillclimb_autoencoder.py
already knows how to load.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, '.')

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_fineweb_fixed_length


def get_layers(model):
    """Find the list of transformer blocks across common model layouts."""
    candidates = [
        ('model.model.layers',                lambda m: m.model.layers),
        ('model.language_model.layers',       lambda m: m.language_model.layers),
        ('model.model.language_model.layers', lambda m: m.model.language_model.layers),
        ('model.transformer.h',               lambda m: m.transformer.h),
        ('model.gpt_neox.layers',             lambda m: m.gpt_neox.layers),
    ]
    for name, fn in candidates:
        try:
            layers = fn(model)
        except AttributeError:
            continue
        if layers is not None and len(layers) > 0:
            print(f"  found transformer blocks at: {name} (n={len(layers)})")
            return layers
    raise RuntimeError("Could not find transformer layers in model")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--out_path', required=True,
                   help='Where to write the memmap (a .meta.json sidecar is also written).')
    p.add_argument('--n_cache', type=int, default=8_000_000)
    p.add_argument('--cache_start', type=int, default=8000,
                   help='Skip first N indices of the shuffled FineWeb stream so '
                        'CE eval (which uses the same seed) can sample disjoint texts.')
    p.add_argument('--seq_len', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--dtype', default='float16', choices=['float16', 'bfloat16', 'float32'])
    p.add_argument('--store_dtype', default='float16', choices=['float16', 'float32'])
    p.add_argument('--fineweb_cache_dir', default='/workspace/fineweb_cache_gemma')
    args = p.parse_args()

    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype], device_map='auto',
    )
    model.eval()
    device = next(model.parameters()).device
    layers = get_layers(model)
    assert 0 <= args.layer < len(layers), f"--layer {args.layer} out of range [0, {len(layers)})"

    n_total = args.cache_start + args.n_cache
    print(f"Loading {n_total} FineWeb sequences (seed={args.seed}, seq_len={args.seq_len})...")
    token_ids = load_fineweb_fixed_length(
        n_total, tok, seq_len=args.seq_len, seed=args.seed,
        cache_dir=args.fineweb_cache_dir,
    )
    cache_tokens = token_ids[args.cache_start:args.cache_start + args.n_cache]
    n_cache = len(cache_tokens)
    print(f"Extracting {n_cache} activations at layer {args.layer} "
          f"(indices [{args.cache_start}, {args.cache_start + n_cache}))...")

    save_dtype = torch.float16 if args.store_dtype == 'float16' else torch.float32
    np_dtype = np.float16 if save_dtype == torch.float16 else np.float32

    captured = {}

    def capture_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured['act'] = h[:, -1, :].to(save_dtype).cpu()

    class EarlyExit(Exception):
        pass

    def exit_hook(module, input, output):
        raise EarlyExit()

    h_cap = layers[args.layer].register_forward_hook(capture_hook)
    h_exit = layers[args.layer + 1].register_forward_hook(exit_hook) if args.layer + 1 < len(layers) else None

    probe = torch.stack(cache_tokens[:1]).to(device)
    with torch.no_grad():
        try:
            model(probe)
        except EarlyExit:
            pass
    d_model = captured['act'].shape[-1]
    print(f"  d_model = {d_model}")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    mm = np.memmap(args.out_path, dtype=np_dtype, mode='w+', shape=(n_cache, d_model))
    nbytes = n_cache * d_model * np.dtype(np_dtype).itemsize
    print(f"Streaming to memmap: ({n_cache}, {d_model}), {np_dtype} -- {nbytes/1e9:.1f} GB")

    t0 = time.time()
    for i in range(0, n_cache, args.batch_size):
        batch = torch.stack(cache_tokens[i:i + args.batch_size]).to(device)
        with torch.no_grad():
            try:
                model(batch)
            except EarlyExit:
                pass
        act = captured['act']
        mm[i:i + act.shape[0]] = act.numpy()
        if (i // args.batch_size) % 50 == 0:
            done = min(i + args.batch_size, n_cache)
            el = time.time() - t0
            rate = done / el if el > 0 else 0
            rem = (n_cache - done) / rate if rate > 0 else 0
            print(f"  {done}/{n_cache} ({rate:.0f}/s, ~{rem/60:.1f}min left)", flush=True)

    mm.flush()
    del mm
    h_cap.remove()
    if h_exit is not None:
        h_exit.remove()

    meta = {'shape': [n_cache, d_model], 'dtype': str(np_dtype().dtype), 'format': 'memmap_raw'}
    with open(args.out_path + '.meta.json', 'w') as f:
        json.dump(meta, f)
    print(f"Done. Wrote memmap + meta to {args.out_path}")


if __name__ == '__main__':
    main()
