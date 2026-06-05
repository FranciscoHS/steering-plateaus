#!/usr/bin/env python3
"""Extract held-out FineWeb activations at a given layer; save to disk.

Used to decouple the (GPU-needing) activation extraction from the (CPU-fine)
dictionary evaluation. Run this once on the pod, then `scp` the .pt file
locally for eval_dictionary_omp.py.

Usage:
    python scripts/superellipse/extract_held_acts.py \\
        --model qwen3-1.7b --layer 14 --n_held 2000 --seed 2024 \\
        --out results/dictionary/qwen3-1.7b_L14_K8000/held_acts.pt
"""

import argparse
import os
import sys
sys.path.insert(0, '.')

import torch

from src.model import load_model
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_held', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=2024,
                    help='seed (must differ from training pool seed)')
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])
    print(f'Extracting {args.n_held} FineWeb activations at L{args.layer} '
          f'(seed={args.seed})', flush=True)
    tok = load_fineweb_fixed_length(args.n_held, tokenizer,
                                    seq_len=args.seq_len, seed=args.seed)
    _, act = extract_context_and_activation(model, tok, args.layer, device)
    A = act.to(dtype=torch.float32).cpu()  # [B, d]
    print(f'  shape={tuple(A.shape)}  mean ||a||={A.norm(dim=-1).mean():.2f}',
          flush=True)
    torch.save({
        'activations': A,
        'model': args.model,
        'layer': args.layer,
        'seq_len': args.seq_len,
        'n_held': args.n_held,
        'seed': args.seed,
    }, args.out)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
