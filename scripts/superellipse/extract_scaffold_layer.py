#!/usr/bin/env python3
"""Extract the contrastive (diff-of-means) scaffold directions at a given layer and
save them in dictionary.pt format (so run_dict_vs_random_sweep.py can sweep them).

Reuses the same registry/extraction path as the dictionary build, so the L2
directions are the exact same concepts as scaffold_qwen_L14, just extracted at a
different layer.

Usage (on the pod):
    python -u scripts/superellipse/extract_scaffold_layer.py \
        --model qwen3-1.7b --layer 2 \
        --names_from results/scaffold_qwen_L14/dictionary.pt \
        --out results/scaffold_qwen_L2/dictionary.pt
"""
import argparse
import os
import sys
sys.path.insert(0, '.')

import torch

from src.model import load_model
from scripts.superellipse.learn_feature_dir_vs_contrastive import load_or_build_scaffold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--names_from', required=True,
                    help='dictionary.pt whose "names" list defines the concept set')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default=None, help='e.g. bfloat16 (needed for big models)')
    args = ap.parse_args()

    names = torch.load(args.names_from, map_location='cpu', weights_only=False)['names']
    print(f"Concepts ({len(names)}): {names}", flush=True)

    dtype = getattr(torch, args.dtype) if args.dtype else None
    model, tokenizer, device = load_model(args.model, dtype=dtype)
    scaffold = load_or_build_scaffold(model, tokenizer, device, args.model, args.layer, names)
    scaffold = scaffold.float().cpu()
    # unit-normalize rows (sweep script re-normalizes anyway, but keep it clean)
    scaffold = scaffold / scaffold.norm(dim=-1, keepdim=True)
    print(f"scaffold at L{args.layer}: {tuple(scaffold.shape)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'dictionary': scaffold, 'names': names, 'n_done': len(names)}, args.out)
    print(f"saved: {args.out}", flush=True)


if __name__ == '__main__':
    main()
