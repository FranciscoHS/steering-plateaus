#!/usr/bin/env python3
"""
Compute projections of 50k FineWeb activations onto 100 MELBO feature directions.

Extracts layer-14 last-token activations from Qwen3 0.6B, projects onto the
100 MELBO directions, and saves projections + decoded prompt texts.

Usage:
    python scripts/autointerp/01_compute_projections.py
    python scripts/autointerp/01_compute_projections.py --n_prompts 1000  # smaller test run
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model
from src.data import load_fineweb_fixed_length


def extract_last_token_activations(model, token_ids_list, layer, device, batch_size=64):
    """Extract last-token residual stream activations at a given layer.

    Args:
        model: HuggingFace model
        token_ids_list: list of [seq_len] tensors
        layer: transformer block index (0-indexed)
        device: torch device
        batch_size: sequences per batch

    Returns:
        activations: [n_sequences, d_model] tensor (CPU)
    """
    all_activations = []

    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)

        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)

        hidden = outputs.hidden_states[layer + 1]  # after block `layer`
        all_activations.append(hidden[:, -1, :].cpu())

        done = min(i + batch_size, len(token_ids_list))
        if (i // batch_size) % 50 == 0:
            print(f"  Extracted {done}/{len(token_ids_list)}", end='\r')

    print(f"  Extracted {len(token_ids_list)}/{len(token_ids_list)}")
    return torch.cat(all_activations, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_prompts', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=123,
                        help='Different seed from MELBO (42) to get fresh prompts')
    parser.add_argument('--directions_path', default=None)
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    if args.directions_path is None:
        args.directions_path = 'results/melbo_diverse_capped/qwen3-0.6b_target1.0_k100/results.pkl'
    if args.results_dir is None:
        args.results_dir = 'results/autointerp/qwen3-0.6b_layer14_k100'

    os.makedirs(args.results_dir, exist_ok=True)
    output_path = os.path.join(args.results_dir, 'projections.pkl')

    # Check if already computed
    if os.path.exists(output_path):
        print(f"Output already exists: {output_path}")
        with open(output_path, 'rb') as f:
            data = pickle.load(f)
        print(f"  prompts: {len(data['prompts_text'])}, projections: {data['projections'].shape}")
        print("Delete the file to recompute. Exiting.")
        return

    # Load MELBO directions
    print(f"Loading directions from {args.directions_path}")
    with open(args.directions_path, 'rb') as f:
        melbo_data = pickle.load(f)
    directions = melbo_data['directions']  # list of [d_model] tensors
    n_dirs = len(directions)
    print(f"  {n_dirs} directions, d_model={directions[0].shape[0]}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"  device={device}, d_model={d_model}")

    # Load FineWeb prompts
    print(f"\nLoading {args.n_prompts} FineWeb sequences (seq_len={args.seq_len}, seed={args.seed})")
    t0 = time.time()
    token_ids_list = load_fineweb_fixed_length(
        args.n_prompts, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Extract activations
    print(f"\nExtracting layer-{args.layer} last-token activations...")
    t0 = time.time()
    activations = extract_last_token_activations(
        model, token_ids_list, args.layer, device, batch_size=args.batch_size)
    print(f"  Shape: {activations.shape}, took {time.time()-t0:.1f}s")

    # Compute projections
    print(f"\nComputing projections onto {n_dirs} directions...")
    direction_matrix = torch.stack(directions)  # [n_dirs, d_model]
    projections = activations @ direction_matrix.T  # [n_prompts, n_dirs] — signed
    abs_projections = projections.abs()
    print(f"  Projections shape: {projections.shape}")
    print(f"  Abs projection range: [{abs_projections.min():.4f}, {abs_projections.max():.4f}]")
    print(f"  Abs projection mean: {abs_projections.mean():.4f}")

    # Decode prompts to text
    print(f"\nDecoding prompts to text...")
    prompts_text = [tokenizer.decode(tids) for tids in token_ids_list]
    print(f"  Example: {prompts_text[0]!r}")

    # Save
    output = {
        'prompts_text': prompts_text,
        'projections': projections.numpy(),       # [n_prompts, n_dirs], signed
        'abs_projections': abs_projections.numpy(),  # [n_prompts, n_dirs]
        'metadata': {
            'model': args.model,
            'layer': args.layer,
            'seq_len': args.seq_len,
            'n_prompts': len(prompts_text),
            'n_directions': n_dirs,
            'seed': args.seed,
            'directions_path': args.directions_path,
        }
    }
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nSaved to {output_path}")

    # Size check
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"File size: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
