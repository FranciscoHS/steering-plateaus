#!/usr/bin/env python3
"""
Compute all-position projections of long FineWeb passages onto 100 MELBO directions.

Extracts layer-14 activations at EVERY token position in 128-token passages,
projects onto the 100 MELBO directions. This tests whether directions learned
from 5-token last-token activations generalize to all positions in longer text.

Usage:
    python scripts/autointerp/01_compute_projections_longctx.py
    python scripts/autointerp/01_compute_projections_longctx.py --n_passages 500  # smaller test
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model, extract_activations_allpos
from src.data import load_fineweb_fixed_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--n_passages', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=456,
                        help='Different seed from short-context (123) and MELBO (42)')
    parser.add_argument('--directions_path', default=None)
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    if args.directions_path is None:
        args.directions_path = 'results/melbo_diverse_capped/qwen3-0.6b_target1.0_k100/results.pkl'
    if args.results_dir is None:
        args.results_dir = 'results/autointerp/qwen3-0.6b_layer14_k100_longctx'

    os.makedirs(args.results_dir, exist_ok=True)
    output_path = os.path.join(args.results_dir, 'projections.pkl')

    # Check if already computed
    if os.path.exists(output_path):
        print(f"Output already exists: {output_path}")
        with open(output_path, 'rb') as f:
            data = pickle.load(f)
        print(f"  projections shape: {data['projections'].shape}")
        print(f"  n_passages: {len(data['token_texts'])}")
        print("Delete the file to recompute. Exiting.")
        return

    # Load MELBO directions
    print(f"Loading directions from {args.directions_path}")
    with open(args.directions_path, 'rb') as f:
        melbo_data = pickle.load(f)
    directions = melbo_data['directions']  # list of [d_model] tensors
    n_dirs = len(directions)
    d_model = directions[0].shape[0]
    print(f"  {n_dirs} directions, d_model={d_model}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    print(f"  device={device}, d_model={model._plateau_info['d_model']}")

    # Load FineWeb passages
    print(f"\nLoading {args.n_passages} FineWeb passages (seq_len={args.seq_len}, seed={args.seed})")
    t0 = time.time()
    token_ids_list = load_fineweb_fixed_length(
        args.n_passages, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Extract all-position activations
    print(f"\nExtracting layer-{args.layer} all-position activations...")
    t0 = time.time()
    activations_flat = extract_activations_allpos(
        model, tokenizer, token_ids_list, args.layer, device, batch_size=args.batch_size)
    # activations_flat: [n_passages, seq_len * d_model]
    print(f"  Flat shape: {activations_flat.shape}, took {time.time()-t0:.1f}s")

    # Reshape to [n_passages, seq_len, d_model]
    activations = activations_flat.reshape(args.n_passages, args.seq_len, d_model)
    print(f"  Reshaped: {activations.shape}")

    # Compute projections: [n_passages, seq_len, n_dirs]
    print(f"\nComputing projections onto {n_dirs} directions...")
    direction_matrix = torch.stack(directions)  # [n_dirs, d_model]
    # activations: [n_passages, seq_len, d_model] @ direction_matrix.T: [d_model, n_dirs]
    # → [n_passages, seq_len, n_dirs]
    projections = (activations @ direction_matrix.T).numpy()
    abs_projections = np.abs(projections)
    print(f"  Projections shape: {projections.shape}")
    print(f"  Abs projection range: [{abs_projections.min():.4f}, {abs_projections.max():.4f}]")
    print(f"  Abs projection mean: {abs_projections.mean():.4f}")

    # Decode each token individually for per-token text labels
    print(f"\nDecoding tokens...")
    token_texts = []
    for tids in token_ids_list:
        texts = [tokenizer.decode([tid]) for tid in tids]
        token_texts.append(texts)
    # Show example
    example_passage = ''.join(token_texts[0])
    print(f"  Example passage: {example_passage[:100]}...")
    print(f"  First 5 tokens: {token_texts[0][:5]}")

    # Save
    output = {
        'projections': projections,        # [n_passages, seq_len, n_dirs], signed
        'abs_projections': abs_projections, # [n_passages, seq_len, n_dirs]
        'token_texts': token_texts,         # list of lists of decoded token strings
        'token_ids': [tids.numpy() for tids in token_ids_list],
        'metadata': {
            'model': args.model,
            'layer': args.layer,
            'seq_len': args.seq_len,
            'n_passages': len(token_ids_list),
            'n_directions': n_dirs,
            'seed': args.seed,
            'directions_path': args.directions_path,
        }
    }
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nSaved to {output_path}")

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"File size: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
