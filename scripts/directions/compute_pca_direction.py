#!/usr/bin/env python3
"""Compute PC1 of FineWeb activations at a given layer.

Saves the direction in the pickle format expected by sweep_collective_threshold.py.

Usage:
    python scripts/directions/compute_pca_direction.py \
        --model qwen3-1.7b --layer 7 --n_samples 1000 \
        --outdir results/pca_directions/qwen3-1.7b_L7
"""
import argparse, os, sys, pickle
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F

from src.model import load_model
from src.data import load_fineweb_fixed_length
from scripts.melbo.run_iterative_melbo_hf import extract_context_and_activation


def main():
    parser = argparse.ArgumentParser(description='Compute PC1 of FineWeb activations')
    parser.add_argument('--model', required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)

    # Collect activations
    print(f"Loading {args.n_samples} FineWeb samples (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_samples, tokenizer, seq_len=args.seq_len, seed=args.seed)

    print(f"Extracting activations at layer {args.layer}...")
    _, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    # activations: [n_samples, d_model]
    print(f"Activations shape: {activations.shape}")

    # Centered PCA via SVD
    acts_np = activations.numpy()
    mean = acts_np.mean(axis=0)
    centered = acts_np - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    var_explained = S**2 / (S**2).sum()
    print(f"Variance explained by PC1: {var_explained[0]:.4f}")
    print(f"Variance explained by top 5: {var_explained[:5]}")

    pc1 = torch.tensor(Vt[0], dtype=torch.float32)
    pc1 = F.normalize(pc1, dim=0)

    # Save in the format expected by sweep_collective_threshold.py
    os.makedirs(args.outdir, exist_ok=True)
    save_data = {
        'directions': pc1.unsqueeze(0),  # [1, d_model]
        'variance_explained': var_explained,
        'singular_values': S,
        'mean': mean,
        'n_samples': args.n_samples,
        'layer': args.layer,
        'model': args.model,
    }
    pkl_path = os.path.join(args.outdir, 'pc1_direction.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved: {pkl_path}")


if __name__ == '__main__':
    main()
