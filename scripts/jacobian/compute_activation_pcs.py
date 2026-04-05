#!/usr/bin/env python3
"""
Compute top PCs of the activation distribution at a given layer.

Loads many FineWeb activations, computes PCA, reports variance explained,
and saves PC directions in a format compatible with run_feature_vs_random_sweep.py.

Usage:
    python scripts/jacobian/compute_activation_pcs.py --model qwen3-1.7b --layer 7 --n_act 500 --k 15
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length


def main():
    parser = argparse.ArgumentParser(description='Compute top PCs of activation distribution')
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=7)
    parser.add_argument('--n_act', type=int, default=500,
                        help='Number of activations to compute PCA over')
    parser.add_argument('--k', type=int, default=15, help='Number of PCs to save')
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default='results/activation_pcs')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Config: model={args.model}, layer={args.layer}, n_act={args.n_act}, k={args.k}")

    # Load model
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"  d_model={d_model}, device={device}")

    # Load activations
    print(f"Loading {args.n_act} FineWeb sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed
    )

    print("Extracting activations...")
    all_activations = []
    batch_size = 8
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[args.layer + 1]
        all_activations.append(hidden[:, -1, :].float().cpu())
        if (i // batch_size) % 10 == 0:
            print(f"  {min(i + batch_size, args.n_act)}/{args.n_act}", end='\r')
    print()

    activations = torch.cat(all_activations, dim=0)  # [n_act, d_model]
    print(f"Activations shape: {activations.shape}, mean norm: {activations.norm(dim=-1).mean():.2f}")

    # PCA (center first)
    mean = activations.mean(dim=0)
    centered = activations - mean
    print("Computing SVD...")
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

    # Variance explained
    total_var = (S ** 2).sum().item()
    var_explained = (S ** 2).cumsum(dim=0) / total_var
    print(f"\nVariance explained by top PCs:")
    for i in [1, 5, 10, 15, 20, 30, 50, 100]:
        if i <= len(S):
            print(f"  {i:3d} PCs: {var_explained[i-1].item()*100:.1f}%")

    # Save top k PCs as directions (compatible with run_feature_vs_random_sweep.py)
    pc_directions = Vt[:args.k].cpu()  # [k, d_model], already unit norm from SVD

    # Save individual PC pkl files for run_feature_vs_random_sweep.py
    for i in range(args.k):
        out = {'directions': [pc_directions[i]]}
        path = os.path.join(args.results_dir, f'pc_{i}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(out, f)

    # Save full results
    results = {
        'args': vars(args),
        'pc_directions': pc_directions,
        'singular_values': S[:args.k].cpu(),
        'var_explained_cumulative': var_explained[:args.k].cpu(),
        'var_explained_individual': (S[:args.k] ** 2 / total_var).cpu(),
        'mean': mean.cpu(),
        'total_var': total_var,
        'n_act': args.n_act,
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved {args.k} PC directions to {args.results_dir}/pc_*.pkl")
    print(f"Full results saved to {results_path}")


if __name__ == '__main__':
    main()
