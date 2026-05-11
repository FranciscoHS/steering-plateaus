#!/usr/bin/env python3
"""
Prune a batch_decomp dictionary via iterative matching pursuit.

Loads all direction vectors from batch_decomp results, then iteratively
removes the least-used atoms (measured by GPU-accelerated matching pursuit
on held-out activations) until the dictionary reaches the target size.

Usage:
    python -u scripts/dictionary_learning/prune_dictionary.py \
        --model meta-llama/Llama-3.2-1B --layer 2 \
        --results_dir results/batch_decomp/llama-3.2-1b_layer2 \
        --output_dir results/pruned_dictionary/llama-3.2-1b_layer2 \
        --target_size 20000 --n_nonzero_coefs 20
"""

import argparse
import glob
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model
from src.data import load_fineweb_fixed_length


# ── Matching pursuit on GPU ───────────────────────────────────────────────

def matching_pursuit_usage(dictionary, activations, n_nonzero_coefs):
    """Count how often each dictionary atom is selected by matching pursuit.

    Args:
        dictionary: (n_atoms, d_model) tensor on GPU, should be unit-normalized
        activations: (n_val, d_model) tensor on GPU
        n_nonzero_coefs: number of MP iterations (atoms selected per activation)

    Returns:
        usage_counts: (n_atoms,) tensor on GPU
    """
    n_atoms = dictionary.shape[0]
    n_val = activations.shape[0]
    device = dictionary.device

    usage_counts = torch.zeros(n_atoms, device=device)
    ones = torch.ones(n_val, device=device)
    residuals = activations.clone()

    for step in range(n_nonzero_coefs):
        # Correlations: (n_val, n_atoms)
        corr = residuals @ dictionary.T
        # Best atom per activation
        best = corr.abs().argmax(dim=1)  # (n_val,)
        # Count usage
        usage_counts.scatter_add_(0, best, ones)
        # Subtract projection
        coefs = corr[torch.arange(n_val, device=device), best]  # (n_val,)
        residuals = residuals - coefs.unsqueeze(1) * dictionary[best]

    return usage_counts


# ── Loading ───────────────────────────────────────────────────────────────

def load_dictionary(results_dir):
    """Load all direction vectors and provenance from batch_decomp results.

    Returns:
        directions: (n_atoms, d_model) tensor
        provenance: list of (act_idx, dir_idx) tuples
    """
    pattern = os.path.join(results_dir, "act_*", "results.pkl")
    result_files = sorted(glob.glob(pattern))

    if not result_files:
        raise FileNotFoundError(f"No results.pkl found in {results_dir}/act_*/")

    all_directions = []
    provenance = []
    n_complete = 0
    n_skipped = 0
    dirs_per_act = []

    for path in result_files:
        with open(path, 'rb') as f:
            res = pickle.load(f)

        if not res.get('complete', False):
            n_skipped += 1
            continue

        act_idx = res['act_idx']
        directions = res['directions']
        n_complete += 1
        dirs_per_act.append(len(directions))

        for dir_idx, d in enumerate(directions):
            all_directions.append(d)
            provenance.append((act_idx, dir_idx))

    directions_tensor = torch.stack(all_directions)  # (n_atoms, d_model)

    print(f"Loaded dictionary from {results_dir}:")
    print(f"  {n_complete} complete activations, {n_skipped} skipped")
    print(f"  {len(provenance)} total atoms, d_model={directions_tensor.shape[1]}")
    print(f"  Dirs per activation: min={min(dirs_per_act)}, "
          f"median={sorted(dirs_per_act)[len(dirs_per_act)//2]}, "
          f"max={max(dirs_per_act)}")

    return directions_tensor, provenance


def extract_val_activations(model, tokenizer, device, args):
    """Extract held-out activations, with caching."""
    cache_path = os.path.join(args.output_dir, "val_activations.pt")

    if args.reuse_val and os.path.exists(cache_path):
        print(f"Loading cached val activations from {cache_path}")
        return torch.load(cache_path, weights_only=True)

    print(f"Loading {args.n_total} FineWeb samples (seed={args.seed}, seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_total, tokenizer, seq_len=args.seq_len, seed=args.seed
    )
    print(f"Loaded {len(token_ids_list)} samples, using indices [{args.val_start}, {args.val_end})")

    val_tokens = token_ids_list[args.val_start:args.val_end]
    n_val = len(val_tokens)

    print(f"Extracting {n_val} activations at layer {args.layer}...")
    all_activations = []
    batch_size = 64
    for i in range(0, n_val, batch_size):
        batch = torch.stack(val_tokens[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
        hidden = outputs.hidden_states[args.layer + 1]
        all_activations.append(hidden[:, -1, :].float().cpu())
        if (i // batch_size) % 10 == 0:
            print(f"  {min(i + batch_size, n_val)}/{n_val}", end='\r')
    print()

    activations = torch.cat(all_activations, dim=0)  # (n_val, d_model)
    print(f"Val activations: shape={activations.shape}, "
          f"mean norm={activations.norm(dim=-1).mean():.2f}")

    torch.save(activations, cache_path)
    print(f"Cached val activations to {cache_path}")

    return activations


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Prune dictionary via matching pursuit')
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_total', type=int, default=7000,
                        help='Total FineWeb samples to load (must be >= val_end)')
    parser.add_argument('--val_start', type=int, default=2000)
    parser.add_argument('--val_end', type=int, default=7000)
    parser.add_argument('--results_dir', default='results/batch_decomp/llama-3.2-1b_layer2',
                        help='Where to find act_*/results.pkl from batch_decomp')
    parser.add_argument('--output_dir', default='results/pruned_dictionary/llama-3.2-1b_layer2')
    parser.add_argument('--target_size', type=int, default=20000)
    parser.add_argument('--n_nonzero_coefs', type=int, default=20,
                        help='Sparsity level for matching pursuit')
    parser.add_argument('--prune_frac', type=float, default=0.25,
                        help='Fraction of atoms to remove per round')
    parser.add_argument('--reuse_val', action='store_true',
                        help='Reuse cached val activations if available')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Dictionary pruning via matching pursuit")
    print(f"  results_dir: {args.results_dir}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  target_size: {args.target_size}")
    print(f"  n_nonzero_coefs: {args.n_nonzero_coefs}")
    print(f"  prune_frac: {args.prune_frac}")
    print()

    # Step 1: Load dictionary
    dictionary, provenance = load_dictionary(args.results_dir)

    if dictionary.shape[0] <= args.target_size:
        print(f"Dictionary already <= target_size ({dictionary.shape[0]} <= {args.target_size}), "
              f"nothing to prune.")
        return

    # Step 2: Extract held-out activations
    model, tokenizer, device = load_model(args.model)
    val_activations = extract_val_activations(model, tokenizer, device, args)

    # Free model memory — only need dictionary + activations for pruning
    del model
    torch.cuda.empty_cache()

    # Move to GPU
    dictionary = dictionary.to(device)
    val_activations = val_activations.to(device)

    # Step 3: Iterative pruning
    provenance = np.array(provenance)  # (n_atoms, 2) for easy indexing
    pruning_log = []
    round_idx = 0

    print(f"\nStarting pruning: {dictionary.shape[0]} atoms → {args.target_size}")
    print("=" * 60)

    while dictionary.shape[0] > args.target_size:
        t0 = time.time()
        n_atoms = dictionary.shape[0]

        # How many to remove this round
        n_remove_frac = int(n_atoms * args.prune_frac)
        n_remove = min(n_remove_frac, n_atoms - args.target_size)

        # Run matching pursuit
        usage_counts = matching_pursuit_usage(
            dictionary, val_activations, args.n_nonzero_coefs
        )

        # Stats
        usage_cpu = usage_counts.cpu().numpy()
        elapsed = time.time() - t0

        log_entry = {
            'round': round_idx,
            'size_before': n_atoms,
            'n_remove': n_remove,
            'size_after': n_atoms - n_remove,
            'usage_min': int(usage_cpu.min()),
            'usage_median': float(np.median(usage_cpu)),
            'usage_max': int(usage_cpu.max()),
            'usage_mean': float(usage_cpu.mean()),
            'n_zero_usage': int((usage_cpu == 0).sum()),
            'elapsed_s': elapsed,
        }
        pruning_log.append(log_entry)

        print(f"Round {round_idx}: {n_atoms} → {n_atoms - n_remove} "
              f"(removing {n_remove})")
        print(f"  Usage: min={log_entry['usage_min']}, "
              f"median={log_entry['usage_median']:.0f}, "
              f"max={log_entry['usage_max']}, "
              f"zero={log_entry['n_zero_usage']}")
        print(f"  Time: {elapsed:.1f}s")

        # Remove bottom atoms by usage
        keep_indices = torch.argsort(usage_counts)[n_remove:]
        keep_indices_cpu = keep_indices.cpu().numpy()
        dictionary = dictionary[keep_indices]
        provenance = provenance[keep_indices_cpu]

        round_idx += 1

    print("=" * 60)
    print(f"Pruning complete: {dictionary.shape[0]} atoms remaining")

    # Step 4: Save results
    output = {
        'dictionary': dictionary.cpu(),
        'provenance': [(int(row[0]), int(row[1])) for row in provenance],
        'pruning_log': pruning_log,
        'args': vars(args),
    }
    output_path = os.path.join(args.output_dir, "pruned_dictionary.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nSaved pruned dictionary to {output_path}")
    print(f"  Shape: {dictionary.shape}")
    print(f"  Provenance entries: {len(output['provenance'])}")


if __name__ == '__main__':
    main()
