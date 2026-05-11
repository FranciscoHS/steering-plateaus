#!/usr/bin/env python3
"""
Prune a batch_decomp dictionary by minimizing frame potential.

Greedy batch approach: each round, compute each direction's frame potential
contribution and plateau-breaking response, then remove the directions with
the worst response/fp_contribution ratio. This encourages the remaining
dictionary to approximate a tight frame while preserving high-response
(plateau-breaking) directions.

Frame potential: FP = Σ_i Σ_j |⟨f_i, f_j⟩|²
For n unit vectors in d dimensions:
  - If n ≤ d: minimized by an orthonormal set (FP = n)
  - If n > d: minimized by a tight frame (FP = n²/d)

Usage:
    python -u scripts/dictionary_learning/prune_dictionary_fp.py \
        --results_dir results/batch_decomp/llama-3.2-1b_layer2 \
        --output_dir results/pruned_dictionary_fp/llama-3.2-1b_layer2 \
        --target_size 20000
"""

import argparse
import glob
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np


# ── Loading ───────────────────────────────────────────────────────────────

def load_dictionary_with_responses(results_dir):
    """Load direction vectors, responses, and provenance from batch_decomp results.

    Returns:
        directions: (n_atoms, d_model) tensor
        responses: (n_atoms,) tensor - plateau-breaking response per direction
        provenance: list of (act_idx, dir_idx) tuples
    """
    pattern = os.path.join(results_dir, "act_*", "results.pkl")
    result_files = sorted(glob.glob(pattern))

    if not result_files:
        raise FileNotFoundError(f"No results.pkl found in {results_dir}/act_*/")

    all_directions = []
    all_responses = []
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
        responses = res['responses']
        n_complete += 1
        dirs_per_act.append(len(directions))

        for dir_idx, (d, r) in enumerate(zip(directions, responses)):
            all_directions.append(d)
            all_responses.append(r)
            provenance.append((act_idx, dir_idx))

    directions_tensor = torch.stack(all_directions)
    responses_tensor = torch.tensor(all_responses, dtype=torch.float32)

    print(f"Loaded dictionary from {results_dir}:")
    print(f"  {n_complete} complete activations, {n_skipped} skipped")
    print(f"  {len(provenance)} total atoms, d_model={directions_tensor.shape[1]}")
    print(f"  Dirs per activation: min={min(dirs_per_act)}, "
          f"median={sorted(dirs_per_act)[len(dirs_per_act)//2]}, "
          f"max={max(dirs_per_act)}")
    print(f"  Response: min={responses_tensor.min():.4f}, "
          f"median={responses_tensor.median():.4f}, "
          f"max={responses_tensor.max():.4f}, "
          f"mean={responses_tensor.mean():.4f}")

    return directions_tensor, responses_tensor, provenance


# ── Frame potential ───────────────────────────────────────────────────────

def compute_fp_contributions(dictionary, chunk_size=1000):
    """Compute per-direction frame potential contribution without forming full Gram matrix.

    For unit-norm direction k, its FP contribution (the FP decrease if removed) is:
        fp_contribution[k] = 2 * Σ_{j≠k} ⟨d_k, d_j⟩² + 1
                            = 2 * (Σ_j ⟨d_k, d_j⟩²) - 1

    Args:
        dictionary: (n, d) tensor, unit-normalized rows
        chunk_size: rows processed per chunk (controls peak memory)

    Returns:
        fp_contributions: (n,) tensor
    """
    n = dictionary.shape[0]
    device = dictionary.device
    fp_contributions = torch.zeros(n, device=device)

    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        # (chunk, d) @ (d, n) -> (chunk, n)
        gram_chunk = dictionary[i:end] @ dictionary.T
        fp_contributions[i:end] = 2.0 * (gram_chunk ** 2).sum(dim=1) - 1.0

    return fp_contributions


def compute_frame_potential(fp_contributions, n):
    """Compute total frame potential from per-direction contributions.

    FP = (Σ_k fp_contribution[k] + n) / 2
    """
    return (fp_contributions.sum().item() + n) / 2.0


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Prune dictionary by minimizing frame potential')
    parser.add_argument('--results_dir', required=True,
                        help='Where to find act_*/results.pkl from batch_decomp')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--target_size', type=int, default=20000)
    parser.add_argument('--prune_frac', type=float, default=0.25,
                        help='Fraction of atoms to remove per round')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Chunk size for Gram matrix computation')
    parser.add_argument('--device', default='auto',
                        help='Device (auto/cpu/cuda)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Dictionary pruning via frame potential")
    print(f"  results_dir: {args.results_dir}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  target_size: {args.target_size}")
    print(f"  prune_frac: {args.prune_frac}")
    print(f"  device: {device}")
    print()

    # Step 1: Load dictionary with responses
    dictionary, responses, provenance = load_dictionary_with_responses(args.results_dir)

    if dictionary.shape[0] <= args.target_size:
        print(f"Dictionary already <= target_size ({dictionary.shape[0]} <= {args.target_size}), "
              f"nothing to prune.")
        return

    # Normalize directions (should already be unit-norm, but be safe)
    dictionary = F.normalize(dictionary, dim=-1)

    # Move to device
    dictionary = dictionary.to(device)
    responses = responses.to(device)

    # Ensure responses are strictly positive for the ratio
    if responses.min() <= 0:
        shift = abs(responses.min().item()) + 1e-6
        print(f"  Shifting responses by +{shift:.6f} (min was {responses.min().item():.4f})")
        responses = responses + shift

    provenance = np.array(provenance)
    pruning_log = []
    round_idx = 0
    d_model = dictionary.shape[1]

    print(f"\nStarting pruning: {dictionary.shape[0]} atoms → {args.target_size}")
    print("=" * 60)

    while dictionary.shape[0] > args.target_size:
        t0 = time.time()
        n_atoms = dictionary.shape[0]

        # How many to remove this round
        n_remove_frac = int(n_atoms * args.prune_frac)
        n_remove = min(n_remove_frac, n_atoms - args.target_size)

        # Compute frame potential contributions
        fp_contribs = compute_fp_contributions(dictionary, chunk_size=args.chunk_size)

        # Score: response / fp_contribution
        # High score = worth keeping (high plateau-breaking, low redundancy)
        scores = responses / fp_contribs

        total_fp = compute_frame_potential(fp_contribs, n_atoms)
        fp_tight = n_atoms ** 2 / d_model
        elapsed = time.time() - t0

        # Log
        fp_cpu = fp_contribs.cpu().numpy()
        scores_cpu = scores.cpu().numpy()

        log_entry = {
            'round': round_idx,
            'size_before': n_atoms,
            'n_remove': n_remove,
            'size_after': n_atoms - n_remove,
            'frame_potential': total_fp,
            'fp_tight_frame': fp_tight,
            'fp_ratio': total_fp / fp_tight,
            'fp_contrib_min': float(fp_cpu.min()),
            'fp_contrib_median': float(np.median(fp_cpu)),
            'fp_contrib_max': float(fp_cpu.max()),
            'score_min': float(scores_cpu.min()),
            'score_median': float(np.median(scores_cpu)),
            'score_max': float(scores_cpu.max()),
            'elapsed_s': elapsed,
        }
        pruning_log.append(log_entry)

        print(f"Round {round_idx}: {n_atoms} -> {n_atoms - n_remove} "
              f"(removing {n_remove})")
        print(f"  FP: {total_fp:.1f} (tight frame: {fp_tight:.1f}, "
              f"ratio: {total_fp/fp_tight:.4f})")
        print(f"  FP contrib: min={fp_cpu.min():.2f}, "
              f"median={np.median(fp_cpu):.2f}, max={fp_cpu.max():.2f}")
        print(f"  Score (resp/fp): min={scores_cpu.min():.4f}, "
              f"median={np.median(scores_cpu):.4f}, max={scores_cpu.max():.4f}")
        print(f"  Time: {elapsed:.1f}s")

        # Remove directions with lowest score
        keep_indices = torch.argsort(scores)[n_remove:]
        keep_indices_cpu = keep_indices.cpu().numpy()
        dictionary = dictionary[keep_indices]
        responses = responses[keep_indices]
        provenance = provenance[keep_indices_cpu]

        round_idx += 1

    # Final stats
    fp_final = compute_fp_contributions(dictionary, chunk_size=args.chunk_size)
    total_fp_final = compute_frame_potential(fp_final, dictionary.shape[0])
    fp_tight_final = dictionary.shape[0] ** 2 / d_model

    print("=" * 60)
    print(f"Pruning complete: {dictionary.shape[0]} atoms remaining")
    print(f"  Final FP: {total_fp_final:.1f} (tight frame: {fp_tight_final:.1f}, "
          f"ratio: {total_fp_final/fp_tight_final:.4f})")
    print(f"  Response range: [{responses.min().item():.4f}, {responses.max().item():.4f}]")

    # Save results (same format as prune_dictionary.py for benchmark compatibility)
    output = {
        'dictionary': dictionary.cpu(),
        'responses': responses.cpu(),
        'provenance': [(int(row[0]), int(row[1])) for row in provenance],
        'pruning_log': pruning_log,
        'final_frame_potential': total_fp_final,
        'final_fp_ratio': total_fp_final / fp_tight_final,
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
