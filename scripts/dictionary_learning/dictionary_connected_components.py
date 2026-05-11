#!/usr/bin/env python3
"""
Analyze connected components in the dictionary's cosine similarity graph.

For each threshold t, build a graph where directions are nodes and edges connect
pairs with |cosine_similarity| > t. Compute connected components and plot:
- Number of connected components vs threshold
- Size of largest component vs threshold
- Mean component size vs threshold (among components with size > 1)

Usage:
    python -u scripts/dictionary_learning/dictionary_connected_components.py \
        --results_dir results/batch_decomp/llama-3.2-1b_layer2 \
        --output_dir results/dictionary_components
"""

import argparse
import glob
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def load_dictionary(results_dir):
    """Load all direction vectors from batch_decomp results.

    Returns:
        directions: (n_atoms, d_model) tensor, unit-normalized
        provenance: list of (act_idx, dir_idx) tuples
    """
    pattern = os.path.join(results_dir, "act_*", "results.pkl")
    result_files = sorted(glob.glob(pattern))

    if not result_files:
        raise FileNotFoundError(f"No results.pkl found in {results_dir}/act_*/")

    all_directions = []
    provenance = []
    n_complete = 0

    for path in result_files:
        with open(path, 'rb') as f:
            res = pickle.load(f)
        if not res.get('complete', False):
            continue
        act_idx = res['act_idx']
        for dir_idx, d in enumerate(res['directions']):
            all_directions.append(d)
            provenance.append((act_idx, dir_idx))
        n_complete += 1

    directions = torch.stack(all_directions)
    directions = F.normalize(directions, dim=-1)

    print(f"Loaded {directions.shape[0]} directions from {n_complete} activations")
    print(f"  d_model = {directions.shape[1]}")
    return directions, provenance


def compute_components_at_threshold(directions, threshold, chunk_size=2000):
    """Build thresholded cosine similarity graph and find connected components.

    Uses chunked matrix multiply to avoid OOM on the full 40k×40k matrix.
    Uses absolute cosine similarity (signs don't matter for feature identity).

    Returns:
        n_components: number of connected components
        component_sizes: sorted list of component sizes (descending)
    """
    n = directions.shape[0]
    device = directions.device

    # Build sparse adjacency matrix in COO format
    rows = []
    cols = []

    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        # (chunk, d) @ (d, n) -> (chunk, n) — cosine sim since dirs are unit-norm
        sim_chunk = directions[i:end_i] @ directions.T
        # Absolute value — antiparallel directions are the same feature
        sim_chunk = sim_chunk.abs()
        # Zero out self-similarities and lower triangle (only need upper triangle)
        for local_idx in range(end_i - i):
            global_idx = i + local_idx
            sim_chunk[local_idx, :global_idx + 1] = 0.0
        # Find edges above threshold
        r, c = torch.where(sim_chunk > threshold)
        rows.append((r + i).cpu().numpy())
        cols.append(c.cpu().numpy())

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(len(rows), dtype=np.int8)

    # Symmetric adjacency matrix
    adj = sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
    adj = adj + adj.T

    n_components, labels = connected_components(adj, directed=False)
    _, counts = np.unique(labels, return_counts=True)
    component_sizes = sorted(counts, reverse=True)

    return n_components, component_sizes, len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--output_dir', default='results/dictionary_components')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--chunk_size', type=int, default=2000)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load dictionary
    directions, provenance = load_dictionary(args.results_dir)
    directions = directions.to(device)
    n_total = directions.shape[0]

    # Sweep thresholds
    results = []
    for t in sorted(args.thresholds):
        print(f"\nThreshold {t:.2f}...")
        t0 = time.time()
        n_comp, sizes, n_edges = compute_components_at_threshold(
            directions, t, chunk_size=args.chunk_size)
        elapsed = time.time() - t0

        # Stats
        largest = sizes[0]
        n_singletons = sum(1 for s in sizes if s == 1)
        n_nontrivial = sum(1 for s in sizes if s > 1)
        mean_nontrivial = np.mean([s for s in sizes if s > 1]) if n_nontrivial > 0 else 0
        n_in_nontrivial = sum(s for s in sizes if s > 1)

        entry = {
            'threshold': t,
            'n_components': n_comp,
            'n_edges': n_edges,
            'largest': largest,
            'n_singletons': n_singletons,
            'n_nontrivial': n_nontrivial,
            'mean_nontrivial': mean_nontrivial,
            'n_in_nontrivial': n_in_nontrivial,
            'top10_sizes': sizes[:10],
            'elapsed': elapsed,
        }
        results.append(entry)

        print(f"  Components: {n_comp} total ({n_singletons} singletons, "
              f"{n_nontrivial} non-trivial)")
        print(f"  Edges: {n_edges}")
        print(f"  Largest component: {largest} ({100*largest/n_total:.1f}%)")
        print(f"  Mean non-trivial size: {mean_nontrivial:.1f}")
        print(f"  Directions in non-trivial components: {n_in_nontrivial} "
              f"({100*n_in_nontrivial/n_total:.1f}%)")
        print(f"  Top 10 sizes: {sizes[:10]}")
        print(f"  Time: {elapsed:.1f}s")

    # Save raw results
    with open(os.path.join(args.output_dir, 'component_results.pkl'), 'wb') as f:
        pickle.dump({'results': results, 'n_total': n_total, 'args': vars(args)}, f)

    # Plot
    thresholds = [r['threshold'] for r in results]
    n_components = [r['n_components'] for r in results]
    largest = [r['largest'] for r in results]
    n_nontrivial = [r['n_nontrivial'] for r in results]
    mean_nontrivial = [r['mean_nontrivial'] for r in results]
    n_in_nontrivial = [r['n_in_nontrivial'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'Connected components in dictionary cosine similarity graph\n'
                 f'({n_total} directions, absolute cosine similarity)', fontsize=13)

    # Top left: number of non-trivial components
    ax = axes[0, 0]
    ax.plot(thresholds, n_nontrivial, 'o-', color='C0')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Number of non-trivial components')
    ax.set_title('Non-trivial components (size > 1)')
    ax.grid(True, alpha=0.3)

    # Top right: largest component size
    ax = axes[0, 1]
    ax.plot(thresholds, largest, 's-', color='C1')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Largest component size')
    ax.set_title('Largest component')
    ax.grid(True, alpha=0.3)

    # Bottom left: fraction of directions in non-trivial components
    ax = axes[1, 0]
    frac = [100 * x / n_total for x in n_in_nontrivial]
    ax.plot(thresholds, frac, 'D-', color='C2')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('% of directions in non-trivial components')
    ax.set_title('Fraction connected')
    ax.grid(True, alpha=0.3)

    # Bottom right: mean non-trivial component size
    ax = axes[1, 1]
    ax.plot(thresholds, mean_nontrivial, '^-', color='C3')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Mean component size')
    ax.set_title('Mean non-trivial component size')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'component_analysis.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")


if __name__ == '__main__':
    main()
