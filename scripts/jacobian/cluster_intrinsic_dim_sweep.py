#!/usr/bin/env python3
"""
Systematic study of local intrinsic dimensionality vs cluster size.

Varies the number of k-means clusters to get clusters of different sizes,
estimates TWO-NN ID for each cluster, and plots ID vs cluster size.

Usage:
    python -u scripts/jacobian/cluster_intrinsic_dim_sweep.py \
        --cache_path results/trained_autoencoder/bottleneck_sweep/cached_activations.pt \
        --output_dir results/cluster_id_sweep
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skdim
from sklearn.cluster import MiniBatchKMeans


def run_twonn(X_np):
    """Run TWO-NN with jitter to handle near-duplicates."""
    X_np = X_np + np.random.randn(*X_np.shape) * 1e-7 * X_np.std()
    est = skdim.id.TwoNN()
    est.fit(X_np)
    return est.dimension_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_path', required=True,
                        help='Path to cached activations tensor')
    parser.add_argument('--n_clusters_list', type=int, nargs='+',
                        default=[5, 10, 20, 50, 100, 200, 500],
                        help='Numbers of clusters to try')
    parser.add_argument('--n_samples', type=int, default=50000,
                        help='Number of activations to use')
    parser.add_argument('--min_cluster_size', type=int, default=100,
                        help='Minimum cluster size for reliable ID estimation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='results/cluster_id_sweep')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load activations
    print(f"Loading activations from {args.cache_path}...")
    activations = torch.load(args.cache_path, weights_only=True)
    print(f"  Shape: {activations.shape}")

    if args.n_samples < activations.shape[0]:
        idx = torch.randperm(activations.shape[0])[:args.n_samples]
        activations = activations[idx]
        print(f"  Subsampled to {args.n_samples}")

    X = activations.float().numpy()

    # Global ID
    print(f"\nGlobal ID ({len(X)} points)...")
    global_id = run_twonn(X)
    print(f"  Global ID = {global_id:.2f}")

    # Collect all (cluster_size, local_id) pairs
    all_sizes = []
    all_ids = []
    results_by_k = {}

    for n_clusters in sorted(args.n_clusters_list):
        expected_size = len(X) // n_clusters
        if expected_size < args.min_cluster_size:
            print(f"\nk={n_clusters}: expected cluster size {expected_size} < {args.min_cluster_size}, skipping")
            continue

        print(f"\nk={n_clusters} (expected ~{expected_size} pts/cluster)...")
        t0 = time.time()
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=args.seed,
                                batch_size=min(10000, len(X)))
        labels = kmeans.fit_predict(X)
        print(f"  Clustering: {time.time() - t0:.1f}s")

        cluster_ids = []
        cluster_sizes = []
        for c in range(n_clusters):
            mask = labels == c
            n_in = mask.sum()
            if n_in < args.min_cluster_size:
                continue
            cid = run_twonn(X[mask])
            cluster_ids.append(cid)
            cluster_sizes.append(int(n_in))
            all_sizes.append(int(n_in))
            all_ids.append(cid)

        cluster_ids = np.array(cluster_ids)
        cluster_sizes = np.array(cluster_sizes)
        print(f"  {len(cluster_ids)} clusters kept, sizes {cluster_sizes.min()}-{cluster_sizes.max()}")
        print(f"  ID: mean={cluster_ids.mean():.2f}, std={cluster_ids.std():.2f}, "
              f"range=[{cluster_ids.min():.2f}, {cluster_ids.max():.2f}]")

        results_by_k[n_clusters] = {
            'cluster_ids': cluster_ids.tolist(),
            'cluster_sizes': cluster_sizes.tolist(),
        }

    all_sizes = np.array(all_sizes)
    all_ids = np.array(all_ids)

    # Plot: scatter of ID vs cluster size
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(all_sizes, all_ids, alpha=0.4, s=15, color='steelblue')
    ax.axhline(global_id, color='red', linestyle='--', linewidth=1.5,
               label=f'Global ID = {global_id:.1f}')

    # Binned median line
    size_bins = np.logspace(np.log10(all_sizes.min()), np.log10(all_sizes.max()), 15)
    bin_centers = []
    bin_medians = []
    bin_q25 = []
    bin_q75 = []
    for i in range(len(size_bins) - 1):
        mask = (all_sizes >= size_bins[i]) & (all_sizes < size_bins[i+1])
        if mask.sum() >= 3:
            bin_centers.append(np.sqrt(size_bins[i] * size_bins[i+1]))
            bin_medians.append(np.median(all_ids[mask]))
            bin_q25.append(np.percentile(all_ids[mask], 25))
            bin_q75.append(np.percentile(all_ids[mask], 75))
    if bin_centers:
        bin_centers = np.array(bin_centers)
        bin_medians = np.array(bin_medians)
        bin_q25 = np.array(bin_q25)
        bin_q75 = np.array(bin_q75)
        ax.plot(bin_centers, bin_medians, 'o-', color='darkblue', linewidth=2,
                markersize=6, label='Binned median')
        ax.fill_between(bin_centers, bin_q25, bin_q75, color='darkblue', alpha=0.15,
                        label='IQR')

    ax.set_xscale('log')
    ax.set_xlabel('Cluster size (number of points)')
    ax.set_ylabel('TWO-NN intrinsic dimensionality')
    ax.set_title('Local ID vs cluster size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, 'id_vs_cluster_size.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")

    # Save results
    results = {
        'global_id': global_id,
        'n_samples': len(X),
        'min_cluster_size': args.min_cluster_size,
        'all_sizes': all_sizes.tolist(),
        'all_ids': all_ids.tolist(),
        'by_k': results_by_k,
    }
    results_path = os.path.join(args.output_dir, 'sweep_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
