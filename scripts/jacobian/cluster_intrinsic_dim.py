#!/usr/bin/env python3
"""
Cluster activations and estimate intrinsic dimensionality per cluster.

Goal: understand whether different regions of activation space have different
local intrinsic dimensionality (vs the ~10 global estimate).

Usage:
    python -u scripts/jacobian/cluster_intrinsic_dim.py \
        --cache_path /workspace/steering-plateaus/results/trained_autoencoder/bottleneck_sweep/cached_activations.pt \
        --output_dir /workspace/steering-plateaus/results/trained_autoencoder/cluster_id
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import skdim


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
    parser.add_argument('--n_clusters', type=int, nargs='+', default=[10, 20, 50],
                        help='Number of clusters to try')
    parser.add_argument('--n_samples', type=int, default=50000,
                        help='Number of activations to use (subsample from cache)')
    parser.add_argument('--min_cluster_size', type=int, default=200,
                        help='Minimum cluster size for reliable ID estimation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='results/trained_autoencoder/cluster_id')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load activations
    print(f"Loading activations from {args.cache_path}...")
    activations = torch.load(args.cache_path, weights_only=True)
    print(f"  Shape: {activations.shape}")

    # Subsample
    if args.n_samples < activations.shape[0]:
        idx = torch.randperm(activations.shape[0])[:args.n_samples]
        activations = activations[idx]
        print(f"  Subsampled to {args.n_samples}")

    X = activations.float().numpy()

    # Global intrinsic dimensionality
    print(f"\nGlobal intrinsic dimensionality ({len(X)} points)...")
    global_id = run_twonn(X)
    print(f"  Global ID = {global_id:.2f}")

    all_results = {'global_id': global_id}

    for n_clusters in args.n_clusters:
        print(f"\n{'='*60}")
        print(f"K-means with {n_clusters} clusters")
        print(f"{'='*60}")

        # K-means clustering
        from sklearn.cluster import MiniBatchKMeans
        t0 = time.time()
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=args.seed,
                                batch_size=min(10000, len(X)))
        labels = kmeans.fit_predict(X)
        print(f"  Clustering took {time.time() - t0:.1f}s")

        # Estimate ID per cluster
        cluster_ids = []
        cluster_sizes = []
        for c in range(n_clusters):
            mask = labels == c
            n_in_cluster = mask.sum()
            if n_in_cluster < args.min_cluster_size:
                print(f"  Cluster {c}: {n_in_cluster} points (skipped, too small)")
                continue
            X_cluster = X[mask]
            cid = run_twonn(X_cluster)
            cluster_ids.append(cid)
            cluster_sizes.append(n_in_cluster)
            print(f"  Cluster {c}: {n_in_cluster:6d} points, ID = {cid:.2f}")

        cluster_ids = np.array(cluster_ids)
        cluster_sizes = np.array(cluster_sizes)

        print(f"\n  Summary ({len(cluster_ids)} clusters with >= {args.min_cluster_size} points):")
        print(f"    Mean ID  = {cluster_ids.mean():.2f}")
        print(f"    Median ID = {np.median(cluster_ids):.2f}")
        print(f"    Std ID   = {cluster_ids.std():.2f}")
        print(f"    Min ID   = {cluster_ids.min():.2f}")
        print(f"    Max ID   = {cluster_ids.max():.2f}")

        all_results[f'k{n_clusters}'] = {
            'cluster_ids': cluster_ids.tolist(),
            'cluster_sizes': cluster_sizes.tolist(),
            'mean_id': float(cluster_ids.mean()),
            'median_id': float(np.median(cluster_ids)),
            'std_id': float(cluster_ids.std()),
            'min_id': float(cluster_ids.min()),
            'max_id': float(cluster_ids.max()),
        }

        # Histogram of cluster IDs
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(cluster_ids, bins=max(10, n_clusters // 3), edgecolor='black', alpha=0.7)
        ax.axvline(global_id, color='red', linestyle='--', label=f'Global ID = {global_id:.1f}')
        ax.axvline(cluster_ids.mean(), color='blue', linestyle='--',
                   label=f'Mean local ID = {cluster_ids.mean():.1f}')
        ax.set_xlabel('Intrinsic dimensionality')
        ax.set_ylabel('Number of clusters')
        ax.set_title(f'Local ID distribution (k={n_clusters} clusters)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = os.path.join(args.output_dir, f'cluster_id_k{n_clusters}.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to {plot_path}")

    # Save results
    results_path = os.path.join(args.output_dir, 'cluster_id_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
