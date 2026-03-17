#!/usr/bin/env python3
"""Run MELBO many times with different seeds, cluster directions, average within clusters.

Idea: each MELBO run finds f + a_i where f is a true feature and a_i is random noise.
By running many times and averaging directions that land near the same feature,
we can estimate f (since the a_i are zero-mean).

Algorithm:
  1. Run N independent MELBO optimizations (1 direction each, different seeds)
  2. Cluster by cosine similarity > threshold (greedy: assign each to most similar
     existing cluster, or start new cluster)
  3. Average within clusters (and normalize)
  4. Sweep the averaged directions + individual directions

Usage:
    python scripts/melbo/run_melbo_multiseed.py --model qwen3-1.7b --inject_layer 7 --measure_layer -2 --n_seeds 20
"""

import argparse, os, pickle, sys, time
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_to_layer, forward_from_layer_to_final,
    _get_blocks,
)
from src.data import load_fineweb_fixed_length

# Reuse core functions from iterative melbo
from scripts.melbo.run_iterative_melbo_hf import (
    extract_context_and_activation,
    forward_to_L,
    capped_melbo_one_direction,
    refine_in_cone,
    sweep_dir,
)


def cluster_directions(directions, threshold=0.5):
    """Greedy clustering by cosine similarity.

    For each direction, find the existing cluster whose centroid has highest
    cosine similarity. If above threshold, add to that cluster. Otherwise,
    start a new cluster.

    Returns list of clusters, each a list of indices into directions.
    """
    if len(directions) == 0:
        return []

    D = torch.stack(directions)  # [N, d_model]
    clusters = []  # list of list of indices
    centroids = []  # normalized mean of each cluster

    for i in range(len(directions)):
        if len(clusters) == 0:
            clusters.append([i])
            centroids.append(D[i])
            continue

        # Compute cosine similarity to all centroids
        C = torch.stack(centroids)  # [n_clusters, d_model]
        cos_sims = F.cosine_similarity(D[i].unsqueeze(0), C, dim=1)  # [n_clusters]
        best_sim, best_idx = cos_sims.max(dim=0)

        if best_sim.item() >= threshold:
            clusters[best_idx].append(i)
            # Update centroid (running mean, normalized)
            cluster_dirs = D[clusters[best_idx]]
            centroids[best_idx] = F.normalize(cluster_dirs.mean(dim=0), dim=0)
        else:
            clusters.append([i])
            centroids.append(D[i])

    return clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--inject_layer', type=int, default=7)
    parser.add_argument('--measure_layer', type=int, default=-2)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--act_idx', type=int, default=0)
    parser.add_argument('--n_load', type=int, default=100)

    # Multi-seed config
    parser.add_argument('--n_seeds', type=int, default=20,
                        help='Number of independent MELBO runs')
    parser.add_argument('--cluster_threshold', type=float, default=0.5,
                        help='Cosine similarity threshold for clustering')

    # MELBO config (per-run)
    parser.add_argument('--opt_angle', type=float, default=5.0)
    parser.add_argument('--target_cos', type=float, default=0.1)
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--n_restarts', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)

    # Refinement
    parser.add_argument('--refine_steps', type=int, default=200)
    parser.add_argument('--refine_restarts', type=int, default=3)
    parser.add_argument('--max_drift', type=float, default=5.0)
    parser.add_argument('--refine_lr', type=float, default=0.01)

    # Sweep config
    parser.add_argument('--max_sweep_angle', type=float, default=60.0)
    parser.add_argument('--n_sweep_angles', type=int, default=60)
    parser.add_argument('--n_rand', type=int, default=20)

    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--data_seed', type=int, default=42,
                        help='Seed for data loading (same across all runs)')
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = (f'results/melbo_multiseed/{args.model}'
                           f'_L{args.inject_layer}_meas{args.measure_layer}'
                           f'_angle{args.opt_angle}_N{args.n_seeds}')
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model and data (deterministic, same for all runs)
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))

    if args.measure_layer < 0:
        args.measure_layer = n_layers + args.measure_layer
    print(f"Device: {device}, d_model: {d_model}, measure_layer: {args.measure_layer}")

    print(f"Loading {args.n_load} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(args.n_load, tokenizer,
                                                seq_len=args.seq_len, seed=args.data_seed)
    print(f"Extracting layer-{args.inject_layer} activations...")
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.inject_layer, device)

    ctx = all_contexts[args.act_idx].unsqueeze(0).to(device)
    act = all_activations[args.act_idx].to(device)
    a_norm = act.norm()
    a_unit = act / a_norm
    print(f"Using activation {args.act_idx}, norm: {a_norm:.2f}")

    with torch.no_grad():
        ref_L = forward_to_L(model, ctx, act.unsqueeze(0),
                             args.inject_layer, args.measure_layer)

    angle_rad = np.deg2rad(args.opt_angle)

    # === Run MELBO with different seeds ===
    all_raw_directions = []
    all_rewards = []
    all_cos_dists = []

    for seed_i in range(args.n_seeds):
        opt_seed = seed_i * 1000 + 7  # deterministic but varied seeds
        torch.manual_seed(opt_seed)
        np.random.seed(opt_seed)

        print(f"\n{'='*60}")
        print(f"MELBO run {seed_i+1}/{args.n_seeds} (opt_seed={opt_seed})")
        print(f"{'='*60}")

        # Find one direction (no previous directions to orthogonalize against)
        direction, reward, history = capped_melbo_one_direction(
            model, ctx, act, ref_L, a_unit, a_norm,
            angle_rad, args.target_cos, [],  # no prev directions
            args.inject_layer, args.measure_layer, device, d_model,
            n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts)

        if direction is None:
            print(f"  Run {seed_i+1}: no direction found, skipping")
            continue

        # Refine
        refined, cd, drift = refine_in_cone(
            model, ctx, act, ref_L, a_unit, a_norm,
            direction, args.max_drift, angle_rad,
            args.inject_layer, args.measure_layer, device, d_model,
            n_steps=args.refine_steps, lr=args.refine_lr,
            n_restarts=args.refine_restarts)

        all_raw_directions.append(refined.cpu())
        all_rewards.append(reward)
        all_cos_dists.append(cd)
        print(f"  reward={reward:.6f}, cos_dist={cd:.4f}, drift={drift:.1f}°")

    n_found = len(all_raw_directions)
    print(f"\n{'='*60}")
    print(f"Found {n_found} directions from {args.n_seeds} runs")
    print(f"{'='*60}")

    if n_found == 0:
        print("No directions found!")
        return

    # === Pairwise cosine similarity matrix ===
    D = torch.stack(all_raw_directions)
    cos_sim_matrix = (D @ D.T)  # [N, N], values in [-1, 1]
    print(f"\nPairwise cosine similarities:")
    print(f"  min={cos_sim_matrix.min().item():.3f}, "
          f"max (off-diag)={cos_sim_matrix.fill_diagonal_(0).abs().max().item():.3f}, "
          f"mean (off-diag)={cos_sim_matrix.abs().sum().item() / max(n_found*(n_found-1), 1):.3f}")

    # === Cluster ===
    clusters = cluster_directions(all_raw_directions, threshold=args.cluster_threshold)
    print(f"\nClustering (threshold={args.cluster_threshold}):")
    print(f"  {len(clusters)} clusters from {n_found} directions")

    # Compute cluster averages
    averaged_directions = []
    cluster_info = []
    for ci, cluster_indices in enumerate(clusters):
        cluster_dirs = D[cluster_indices]
        avg = F.normalize(cluster_dirs.mean(dim=0), dim=0)
        averaged_directions.append(avg)

        # Within-cluster stats
        if len(cluster_indices) > 1:
            pairwise = cluster_dirs @ cluster_dirs.T
            pairwise.fill_diagonal_(0)
            mean_sim = pairwise.sum().item() / (len(cluster_indices) * (len(cluster_indices) - 1))
        else:
            mean_sim = 1.0

        info = {
            'cluster_idx': ci,
            'n_members': len(cluster_indices),
            'member_indices': cluster_indices,
            'mean_cos_sim': mean_sim,
            'member_rewards': [all_rewards[j] for j in cluster_indices],
            'member_cos_dists': [all_cos_dists[j] for j in cluster_indices],
        }
        cluster_info.append(info)
        print(f"  Cluster {ci}: {len(cluster_indices)} members, "
              f"mean_cos_sim={mean_sim:.3f}, "
              f"rewards={[f'{r:.4f}' for r in info['member_rewards']]}")

    # === Sweep ===
    angles_deg = np.linspace(0, args.max_sweep_angle, args.n_sweep_angles + 1)[1:]
    n_avg = len(averaged_directions)

    # Sweep averaged directions
    print(f"\nSweeping {n_avg} averaged directions...")
    avg_cos = np.zeros((n_avg, len(angles_deg)))
    avg_l2 = np.zeros((n_avg, len(angles_deg)))
    for i, d in enumerate(averaged_directions):
        avg_cos[i], avg_l2[i] = sweep_dir(
            model, ctx, act, ref_L, a_unit, a_norm, d.to(device),
            args.inject_layer, args.measure_layer, device, angles_deg)

    # Sweep individual directions
    print(f"Sweeping {n_found} individual directions...")
    indiv_cos = np.zeros((n_found, len(angles_deg)))
    indiv_l2 = np.zeros((n_found, len(angles_deg)))
    for i, d in enumerate(all_raw_directions):
        indiv_cos[i], indiv_l2[i] = sweep_dir(
            model, ctx, act, ref_L, a_unit, a_norm, d.to(device),
            args.inject_layer, args.measure_layer, device, angles_deg)

    # Sweep random
    print(f"Sweeping {args.n_rand} random directions...")
    torch.manual_seed(args.data_seed + 999)
    rand_dirs = F.normalize(torch.randn(args.n_rand, d_model), dim=-1)
    rand_cos = np.zeros((args.n_rand, len(angles_deg)))
    rand_l2 = np.zeros((args.n_rand, len(angles_deg)))
    for k in range(args.n_rand):
        rand_cos[k], rand_l2[k] = sweep_dir(
            model, ctx, act, ref_L, a_unit, a_norm,
            rand_dirs[k], args.inject_layer,
            args.measure_layer, device, angles_deg)

    # === Save ===
    results = {
        'raw_directions': [d.cpu() for d in all_raw_directions],
        'raw_rewards': all_rewards,
        'raw_cos_dists': all_cos_dists,
        'clusters': cluster_info,
        'averaged_directions': [d.cpu() for d in averaged_directions],
        'cos_sim_matrix': cos_sim_matrix.cpu().numpy(),
        'angles_deg': angles_deg,
        'avg_cos': avg_cos,
        'avg_l2': avg_l2,
        'indiv_cos': indiv_cos,
        'indiv_l2': indiv_l2,
        'rand_cos': rand_cos,
        'rand_l2': rand_l2,
        'args': vars(args),
    }
    save_path = os.path.join(args.results_dir, 'results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    # === Plot ===
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax_idx, (metric_name, avg_data, indiv_data, rand_data) in enumerate([
        ('Cosine distance', avg_cos, indiv_cos, rand_cos),
        ('L2 distance', avg_l2, indiv_l2, rand_l2),
    ]):
        ax = axes[ax_idx]

        # Random baseline
        rand_med = np.median(rand_data, axis=0)
        rand_q25 = np.percentile(rand_data, 25, axis=0)
        rand_q75 = np.percentile(rand_data, 75, axis=0)
        ax.fill_between(angles_deg, rand_q25, rand_q75, color='gray', alpha=0.15)
        ax.plot(angles_deg, rand_med, color='gray', lw=2, label='Random')

        # Individual directions (thin, colored by cluster)
        colors = plt.cm.tab10(np.arange(len(clusters)) % 10)
        for ci, cl in enumerate(clusters):
            for j, idx in enumerate(cl):
                ax.plot(angles_deg, indiv_data[idx], color=colors[ci],
                        lw=0.8, alpha=0.3,
                        label=f'Cluster {ci} indiv' if j == 0 else None)

        # Averaged directions (thick)
        for ci in range(n_avg):
            n_members = cluster_info[ci]['n_members']
            ax.plot(angles_deg, avg_data[ci], color=colors[ci],
                    lw=2.5, label=f'Cluster {ci} avg (n={n_members})')

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(metric_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(metric_name)

    # Right: cosine similarity matrix
    ax = axes[2]
    # Restore diagonal for display
    cos_sim_display = (D @ D.T).numpy()
    im = ax.imshow(cos_sim_display, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Cosine similarity ({n_found} dirs)')

    plt.suptitle(f'{args.model} L{args.inject_layer}→L{args.measure_layer}: '
                 f'{args.n_seeds} seeds → {n_found} dirs → {len(clusters)} clusters '
                 f'(threshold={args.cluster_threshold})',
                 fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'multiseed_melbo.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {plot_path}")
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    main()
