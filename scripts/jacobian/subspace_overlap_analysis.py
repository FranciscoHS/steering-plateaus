#!/usr/bin/env python3
"""
Compute pairwise subspace overlap between decomposition runs.

For runs i, j with direction matrices U_i, U_j (each k×d):
    overlap(i,j) = (1/k) ||U_i @ U_j^T||_F^2

This equals the mean squared cosine of the principal angles between the two
subspaces. It's 1 if the subspaces are identical, and k/d for random subspaces.

Loads results from reproducibility_svd.py and computes overlaps for various
truncations (top-5, top-10, all directions per run).

Usage:
    python scripts/jacobian/subspace_overlap_analysis.py \
        --results_pkl results/reproducibility_svd/eiffel/results.pkl
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np


def compute_pairwise_overlaps(run_directions, k_trunc=None):
    """
    Compute pairwise subspace overlap between runs.

    Args:
        run_directions: list of lists of direction tensors (one list per run)
        k_trunc: if set, only use the top k_trunc directions per run

    Returns:
        overlap_matrix: [n_runs, n_runs] array of overlap values
        k_used: actual number of directions used per run (min across runs)
    """
    n_runs = len(run_directions)

    # Truncate and stack into matrices
    if k_trunc is not None:
        k_used = min(k_trunc, min(len(dirs) for dirs in run_directions))
    else:
        k_used = min(len(dirs) for dirs in run_directions)

    matrices = []
    for dirs in run_directions:
        U = torch.stack(dirs[:k_used]).float()  # [k, d]
        matrices.append(U)

    overlap = np.zeros((n_runs, n_runs))
    for i in range(n_runs):
        for j in range(n_runs):
            # overlap = (1/k) ||U_i @ U_j^T||_F^2
            cross = matrices[i] @ matrices[j].T  # [k, k]
            overlap[i, j] = (cross ** 2).sum().item() / k_used

    return overlap, k_used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_pkl', required=True)
    parser.add_argument('--truncations', type=str, default='5,10,20',
                        help='Comma-separated k values to truncate at')
    args = parser.parse_args()

    with open(args.results_pkl, 'rb') as f:
        results = pickle.load(f)

    run_results = results['all_run_results']
    n_runs = len(run_results)
    d_model = results['direction_matrix'].shape[1]
    target_prompt = results.get('target_prompt', '(unknown)')

    truncations = [int(x) for x in args.truncations.split(',')]

    print(f"=== Pairwise subspace overlap analysis ===")
    print(f"Prompt: '{target_prompt}'")
    print(f"Runs: {n_runs}, d_model: {d_model}")
    print(f"Directions per run: {[len(r['directions']) for r in run_results]}")

    run_directions = [r['directions'] for r in run_results]

    # Random baseline
    print(f"\nRandom subspace baseline: k/d")
    for k in truncations:
        print(f"  k={k}: {k/d_model:.4f}")

    # Compute overlaps at each truncation
    all_overlaps = {}
    for k_trunc in truncations:
        overlap, k_used = compute_pairwise_overlaps(run_directions, k_trunc)
        all_overlaps[k_used] = overlap

        # Extract off-diagonal elements
        mask = ~np.eye(n_runs, dtype=bool)
        off_diag = overlap[mask]

        print(f"\nSubspace overlap (top {k_used} directions per run):")
        print(f"  Random baseline: {k_used/d_model:.4f}")
        print(f"  Mean overlap:    {off_diag.mean():.4f}")
        print(f"  Min overlap:     {off_diag.min():.4f}")
        print(f"  Max overlap:     {off_diag.max():.4f}")
        print(f"  Std overlap:     {off_diag.std():.4f}")
        print(f"  Ratio vs random: {off_diag.mean() / (k_used/d_model):.1f}x")

    # Also compute principal angles for the default truncation (middle value)
    k_main = truncations[len(truncations)//2]
    k_main = min(k_main, min(len(dirs) for dirs in run_directions))
    print(f"\n{'='*60}")
    print(f"Principal angles between run pairs (top {k_main} dirs)")
    print(f"{'='*60}")
    matrices = [torch.stack(dirs[:k_main]).float() for dirs in run_directions]
    for i in range(min(n_runs, 5)):  # show first 5 pairs with run 0
        for j in range(i+1, min(n_runs, 5)):
            cross = matrices[i] @ matrices[j].T
            # SVD of cross gives cos(principal angles)
            cos_angles = torch.linalg.svdvals(cross).numpy()
            angles_deg = np.degrees(np.arccos(np.clip(cos_angles, -1, 1)))
            print(f"  Runs {i+1}-{j+1}: angles = {', '.join(f'{a:.1f}°' for a in angles_deg)}")

    # ── Save & Plot ──
    results_dir = os.path.dirname(args.results_pkl)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_trunc = len(all_overlaps)
    fig, axes = plt.subplots(1, n_trunc + 1, figsize=(5 * (n_trunc + 1), 4.5))

    for idx, (k_used, overlap) in enumerate(all_overlaps.items()):
        ax = axes[idx]
        random_baseline = k_used / d_model
        im = ax.imshow(overlap, cmap='YlOrRd', vmin=0,
                       vmax=max(overlap.max(), 0.5))
        for i in range(n_runs):
            for j in range(n_runs):
                ax.text(j, i, f'{overlap[i,j]:.2f}', ha='center', va='center',
                        fontsize=7, color='white' if overlap[i,j] > 0.3 else 'black')
        ax.set_title(f'Top {k_used} dirs\n(random={random_baseline:.4f})')
        ax.set_xlabel('Run')
        ax.set_ylabel('Run')
        ax.set_xticks(range(n_runs))
        ax.set_yticks(range(n_runs))
        ax.set_xticklabels(range(1, n_runs+1), fontsize=8)
        ax.set_yticklabels(range(1, n_runs+1), fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Last panel: summary bar chart
    ax = axes[-1]
    ks = sorted(all_overlaps.keys())
    means = [all_overlaps[k][~np.eye(n_runs, dtype=bool)].mean() for k in ks]
    baselines = [k / d_model for k in ks]
    x = np.arange(len(ks))
    ax.bar(x - 0.15, means, 0.3, label='Measured', color='steelblue')
    ax.bar(x + 0.15, baselines, 0.3, label='Random baseline', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in ks])
    ax.set_ylabel('Mean pairwise overlap')
    ax.set_title('Overlap vs random')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f"Pairwise subspace overlap: '{target_prompt}'", fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'subspace_overlap.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {plot_path}")

    # Save overlap data
    overlap_path = os.path.join(results_dir, 'subspace_overlaps.pkl')
    with open(overlap_path, 'wb') as f:
        pickle.dump({'overlaps': all_overlaps, 'truncations': truncations,
                     'd_model': d_model, 'n_runs': n_runs}, f)
    print(f"Overlap data saved to {overlap_path}")


if __name__ == '__main__':
    main()
