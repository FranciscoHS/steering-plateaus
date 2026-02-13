#!/usr/bin/env python3
"""
Plot relative distance along slerp vs optimized path.

Computes d = ||logits(t) - logits_A|| / (||logits(t) - logits_A|| + ||logits(t) - logits_B||)
at evenly-spaced points along both paths, matching the interpolation experiment plot.
"""

import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer


def slerp_point(v0, v1, t):
    """Slerp between two vectors (direction) + linear magnitude."""
    norm0 = v0.norm()
    norm1 = v1.norm()
    u0 = v0 / norm0
    u1 = v1 / norm1
    dot = torch.clamp(torch.dot(u0, u1), -1.0, 1.0)
    theta = torch.acos(dot)
    if theta.abs() < 1e-6:
        direction = u0
    else:
        sin_theta = torch.sin(theta)
        direction = (torch.sin((1 - t) * theta) / sin_theta) * u0 + \
                    (torch.sin(t * theta) / sin_theta) * u1
    mag = (1 - t) * norm0 + t * norm1
    return direction * mag


def compute_relative_distances(model, tokenizer, context, waypoints, start_layer,
                               logits_A, logits_B, n_points=100):
    """Compute relative logit distance and top token at n_points along a path."""
    device = waypoints.device
    n_segments = waypoints.shape[0] - 1

    t_global = np.linspace(0, 1, n_points)
    rel_distances = []
    top_tokens = []

    for tg in t_global:
        seg_float = tg * n_segments
        seg_idx = min(int(seg_float), n_segments - 1)
        t_local = seg_float - seg_idx

        with torch.no_grad():
            point = slerp_point(waypoints[seg_idx], waypoints[seg_idx + 1], t_local)
            point_3d = point.reshape(1, 1, -1)
            logits = forward_from_layer(model, context, point_3d, start_layer).squeeze(0)

            dist_A = (logits - logits_A).norm().item()
            dist_B = (logits - logits_B).norm().item()
            rel_dist = dist_A / (dist_A + dist_B + 1e-10)

            top_idx = logits.argmax().item()
            top_tok = tokenizer.decode([top_idx])

        rel_distances.append(rel_dist)
        top_tokens.append(top_tok)

    return t_global, np.array(rel_distances), top_tokens


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/path_optimization_v2')
    args = parser.parse_args()
    results_dir = args.results_dir

    # Load saved results
    with open(os.path.join(results_dir, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)

    start_layer = results['args']['start_layer']

    print(f"Loading model: {results['args']['model']}")
    model, tokenizer, device = load_model(results['args']['model'])

    context = results['opt_eval']['predictions'][0]  # won't work, need raw context
    # Actually, load from checkpoint which has the context tensor
    with open(os.path.join(results_dir, 'checkpoint.pkl'), 'rb') as f:
        cp = pickle.load(f)

    context = cp['context'].to(device)
    start_act = cp['start_act'].to(device)
    end_act = cp['end_act'].to(device)
    opt_knots = cp['opt_state']['knots'].to(device)

    n_knots = opt_knots.shape[0]
    print(f"Start layer: {start_layer}, knots: {n_knots}")

    # Get endpoint logits
    with torch.no_grad():
        logits_A = forward_from_layer(
            model, context, start_act.reshape(1, 1, -1), start_layer
        ).squeeze(0)
        logits_B = forward_from_layer(
            model, context, end_act.reshape(1, 1, -1), start_layer
        ).squeeze(0)

    # Slerp path: same number of waypoints as optimized for fair comparison
    n_waypoints = n_knots + 2
    t_init = torch.linspace(0, 1, n_waypoints, device=device)
    slerp_waypoints = torch.stack([
        slerp_point(start_act, end_act, t.item()) for t in t_init
    ])

    # Optimized path
    opt_waypoints = torch.cat([
        start_act.unsqueeze(0), opt_knots, end_act.unsqueeze(0)
    ], dim=0)

    # Compute relative distances
    n_points = 200
    print(f"Computing relative distances ({n_points} points per path)...")

    print("  Slerp path...")
    t_slerp, d_slerp, tok_slerp = compute_relative_distances(
        model, tokenizer, context, slerp_waypoints, start_layer,
        logits_A, logits_B, n_points
    )

    print("  Optimized path...")
    t_opt, d_opt, tok_opt = compute_relative_distances(
        model, tokenizer, context, opt_waypoints, start_layer,
        logits_A, logits_B, n_points
    )

    # Identify stable token regions for annotation
    def get_token_regions(t_vals, tokens):
        """Find contiguous regions with the same top token."""
        regions = []
        current_tok = tokens[0]
        start_idx = 0
        for i in range(1, len(tokens)):
            if tokens[i] != current_tok:
                regions.append((t_vals[start_idx], t_vals[i-1], current_tok))
                current_tok = tokens[i]
                start_idx = i
        regions.append((t_vals[start_idx], t_vals[-1], current_tok))
        return regions

    slerp_regions = get_token_regions(t_slerp, tok_slerp)
    opt_regions = get_token_regions(t_opt, tok_opt)

    print("\nSlerp token regions:")
    for t0, t1, tok in slerp_regions:
        if t1 - t0 > 0.02:  # skip tiny regions
            print(f"  t=[{t0:.2f}, {t1:.2f}]: '{tok}'")

    print("\nOptimized token regions:")
    for t0, t1, tok in opt_regions:
        if t1 - t0 > 0.02:
            print(f"  t=[{t0:.2f}, {t1:.2f}]: '{tok}'")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(t_slerp, d_slerp, 'b-', linewidth=2, label='Slerp path', alpha=0.8)
    ax.plot(t_opt, d_opt, 'r-', linewidth=2, label='Optimized path', alpha=0.8)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Linear (d = t)')

    # Annotate token regions (only regions longer than 5% of path)
    def annotate_regions(regions, d_vals, t_vals, color, y_offset):
        """Add token labels at the center of each stable region."""
        for t0, t1, tok in regions:
            if t1 - t0 < 0.05:
                continue
            t_mid = (t0 + t1) / 2
            # Find d at midpoint
            idx = np.argmin(np.abs(t_vals - t_mid))
            d_mid = d_vals[idx]
            tok_display = repr(tok).strip("'")
            ax.annotate(
                tok_display,
                xy=(t_mid, d_mid + y_offset),
                fontsize=9, fontweight='bold', color=color,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec=color, alpha=0.85, linewidth=0.8),
            )

    annotate_regions(slerp_regions, d_slerp, t_slerp, 'blue', -0.07)
    annotate_regions(opt_regions, d_opt, t_opt, 'red', +0.07)

    ax.set_xlabel('Path parameter t', fontsize=12)
    ax.set_ylabel('Relative distance d', fontsize=12)
    ax.set_title("Relative Distance Along Path\n"
                 "'The house was big' → 'The house was in'",
                 fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 1.08)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'relative_distance_with_tokens.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")

    # Print summary
    print(f"\nSlerp: d at midpoint = {d_slerp[n_points//2]:.3f}")
    print(f"Optimized: d at midpoint = {d_opt[n_points//2]:.3f}")

    # Find transition point (where d crosses 0.5) for each path
    for name, d_vals in [('Slerp', d_slerp), ('Optimized', d_opt)]:
        crossings = np.where(np.diff(np.sign(d_vals - 0.5)))[0]
        if len(crossings) > 0:
            t_cross = t_slerp[crossings[0]]
            print(f"{name}: crosses d=0.5 at t≈{t_cross:.3f}")


if __name__ == '__main__':
    main()
