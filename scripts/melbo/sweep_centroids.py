#!/usr/bin/env python3
"""Sweep pre-computed centroid directions and plot against random baseline.

Usage:
    python scripts/melbo/sweep_centroids.py --model qwen3-1.7b --inject_layer 7 --measure_layer -2 \
        --centroids centroids_thresh005.pt --results_dir results/melbo_multiseed/sweep_2clusters
"""

import argparse, os, pickle, sys
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.melbo.run_iterative_melbo_hf import (
    extract_context_and_activation, forward_to_L, sweep_dir,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--inject_layer', type=int, default=7)
    parser.add_argument('--measure_layer', type=int, default=-2)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--act_idx', type=int, default=0)
    parser.add_argument('--n_load', type=int, default=100)
    parser.add_argument('--centroids', type=str, required=True)
    parser.add_argument('--max_sweep_angle', type=float, default=60.0)
    parser.add_argument('--n_sweep_angles', type=int, default=60)
    parser.add_argument('--n_rand', type=int, default=20)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--title', type=str, default=None,
                        help='Custom plot title (default: auto-generated)')
    parser.add_argument('--label_prefix', type=str, default='Dir',
                        help='Label prefix for directions in legend (default: Dir)')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))

    if args.measure_layer < 0:
        args.measure_layer = n_layers + args.measure_layer

    token_ids_list = load_fineweb_fixed_length(args.n_load, tokenizer,
                                                seq_len=args.seq_len, seed=args.seed)
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.inject_layer, device)

    ctx = all_contexts[args.act_idx].unsqueeze(0).to(device)
    act = all_activations[args.act_idx].to(device)
    a_norm = act.norm()
    a_unit = act / a_norm

    with torch.no_grad():
        ref_L = forward_to_L(model, ctx, act.unsqueeze(0),
                             args.inject_layer, args.measure_layer)

    # Load centroids
    centroids = torch.load(args.centroids, weights_only=True)
    n_cent = len(centroids)
    print(f"Loaded {n_cent} centroids from {args.centroids}")

    angles_deg = np.linspace(0, args.max_sweep_angle, args.n_sweep_angles + 1)[1:]

    # Sweep centroids
    cent_cos = np.zeros((n_cent, len(angles_deg)))
    cent_l2 = np.zeros((n_cent, len(angles_deg)))
    for i in range(n_cent):
        print(f"Sweeping centroid {i}...")
        cent_cos[i], cent_l2[i] = sweep_dir(
            model, ctx, act, ref_L, a_unit, a_norm, centroids[i].to(device),
            args.inject_layer, args.measure_layer, device, angles_deg)

    # Sweep random
    print(f"Sweeping {args.n_rand} random directions...")
    rand_dirs = F.normalize(torch.randn(args.n_rand, d_model), dim=-1)
    rand_cos = np.zeros((args.n_rand, len(angles_deg)))
    rand_l2 = np.zeros((args.n_rand, len(angles_deg)))
    for k in range(args.n_rand):
        rand_cos[k], rand_l2[k] = sweep_dir(
            model, ctx, act, ref_L, a_unit, a_norm,
            rand_dirs[k], args.inject_layer,
            args.measure_layer, device, angles_deg)

    # Save
    results = {
        'centroids': centroids.cpu(),
        'angles_deg': angles_deg,
        'cent_cos': cent_cos, 'cent_l2': cent_l2,
        'rand_cos': rand_cos, 'rand_l2': rand_l2,
        'args': vars(args),
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.arange(n_cent))

    for ax_idx, (metric_name, cent_data, rand_data) in enumerate([
        ('Cosine distance', cent_cos, rand_cos),
        ('L2 distance', cent_l2, rand_l2),
    ]):
        ax = axes[ax_idx]
        rand_med = np.median(rand_data, axis=0)
        rand_q25 = np.percentile(rand_data, 25, axis=0)
        rand_q75 = np.percentile(rand_data, 75, axis=0)
        ax.fill_between(angles_deg, rand_q25, rand_q75, color='gray', alpha=0.15)
        ax.plot(angles_deg, rand_med, color='gray', lw=2, label='Random (median)')

        for i in range(n_cent):
            ax.plot(angles_deg, cent_data[i], color=colors[i],
                    lw=2.5, label=f'{args.label_prefix} {i}')

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(metric_name)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(metric_name)

    title = args.title or (f'{args.model} L{args.inject_layer}→L{args.measure_layer}: '
                           f'{n_cent} directions')
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'sweep_centroids.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


if __name__ == '__main__':
    main()
