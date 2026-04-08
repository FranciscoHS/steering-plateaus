#!/usr/bin/env python3
"""Perturbation sweep: MELBO directions vs random directions.

Sweeps each direction across angles 0-60°, measuring L2 at penultimate layer.
Compares top-k MELBO directions (from a saved results.pkl) to random directions.

Usage:
    python scripts/melbo/sweep_directions_comparison.py \
        --melbo_results results/melbo_spectrum/qwen3-1.7b_layer2_l2_tgt26_act1/results.pkl \
        --melbo_angle 1.0 --n_melbo 5 --n_random 20
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks, forward_from_layer_to_layer
from src.data import load_fineweb_fixed_length


def sweep_single_direction(model, ctx, act, direction, layer, measure_layer,
                           device, angles_rad, batch_size=128):
    """Sweep a single direction across angles, measuring L2 at measure_layer.

    Args:
        ctx: [1, seq-1, d_model] context hidden states
        act: [d_model] activation vector
        direction: [d_model] unit direction
        angles_rad: 1D array of angles

    Returns:
        l2_curve: [n_angles] array
    """
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=torch.float32)
    R = a_d.norm()
    a_unit = a_d / R

    d_d = direction.to(device=device, dtype=torch.float32)
    # Orthogonalize against activation
    d_orth = d_d - (d_d @ a_unit) * a_unit
    d_orth = F.normalize(d_orth, dim=0)

    n = len(angles_rad)
    s = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)
    c = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)

    # Perturbed points: R * (cos(α)·â + sin(α)·d_orth)
    all_points = (R * (c.unsqueeze(1) * a_unit.unsqueeze(0)
                       + s.unsqueeze(1) * d_orth.unsqueeze(0))).to(model_dtype)

    # Reference
    with torch.no_grad():
        ref_act = forward_from_layer_to_layer(
            model, ctx_d, a_d[None, None].to(model_dtype), layer, measure_layer).float()

    l2_curve = np.zeros(n)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = all_points[start:end].unsqueeze(1)
            pf = forward_from_layer_to_layer(
                model, ctx_d, batch, layer, measure_layer).float()
            l2_curve[start:end] = (pf - ref_act).norm(dim=-1).cpu().numpy()

    return l2_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--melbo_results', required=True,
                        help='Path to MELBO spectrum results.pkl')
    parser.add_argument('--melbo_angle', type=float, default=1.0,
                        help='Which optimization angle to take directions from')
    parser.add_argument('--n_melbo', type=int, default=5,
                        help='Number of top MELBO directions to sweep')
    parser.add_argument('--n_random', type=int, default=20,
                        help='Number of random directions to sweep')
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angle_steps', type=int, default=120)
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    # Load MELBO results to get directions and args
    with open(args.melbo_results, 'rb') as f:
        melbo = pickle.load(f)

    melbo_args = melbo['args']
    model_name = melbo_args['model']
    layer = melbo_args['layer']
    act_idx = melbo_args['act_idx']
    seq_len = melbo_args['seq_len']
    n_act = melbo_args['n_act']
    seed = melbo_args['seed']

    # Get MELBO directions at the requested angle
    melbo_results = melbo['melbo_results']
    if args.melbo_angle not in melbo_results:
        available = list(melbo_results.keys())
        print(f"Angle {args.melbo_angle} not found. Available: {available}")
        sys.exit(1)

    melbo_dirs = melbo_results[args.melbo_angle]['directions']
    melbo_objs = melbo_results[args.melbo_angle]['objs']
    n_melbo = min(args.n_melbo, len(melbo_dirs))
    print(f"Using top-{n_melbo} MELBO directions from {args.melbo_angle}° optimization")
    for i in range(n_melbo):
        print(f"  d_{i+1}: L2 @ {args.melbo_angle}° = {melbo_objs[i]:.2f}")

    if args.results_dir is None:
        base = os.path.dirname(args.melbo_results)
        args.results_dir = base
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer, device = load_model(model_name)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    print(f"Device: {device}, d_model: {d_model}, perturb@L{layer}, measure@L{measure_layer}")

    # Extract activations (same params as MELBO run)
    print(f"\nLoading {n_act} FineWeb sequences (seq_len={seq_len})...")
    token_ids_list = load_fineweb_fixed_length(n_act, tokenizer, seq_len=seq_len, seed=seed)

    all_contexts = []
    all_activations = []
    for i in range(0, len(token_ids_list), 64):
        batch_tokens = torch.stack(token_ids_list[i:i+64]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].cpu())
        all_activations.append(hidden[:, -1, :].cpu())
    contexts_all = torch.cat(all_contexts, 0)
    activations_all = torch.cat(all_activations, 0)

    ctx = contexts_all[act_idx:act_idx+1]
    act = activations_all[act_idx]
    print(f"Using activation {act_idx}, norm: {act.norm():.2f}")

    # Angle grid
    angles_deg = np.linspace(0, args.max_angle, args.n_angle_steps + 1)
    angles_rad = np.deg2rad(angles_deg)

    # Sweep MELBO directions
    print(f"\nSweeping {n_melbo} MELBO directions...")
    melbo_curves = []
    for i in range(n_melbo):
        d = melbo_dirs[i]
        if isinstance(d, np.ndarray):
            d = torch.tensor(d, dtype=torch.float32)
        d = d / d.norm()
        curve = sweep_single_direction(model, ctx, act, d, layer, measure_layer,
                                       device, angles_rad)
        melbo_curves.append(curve)
        print(f"  d_{i+1}: L2 @ 10° = {curve[np.argmin(np.abs(angles_deg - 10))]:.1f}, "
              f"@ 30° = {curve[np.argmin(np.abs(angles_deg - 30))]:.1f}")

    # Sweep random directions
    print(f"\nSweeping {args.n_random} random directions...")
    torch.manual_seed(seed + 1000)
    random_dirs = torch.randn(args.n_random, d_model)
    random_dirs = F.normalize(random_dirs, dim=-1)
    random_curves = []
    for i in range(args.n_random):
        curve = sweep_single_direction(model, ctx, act, random_dirs[i], layer,
                                       measure_layer, device, angles_rad)
        random_curves.append(curve)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_random}")

    melbo_curves = np.array(melbo_curves)
    random_curves = np.array(random_curves)

    # Save
    results = {
        'args': vars(args),
        'melbo_args': melbo_args,
        'angles_deg': angles_deg,
        'melbo_curves': melbo_curves,
        'random_curves': random_curves,
        'melbo_angle': args.melbo_angle,
        'n_melbo': n_melbo,
    }
    pkl_path = os.path.join(args.results_dir, 'sweep_directions_comparison.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # Plot
    plot(results, args.results_dir)


def plot(results, out_dir):
    angles_deg = results['angles_deg']
    melbo_curves = results['melbo_curves']
    random_curves = results['random_curves']
    n_melbo = results['n_melbo']
    melbo_angle = results['melbo_angle']

    blue = '#2B4E72'
    orange = '#E8820C'

    fig, ax = plt.subplots(figsize=(8, 5))

    # Random: median + IQR
    rand_median = np.median(random_curves, axis=0)
    rand_q25 = np.percentile(random_curves, 25, axis=0)
    rand_q75 = np.percentile(random_curves, 75, axis=0)
    ax.plot(angles_deg, rand_median, color=blue, linewidth=2,
            label='Random directions')
    ax.fill_between(angles_deg, rand_q25, rand_q75, color=blue, alpha=0.15)

    # MELBO: individual lines in orange tones
    cmap = plt.cm.Oranges
    for i in range(n_melbo):
        c = cmap(0.4 + 0.5 * i / max(n_melbo - 1, 1))
        ax.plot(angles_deg, melbo_curves[i], color=c, linewidth=1.5,
                label='Optimized directions' if i == 0 else None)

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel('L2 distance', fontsize=12)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, angles_deg[-1])

    plt.tight_layout()
    path = os.path.join(out_dir, 'melbo_vs_random_sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
