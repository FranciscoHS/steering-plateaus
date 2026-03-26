#!/usr/bin/env python3
"""
Spherical perturbation sweep: random directions vs feature directions.

For each real activation, perturbs along:
  (i)   random orthogonal directions (baseline)
  (ii)  named feature directions (loaded from contrastive results or --direction_pkl)

Measures KL divergence at each angle. Plots all conditions on one figure.

Usage:
    python scripts/perturbation/run_feature_vs_random_sweep.py
    python scripts/perturbation/run_feature_vs_random_sweep.py \
        --direction_pkl results/iterative_melbo_hf/.../results.pkl \
        --model qwen3-1.7b --layer 7
"""

import sys
import os
import pickle
import time
import argparse
import numpy as np
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_to_final,
    _get_final_norm, _get_lm_head, _get_blocks,
)
from src.data import load_fineweb_fixed_length


def project_orthogonal(direction, activation):
    """Project direction onto subspace orthogonal to activation, then normalize.

    Args:
        direction: [d_model] unit vector (the feature direction)
        activation: [d_model] the activation vector

    Returns:
        v_perp: [d_model] unit vector in direction's component orthogonal to activation
    """
    a_unit = activation / activation.norm()
    proj = (direction @ a_unit) * a_unit
    v_perp = direction - proj
    v_perp = v_perp / (v_perp.norm() + 1e-10)
    return v_perp


def sweep_single_direction(model, context, activation, v_perp, angles_rad,
                           layer, device):
    """Sweep a single perturbation direction across angles.

    Returns:
        kl_divs: [n_angles] array of KL divergences
        l2_dists: [n_angles] array of L2 distances in final-layer activation space
    """
    n_angles = len(angles_rad)
    kl_divs = np.zeros(n_angles)
    l2_dists = np.zeros(n_angles)

    model_dtype = next(model.parameters()).dtype
    ctx = context.unsqueeze(0).to(device=device, dtype=model_dtype)
    a = activation.to(device=device, dtype=model_dtype)
    R = a.norm()
    a_unit = a / R
    vp = v_perp.to(device=device, dtype=model_dtype)

    # Reference
    with torch.no_grad():
        ref_point = a.unsqueeze(0).unsqueeze(0)
        ref_final = forward_from_layer_to_final(model, ctx, ref_point, layer)
        ref_ln = _get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1)
        ref_logits = _get_lm_head(model)(ref_ln)
        ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
        ref_probs = ref_log_probs.exp()

    for ang_idx, theta in enumerate(angles_rad):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        perturbed = R * (cos_t * a_unit + sin_t * vp)
        perturbed = perturbed.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pert_final = forward_from_layer_to_final(model, ctx, perturbed, layer)
            # L2 distance in final-layer activation space
            l2_dists[ang_idx] = (pert_final.float() - ref_final.float()).norm(dim=-1).item()
            # KL divergence
            pert_ln = _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)
            pert_logits = _get_lm_head(model)(pert_ln)
            pert_log_probs = F.log_softmax(pert_logits.float(), dim=-1)
            kl = (ref_probs * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            kl_divs[ang_idx] = kl.item()

    return kl_divs, l2_dists


def sweep_random_directions(model, context, activation, n_dirs, angles_rad,
                            layer, device):
    """Sweep multiple random orthogonal directions. Returns KL and L2 arrays."""
    d_model = activation.shape[0]
    a_unit = activation / activation.norm()

    # Sample random orthogonal directions
    v = torch.randn(n_dirs, d_model)
    v = v - (v @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
    v = F.normalize(v, dim=-1)

    all_kl = np.zeros((n_dirs, len(angles_rad)))
    all_l2 = np.zeros((n_dirs, len(angles_rad)))
    for d_idx in range(n_dirs):
        all_kl[d_idx], all_l2[d_idx] = sweep_single_direction(
            model, context, activation, v[d_idx], angles_rad, layer, device
        )

    return all_kl, all_l2  # each [n_dirs, n_angles]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-4b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--n_act', type=int, default=50,
                        help='Number of real activations to sweep')
    parser.add_argument('--n_random_dirs', type=int, default=10,
                        help='Random directions per activation')
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=40)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--direction_pkl', type=str, default=None,
                        help='Path to results.pkl from iterative_melbo_hf (overrides contrastive dirs)')
    parser.add_argument('--dir_idx', type=int, default=0,
                        help='Which direction to use from direction_pkl (default: 0)')
    parser.add_argument('--custom_prompts', type=str, default=None,
                        help='Pipe-separated custom prompts to use instead of FineWeb (e.g. "prompt1|prompt2")')
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/feature_vs_random_sweep/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    print(f"Config: model={args.model}, layer={args.layer}, n_act={args.n_act}, "
          f"n_random_dirs={args.n_random_dirs}, angles=[0, {args.max_angle}]° ({args.n_angles} points)")

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    print(f"Device: {device}")

    # Load feature directions
    if args.direction_pkl:
        print(f"Loading direction from {args.direction_pkl}")
        with open(args.direction_pkl, 'rb') as f:
            dir_data = pickle.load(f)
        direction = dir_data['directions'][args.dir_idx].float()
        direction = direction / direction.norm()
        opt_angle = dir_data['args'].get('opt_angle', '?')
        dir_label = f'MELBO (opt@{opt_angle}°)'
        named_directions = {dir_label: direction}
        print(f"  Direction {args.dir_idx}: opt_angle={opt_angle}°")
    else:
        lang_path = f'results/allpairs_contrastive/{args.model}/directions.pkl'
        prog_path = f'results/programming_contrastive/{args.model}/directions_6lang_with_python.pkl'
        print(f"Loading features: {lang_path}, {prog_path}")

        with open(lang_path, 'rb') as f:
            lang_data = pickle.load(f)
        with open(prog_path, 'rb') as f:
            prog_data = pickle.load(f)

        ru_vec = lang_data['features']['ru'].float()
        c_vec = prog_data['features']['c'].float()

        combined = ru_vec + c_vec
        combined = combined / combined.norm()

        cos_rc = (ru_vec @ c_vec).item()
        print(f"cos(RU, C) = {cos_rc:.4f}")

        named_directions = {
            'C': c_vec,
            'RU': ru_vec,
            'RU+C': combined,
        }

    # Extract activations
    if args.custom_prompts is not None:
        prompts = [p.strip() for p in args.custom_prompts.split('|')]
        args.n_act = len(prompts)
        print(f"\nUsing {len(prompts)} custom prompts:")
        for p in prompts:
            print(f"  - {p!r}")
        print(f"Extracting layer-{args.layer} activations...")
        all_contexts = []
        all_activations = []
        for prompt in prompts:
            tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[args.layer + 1]
            all_contexts.append(hidden[0, :-1, :].float().cpu())
            all_activations.append(hidden[0, -1, :].float().cpu())
        # Contexts have variable length, store as list
        contexts = all_contexts  # list of [seq-1, d]
        activations = torch.stack(all_activations, 0)
    else:
        print(f"\nLoading {args.n_act} FineWeb sequences (seq_len={args.seq_len})...")
        token_ids_list = load_fineweb_fixed_length(
            args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed
        )

        print(f"Extracting layer-{args.layer} activations...")
        all_contexts = []
        all_activations = []
        batch_size = 4
        for i in range(0, len(token_ids_list), batch_size):
            batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
            with torch.no_grad():
                outputs = model(batch_tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[args.layer + 1]
            all_contexts.append(hidden[:, :-1, :].float().cpu())
            all_activations.append(hidden[:, -1, :].float().cpu())
        contexts = torch.cat(all_contexts, 0)
        activations = torch.cat(all_activations, 0)
    print(f"Activations shape: {activations.shape}, mean norm: {activations.norm(dim=-1).mean():.2f}")

    # --- Run sweeps ---
    n_angles = len(angles_deg)

    # Storage: random gets [n_act, n_random_dirs, n_angles], named get [n_act, n_angles]
    results_kl = {'random': np.zeros((args.n_act, args.n_random_dirs, n_angles))}
    results_l2 = {'random': np.zeros((args.n_act, args.n_random_dirs, n_angles))}
    for name in named_directions:
        results_kl[name] = np.zeros((args.n_act, n_angles))
        results_l2[name] = np.zeros((args.n_act, n_angles))

    t0 = time.time()
    for i in range(args.n_act):
        # Random directions
        rand_kl, rand_l2 = sweep_random_directions(
            model, contexts[i], activations[i], args.n_random_dirs,
            angles_rad, args.layer, device
        )
        results_kl['random'][i] = rand_kl
        results_l2['random'][i] = rand_l2

        # Named directions
        for name, direction in named_directions.items():
            v_perp = project_orthogonal(direction, activations[i])
            kl, l2 = sweep_single_direction(
                model, contexts[i], activations[i], v_perp,
                angles_rad, args.layer, device
            )
            results_kl[name][i] = kl
            results_l2[name][i] = l2

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (args.n_act - i - 1)
            print(f"  {i+1}/{args.n_act}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.0f}s")

    # Save raw results
    save_data = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'kl': results_kl,
        'l2': results_l2,
    }
    pkl_path = os.path.join(args.results_dir, 'feature_vs_random_sweep.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved: {pkl_path}")

    # --- Plot ---
    dir_names = list(named_directions.keys())
    palette = ['#4a86c8', '#c84a5a', '#7b4ac8', '#4ac87b', '#c8a84a']
    colors = {name: palette[i % len(palette)] for i, name in enumerate(dir_names)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (metric_data, metric_name) in zip(axes, [
        (results_kl, 'KL divergence (nats)'),
        (results_l2, 'L2 distance (final layer)'),
    ]):
        # Random: median over directions per activation, then across activations
        rand = metric_data['random']  # [n_act, n_dirs, n_angles]
        rand_per_act = np.median(rand, axis=1)  # [n_act, n_angles]
        rand_median = np.median(rand_per_act, axis=0)
        rand_q25 = np.percentile(rand_per_act, 25, axis=0)
        rand_q75 = np.percentile(rand_per_act, 75, axis=0)
        ax.plot(angles_deg, rand_median, color='gray', linewidth=2.5, label='Random')
        ax.fill_between(angles_deg, rand_q25, rand_q75, color='gray', alpha=0.15)

        # Named directions
        for name in dir_names:
            d = metric_data[name]  # [n_act, n_angles]
            median = np.median(d, axis=0)
            q25 = np.percentile(d, 25, axis=0)
            q75 = np.percentile(d, 75, axis=0)
            ax.plot(angles_deg, median, color=colors[name], linewidth=2.5, label=name)
            ax.fill_between(angles_deg, q25, q75, color=colors[name], alpha=0.15)

        ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, args.max_angle)

    fig.suptitle(f'{args.model}, layer {args.layer}: Feature vs Random Perturbation\n'
                 f'n={args.n_act} activations',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(args.results_dir, 'feature_vs_random_sweep.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == '__main__':
    main()
