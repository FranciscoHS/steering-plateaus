#!/usr/bin/env python3
"""
Perturbation sweep of the 5 MELBO directions from manifold v2,
each on its own activation (the one it was optimized at),
plus 10 random directions as baseline.

Usage:
    python scripts/melbo/sweep_manifold_v2_directions.py \
        --melbo_results results/melbo_manifold_v2/qwen3-0.6b/results.pkl
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_to_final,
    _get_final_norm, _get_lm_head,
)
from src.data import load_fineweb_fixed_length


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device,
                                   batch_size=64):
    all_contexts = []
    all_activations = []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].cpu())
        all_activations.append(hidden[:, -1, :].cpu())
    return torch.cat(all_contexts, 0), torch.cat(all_activations, 0)


def sweep_one(model, context, activation, directions, angles_rad, layer, device):
    """Sweep for one activation along multiple directions.

    Args:
        context: [1, seq_len-1, d]
        activation: [d]
        directions: [n_dirs, d]

    Returns:
        kl_divs, cos_dists: [n_angles, n_dirs]
    """
    n_angles = len(angles_rad)
    n_dirs = directions.shape[0]
    kl_divs = np.zeros((n_angles, n_dirs))
    cos_dists = np.zeros((n_angles, n_dirs))

    model_dtype = next(model.parameters()).dtype
    ctx = context.to(device=device, dtype=model_dtype)
    a = activation.to(device=device, dtype=model_dtype)
    R = a.norm()
    a_unit = a / R
    dirs = directions.to(device=device, dtype=model_dtype)

    # Orthogonalize directions to activation
    proj = (dirs @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
    v_perps = F.normalize(dirs - proj, dim=-1)

    with torch.no_grad():
        ref_point = a.unsqueeze(0).unsqueeze(0)
        ref_final = forward_from_layer_to_final(model, ctx, ref_point, layer)
        ref_ln = _get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1)
        ref_logits = _get_lm_head(model)(ref_ln)
        ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)

    for ang_idx, theta in enumerate(angles_rad):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        perturbed = R * (cos_t * a_unit.unsqueeze(0) + sin_t * v_perps)
        perturbed = perturbed.unsqueeze(1)

        with torch.no_grad():
            pert_final = forward_from_layer_to_final(model, ctx, perturbed, layer)

            cos_sim = F.cosine_similarity(pert_final.float(), ref_final.float(), dim=-1)
            cos_dists[ang_idx] = (1 - cos_sim).cpu().numpy()

            pert_ln = _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)
            pert_logits = _get_lm_head(model)(pert_ln)
            pert_log_probs = F.log_softmax(pert_logits.float(), dim=-1)
            kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            kl_divs[ang_idx] = kl.cpu().numpy()

    return kl_divs, cos_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--melbo_results', required=True)
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--max_angle', type=float, default=90)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act_pool', type=int, default=50)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load MELBO results
    print(f"Loading MELBO results from {args.melbo_results}")
    with open(args.melbo_results, 'rb') as f:
        melbo_data = pickle.load(f)
    act_results = melbo_data['act_results']
    test_indices = melbo_data['test_indices']
    n_acts = len(act_results)
    print(f"Loaded {n_acts} MELBO directions at indices {list(test_indices)}")

    if args.results_dir is None:
        args.results_dir = os.path.join(
            os.path.dirname(args.melbo_results), 'sweep')
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']

    # Load FineWeb activations (same seed as v2 experiment)
    print(f"Loading {args.n_act_pool} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act_pool, tokenizer, seq_len=args.seq_len, seed=args.seed)
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    colors = ['red', 'orange', 'purple', 'green', 'blue']

    # One subplot per activation
    fig, axes = plt.subplots(2, n_acts, figsize=(5 * n_acts, 8), squeeze=False)

    all_sweep_results = []

    for i, ar in enumerate(act_results):
        act_idx = ar['act_idx']
        melbo_dir = ar['melbo_direction']
        ctx_i = contexts[act_idx:act_idx+1]
        act_i = activations[act_idx]

        print(f"\nActivation {i+1}/{n_acts} (idx {act_idx})...")

        # Random directions
        rand_dirs = torch.randn(args.n_random_dirs, d_model)
        rand_dirs = F.normalize(rand_dirs, dim=-1)
        rand_kl, rand_cos = sweep_one(
            model, ctx_i, act_i, rand_dirs, angles_rad, args.layer, device)

        # MELBO direction
        melbo_kl, melbo_cos = sweep_one(
            model, ctx_i, act_i, melbo_dir.unsqueeze(0), angles_rad, args.layer, device)

        all_sweep_results.append({
            'act_idx': act_idx,
            'rand_kl': rand_kl, 'rand_cos': rand_cos,
            'melbo_kl': melbo_kl, 'melbo_cos': melbo_cos,
        })

        for row, (metric, rand_data, melbo_data, ylabel) in enumerate([
            ('KL', rand_kl, melbo_kl, 'KL divergence'),
            ('cos', rand_cos, melbo_cos, 'Cosine distance'),
        ]):
            ax = axes[row, i]
            # Random: median + IQR across dirs
            rand_median = np.median(rand_data, axis=1)
            rand_q25 = np.percentile(rand_data, 25, axis=1)
            rand_q75 = np.percentile(rand_data, 75, axis=1)
            ax.fill_between(angles_deg, rand_q25, rand_q75, color='gray', alpha=0.2)
            ax.plot(angles_deg, rand_median, color='gray', lw=2,
                    label='Random', linestyle='--')

            # MELBO
            ax.plot(angles_deg, melbo_data[:, 0], color=colors[i], lw=2,
                    label=f'MELBO (cos_d={ar["melbo_cos_dist"]:.4f})')

            ax.set_xlabel('Angle (deg)')
            if i == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(f'Act {i+1} (idx {act_idx})')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f'Manifold v2 sweep: each MELBO dir on its own activation\n'
                 f'{args.model}, layer {args.layer}', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'sweep_per_activation.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {plot_path}")

    # Save
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'sweep_results': all_sweep_results,
                     'angles_deg': angles_deg}, f)

    print("Done!")


if __name__ == '__main__':
    main()
