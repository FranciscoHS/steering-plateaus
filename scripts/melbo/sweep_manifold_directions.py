#!/usr/bin/env python3
"""
Perturbation sweep of Mandarin activations along:
  - 10 random directions (median shown)
  - Each of the 5 MELBO directions from the manifold test

Uses the same spherical perturbation as run_perturbation_sweep_spherical.py.

Usage:
    python scripts/melbo/sweep_manifold_directions.py \
        --melbo_results results/melbo_manifold_test_cosdist/qwen3-0.6b/results.pkl
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
    _get_final_norm, _get_lm_head, _get_blocks,
)
from data.bilingual_prompts import CHINESE_PROMPTS


def extract_mandarin_activations(model, tokenizer, device, layer, n_prompts=10):
    """Extract activations for Mandarin prompts at given layer.

    Returns:
        contexts: [n, seq_len-1, d_model]
        activations: [n, d_model]
    """
    contexts = []
    activations = []
    for p in CHINESE_PROMPTS[:n_prompts]:
        tokens = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        contexts.append(hidden[:, :-1, :].cpu())
        activations.append(hidden[:, -1, :].cpu())
    return contexts, torch.stack([a.squeeze(0) for a in activations])


def sweep_one_activation(model, context, activation, directions, angles_rad,
                         layer, device):
    """Sweep perturbation for one activation along multiple fixed directions.

    Args:
        context: [1, seq_len-1, d] for this activation
        activation: [d]
        directions: [n_dirs, d] unit vectors
        angles_rad: [n_angles]

    Returns:
        kl_divs: [n_angles, n_dirs]
        cos_dists: [n_angles, n_dirs]
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

    # Make directions orthogonal to activation
    proj = (dirs @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
    v_perps = F.normalize(dirs - proj, dim=-1)

    # Reference
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

            # Cosine distance at final layer
            cos_sim = F.cosine_similarity(pert_final.float(), ref_final.float(), dim=-1)
            cos_dists[ang_idx] = (1 - cos_sim).cpu().numpy()

            # KL divergence
            pert_ln = _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)
            pert_logits = _get_lm_head(model)(pert_ln)
            pert_log_probs = F.log_softmax(pert_logits.float(), dim=-1)
            kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            kl_divs[ang_idx] = kl.cpu().numpy()

    return kl_divs, cos_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--melbo_results', required=True,
                        help='Path to manifold test results.pkl')
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--n_mandarin', type=int, default=10,
                        help='Number of Mandarin prompts to use')
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--max_angle', type=float, default=90)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load MELBO results
    print(f"Loading MELBO results from {args.melbo_results}")
    with open(args.melbo_results, 'rb') as f:
        melbo_data = pickle.load(f)
    group_results = melbo_data['group_results']
    n_groups = len(group_results)
    melbo_dirs = torch.stack([r['melbo_direction'] for r in group_results])
    print(f"Loaded {n_groups} MELBO directions")

    if args.results_dir is None:
        args.results_dir = os.path.join(
            os.path.dirname(args.melbo_results), 'sweep')
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    # Extract Mandarin activations
    print(f"Extracting activations for {args.n_mandarin} Mandarin prompts...")
    contexts, activations = extract_mandarin_activations(
        model, tokenizer, device, args.layer, n_prompts=args.n_mandarin)
    print(f"Activation shape: {activations.shape}, "
          f"mean norm: {activations.norm(dim=-1).mean():.2f}")

    # Angles
    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # === Sweep random directions ===
    print(f"\nSweeping {args.n_random_dirs} random directions per activation...")
    random_kl_all = []
    random_cos_all = []
    for i in range(args.n_mandarin):
        rand_dirs = torch.randn(args.n_random_dirs, d_model)
        rand_dirs = F.normalize(rand_dirs, dim=-1)
        kl, cos = sweep_one_activation(
            model, contexts[i], activations[i], rand_dirs,
            angles_rad, args.layer, device)
        random_kl_all.append(kl)
        random_cos_all.append(cos)
        print(f"  Activation {i+1}/{args.n_mandarin} done")
    # Stack: [n_mandarin, n_angles, n_random_dirs]
    random_kl_all = np.stack(random_kl_all)
    random_cos_all = np.stack(random_cos_all)

    # === Sweep MELBO directions ===
    print(f"\nSweeping {n_groups} MELBO directions...")
    melbo_kl_all = {}
    melbo_cos_all = {}
    for g in range(n_groups):
        d = melbo_dirs[g].unsqueeze(0)  # [1, d_model]
        kl_list = []
        cos_list = []
        for i in range(args.n_mandarin):
            kl, cos = sweep_one_activation(
                model, contexts[i], activations[i], d,
                angles_rad, args.layer, device)
            kl_list.append(kl[:, 0])   # [n_angles]
            cos_list.append(cos[:, 0])
        melbo_kl_all[g] = np.stack(kl_list)    # [n_mandarin, n_angles]
        melbo_cos_all[g] = np.stack(cos_list)
        print(f"  Group {g+1}/{n_groups} done")

    # === Save ===
    results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'random_kl': random_kl_all,
        'random_cos': random_cos_all,
        'melbo_kl': melbo_kl_all,
        'melbo_cos': melbo_cos_all,
        'melbo_dirs': melbo_dirs,
    }
    pkl_path = os.path.join(args.results_dir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # === Plot ===
    colors = ['red', 'orange', 'purple', 'green', 'blue']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (metric_name, rand_data, melbo_data_dict) in zip(axes, [
        ('KL divergence', random_kl_all, melbo_kl_all),
        ('Cosine distance', random_cos_all, melbo_cos_all),
    ]):
        # Random: median across dirs, then median across activations
        # rand_data: [n_mandarin, n_angles, n_random_dirs]
        rand_per_act = np.median(rand_data, axis=2)  # [n_mandarin, n_angles]
        rand_median = np.median(rand_per_act, axis=0)
        rand_q25 = np.percentile(rand_per_act, 25, axis=0)
        rand_q75 = np.percentile(rand_per_act, 75, axis=0)
        ax.fill_between(angles_deg, rand_q25, rand_q75, color='gray', alpha=0.2)
        ax.plot(angles_deg, rand_median, color='gray', lw=2,
                label='Random (10 dirs)', linestyle='--')

        # MELBO directions
        for g in range(n_groups):
            data = melbo_data_dict[g]  # [n_mandarin, n_angles]
            median = np.median(data, axis=0)
            q25 = np.percentile(data, 25, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            cos_dist_val = group_results[g].get('melbo_cos_dist',
                                                 group_results[g].get('melbo_kl', '?'))
            label = f'Group {g+1} (obj={cos_dist_val:.4f})'
            ax.plot(angles_deg, median, color=colors[g], lw=2, label=label)
            ax.fill_between(angles_deg, q25, q75, color=colors[g], alpha=0.1)

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, args.max_angle)

    fig.suptitle(f'Manifold MELBO sweep: {args.model}, layer {args.layer}\n'
                 f'{args.n_mandarin} Mandarin activations', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'sweep_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
