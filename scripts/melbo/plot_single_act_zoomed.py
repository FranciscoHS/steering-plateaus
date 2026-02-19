#!/usr/bin/env python3
"""
Replot single-activation MELBO sweep zoomed to 0-50 deg, with random baselines.
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
    load_model, forward_from_layer_allpos,
    _get_blocks, _is_rope_model, _get_position_embeddings, _run_block,
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


def forward_to_hidden_and_logits(model, hidden_states, start_layer):
    hidden = hidden_states
    blocks = _get_blocks(model)
    rope = _is_rope_model(model)
    pos_emb = _get_position_embeddings(model, hidden) if rope else None
    for block in blocks[start_layer + 1:]:
        hidden = _run_block(block, hidden, model, pos_emb)
    final_hidden = hidden[:, -1, :]
    normed = _get_final_norm(model)(final_hidden)
    logits = _get_lm_head(model)(normed)
    return final_hidden, logits


def build_perturbed_hidden(contexts, activations, direction, angle_rad):
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms
    proj = (direction.unsqueeze(0) * a_unit).sum(dim=-1, keepdim=True)
    u_perp = direction.unsqueeze(0) - proj * a_unit
    u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)
    a_pert = a_norms * (cos_t * a_unit + sin_t * u_perp)
    return torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir',
                        default='results/melbo_single_act/qwen3-0.6b_target1.0_seed42')
    parser.add_argument('--max_angle', type=float, default=50.0)
    parser.add_argument('--n_random_dirs', type=int, default=50,
                        help='Number of random directions to sweep')
    parser.add_argument('--n_angles', type=int, default=100,
                        help='Number of angles in sweep')
    args = parser.parse_args()

    # Load saved results
    results_path = os.path.join(args.results_dir, 'results.pkl')
    print(f"Loading results from {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    saved_args = results['args']
    directions = results['directions']
    n_good = results['n_good']
    token_ids_list = results['token_ids']

    print(f"Loaded {n_good} good directions")
    print(f"Model: {saved_args['model']}, layer: {saved_args['layer']}")

    # Load model to compute random baselines
    print(f"Loading model: {saved_args['model']}")
    model, tokenizer, device = load_model(saved_args['model'])
    d_model = model._plateau_info['d_model']
    layer = saved_args['layer']

    # Reconstruct activations
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, layer, device)
    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)

    with torch.no_grad():
        ref_hidden = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_hidden_and_logits(model, ref_hidden, layer)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # Fine-grained sweep angles up to max_angle
    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Sweep MELBO directions
    print(f"Sweeping {n_good} MELBO directions (0-{args.max_angle} deg)...")
    melbo_kl = np.zeros((n_good, len(angles_deg)))
    for i in range(n_good):
        d = directions[i].to(device)
        for ang_idx, theta in enumerate(angles_rad):
            with torch.no_grad():
                hidden = build_perturbed_hidden(contexts_dev, activations_dev, d, theta)
                _, pert_logits = forward_to_hidden_and_logits(model, hidden, layer)
                pert_log_probs = F.log_softmax(pert_logits, dim=-1)
                kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
                melbo_kl[i, ang_idx] = kl.item()
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_good} done")

    # Sweep random directions
    print(f"Sweeping {args.n_random_dirs} random directions...")
    a_unit = activations_dev / activations_dev.norm(dim=-1, keepdim=True)
    random_kl = np.zeros((args.n_random_dirs, len(angles_deg)))
    for r in range(args.n_random_dirs):
        # Random direction orthogonal to the activation
        v = torch.randn(d_model, device=device)
        v = v - (v @ a_unit[0]) * a_unit[0]
        v = F.normalize(v, dim=0)
        for ang_idx, theta in enumerate(angles_rad):
            with torch.no_grad():
                hidden = build_perturbed_hidden(contexts_dev, activations_dev, v, theta)
                _, pert_logits = forward_to_hidden_and_logits(model, hidden, layer)
                pert_log_probs = F.log_softmax(pert_logits, dim=-1)
                kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
                random_kl[r, ang_idx] = kl.item()

    # ===== Plot =====
    fig, ax = plt.subplots(figsize=(12, 7))

    # Random: show individual traces + median
    for r in range(args.n_random_dirs):
        ax.plot(angles_deg, random_kl[r], color='gray', alpha=0.15, lw=0.8)
    rand_median = np.median(random_kl, axis=0)
    rand_p25 = np.percentile(random_kl, 25, axis=0)
    rand_p75 = np.percentile(random_kl, 75, axis=0)
    ax.plot(angles_deg, rand_median, color='gray', lw=2.5, ls='--', label='Random (median)')
    ax.fill_between(angles_deg, rand_p25, rand_p75, color='gray', alpha=0.15)

    # MELBO directions
    cmap = plt.cm.viridis
    for i in range(n_good):
        alpha = max(0.15, 1.0 - i * 0.004)
        lw = 1.8 if i < 10 else 0.8
        color = cmap(i / max(n_good, 1))
        label = f'MELBO_{i+1}' if i < 5 else (f'MELBO_{i+1}' if i % 50 == 49 else None)
        ax.plot(angles_deg, melbo_kl[i], color=color, alpha=alpha, lw=lw, label=label)

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel('KL divergence', fontsize=12)
    ax.set_title(f'Single-activation MELBO: {n_good} directions vs random\n'
                 f'{saved_args["model"]}, layer {layer}, '
                 f'activation: "{tokenizer.decode(token_ids_list[0])}"',
                 fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, args.max_angle)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    out_path = os.path.join(args.results_dir, f'sweep_zoomed_{int(args.max_angle)}deg.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # Print some stats
    opt_ang = saved_args['opt_angle']
    opt_idx = np.argmin(np.abs(angles_deg - opt_ang))
    print(f"\nAt {opt_ang} deg:")
    print(f"  Random median KL: {rand_median[opt_idx]:.4f}")
    print(f"  MELBO range: {melbo_kl[:, opt_idx].min():.4f} - {melbo_kl[:, opt_idx].max():.4f}")
    print(f"  MELBO median KL: {np.median(melbo_kl[:, opt_idx]):.4f}")


if __name__ == '__main__':
    main()
