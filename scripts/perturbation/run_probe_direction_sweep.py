#!/usr/bin/env python3
"""
Perturbation sweep along known probe directions vs random directions.

Loads existing sweep results, then runs additional sweeps perturbing along
probe-derived directions (orthogonalized to stay on the sphere).
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
from src.data import generate_chess_games


def run_sweep_single_direction(model, context, activation, v_perp, angles_rad, layer, device):
    """Sweep perturbation along a single direction for one activation."""
    n_angles = len(angles_rad)
    l2_dists = np.zeros(n_angles)
    kl_divs = np.zeros(n_angles)

    model_dtype = next(model.parameters()).dtype
    ctx = context.unsqueeze(0).to(device=device, dtype=model_dtype)
    a = activation.to(device=device, dtype=model_dtype)
    R = a.norm()
    a_unit = a / R
    vp = v_perp.to(device=device, dtype=model_dtype)

    with torch.no_grad():
        ref_point = a.unsqueeze(0).unsqueeze(0)
        ref_final = forward_from_layer_to_final(model, ctx, ref_point, layer)
        ref_ln = _get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1)
        ref_logits = _get_lm_head(model)(ref_ln)
        ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)

    for ang_idx, theta in enumerate(angles_rad):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        perturbed = R * (cos_t * a_unit + sin_t * vp)
        perturbed = perturbed.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pert_final = forward_from_layer_to_final(model, ctx, perturbed, layer)
            l2_dists[ang_idx] = (pert_final - ref_final).float().norm().item()

            pert_ln = _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)
            pert_logits = _get_lm_head(model)(pert_ln)
            pert_log_probs = F.log_softmax(pert_logits.float(), dim=-1)
            ref_probs = ref_log_probs.exp()
            kl = (ref_probs * (ref_log_probs - pert_log_probs)).sum().item()
            kl_divs[ang_idx] = kl

    return l2_dists, kl_divs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing_results', type=str, required=True,
                        help='Path to existing results.pkl from sweep')
    parser.add_argument('--probe_path', type=str, required=True,
                        help='Path to saved probe .pth file')
    parser.add_argument('--n_probe_dirs', type=int, default=10,
                        help='Number of probe directions to sample per activation')
    parser.add_argument('--results_dir', type=str, default=None)
    args = parser.parse_args()

    # Load existing results
    with open(args.existing_results, 'rb') as f:
        results = pickle.load(f)
    old_args = results['args']
    layer = old_args['layer']
    n_act = old_args['n_act']
    seq_len = old_args['seq_len']
    model_name = old_args['model']
    angles_deg = results['angles_deg']
    angles_rad = results['angles_rad']
    act_norm_mean = results['act_norm_mean']

    if args.results_dir is None:
        args.results_dir = os.path.dirname(args.existing_results)

    # Load probe
    probe_ckpt = torch.load(args.probe_path, map_location='cpu', weights_only=False)
    probe = probe_ckpt['linear_probe']  # [1, 512, 8, 8, 13]
    print(f"Probe shape: {probe.shape}, accuracy: {probe_ckpt.get('acc', 'N/A'):.4f}")

    # Flatten probe directions: [832, 512]
    # probe[0, :, r, c, k] is a 512-dim direction
    probe_dirs = probe[0].permute(1, 2, 3, 0).reshape(-1, probe.shape[1])  # [8*8*13, 512]
    probe_norms = probe_dirs.norm(dim=-1)
    print(f"Probe directions: {probe_dirs.shape[0]}, norm range: [{probe_norms.min():.3f}, {probe_norms.max():.3f}]")

    # Load model and generate data (same seed as original)
    seed = old_args.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Loading model: {model_name}")
    model, tokenizer, device = load_model(model_name)

    n_fit = old_args.get('n_fit', n_act)
    n_load = max(n_act, n_fit)
    print(f"Generating {n_load} chess game sequences...")
    token_ids_list = generate_chess_games(n_load, seq_len, seed=seed)

    # Extract activations (same as original sweep)
    print(f"Extracting layer-{layer} activations...")
    all_contexts = []
    all_activations = []
    for i in range(0, len(token_ids_list), 64):
        batch_tokens = torch.stack(token_ids_list[i:i+64]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].float().cpu())
        all_activations.append(hidden[:, -1, :].float().cpu())
    contexts = torch.cat(all_contexts, 0)[:n_act]
    activations = torch.cat(all_activations, 0)[:n_act]
    print(f"Mean activation norm: {activations.norm(dim=-1).mean():.2f}")

    # For each activation, pick n_probe_dirs probe directions, orthogonalize
    # Pick the highest-norm (strongest) probe directions
    top_indices = probe_norms.argsort(descending=True)[:args.n_probe_dirs]
    selected_dirs = probe_dirs[top_indices]  # [n_probe_dirs, 512]
    selected_dirs = F.normalize(selected_dirs, dim=-1)

    # Decode which features these correspond to
    piece_names = ['blank', 'K', 'Q', 'R', 'B', 'N', 'P', 'k', 'q', 'r', 'b', 'n', 'p']
    files = 'abcdefgh'
    ranks = '12345678'
    print(f"\nTop {args.n_probe_dirs} probe directions by norm:")
    for idx in top_indices:
        r = idx // (8 * 13)
        c = (idx % (8 * 13)) // 13
        k = idx % 13
        square = f"{files[c]}{ranks[r]}"
        print(f"  {piece_names[k]} on {square} (norm={probe_norms[idx]:.3f})")

    # Sweep along probe directions
    print(f"\n=== Probe direction sweeps ===")
    probe_l2 = np.zeros((n_act, len(angles_deg), args.n_probe_dirs))
    probe_kl = np.zeros((n_act, len(angles_deg), args.n_probe_dirs))

    import time
    t0 = time.time()
    for i in range(n_act):
        a = activations[i]
        a_unit = a / a.norm()
        for j in range(args.n_probe_dirs):
            # Orthogonalize probe direction w.r.t. activation
            d = selected_dirs[j].clone()
            d = d - (d @ a_unit) * a_unit
            d = F.normalize(d, dim=-1)
            probe_l2[i, :, j], probe_kl[i, :, j] = run_sweep_single_direction(
                model, contexts[i], activations[i], d, angles_rad, layer, device
            )
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_act}  ({(i+1)/elapsed:.1f} act/s)")
    print(f"Probe sweeps done in {time.time()-t0:.1f}s")

    # Save combined results
    results['probe'] = {'l2': probe_l2, 'kl': probe_kl}
    results['probe_info'] = {
        'n_probe_dirs': args.n_probe_dirs,
        'top_indices': top_indices.numpy(),
    }
    save_path = os.path.join(args.results_dir, 'results_with_probe.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    # Plot: 3 conditions
    comparison_label = results.get('comparison_label', 'Random')
    conditions = [
        ('real', 'Real', 'blue'),
        ('random', comparison_label, 'red'),
        ('probe', 'Probe direction', 'green'),
    ]

    for plot_type in ['median', 'individual']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for metric_idx, (key, name) in enumerate([('l2', 'L2 distance'), ('kl', 'KL divergence')]):
            ax = axes[metric_idx]
            for cond_key, label, color in conditions:
                d = results[cond_key][key]
                per_act = np.median(d, axis=2)

                if plot_type == 'median':
                    median = np.median(per_act, axis=0)
                    q25 = np.percentile(per_act, 25, axis=0)
                    q75 = np.percentile(per_act, 75, axis=0)
                    ax.plot(angles_deg, median, color=color, label=label, linewidth=2)
                    ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.15)
                else:
                    n_individual = min(10, per_act.shape[0])
                    np.random.seed(42)
                    chosen = np.random.choice(per_act.shape[0], n_individual, replace=False)
                    for jj, idx in enumerate(chosen):
                        ax.plot(angles_deg, per_act[idx], color=color, alpha=0.4, linewidth=1.5,
                                label=label if jj == 0 else None)

            ax.set_xlabel('Perturbation angle (degrees)')
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 180)

            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            tick_angles = [0, 30, 60, 90, 120, 150, 180]
            tick_chords = [2 * act_norm_mean * np.sin(np.deg2rad(a) / 2) for a in tick_angles]
            ax2.set_xticks(tick_angles)
            ax2.set_xticklabels([f'{c:.0f}' for c in tick_chords])
            ax2.set_xlabel('Chord distance')

        fig.suptitle(f'{model_name}, layer {layer} (spherical, with probe directions, {plot_type})', fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(args.results_dir, f'sweep_with_probe_{plot_type}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
