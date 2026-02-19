#!/usr/bin/env python3
"""
Direction decomposition v2: fixed rotation (PC1 is anti-parallel to mean!)

cos(PC1, mean) = -0.979 (not +0.979). PC1 is ~168° from mean, not ~12°.
The rotation in the {mean, orth} plane goes:
  0° = mean, 90° = orth, 168° ≈ PC1, 180° = -mean

Usage:
    python scripts/run_direction_decomposition_v2.py --model qwen3-0.6b --layer 14
"""

import argparse
import os
import pickle
import sys
import time
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


# ── Reused infrastructure ──

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


def sweep_single_direction(model, contexts, activations, ref_final, ref_log_probs,
                           direction, angles_rad, layer, device):
    N = activations.shape[0]
    n_angles = len(angles_rad)
    kl_divs = np.zeros((N, n_angles))
    l2_dists = np.zeros((N, n_angles))
    direction_dev = direction.to(device)
    for ang_idx, theta in enumerate(angles_rad):
        with torch.no_grad():
            hidden = build_perturbed_hidden(contexts, activations, direction_dev, theta)
            pert_final, pert_logits = forward_to_hidden_and_logits(model, hidden, layer)
            pert_log_probs = F.log_softmax(pert_logits, dim=-1)
            l2 = (pert_final - ref_final).norm(dim=-1)
            kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            l2_dists[:, ang_idx] = l2.cpu().numpy()
            kl_divs[:, ang_idx] = kl.cpu().numpy()
    return kl_divs, l2_dists


def sweep_random_directions(model, contexts, activations, ref_final, ref_log_probs,
                            angles_rad, layer, device, n_dirs=10):
    N, d = activations.shape
    n_angles = len(angles_rad)
    kl_divs = np.zeros((N, n_angles, n_dirs))
    l2_dists = np.zeros((N, n_angles, n_dirs))
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms
    v_perps = torch.zeros(N, n_dirs, d, device=device)
    for i in range(N):
        v = torch.randn(n_dirs, d, device=device)
        v = v - (v @ a_unit[i]).unsqueeze(1) * a_unit[i].unsqueeze(0)
        v_perps[i] = F.normalize(v, dim=-1)
    for ang_idx, theta in enumerate(angles_rad):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        for dir_idx in range(n_dirs):
            v = v_perps[:, dir_idx, :]
            a_pert = a_norms * (cos_t * a_unit + sin_t * v)
            with torch.no_grad():
                hidden = torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)
                pert_final, pert_logits = forward_to_hidden_and_logits(model, hidden, layer)
                pert_log_probs = F.log_softmax(pert_logits, dim=-1)
                l2 = (pert_final - ref_final).norm(dim=-1)
                kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
                l2_dists[:, ang_idx, dir_idx] = l2.cpu().numpy()
                kl_divs[:, ang_idx, dir_idx] = kl.cpu().numpy()
    return kl_divs, l2_dists


def project_out(v, u):
    """Component of v orthogonal to u (both unit norm), re-normalized."""
    residual = v - (v @ u) * u
    return residual / residual.norm()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--n_act', type=int, default=50)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--directions_path', default=None)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.directions_path is None:
        args.directions_path = f'results/melbo_constrained/{args.model}/optimization_results.pkl'
    if args.results_dir is None:
        args.results_dir = f'results/direction_decomposition_v2/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Load directions =====
    print(f"Loading saved directions from {args.directions_path}")
    with open(args.directions_path, 'rb') as f:
        saved = pickle.load(f)

    mean_dir = saved['mean_dir']
    pc1 = saved['pc1']
    melbo1 = saved['melbo_directions'][0]

    # ===== Decompositions (with correct signs!) =====
    print("\n" + "="*60)
    print("DECOMPOSITIONS (with signs)")
    print("="*60)

    cos_pc1_mean = (pc1 @ mean_dir).item()
    cos_melbo1_mean = (melbo1 @ mean_dir).item()
    cos_melbo1_pc1 = (melbo1 @ pc1).item()

    print(f"cos(PC1, mean) = {cos_pc1_mean:.4f}  (NEGATIVE! Anti-parallel!)")
    print(f"cos(MELBO_1, mean) = {cos_melbo1_mean:.4f}")
    print(f"cos(MELBO_1, PC1) = {cos_melbo1_pc1:.4f}")

    pc1_angle = np.degrees(np.arccos(np.clip(cos_pc1_mean, -1, 1)))
    melbo1_angle_from_mean = np.degrees(np.arccos(np.clip(cos_melbo1_mean, -1, 1)))
    print(f"\nAngle(PC1, mean) = {pc1_angle:.1f} deg")
    print(f"Angle(MELBO_1, mean) = {melbo1_angle_from_mean:.1f} deg")

    # Orthogonal component of PC1 relative to mean
    pc1_orth = project_out(pc1, mean_dir)
    melbo1_orth = project_out(melbo1, mean_dir)

    print(f"\nIn the {{mean, pc1_orth}} plane:")
    print(f"  PC1 = {cos_pc1_mean:.4f}*mean + {(pc1 - cos_pc1_mean*mean_dir).norm().item():.4f}*pc1_orth")
    print(f"  MELBO_1 along mean: {cos_melbo1_mean:.4f}")
    print(f"  MELBO_1 along pc1_orth: {(melbo1 @ pc1_orth).item():.4f}")
    print(f"  cos(melbo1_orth, pc1_orth) = {(melbo1_orth @ pc1_orth).item():.4f}")

    # ===== Rotation directions in {mean, pc1_orth} plane =====
    # rot(phi) = cos(phi)*mean + sin(phi)*pc1_orth
    # PC1 is at phi = arctan2(sin_component, cos_component) where:
    #   cos_component = cos(PC1, mean) = -0.979
    #   sin_component = (PC1 @ pc1_orth) = +0.203
    pc1_phi = np.arctan2((pc1 @ pc1_orth).item(), cos_pc1_mean)
    melbo1_phi = np.arctan2((melbo1 @ pc1_orth).item(), cos_melbo1_mean)

    print(f"\nRotation angle in {{mean, pc1_orth}} plane:")
    print(f"  PC1 at phi = {np.degrees(pc1_phi):.1f} deg")
    print(f"  MELBO_1 at phi = {np.degrees(melbo1_phi):.1f} deg")

    # Pick rotation angles: 0 (mean), some intermediates, PC1, past PC1
    rotation_phis_deg = [0, 30, 60, 90, 120, 150, np.degrees(pc1_phi), 180]
    rotation_phis_deg = sorted(set([round(p, 1) for p in rotation_phis_deg]))

    rotation_dirs = {}
    for phi_d in rotation_phis_deg:
        phi_r = np.radians(phi_d)
        d = np.cos(phi_r) * mean_dir + np.sin(phi_r) * pc1_orth
        d = d / d.norm()
        rotation_dirs[phi_d] = d
    print(f"\nRotation angles: {rotation_phis_deg}")

    # Verify PC1 reconstruction
    pc1_rot = rotation_dirs[round(np.degrees(pc1_phi), 1)]
    print(f"  cos(rot@PC1_angle, PC1) = {(pc1_rot @ pc1).item():.6f}  (should be ~1.0)")

    # ===== Load model + activations =====
    print(f"\n{'='*60}")
    print(f"LOADING MODEL AND ACTIVATIONS (seed={args.seed})")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Activations: {activations.shape}, mean norm: {activations.norm(dim=-1).mean():.2f}")

    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)

    with torch.no_grad():
        ref_hidden = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_hidden_and_logits(model, ref_hidden, args.layer)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # ===== Sweeps =====
    angles_deg = np.linspace(0, 180, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    print(f"\n{'='*60}")
    print("PERTURBATION SWEEPS")
    print(f"{'='*60}")

    sweep_results = {}

    # Named directions
    named_dirs = {
        'mean': mean_dir,
        'PC1': pc1,
        'MELBO_1': melbo1,
        'melbo1_orth_to_mean': melbo1_orth,
        'pc1_orth_to_mean': pc1_orth,
    }
    for name, direction in named_dirs.items():
        print(f"Sweeping: {name}...", end=' ', flush=True)
        t0 = time.time()
        kl, l2 = sweep_single_direction(
            model, contexts_dev, activations_dev, ref_final, ref_log_probs,
            direction, angles_rad, args.layer, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}
        print(f"{time.time()-t0:.1f}s")

    # Rotation directions
    for phi_d, direction in rotation_dirs.items():
        name = f'rot_{phi_d}'
        print(f"Sweeping: {name}...", end=' ', flush=True)
        t0 = time.time()
        kl, l2 = sweep_single_direction(
            model, contexts_dev, activations_dev, ref_final, ref_log_probs,
            direction, angles_rad, args.layer, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}
        print(f"{time.time()-t0:.1f}s")

    # Random
    print(f"Sweeping: random ({args.n_random_dirs} dirs/act)...", end=' ', flush=True)
    t0 = time.time()
    rand_kl, rand_l2 = sweep_random_directions(
        model, contexts_dev, activations_dev, ref_final, ref_log_probs,
        angles_rad, args.layer, device, n_dirs=args.n_random_dirs)
    sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}
    print(f"{time.time()-t0:.1f}s")

    # ===== Save =====
    all_results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'rotation_phis_deg': rotation_phis_deg,
        'pc1_phi_deg': np.degrees(pc1_phi),
        'melbo1_phi_deg': np.degrees(melbo1_phi),
        'signed_cosines': {
            'pc1_mean': cos_pc1_mean,
            'melbo1_mean': cos_melbo1_mean,
            'melbo1_pc1': cos_melbo1_pc1,
        },
        'sweep_results': sweep_results,
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved results to {args.results_dir}/results.pkl")

    # ===== Plots =====
    max_angle = 60
    mask = angles_deg <= max_angle
    x = angles_deg[mask]

    def get_stats(name, key):
        """Return median, q25, q75 for a direction."""
        data = sweep_results[name][key]
        if data.ndim == 3:  # random: [N, angles, dirs]
            per_act = np.median(data, axis=2)
        else:
            per_act = data
        return (np.median(per_act, axis=0),
                np.percentile(per_act, 25, axis=0),
                np.percentile(per_act, 75, axis=0))

    # --- Plot 1: Rotation from mean to PC1 (corrected, both panels) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = plt.cm.coolwarm

    for metric_idx, (key, ylabel) in enumerate([('kl', 'KL divergence'), ('l2', 'L2 distance')]):
        ax = axes[metric_idx]

        # Random baseline
        rm, rq25, rq75 = get_stats('random', key)
        ax.fill_between(x, rq25[mask], rq75[mask], color='gray', alpha=0.2)
        ax.plot(x, rm[mask], color='gray', lw=2, ls='--', label='Random')

        # Rotation directions
        for i, phi_d in enumerate(rotation_phis_deg):
            name = f'rot_{phi_d}'
            med, q25, q75 = get_stats(name, key)
            t = phi_d / 180.0
            color = cmap(t)
            is_pc1 = abs(phi_d - np.degrees(pc1_phi)) < 1.0
            label = f'{phi_d:.0f}°'
            if phi_d == 0:
                label += ' (=mean)'
            elif is_pc1:
                label += ' (=PC1)'
            elif phi_d == 90:
                label += ' (=orth)'
            elif phi_d == 180:
                label += ' (=-mean)'
            lw = 2.5 if is_pc1 else 1.8
            ax.fill_between(x, q25[mask], q75[mask], color=color, alpha=0.1)
            ax.plot(x, med[mask], color=color, lw=lw, label=label)

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_angle)

    fig.suptitle(f'Rotation from mean to -mean in {{mean, orth}} plane\n'
                 f'{args.model}, layer {args.layer} | '
                 f'PC1 at {np.degrees(pc1_phi):.0f}°, nearly anti-parallel to mean')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'rotation_sweep.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: rotation_sweep.png")

    # --- Plot 2: Decomposition comparison (both panels) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    decomp_styles = {
        'mean':               ('green',  '-',  'mean (diff-of-means)'),
        'PC1':                ('blue',   '-',  'PC1'),
        'MELBO_1':            ('red',    '-',  'MELBO_1'),
        'melbo1_orth_to_mean':('red',    ':',  'MELBO_1 orth to mean'),
        'pc1_orth_to_mean':   ('blue',   ':',  'PC1 orth to mean'),
    }

    for metric_idx, (key, ylabel) in enumerate([('kl', 'KL divergence'), ('l2', 'L2 distance')]):
        ax = axes[metric_idx]

        # Random baseline
        rm, rq25, rq75 = get_stats('random', key)
        ax.fill_between(x, rq25[mask], rq75[mask], color='gray', alpha=0.2)
        ax.plot(x, rm[mask], color='gray', lw=2, ls='--', label='Random')

        for name, (color, ls, label) in decomp_styles.items():
            med, q25, q75 = get_stats(name, key)
            ax.fill_between(x, q25[mask], q75[mask], color=color, alpha=0.08)
            ax.plot(x, med[mask], color=color, ls=ls, lw=2, label=label)

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_angle)

    fig.suptitle(f'Direction decompositions | {args.model}, layer {args.layer}\n'
                 f'cos(PC1,mean)={cos_pc1_mean:.3f}, cos(MELBO1,mean)={cos_melbo1_mean:.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'decomposition_sweep.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: decomposition_sweep.png")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Median KL at selected perturbation angles")
    print(f"{'='*60}")
    for target in [30, 45, 60]:
        idx = np.argmin(np.abs(angles_deg - target))
        print(f"\n  At {angles_deg[idx]:.0f} deg perturbation:")
        for name in ['mean', 'PC1', 'MELBO_1', 'melbo1_orth_to_mean', 'pc1_orth_to_mean']:
            kl = np.median(sweep_results[name]['kl'][:, idx])
            print(f"    {name:25s}: {kl:.4f}")
        rand_kl = np.median(np.median(sweep_results['random']['kl'][:, idx, :], axis=1))
        print(f"    {'random':25s}: {rand_kl:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
