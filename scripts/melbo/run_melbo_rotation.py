#!/usr/bin/env python3
"""
Rotation in the {MELBO_1, mean} plane.

Parameterization:
  0°  = -MELBO_1 (≈ mean direction, since cos(mean, melbo1) ≈ -0.90)
  90° = orth to MELBO_1 (the component of mean orthogonal to melbo1)
  180° = MELBO_1

diff-of-means falls at ~26° (near -MELBO_1, as expected).
PC1 and -mean are also projected onto this plane.

Usage:
    python scripts/run_melbo_rotation.py --model qwen3-0.6b --layer 14
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
        args.results_dir = f'results/melbo_rotation/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Load directions =====
    print(f"Loading saved directions from {args.directions_path}")
    with open(args.directions_path, 'rb') as f:
        saved = pickle.load(f)

    mean_dir = saved['mean_dir']
    pc1 = saved['pc1']
    melbo1 = saved['melbo_directions'][0]

    # ===== Define the {melbo1, mean} plane =====
    print("\n" + "="*60)
    print("PLANE DEFINITION: {MELBO_1, mean}")
    print("="*60)

    cos_mean_melbo1 = (mean_dir @ melbo1).item()
    print(f"cos(mean, MELBO_1) = {cos_mean_melbo1:.4f}")

    # Orthonormal basis for the plane:
    # e1 = -melbo1  (so 0° starts near mean)
    # e2 = component of mean orthogonal to melbo1, normalized
    e1 = -melbo1
    mean_orth_to_melbo1 = mean_dir - (mean_dir @ melbo1) * melbo1
    e2 = mean_orth_to_melbo1 / mean_orth_to_melbo1.norm()

    print(f"e1 = -MELBO_1")
    print(f"e2 = normalize(mean - (mean·MELBO_1)·MELBO_1)")
    print(f"  e1·e2 = {(e1 @ e2).item():.6f} (should be ~0)")

    # Rotation: rot(φ) = cos(φ)*e1 + sin(φ)*e2
    # 0° = -MELBO_1, 90° = e2 (orth to melbo1), 180° = MELBO_1

    # Project mean, PC1, -mean onto this plane
    def plane_angle(v):
        c1 = (v @ e1).item()
        c2 = (v @ e2).item()
        in_plane_norm = np.sqrt(c1**2 + c2**2)
        phi = np.degrees(np.arctan2(c2, c1))
        return phi, in_plane_norm

    mean_phi, mean_inplane = plane_angle(mean_dir)
    pc1_phi, pc1_inplane = plane_angle(pc1)
    neg_mean_phi, neg_mean_inplane = plane_angle(-mean_dir)

    print(f"\nProjections onto {{e1, e2}} plane:")
    print(f"  mean:  φ={mean_phi:.1f}°, in-plane={mean_inplane:.3f} ({mean_inplane**2*100:.0f}% variance)")
    print(f"  PC1:   φ={pc1_phi:.1f}°, in-plane={pc1_inplane:.3f} ({pc1_inplane**2*100:.0f}% variance)")
    print(f"  -mean: φ={neg_mean_phi:.1f}°, in-plane={neg_mean_inplane:.3f}")

    # Rotation angles: 0 to 180 in steps, plus exact positions of mean and PC1
    rotation_phis_deg = list(range(0, 181, 15))
    # Add mean and PC1 positions if they're in range
    for special_phi in [mean_phi, pc1_phi]:
        if 0 <= special_phi <= 180:
            rotation_phis_deg.append(round(special_phi, 1))
    rotation_phis_deg = sorted(set([round(p, 1) for p in rotation_phis_deg]))

    rotation_dirs = {}
    for phi_d in rotation_phis_deg:
        phi_r = np.radians(phi_d)
        d = np.cos(phi_r) * e1 + np.sin(phi_r) * e2
        d = d / d.norm()
        rotation_dirs[phi_d] = d
    print(f"\nRotation angles: {rotation_phis_deg}")

    # Verify reconstructions
    if 0 <= mean_phi <= 180:
        mean_rot = rotation_dirs[round(mean_phi, 1)]
        print(f"  cos(rot@mean_angle, mean) = {(mean_rot @ mean_dir).item():.4f}")
    if 0 <= pc1_phi <= 180:
        pc1_rot = rotation_dirs[round(pc1_phi, 1)]
        print(f"  cos(rot@pc1_angle, PC1) = {(pc1_rot @ pc1).item():.4f}")

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
    print(f"Activations: {activations.shape}")

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

    # Named directions for comparison
    named_dirs = {
        'mean': mean_dir,
        'PC1': pc1,
        'MELBO_1': melbo1,
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
        'mean_phi_deg': mean_phi,
        'pc1_phi_deg': pc1_phi,
        'neg_mean_phi_deg': neg_mean_phi,
        'mean_inplane': mean_inplane,
        'pc1_inplane': pc1_inplane,
        'sweep_results': sweep_results,
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved results to {args.results_dir}/results.pkl")

    # ===== Plot =====
    max_angle = 60
    mask = angles_deg <= max_angle
    x = angles_deg[mask]

    def get_stats(name, key):
        data = sweep_results[name][key]
        if data.ndim == 3:
            per_act = np.median(data, axis=2)
        else:
            per_act = data
        return (np.median(per_act, axis=0),
                np.percentile(per_act, 25, axis=0),
                np.percentile(per_act, 75, axis=0))

    # Distinct colors for rotation angles (0° to 180° in 15° steps + specials)
    # Use a perceptually uniform colormap with enough separation
    n_rot = len(rotation_phis_deg)
    cmap = plt.cm.viridis
    rot_colors = {}
    for i, phi_d in enumerate(rotation_phis_deg):
        rot_colors[phi_d] = cmap(i / max(n_rot - 1, 1))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for metric_idx, (key, ylabel) in enumerate([('kl', 'KL divergence'), ('l2', 'L2 distance')]):
        ax = axes[metric_idx]

        # Random baseline
        rm, rq25, rq75 = get_stats('random', key)
        ax.fill_between(x, rq25[mask], rq75[mask], color='gray', alpha=0.15)
        ax.plot(x, rm[mask], color='gray', lw=2, ls='--', label='Random', zorder=1)

        # Rotation directions
        for phi_d in rotation_phis_deg:
            name = f'rot_{phi_d}'
            med, q25, q75 = get_stats(name, key)
            color = rot_colors[phi_d]

            is_mean = abs(phi_d - mean_phi) < 1.0
            is_pc1 = abs(phi_d - pc1_phi) < 1.0

            label = f'{phi_d:.0f}°'
            if phi_d == 0:
                label += ' (−MELBO_1)'
            elif phi_d == 180:
                label += ' (MELBO_1)'
            elif phi_d == 90:
                label += ' (⊥ MELBO_1)'
            elif is_mean:
                label = f'{phi_d:.0f}° (≈mean)'
            elif is_pc1:
                label = f'{phi_d:.0f}° (≈PC1)'

            lw = 2.5 if (phi_d == 180 or is_mean) else 1.5
            zorder = 3 if (phi_d == 180 or is_mean) else 2
            ax.fill_between(x, q25[mask], q75[mask], color=color, alpha=0.08)
            ax.plot(x, med[mask], color=color, lw=lw, label=label, zorder=zorder)

        # Overlay actual mean and MELBO_1 as dashed lines for comparison
        for dname, dcolor, dlabel in [
            ('mean', '#2ca02c', 'mean (actual)'),
            ('MELBO_1', '#d62728', 'MELBO_1 (actual)'),
        ]:
            med, q25, q75 = get_stats(dname, key)
            ax.fill_between(x, q25[mask], q75[mask], color=dcolor, alpha=0.06)
            ax.plot(x, med[mask], color=dcolor, lw=2.5, ls='--', label=dlabel, zorder=5)

        ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_angle)

    fig.suptitle(
        f'Rotation in {{MELBO_1, mean}} plane: −MELBO_1 (0°) → ⊥ (90°) → MELBO_1 (180°)\n'
        f'Qwen3-0.6B, layer 14 | '
        f'mean at {mean_phi:.0f}° ({mean_inplane**2*100:.0f}% in-plane), '
        f'PC1 at {pc1_phi:.0f}° ({pc1_inplane**2*100:.0f}% in-plane)',
        fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'melbo_rotation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: melbo_rotation.png")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Median KL at 30° perturbation")
    print(f"{'='*60}")
    idx_30 = np.argmin(np.abs(angles_deg - 30))
    for name in ['mean', 'PC1', 'MELBO_1']:
        kl = np.median(sweep_results[name]['kl'][:, idx_30])
        print(f"  {name:20s}: {kl:.4f}")
    for phi_d in rotation_phis_deg:
        name = f'rot_{phi_d}'
        kl = np.median(sweep_results[name]['kl'][:, idx_30])
        tag = ''
        if abs(phi_d - mean_phi) < 1.0:
            tag = ' (≈mean)'
        elif abs(phi_d - pc1_phi) < 1.0:
            tag = ' (≈PC1)'
        elif phi_d == 0:
            tag = ' (−MELBO_1)'
        elif phi_d == 90:
            tag = ' (⊥)'
        elif phi_d == 180:
            tag = ' (MELBO_1)'
        print(f"  rot_{phi_d:>5.0f}°{tag:15s}: {kl:.4f}")
    rand_kl = np.median(np.median(sweep_results['random']['kl'][:, idx_30, :], axis=1))
    print(f"  {'random':20s}: {rand_kl:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
