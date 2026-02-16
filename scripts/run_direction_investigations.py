#!/usr/bin/env python3
"""
Direction investigation experiments:
  Q1: Does the PC1/MELBO/mean differential behavior generalize to new activations?
  Q2: Decompose PC1 into mean + orthogonal. What does orthogonal alone do?
      Rotate from mean toward PC1 — when does plateau-breaking emerge?
  Q3: How aligned are MELBO and PC1?

Loads saved directions from constrained MELBO results, runs perturbation sweeps
on fresh FineWeb activations (different seed from original experiment).

Usage:
    python scripts/run_direction_investigations.py --model qwen3-0.6b --layer 14
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


# ── Reused from run_melbo_constrained.py ──

def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device,
                                   batch_size=64):
    """Extract layer-L context and final-token activation for fixed-length sequences."""
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
    """Forward pass returning both final-block hidden state and logits (last token)."""
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
    """Construct hidden states after spherical perturbation along direction."""
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
    """Perturbation sweep for one fixed direction across all activations."""
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
    """Perturbation sweep with random directions (different per activation)."""
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
                pert_final, pert_logits = forward_to_hidden_and_logits(
                    model, hidden, layer)
                pert_log_probs = F.log_softmax(pert_logits, dim=-1)
                l2 = (pert_final - ref_final).norm(dim=-1)
                kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
                l2_dists[:, ang_idx, dir_idx] = l2.cpu().numpy()
                kl_divs[:, ang_idx, dir_idx] = kl.cpu().numpy()

    return kl_divs, l2_dists


# ── New analysis code ──

def decompose_pc1(pc1, mean_dir):
    """Decompose PC1 = alpha * mean_dir + beta * orth_dir.

    Returns alpha, beta, orth_dir (all unit-norm where applicable).
    """
    alpha = (pc1 @ mean_dir).item()
    orth_raw = pc1 - alpha * mean_dir
    beta = orth_raw.norm().item()
    orth_dir = orth_raw / orth_raw.norm()
    return alpha, beta, orth_dir


def make_rotation_directions(mean_dir, orth_dir, n_steps=20, max_phi_deg=45.0):
    """Create directions rotating from mean_dir toward orth_dir in the {mean, orth} plane.

    dir(phi) = cos(phi) * mean_dir + sin(phi) * orth_dir  (then normalize, though already unit norm)

    Returns: phi_degrees [n_steps], directions [n_steps, d_model]
    """
    phi_deg = np.linspace(0, max_phi_deg, n_steps + 1)[1:]  # skip 0 (= mean_dir itself)
    phi_rad = np.deg2rad(phi_deg)
    directions = []
    for phi in phi_rad:
        d = np.cos(phi) * mean_dir + np.sin(phi) * orth_dir
        d = d / d.norm()
        directions.append(d)
    return phi_deg, torch.stack(directions)


def print_cosine_matrix(names, directions):
    """Print pairwise cosine similarity matrix."""
    n = len(names)
    print(f"\n{'':>12s}", end='')
    for name in names:
        print(f"{name:>10s}", end='')
    print()
    for i in range(n):
        print(f"{names[i]:>12s}", end='')
        for j in range(n):
            cos = (directions[i] @ directions[j]).item()
            print(f"{cos:>10.4f}", end='')
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--n_act', type=int, default=50)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--n_rotation_steps', type=int, default=20,
                        help='Number of rotation steps from mean toward PC1')
    parser.add_argument('--max_rotation_deg', type=float, default=45.0,
                        help='Maximum rotation angle in degrees')
    parser.add_argument('--directions_path', default=None,
                        help='Path to saved optimization_results.pkl')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=123,
                        help='Seed for FineWeb loading (different from original 42)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.directions_path is None:
        args.directions_path = f'results/melbo_constrained/{args.model}/optimization_results.pkl'
    if args.results_dir is None:
        args.results_dir = f'results/direction_investigations/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Phase A: Load directions and compute similarities (Q3) =====
    print(f"Loading saved directions from {args.directions_path}")
    with open(args.directions_path, 'rb') as f:
        saved = pickle.load(f)

    mean_dir = saved['mean_dir']
    pc1 = saved['pc1']
    melbo_dirs = saved['melbo_directions']

    names = ['mean', 'PC1'] + [f'MELBO_{i+1}' for i in range(len(melbo_dirs))]
    all_dirs = [mean_dir, pc1] + melbo_dirs

    print("\n" + "="*60)
    print("Q3: PAIRWISE COSINE SIMILARITIES")
    print("="*60)
    print_cosine_matrix(names, all_dirs)

    # ===== Phase B: PC1 decomposition (Q2) =====
    print("\n" + "="*60)
    print("Q2: PC1 DECOMPOSITION")
    print("="*60)

    alpha, beta, orth_dir = decompose_pc1(pc1, mean_dir)
    print(f"PC1 = {alpha:.4f} * mean + {beta:.4f} * orth")
    print(f"Angle between PC1 and mean: {np.degrees(np.arccos(abs(alpha))):.1f} deg")

    # Alignment of orth with other directions
    print(f"\nOrthogonal component alignment:")
    for name, d in zip(names, all_dirs):
        print(f"  cos(orth, {name}) = {(orth_dir @ d).abs().item():.4f}")

    # MELBO_1 decomposition in {mean, orth} plane
    m1 = melbo_dirs[0]
    m1_mean = (m1 @ mean_dir).item()
    m1_orth = (m1 @ orth_dir).item()
    m1_plane_frac = (m1_mean**2 + m1_orth**2)**0.5
    print(f"\nMELBO_1 in {{mean, orth}} plane:")
    print(f"  along mean: {m1_mean:.4f}")
    print(f"  along orth: {m1_orth:.4f}")
    print(f"  plane fraction: {m1_plane_frac:.4f}")
    print(f"  out-of-plane: {(1 - m1_plane_frac**2)**0.5:.4f}")

    # Rotation directions
    phi_deg, rotation_dirs = make_rotation_directions(
        mean_dir, orth_dir, n_steps=args.n_rotation_steps,
        max_phi_deg=args.max_rotation_deg)

    # Mark where PC1 sits in the rotation
    pc1_phi = np.degrees(np.arcsin(beta))  # sin(phi) = beta for PC1
    print(f"\nRotation from mean to orth:")
    print(f"  {args.n_rotation_steps} steps from 0 to {args.max_rotation_deg} deg")
    print(f"  PC1 is at phi = {pc1_phi:.1f} deg")

    # ===== Phase C: Load model + new activations =====
    print(f"\n{'='*60}")
    print(f"LOADING MODEL AND ACTIVATIONS (seed={args.seed})")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    print(f"Loading {args.n_act} FineWeb sequences (seed={args.seed})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Activation shape: {activations.shape}, "
          f"mean norm: {activations.norm(dim=-1).mean():.2f}")

    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)

    # Reference (unperturbed)
    with torch.no_grad():
        ref_hidden = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_hidden_and_logits(model, ref_hidden, args.layer)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # ===== Phase D: Perturbation sweeps =====
    angles_deg = np.linspace(0, 180, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    print(f"\n{'='*60}")
    print(f"PERTURBATION SWEEPS ({len(angles_deg)} angles, 0-180 deg)")
    print(f"{'='*60}")

    sweep_results = {}

    # Named directions: mean, PC1, MELBO_1, orth
    named_dirs = [
        ('mean', mean_dir),
        ('PC1', pc1),
        ('MELBO_1', melbo_dirs[0]),
        ('orth', orth_dir),
    ]
    for name, direction in named_dirs:
        print(f"Sweeping: {name}...", end=' ', flush=True)
        t0 = time.time()
        kl, l2 = sweep_single_direction(
            model, contexts_dev, activations_dev, ref_final, ref_log_probs,
            direction, angles_rad, args.layer, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}
        print(f"{time.time()-t0:.1f}s")

    # Rotation directions
    for i, (phi, direction) in enumerate(zip(phi_deg, rotation_dirs)):
        name = f'rot_{phi:.1f}deg'
        print(f"Sweeping: {name}...", end=' ', flush=True)
        t0 = time.time()
        kl, l2 = sweep_single_direction(
            model, contexts_dev, activations_dev, ref_final, ref_log_probs,
            direction, angles_rad, args.layer, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}
        print(f"{time.time()-t0:.1f}s")

    # Random directions
    print(f"Sweeping: random ({args.n_random_dirs} dirs/act)...", end=' ', flush=True)
    t0 = time.time()
    rand_kl, rand_l2 = sweep_random_directions(
        model, contexts_dev, activations_dev, ref_final, ref_log_probs,
        angles_rad, args.layer, device, n_dirs=args.n_random_dirs)
    sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}
    print(f"{time.time()-t0:.1f}s")

    # ===== Phase E: Save and plot =====
    all_results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'pc1_decomposition': {
            'alpha': alpha, 'beta': beta, 'orth_dir': orth_dir,
            'pc1_phi_deg': pc1_phi,
        },
        'rotation_phi_deg': phi_deg,
        'rotation_dirs': rotation_dirs,
        'sweep_results': sweep_results,
        'cosine_matrix': {
            'names': names,
            'matrix': torch.stack(all_dirs) @ torch.stack(all_dirs).T,
        },
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved results to {args.results_dir}/results.pkl")

    # --- Plot 1: Generalization (Q1) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {'random': 'gray', 'mean': 'green', 'PC1': 'blue', 'MELBO_1': 'red'}

    for metric_idx, (key, ylabel) in enumerate([('kl', 'KL divergence'), ('l2', 'L2 distance')]):
        ax = axes[metric_idx]
        # Random baseline
        rand_data = sweep_results['random'][key]
        rand_per_act = np.median(rand_data, axis=2)
        ax.fill_between(angles_deg,
                        np.percentile(rand_per_act, 25, axis=0),
                        np.percentile(rand_per_act, 75, axis=0),
                        color='gray', alpha=0.2)
        ax.plot(angles_deg, np.median(rand_per_act, axis=0),
                color='gray', lw=2, label='Random', ls='--')

        for name in ['mean', 'PC1', 'MELBO_1']:
            data = sweep_results[name][key]
            median = np.median(data, axis=0)
            ax.plot(angles_deg, median, color=colors[name], lw=2, label=name)
            ax.fill_between(angles_deg,
                            np.percentile(data, 25, axis=0),
                            np.percentile(data, 75, axis=0),
                            color=colors[name], alpha=0.1)

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)

    fig.suptitle(f'Q1: Generalization to new activations (seed={args.seed})\n'
                 f'{args.model}, layer {args.layer}')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'q1_generalization.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: q1_generalization.png")

    # --- Plot 2: Orthogonal component (Q2a) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors2 = {'random': 'gray', 'mean': 'green', 'PC1': 'blue',
                'MELBO_1': 'red', 'orth': 'orange'}

    for metric_idx, (key, ylabel) in enumerate([('kl', 'KL divergence'), ('l2', 'L2 distance')]):
        ax = axes[metric_idx]
        rand_data = sweep_results['random'][key]
        rand_per_act = np.median(rand_data, axis=2)
        ax.fill_between(angles_deg,
                        np.percentile(rand_per_act, 25, axis=0),
                        np.percentile(rand_per_act, 75, axis=0),
                        color='gray', alpha=0.2)
        ax.plot(angles_deg, np.median(rand_per_act, axis=0),
                color='gray', lw=2, label='Random', ls='--')

        for name in ['mean', 'PC1', 'MELBO_1', 'orth']:
            data = sweep_results[name][key]
            median = np.median(data, axis=0)
            ax.plot(angles_deg, median, color=colors2[name], lw=2, label=name)
            ax.fill_between(angles_deg,
                            np.percentile(data, 25, axis=0),
                            np.percentile(data, 75, axis=0),
                            color=colors2[name], alpha=0.1)

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)

    fig.suptitle(f'Q2a: Orthogonal component of PC1\n'
                 f'PC1 = {alpha:.3f}*mean + {beta:.3f}*orth')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'q2a_orthogonal.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: q2a_orthogonal.png")

    # --- Plot 3: Rotation heatmap (Q2b) ---
    # Build KL matrix: rows = rotation angle, cols = perturbation angle
    rot_names = [k for k in sweep_results if k.startswith('rot_')]
    rot_names.sort(key=lambda x: float(x.split('_')[1].replace('deg', '')))
    rot_phis = [float(k.split('_')[1].replace('deg', '')) for k in rot_names]

    kl_matrix = np.zeros((len(rot_names), len(angles_deg)))
    for i, name in enumerate(rot_names):
        kl_matrix[i] = np.median(sweep_results[name]['kl'], axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(kl_matrix, aspect='auto', origin='lower',
                   extent=[angles_deg[0], angles_deg[-1], rot_phis[0], rot_phis[-1]],
                   cmap='hot_r')
    ax.axhline(pc1_phi, color='cyan', ls='--', lw=1.5, label=f'PC1 ({pc1_phi:.1f} deg)')
    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('Rotation from mean toward orth (degrees)')
    ax.legend(loc='upper left')
    plt.colorbar(im, ax=ax, label='Median KL divergence')
    ax.set_title(f'Q2b: Rotation from mean to PC1 direction\n'
                 f'{args.model}, layer {args.layer}')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'q2b_rotation_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: q2b_rotation_heatmap.png")

    # --- Plot 4: Rotation line plot (selected angles) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    # Pick a few perturbation angles to show KL vs rotation
    target_pert_angles = [15, 30, 45, 60, 75, 90]
    cmap = plt.cm.viridis
    for j, target in enumerate(target_pert_angles):
        idx = np.argmin(np.abs(angles_deg - target))
        kl_vs_rot = kl_matrix[:, idx]
        color = cmap(j / (len(target_pert_angles) - 1))
        ax.plot(rot_phis, kl_vs_rot, color=color, lw=2,
                label=f'pert={angles_deg[idx]:.0f} deg')
    ax.axvline(pc1_phi, color='red', ls='--', lw=1.5, alpha=0.7, label=f'PC1 ({pc1_phi:.1f} deg)')
    ax.set_xlabel('Rotation from mean toward orth (degrees)')
    ax.set_ylabel('Median KL divergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Q2b: KL vs rotation angle at different perturbation angles')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'q2b_rotation_lines.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: q2b_rotation_lines.png")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Activations: {args.n_act} FineWeb sequences, seed={args.seed}")
    print(f"PC1 decomposition: {alpha:.4f}*mean + {beta:.4f}*orth (angle={np.degrees(np.arccos(abs(alpha))):.1f} deg)")
    print(f"\nMedian KL at 60 deg perturbation:")
    idx_60 = np.argmin(np.abs(angles_deg - 60))
    for name in ['mean', 'PC1', 'MELBO_1', 'orth']:
        kl_60 = np.median(sweep_results[name]['kl'][:, idx_60])
        print(f"  {name}: {kl_60:.4f}")
    rand_60 = np.median(np.median(sweep_results['random']['kl'][:, idx_60, :], axis=1))
    print(f"  random: {rand_60:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
