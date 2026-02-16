#!/usr/bin/env python3
"""
Deeper investigation into perturbation direction decompositions.

Key questions:
- Does MELBO_1's component orthogonal to mean break the plateau? (obs 8)
- Which component of MELBO_1/PC1 carries the plateau-breaking?
- Better rotation plot: standard sweep lines from mean→PC1, 0-60 deg

Loads existing sweep results where available, runs new sweeps for new directions.

Usage:
    python scripts/run_direction_decomposition.py --model qwen3-0.6b --layer 14
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


# ── Direction decomposition helpers ──

def project_out(v, u):
    """Return component of v orthogonal to u (both assumed unit norm)."""
    residual = v - (v @ u) * u
    return residual / residual.norm()


def project_onto(v, u):
    """Return component of v along u (both assumed unit norm), re-normalized."""
    proj = (v @ u) * u
    return proj / proj.norm()


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
        args.results_dir = f'results/direction_decomposition/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Load directions =====
    print(f"Loading saved directions from {args.directions_path}")
    with open(args.directions_path, 'rb') as f:
        saved = pickle.load(f)

    mean_dir = saved['mean_dir']
    pc1 = saved['pc1']
    melbo1 = saved['melbo_directions'][0]

    # ===== Decompositions =====
    print("\n" + "="*60)
    print("DECOMPOSITIONS")
    print("="*60)

    # PC1 decomposition (recap)
    alpha_pc1 = (pc1 @ mean_dir).item()
    pc1_orth_to_mean = project_out(pc1, mean_dir)
    print(f"PC1 = {alpha_pc1:.4f}*mean + {(1-alpha_pc1**2)**0.5:.4f}*pc1_orth_to_mean")

    # MELBO_1 decomposition relative to mean (obs 8)
    alpha_m1 = (melbo1 @ mean_dir).item()
    melbo1_orth_to_mean = project_out(melbo1, mean_dir)
    print(f"MELBO_1 = {alpha_m1:.4f}*mean + {(1-alpha_m1**2)**0.5:.4f}*melbo1_orth_to_mean")
    print(f"  cos(melbo1_orth_to_mean, pc1_orth_to_mean) = "
          f"{(melbo1_orth_to_mean @ pc1_orth_to_mean).abs().item():.4f}")

    # MELBO_1 decomposition relative to PC1 (obs 7)
    alpha_m1_pc1 = (melbo1 @ pc1).item()
    melbo1_orth_to_pc1 = project_out(melbo1, pc1)
    melbo1_proj_on_pc1 = project_onto(melbo1, pc1)
    print(f"\nMELBO_1 = {alpha_m1_pc1:.4f}*PC1 + {(1-alpha_m1_pc1**2)**0.5:.4f}*melbo1_orth_to_pc1")

    # PC1 decomposition relative to MELBO_1
    alpha_pc1_m1 = (pc1 @ melbo1).item()
    pc1_orth_to_melbo1 = project_out(pc1, melbo1)
    pc1_proj_on_melbo1 = project_onto(pc1, melbo1)
    print(f"PC1 = {alpha_pc1_m1:.4f}*MELBO_1 + {(1-alpha_pc1_m1**2)**0.5:.4f}*pc1_orth_to_melbo1")

    # Cross-check all the orthogonal components
    print(f"\nOrthogonal component alignments:")
    print(f"  cos(melbo1_orth_to_mean, melbo1_orth_to_pc1) = "
          f"{(melbo1_orth_to_mean @ melbo1_orth_to_pc1).abs().item():.4f}")
    print(f"  cos(pc1_orth_to_mean, pc1_orth_to_melbo1) = "
          f"{(pc1_orth_to_mean @ pc1_orth_to_melbo1).abs().item():.4f}")
    print(f"  cos(melbo1_orth_to_mean, mean_dir) = "
          f"{(melbo1_orth_to_mean @ mean_dir).abs().item():.4f}  (sanity: should be 0)")

    # Rotation directions: mean → PC1 in the {mean, pc1_orth_to_mean} plane
    # Pick 5 evenly spaced including endpoints
    pc1_phi = np.arccos(min(abs(alpha_pc1), 1.0))  # angle of PC1 from mean
    rotation_phis_deg = np.linspace(0, 2 * np.degrees(pc1_phi), 5)  # 0 to ~2x PC1's angle
    rotation_dirs = {}
    for phi_d in rotation_phis_deg:
        phi_r = np.radians(phi_d)
        d = np.cos(phi_r) * mean_dir + np.sin(phi_r) * pc1_orth_to_mean
        d = d / d.norm()
        rotation_dirs[f'rot_{phi_d:.1f}'] = d
    print(f"\nRotation directions: {[f'{p:.1f}' for p in rotation_phis_deg]} deg")
    print(f"  (PC1 is at {np.degrees(pc1_phi):.1f} deg)")

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

    # All directions to sweep
    all_dirs = {
        # Baselines
        'mean': mean_dir,
        'PC1': pc1,
        'MELBO_1': melbo1,
        # Obs 8: MELBO_1's orth-to-mean component
        'melbo1_orth_to_mean': melbo1_orth_to_mean,
        # Obs 5: PC1's orth-to-mean component (recap)
        'pc1_orth_to_mean': pc1_orth_to_mean,
        # Obs 7: cross-decompositions
        'melbo1_orth_to_pc1': melbo1_orth_to_pc1,
        'pc1_orth_to_melbo1': pc1_orth_to_melbo1,
    }
    # Add rotation directions
    all_dirs.update(rotation_dirs)

    for name, direction in all_dirs.items():
        print(f"Sweeping: {name}...", end=' ', flush=True)
        t0 = time.time()
        kl, l2 = sweep_single_direction(
            model, contexts_dev, activations_dev, ref_final, ref_log_probs,
            direction, angles_rad, args.layer, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}
        print(f"{time.time()-t0:.1f}s")

    # Random baseline
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
        'decompositions': {
            'alpha_pc1_mean': alpha_pc1,
            'alpha_melbo1_mean': alpha_m1,
            'alpha_melbo1_pc1': alpha_m1_pc1,
        },
        'sweep_results': sweep_results,
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved results to {args.results_dir}/results.pkl")

    # ===== Plots =====

    # Helper: get random baseline median + IQR
    def random_stats(key):
        rd = sweep_results['random'][key]
        per_act = np.median(rd, axis=2)
        return np.median(per_act, axis=0), np.percentile(per_act, 25, axis=0), np.percentile(per_act, 75, axis=0)

    max_angle = 60  # only plot to 60 deg
    mask = angles_deg <= max_angle
    x = angles_deg[mask]

    # --- Plot 1: Rotation from mean to PC1 (standard sweep, 0-60 deg) ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Random baseline
    rm, rq25, rq75 = random_stats('kl')
    ax.fill_between(x, rq25[mask], rq75[mask], color='gray', alpha=0.2)
    ax.plot(x, rm[mask], color='gray', lw=2, ls='--', label='Random')

    # Rotation directions
    cmap = plt.cm.coolwarm
    for i, phi_d in enumerate(rotation_phis_deg):
        name = f'rot_{phi_d:.1f}'
        data = sweep_results[name]['kl']
        median = np.median(data, axis=0)
        color = cmap(i / (len(rotation_phis_deg) - 1))
        is_pc1 = abs(phi_d - np.degrees(pc1_phi)) < 1.0
        label = f'{phi_d:.1f}° from mean'
        if is_pc1:
            label += ' (≈PC1)'
        ax.plot(x, median[mask], color=color, lw=2.5 if is_pc1 else 2, label=label,
                ls='-' if not is_pc1 else '--')

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('Median KL divergence')
    ax.set_title(f'Rotation from mean toward PC1\n{args.model}, layer {args.layer}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_angle)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'rotation_sweep.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: rotation_sweep.png")

    # --- Plot 2: Decomposition comparison (obs 7, 8) ---
    fig, ax = plt.subplots(figsize=(10, 6))

    rm, rq25, rq75 = random_stats('kl')
    ax.fill_between(x, rq25[mask], rq75[mask], color='gray', alpha=0.2)
    ax.plot(x, rm[mask], color='gray', lw=2, ls='--', label='Random')

    decomp_dirs = {
        'mean': ('green', '-', 'mean (diff-of-means)'),
        'PC1': ('blue', '-', 'PC1'),
        'MELBO_1': ('red', '-', 'MELBO_1'),
        'melbo1_orth_to_mean': ('red', ':', 'MELBO_1 ⊥ mean'),
        'pc1_orth_to_mean': ('blue', ':', 'PC1 ⊥ mean'),
        'melbo1_orth_to_pc1': ('orange', ':', 'MELBO_1 ⊥ PC1'),
        'pc1_orth_to_melbo1': ('purple', ':', 'PC1 ⊥ MELBO_1'),
    }
    for name, (color, ls, label) in decomp_dirs.items():
        data = sweep_results[name]['kl']
        median = np.median(data, axis=0)
        ax.plot(x, median[mask], color=color, ls=ls, lw=2, label=label)

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('Median KL divergence')
    ax.set_title(f'Direction decompositions\n{args.model}, layer {args.layer}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_angle)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'decomposition_sweep.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: decomposition_sweep.png")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Median KL at 45 deg perturbation")
    print(f"{'='*60}")
    idx_45 = np.argmin(np.abs(angles_deg - 45))
    for name in ['mean', 'PC1', 'MELBO_1', 'melbo1_orth_to_mean', 'pc1_orth_to_mean',
                  'melbo1_orth_to_pc1', 'pc1_orth_to_melbo1']:
        kl_45 = np.median(sweep_results[name]['kl'][:, idx_45])
        print(f"  {name:25s}: {kl_45:.4f}")
    rand_45 = np.median(np.median(sweep_results['random']['kl'][:, idx_45, :], axis=1))
    print(f"  {'random':25s}: {rand_45:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
