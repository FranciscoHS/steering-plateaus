#!/usr/bin/env python3
"""
Test whether the stable-region structure holds off-manifold.

From the existing MELBO perturbation sweep, identify the boundary magnitude (alpha*)
as the angle of maximum KL derivative. Then from the boundary point, do perturbation
sweeps in random directions orthogonal to m_hat and compare with sweeps from the
original activation.

Key prediction of boundary hypothesis:
  Random orthogonal directions remain inert from a_boundary
  (similar to how they're inert from a_original).

Alternative (structure breaks down off-manifold):
  Random orthogonal directions become active from a_boundary
  (geometric structure is a local property of the data manifold).

Usage:
    python scripts/run_boundary_orthogonal.py --model qwen3-0.6b --layer 14
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
    """Perturbation sweep for one fixed direction. Returns kl [N, n_angles], l2 [N, n_angles]."""
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


def project_out(v, u):
    """Component of v orthogonal to u (both unit norm), re-normalized."""
    residual = v - (v @ u) * u
    return residual / residual.norm()


# ── New functions ──

def compute_boundary_activations(activations, direction, angle_rad):
    """Compute boundary activations via spherical perturbation along direction.

    Returns [N, d] boundary activation vectors (same norm as originals).
    """
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms
    d = direction.unsqueeze(0)  # [1, d_model]
    proj = (d * a_unit).sum(dim=-1, keepdim=True)
    u_perp = d - proj * a_unit
    u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)
    return a_norms * (cos_t * a_unit + sin_t * u_perp)


def sample_directions_orthogonal_to(direction, n_dirs, d_model, device):
    """Sample n_dirs random unit directions orthogonal to direction."""
    v = torch.randn(n_dirs, d_model, device=device)
    d = direction.to(device)
    v = v - (v @ d).unsqueeze(1) * d.unsqueeze(0)
    return F.normalize(v, dim=-1)


def find_boundary_angle(melbo_results_path):
    """Find boundary angle (max KL derivative) from saved MELBO results.

    Returns angle in degrees at the midpoint of the steepest segment.
    """
    with open(melbo_results_path, 'rb') as f:
        results = pickle.load(f)

    angles_deg = results['angles_deg']
    kl_data = results['sweep_results']['MELBO_1']['kl']  # [N, n_angles]
    median_kl = np.median(kl_data, axis=0)

    # Finite difference derivative
    dkl = np.diff(median_kl) / np.diff(angles_deg)
    max_idx = np.argmax(dkl)
    boundary_angle = (angles_deg[max_idx] + angles_deg[max_idx + 1]) / 2

    print(f"  Max KL derivative: {dkl[max_idx]:.4f} at {boundary_angle:.1f} deg")
    print(f"  KL at {angles_deg[max_idx]:.0f} deg: {median_kl[max_idx]:.4f}")
    print(f"  KL at {angles_deg[max_idx+1]:.0f} deg: {median_kl[max_idx+1]:.4f}")

    return boundary_angle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--n_act', type=int, default=30)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--n_orth_dirs', type=int, default=10,
                        help='Number of random directions orthogonal to m_hat')
    parser.add_argument('--boundary_angle', type=float, default=None,
                        help='Boundary angle in degrees (default: auto from max KL derivative)')
    parser.add_argument('--directions_path', default=None)
    parser.add_argument('--melbo_results_path', default=None)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.directions_path is None:
        args.directions_path = f'results/melbo_constrained/{args.model}/optimization_results.pkl'
    if args.melbo_results_path is None:
        args.melbo_results_path = f'results/melbo_constrained/{args.model}/results.pkl'
    if args.results_dir is None:
        args.results_dir = f'results/boundary_orthogonal/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Phase 1: Load directions and find boundary =====
    print("="*60)
    print("PHASE 1: Load directions and compute boundary angle")
    print("="*60)

    with open(args.directions_path, 'rb') as f:
        saved = pickle.load(f)

    mean_dir = saved['mean_dir']  # diff-of-means (d_hat)
    m_hat = saved['melbo_directions'][0]  # MELBO_1 direction

    # m_perp: MELBO component orthogonal to diff-of-means
    m_perp = project_out(m_hat, mean_dir)

    print(f"cos(m_hat, mean_dir) = {(m_hat @ mean_dir).item():.4f}")
    print(f"cos(m_hat, m_perp) = {(m_hat @ m_perp).item():.6f}")  # should be ~0
    print(f"cos(m_perp, mean_dir) = {(m_perp @ mean_dir).item():.6f}")  # should be ~0

    # Find boundary angle
    if args.boundary_angle is not None:
        alpha_star = args.boundary_angle
        print(f"\nUsing user-specified boundary angle: {alpha_star:.1f} deg")
    else:
        print(f"\nFinding boundary angle from {args.melbo_results_path}...")
        alpha_star = find_boundary_angle(args.melbo_results_path)
        print(f"Auto-detected boundary angle: {alpha_star:.1f} deg")

    alpha_star_rad = np.deg2rad(alpha_star)

    # ===== Phase 2: Sample random orthogonal directions =====
    print(f"\n{'='*60}")
    print(f"PHASE 2: Sample {args.n_orth_dirs} random directions orthogonal to m_hat")
    print("="*60)

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_model = m_hat.shape[0]
    orth_dirs = sample_directions_orthogonal_to(m_hat, args.n_orth_dirs, d_model,
                                                 torch.device(device_str))

    # Verify orthogonality
    dots_with_mhat = (orth_dirs @ m_hat.to(device_str)).abs()
    print(f"Max |cos(orth_dir, m_hat)|: {dots_with_mhat.max().item():.2e}")

    # Also check pairwise angles between orthogonal directions
    pairwise = (orth_dirs @ orth_dirs.T)
    off_diag = pairwise[~torch.eye(args.n_orth_dirs, dtype=bool, device=device_str)]
    print(f"Pairwise |cos| of orth dirs: mean={off_diag.abs().mean():.4f}, "
          f"max={off_diag.abs().max():.4f}")

    # ===== Phase 3: Load model and extract activations =====
    print(f"\n{'='*60}")
    print(f"PHASE 3: Load model and extract activations")
    print("="*60)

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

    # Move directions to device
    m_hat_dev = m_hat.to(device)
    m_perp_dev = m_perp.to(device)
    orth_dirs = orth_dirs.to(device)

    # ===== Phase 4: Compute boundary activations =====
    print(f"\n{'='*60}")
    print(f"PHASE 4: Compute boundary activations (alpha* = {alpha_star:.1f} deg)")
    print("="*60)

    boundary_acts = compute_boundary_activations(activations_dev, m_hat_dev, alpha_star_rad)
    print(f"Boundary activations shape: {boundary_acts.shape}")
    print(f"  Norm preservation: orig={activations_dev.norm(dim=-1).mean():.2f}, "
          f"boundary={boundary_acts.norm(dim=-1).mean():.2f}")

    # Cosine distance between original and boundary
    cos_sim = F.cosine_similarity(activations_dev, boundary_acts, dim=-1)
    print(f"  Cos(original, boundary): mean={cos_sim.mean():.4f} "
          f"(expected ~cos({alpha_star:.0f} deg)={np.cos(alpha_star_rad):.4f})")

    # ===== Phase 5: Compute references =====
    print(f"\n{'='*60}")
    print("PHASE 5: Compute reference outputs")
    print("="*60)

    # Reference at original
    with torch.no_grad():
        ref_hidden_orig = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_final_orig, ref_logits_orig = forward_to_hidden_and_logits(
            model, ref_hidden_orig, args.layer)
        ref_lp_orig = F.log_softmax(ref_logits_orig, dim=-1)
    print("  Computed reference at a_original")

    # Reference at boundary
    with torch.no_grad():
        ref_hidden_bnd = torch.cat([contexts_dev, boundary_acts.unsqueeze(1)], dim=1)
        ref_final_bnd, ref_logits_bnd = forward_to_hidden_and_logits(
            model, ref_hidden_bnd, args.layer)
        ref_lp_bnd = F.log_softmax(ref_logits_bnd, dim=-1)
    print("  Computed reference at a_boundary")

    # KL between original and boundary (sanity check)
    kl_orig_to_bnd = (ref_lp_orig.exp() * (ref_lp_orig - ref_lp_bnd)).sum(dim=-1)
    print(f"  KL(original || boundary): mean={kl_orig_to_bnd.mean():.4f}, "
          f"median={kl_orig_to_bnd.median():.4f}")

    # ===== Phase 6: Perturbation sweeps =====
    angles_deg = np.linspace(0, 180, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    print(f"\n{'='*60}")
    print(f"PHASE 6: Perturbation sweeps ({len(angles_deg)} angles, 0-180 deg)")
    print("="*60)

    sweep_results = {}

    # --- Sweeps from boundary ---
    print("\n--- From a_boundary ---")

    print(f"  m_hat...", end=' ', flush=True)
    t0 = time.time()
    kl, l2 = sweep_single_direction(
        model, contexts_dev, boundary_acts, ref_final_bnd, ref_lp_bnd,
        m_hat_dev, angles_rad, args.layer, device)
    sweep_results['boundary_mhat'] = {'kl': kl, 'l2': l2}
    print(f"{time.time()-t0:.1f}s")

    print(f"  m_perp...", end=' ', flush=True)
    t0 = time.time()
    kl, l2 = sweep_single_direction(
        model, contexts_dev, boundary_acts, ref_final_bnd, ref_lp_bnd,
        m_perp_dev, angles_rad, args.layer, device)
    sweep_results['boundary_mperp'] = {'kl': kl, 'l2': l2}
    print(f"{time.time()-t0:.1f}s")

    print(f"  {args.n_orth_dirs} random dirs ⊥ m_hat...", end=' ', flush=True)
    t0 = time.time()
    kl_all = np.zeros((args.n_act, len(angles_deg), args.n_orth_dirs))
    l2_all = np.zeros((args.n_act, len(angles_deg), args.n_orth_dirs))
    for d_idx in range(args.n_orth_dirs):
        kl, l2 = sweep_single_direction(
            model, contexts_dev, boundary_acts, ref_final_bnd, ref_lp_bnd,
            orth_dirs[d_idx], angles_rad, args.layer, device)
        kl_all[:, :, d_idx] = kl
        l2_all[:, :, d_idx] = l2
    sweep_results['boundary_orth'] = {'kl': kl_all, 'l2': l2_all}
    print(f"{time.time()-t0:.1f}s")

    # --- Sweeps from original ---
    print("\n--- From a_original ---")

    print(f"  m_hat...", end=' ', flush=True)
    t0 = time.time()
    kl, l2 = sweep_single_direction(
        model, contexts_dev, activations_dev, ref_final_orig, ref_lp_orig,
        m_hat_dev, angles_rad, args.layer, device)
    sweep_results['original_mhat'] = {'kl': kl, 'l2': l2}
    print(f"{time.time()-t0:.1f}s")

    print(f"  m_perp...", end=' ', flush=True)
    t0 = time.time()
    kl, l2 = sweep_single_direction(
        model, contexts_dev, activations_dev, ref_final_orig, ref_lp_orig,
        m_perp_dev, angles_rad, args.layer, device)
    sweep_results['original_mperp'] = {'kl': kl, 'l2': l2}
    print(f"{time.time()-t0:.1f}s")

    print(f"  {args.n_orth_dirs} random dirs ⊥ m_hat...", end=' ', flush=True)
    t0 = time.time()
    kl_all = np.zeros((args.n_act, len(angles_deg), args.n_orth_dirs))
    l2_all = np.zeros((args.n_act, len(angles_deg), args.n_orth_dirs))
    for d_idx in range(args.n_orth_dirs):
        kl, l2 = sweep_single_direction(
            model, contexts_dev, activations_dev, ref_final_orig, ref_lp_orig,
            orth_dirs[d_idx], angles_rad, args.layer, device)
        kl_all[:, :, d_idx] = kl
        l2_all[:, :, d_idx] = l2
    sweep_results['original_orth'] = {'kl': kl_all, 'l2': l2_all}
    print(f"{time.time()-t0:.1f}s")

    # ===== Phase 7: Save results =====
    all_results = {
        'args': vars(args),
        'alpha_star_deg': alpha_star,
        'm_hat': m_hat.cpu(),
        'm_perp': m_perp.cpu(),
        'mean_dir': mean_dir.cpu(),
        'orth_dirs': orth_dirs.cpu(),
        'angles_deg': angles_deg,
        'sweep_results': sweep_results,
        'kl_orig_to_boundary': kl_orig_to_bnd.cpu().numpy(),
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved results to {results_path}")

    # ===== Phase 8: Plots =====
    print(f"\n{'='*60}")
    print("PHASE 8: Plotting")
    print("="*60)

    max_angle = 60  # Focus on 0-60 degrees for clarity
    mask = angles_deg <= max_angle
    x = angles_deg[mask]

    def get_median_iqr(data):
        """Median and IQR across activations (axis=0). Returns (median, q25, q75)."""
        if data.ndim == 3:
            # [N, n_angles, n_dirs] → mean over dirs first, then median over N
            per_act = data.mean(axis=2)  # mean over directions
        else:
            per_act = data
        return (np.median(per_act, axis=0),
                np.percentile(per_act, 25, axis=0),
                np.percentile(per_act, 75, axis=0))

    def get_orth_stats(data):
        """For orthogonal dirs: mean and std over directions, then median over activations.

        Returns (median_of_mean, std_lower, std_upper) for shading.
        """
        # data: [N, n_angles, n_dirs]
        per_act_mean = data.mean(axis=2)   # [N, n_angles]
        per_act_std = data.std(axis=2)     # [N, n_angles]
        med_mean = np.median(per_act_mean, axis=0)
        med_std = np.median(per_act_std, axis=0)
        return med_mean, med_mean - med_std, med_mean + med_std

    colors = {
        'mhat': '#d62728',       # red
        'mperp': '#ff7f0e',      # orange
        'orth': '#1f77b4',       # blue
    }

    # --- Plot 1: From a_boundary ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # m_hat from boundary
    med, q25, q75 = get_median_iqr(sweep_results['boundary_mhat']['kl'])
    ax.plot(x, med[mask], color=colors['mhat'], lw=2.5, label='m_hat (MELBO₁)')
    ax.fill_between(x, q25[mask], q75[mask], color=colors['mhat'], alpha=0.1)

    # m_perp from boundary
    med, q25, q75 = get_median_iqr(sweep_results['boundary_mperp']['kl'])
    ax.plot(x, med[mask], color=colors['mperp'], lw=2.5, label='m_perp (MELBO₁ ⊥ mean)')
    ax.fill_between(x, q25[mask], q75[mask], color=colors['mperp'], alpha=0.1)

    # Random ⊥ m_hat from boundary
    med_mean, lo, hi = get_orth_stats(sweep_results['boundary_orth']['kl'])
    ax.plot(x, med_mean[mask], color=colors['orth'], lw=2.5,
            label=f'Random ⊥ m_hat (mean of {args.n_orth_dirs})')
    ax.fill_between(x, lo[mask], hi[mask], color=colors['orth'], alpha=0.15)

    ax.set_xlabel('Perturbation angle from a_boundary (degrees)', fontsize=12)
    ax.set_ylabel('KL divergence (vs a_boundary output)', fontsize=12)
    ax.set_title(f'Perturbation from boundary point (alpha* = {alpha_star:.1f} deg)\n'
                 f'{args.model}, layer {args.layer}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_angle)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'from_boundary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # --- Plot 2: From a_original ---
    fig, ax = plt.subplots(figsize=(10, 7))

    med, q25, q75 = get_median_iqr(sweep_results['original_mhat']['kl'])
    ax.plot(x, med[mask], color=colors['mhat'], lw=2.5, label='m_hat (MELBO₁)')
    ax.fill_between(x, q25[mask], q75[mask], color=colors['mhat'], alpha=0.1)

    med, q25, q75 = get_median_iqr(sweep_results['original_mperp']['kl'])
    ax.plot(x, med[mask], color=colors['mperp'], lw=2.5, label='m_perp (MELBO₁ ⊥ mean)')
    ax.fill_between(x, q25[mask], q75[mask], color=colors['mperp'], alpha=0.1)

    med_mean, lo, hi = get_orth_stats(sweep_results['original_orth']['kl'])
    ax.plot(x, med_mean[mask], color=colors['orth'], lw=2.5,
            label=f'Random ⊥ m_hat (mean of {args.n_orth_dirs})')
    ax.fill_between(x, lo[mask], hi[mask], color=colors['orth'], alpha=0.15)

    ax.set_xlabel('Perturbation angle from a_original (degrees)', fontsize=12)
    ax.set_ylabel('KL divergence (vs a_original output)', fontsize=12)
    ax.set_title(f'Perturbation from original activation\n'
                 f'{args.model}, layer {args.layer}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_angle)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'from_original.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # --- Plot 3: Overlay random ⊥ curves from both base points ---
    fig, ax = plt.subplots(figsize=(10, 7))

    med_bnd, lo_bnd, hi_bnd = get_orth_stats(sweep_results['boundary_orth']['kl'])
    med_orig, lo_orig, hi_orig = get_orth_stats(sweep_results['original_orth']['kl'])

    ax.plot(x, med_bnd[mask], color='#d62728', lw=2.5,
            label='From a_boundary')
    ax.fill_between(x, lo_bnd[mask], hi_bnd[mask], color='#d62728', alpha=0.15)

    ax.plot(x, med_orig[mask], color='#1f77b4', lw=2.5,
            label='From a_original')
    ax.fill_between(x, lo_orig[mask], hi_orig[mask], color='#1f77b4', alpha=0.15)

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel('KL divergence (vs base point output)', fontsize=12)
    ax.set_title(f'Random directions ⊥ m_hat: boundary vs original\n'
                 f'{args.model}, layer {args.layer}, {args.n_orth_dirs} directions', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_angle)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'overlay_orth.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ===== Summary =====
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)

    idx_15 = np.argmin(np.abs(angles_deg - 15))
    idx_30 = np.argmin(np.abs(angles_deg - 30))

    print(f"Boundary angle alpha* = {alpha_star:.1f} deg")
    print(f"KL(original || boundary) = {kl_orig_to_bnd.median().item():.4f}")

    for angle_label, idx in [('15 deg', idx_15), ('30 deg', idx_30)]:
        print(f"\nMedian KL at {angle_label}:")

        print(f"  From boundary:")
        bnd_mhat = np.median(sweep_results['boundary_mhat']['kl'][:, idx])
        bnd_mperp = np.median(sweep_results['boundary_mperp']['kl'][:, idx])
        bnd_orth = np.median(sweep_results['boundary_orth']['kl'][:, idx, :].mean(axis=1))
        print(f"    m_hat:      {bnd_mhat:.4f}")
        print(f"    m_perp:     {bnd_mperp:.4f}")
        print(f"    random ⊥:   {bnd_orth:.4f}")

        print(f"  From original:")
        orig_mhat = np.median(sweep_results['original_mhat']['kl'][:, idx])
        orig_mperp = np.median(sweep_results['original_mperp']['kl'][:, idx])
        orig_orth = np.median(sweep_results['original_orth']['kl'][:, idx, :].mean(axis=1))
        print(f"    m_hat:      {orig_mhat:.4f}")
        print(f"    m_perp:     {orig_mperp:.4f}")
        print(f"    random ⊥:   {orig_orth:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
