#!/usr/bin/env python3
"""
Boundary orthogonal experiment with fixed reference: all KL measured relative to a_original.

This answers: when you perturb from the boundary point in an orthogonal direction,
does the KL-from-original stay constant (boundary is a flat wall) or change
(boundary is a narrow ridge)?

Usage:
    python scripts/run_boundary_fixedref.py --model qwen3-0.6b --layer 14
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
    load_model,
    _get_blocks, _is_rope_model, _get_position_embeddings, _run_block,
    _get_final_norm, _get_lm_head,
)
from src.data import load_fineweb_fixed_length


# ── Reused infrastructure ──

def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device,
                                   batch_size=64):
    all_contexts, all_activations = [], []
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
    direction_dev = direction.to(device)
    for ang_idx, theta in enumerate(angles_rad):
        with torch.no_grad():
            hidden = build_perturbed_hidden(contexts, activations, direction_dev, theta)
            _, pert_logits = forward_to_hidden_and_logits(model, hidden, layer)
            pert_log_probs = F.log_softmax(pert_logits, dim=-1)
            kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            kl_divs[:, ang_idx] = kl.cpu().numpy()
    return kl_divs


def compute_boundary_activations(activations, direction, angle_rad):
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms
    d = direction.unsqueeze(0)
    proj = (d * a_unit).sum(dim=-1, keepdim=True)
    u_perp = d - proj * a_unit
    u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)
    return a_norms * (cos_t * a_unit + sin_t * u_perp)


def project_out(v, u):
    residual = v - (v @ u) * u
    return residual / residual.norm()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--n_act', type=int, default=30)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--n_orth_dirs', type=int, default=10)
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--boundary_results', default=None)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.boundary_results is None:
        args.boundary_results = f'results/boundary_orthogonal/{args.model}/results.pkl'
    if args.results_dir is None:
        args.results_dir = f'results/boundary_orthogonal/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Load previous results =====
    print("Loading previous boundary results...")
    with open(args.boundary_results, 'rb') as f:
        prev = pickle.load(f)
    alpha_star = prev['alpha_star_deg']
    alpha_star_rad = np.deg2rad(alpha_star)
    m_hat = prev['m_hat']
    m_perp = prev['m_perp']
    mean_dir = prev['mean_dir']
    # Reuse same orthogonal directions for comparability
    orth_dirs = prev['orth_dirs']
    print(f"  alpha* = {alpha_star:.1f} deg, reusing {orth_dirs.shape[0]} orth dirs from prev run")

    # Sample unrestricted random dirs (same seed as previous unrestricted run)
    torch.manual_seed(args.seed + 999)
    d_model = m_hat.shape[0]
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_dirs = F.normalize(torch.randn(args.n_random_dirs, d_model, device=device_str), dim=-1)

    # ===== Load model and data =====
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    m_hat_dev = m_hat.to(device)
    m_perp_dev = m_perp.to(device)
    orth_dirs = orth_dirs.to(device)
    random_dirs = random_dirs.to(device)

    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Activations: {activations.shape}")

    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)

    # Boundary activations
    boundary_acts = compute_boundary_activations(activations_dev, m_hat_dev, alpha_star_rad)

    # Reference: ALWAYS a_original
    with torch.no_grad():
        ref_hidden = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_hidden_and_logits(model, ref_hidden, args.layer)
        ref_lp = F.log_softmax(ref_logits, dim=-1)
    print("Reference: a_original (used for ALL KL measurements)")

    # Baseline KL at boundary
    with torch.no_grad():
        bnd_hidden = torch.cat([contexts_dev, boundary_acts.unsqueeze(1)], dim=1)
        _, bnd_logits = forward_to_hidden_and_logits(model, bnd_hidden, args.layer)
        bnd_lp = F.log_softmax(bnd_logits, dim=-1)
        kl_at_boundary = (ref_lp.exp() * (ref_lp - bnd_lp)).sum(dim=-1)
    print(f"KL(original || boundary): median={kl_at_boundary.median().item():.4f}")

    # ===== Sweeps (all KL relative to a_original) =====
    angles_deg = np.linspace(0, 180, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    sweep = {}

    # From boundary
    print("\n--- From a_boundary (KL vs a_original) ---")

    print("  m_hat...", end=' ', flush=True)
    t0 = time.time()
    sweep['bnd_mhat'] = sweep_single_direction(
        model, contexts_dev, boundary_acts, ref_final, ref_lp,
        m_hat_dev, angles_rad, args.layer, device)
    print(f"{time.time()-t0:.1f}s")

    print("  m_perp...", end=' ', flush=True)
    t0 = time.time()
    sweep['bnd_mperp'] = sweep_single_direction(
        model, contexts_dev, boundary_acts, ref_final, ref_lp,
        m_perp_dev, angles_rad, args.layer, device)
    print(f"{time.time()-t0:.1f}s")

    print(f"  {args.n_orth_dirs} dirs ⊥ m_hat...", end=' ', flush=True)
    t0 = time.time()
    kl_all = np.zeros((args.n_act, len(angles_deg), args.n_orth_dirs))
    for d_idx in range(args.n_orth_dirs):
        kl_all[:, :, d_idx] = sweep_single_direction(
            model, contexts_dev, boundary_acts, ref_final, ref_lp,
            orth_dirs[d_idx], angles_rad, args.layer, device)
    sweep['bnd_orth'] = kl_all
    print(f"{time.time()-t0:.1f}s")

    print(f"  {args.n_random_dirs} unrestricted random...", end=' ', flush=True)
    t0 = time.time()
    kl_all = np.zeros((args.n_act, len(angles_deg), args.n_random_dirs))
    for d_idx in range(args.n_random_dirs):
        kl_all[:, :, d_idx] = sweep_single_direction(
            model, contexts_dev, boundary_acts, ref_final, ref_lp,
            random_dirs[d_idx], angles_rad, args.layer, device)
    sweep['bnd_random'] = kl_all
    print(f"{time.time()-t0:.1f}s")

    # From original
    print("\n--- From a_original (KL vs a_original) ---")

    print("  m_hat...", end=' ', flush=True)
    t0 = time.time()
    sweep['orig_mhat'] = sweep_single_direction(
        model, contexts_dev, activations_dev, ref_final, ref_lp,
        m_hat_dev, angles_rad, args.layer, device)
    print(f"{time.time()-t0:.1f}s")

    print("  m_perp...", end=' ', flush=True)
    t0 = time.time()
    sweep['orig_mperp'] = sweep_single_direction(
        model, contexts_dev, activations_dev, ref_final, ref_lp,
        m_perp_dev, angles_rad, args.layer, device)
    print(f"{time.time()-t0:.1f}s")

    print(f"  {args.n_orth_dirs} dirs ⊥ m_hat...", end=' ', flush=True)
    t0 = time.time()
    kl_all = np.zeros((args.n_act, len(angles_deg), args.n_orth_dirs))
    for d_idx in range(args.n_orth_dirs):
        kl_all[:, :, d_idx] = sweep_single_direction(
            model, contexts_dev, activations_dev, ref_final, ref_lp,
            orth_dirs[d_idx], angles_rad, args.layer, device)
    sweep['orig_orth'] = kl_all
    print(f"{time.time()-t0:.1f}s")

    print(f"  {args.n_random_dirs} unrestricted random...", end=' ', flush=True)
    t0 = time.time()
    kl_all = np.zeros((args.n_act, len(angles_deg), args.n_random_dirs))
    for d_idx in range(args.n_random_dirs):
        kl_all[:, :, d_idx] = sweep_single_direction(
            model, contexts_dev, activations_dev, ref_final, ref_lp,
            random_dirs[d_idx], angles_rad, args.layer, device)
    sweep['orig_random'] = kl_all
    print(f"{time.time()-t0:.1f}s")

    # ===== Save =====
    results = {
        'args': vars(args),
        'alpha_star_deg': alpha_star,
        'angles_deg': angles_deg,
        'kl_at_boundary': kl_at_boundary.cpu().numpy(),
        'sweep': sweep,
    }
    path = os.path.join(args.results_dir, 'fixedref_results.pkl')
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {path}")

    # ===== Plots =====
    max_angle = 60
    mask = angles_deg <= max_angle
    x = angles_deg[mask]

    kl_bnd_median = float(np.median(kl_at_boundary.cpu().numpy()))

    def med(data):
        """Median over activations. For 3D, mean over dirs first."""
        if data.ndim == 3:
            return np.median(data.mean(axis=2), axis=0)
        return np.median(data, axis=0)

    def iqr(data):
        if data.ndim == 3:
            d = data.mean(axis=2)
        else:
            d = data
        return np.percentile(d, 25, axis=0), np.percentile(d, 75, axis=0)

    def dir_spread(data):
        """For multi-dir data: median of (per-activation std across dirs)."""
        return np.median(data.std(axis=2), axis=0)

    # --- Plot 1: From boundary, all directions, KL vs original ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # Horizontal line: KL at boundary point itself
    ax.axhline(kl_bnd_median, color='gray', ls=':', lw=1.5, label=f'KL at boundary ({kl_bnd_median:.2f})')

    y = med(sweep['bnd_mhat'])
    q25, q75 = iqr(sweep['bnd_mhat'])
    ax.plot(x, y[mask], color='#d62728', lw=2.5, label='m_hat')
    ax.fill_between(x, q25[mask], q75[mask], color='#d62728', alpha=0.1)

    y = med(sweep['bnd_mperp'])
    q25, q75 = iqr(sweep['bnd_mperp'])
    ax.plot(x, y[mask], color='#ff7f0e', lw=2.5, label='m_perp')
    ax.fill_between(x, q25[mask], q75[mask], color='#ff7f0e', alpha=0.1)

    y = med(sweep['bnd_orth'])
    sd = dir_spread(sweep['bnd_orth'])
    ax.plot(x, y[mask], color='#1f77b4', lw=2.5, label=f'Random ⊥ m_hat ({args.n_orth_dirs})')
    ax.fill_between(x, (y - sd)[mask], (y + sd)[mask], color='#1f77b4', alpha=0.15)

    y = med(sweep['bnd_random'])
    sd = dir_spread(sweep['bnd_random'])
    ax.plot(x, y[mask], color='#2ca02c', lw=2.5, ls='--', label=f'Unrestricted random ({args.n_random_dirs})')
    ax.fill_between(x, (y - sd)[mask], (y + sd)[mask], color='#2ca02c', alpha=0.1)

    ax.set_xlabel('Perturbation angle from a_boundary (degrees)', fontsize=12)
    ax.set_ylabel('KL divergence (vs a_original)', fontsize=12)
    ax.set_title(f'From boundary: KL relative to original\n'
                 f'{args.model}, layer {args.layer}, alpha*={alpha_star:.1f} deg', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_angle)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'fixedref_from_boundary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # --- Plot 2: From original, all directions ---
    fig, ax = plt.subplots(figsize=(10, 7))

    y = med(sweep['orig_mhat'])
    q25, q75 = iqr(sweep['orig_mhat'])
    ax.plot(x, y[mask], color='#d62728', lw=2.5, label='m_hat')
    ax.fill_between(x, q25[mask], q75[mask], color='#d62728', alpha=0.1)

    y = med(sweep['orig_mperp'])
    q25, q75 = iqr(sweep['orig_mperp'])
    ax.plot(x, y[mask], color='#ff7f0e', lw=2.5, label='m_perp')
    ax.fill_between(x, q25[mask], q75[mask], color='#ff7f0e', alpha=0.1)

    y = med(sweep['orig_orth'])
    sd = dir_spread(sweep['orig_orth'])
    ax.plot(x, y[mask], color='#1f77b4', lw=2.5, label=f'Random ⊥ m_hat ({args.n_orth_dirs})')
    ax.fill_between(x, (y - sd)[mask], (y + sd)[mask], color='#1f77b4', alpha=0.15)

    y = med(sweep['orig_random'])
    sd = dir_spread(sweep['orig_random'])
    ax.plot(x, y[mask], color='#2ca02c', lw=2.5, ls='--', label=f'Unrestricted random ({args.n_random_dirs})')
    ax.fill_between(x, (y - sd)[mask], (y + sd)[mask], color='#2ca02c', alpha=0.1)

    ax.set_xlabel('Perturbation angle from a_original (degrees)', fontsize=12)
    ax.set_ylabel('KL divergence (vs a_original)', fontsize=12)
    ax.set_title(f'From original: KL relative to original\n'
                 f'{args.model}, layer {args.layer}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_angle)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'fixedref_from_original.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # --- Plot 3: Overlay — random ⊥ m_hat from both base points ---
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.axhline(kl_bnd_median, color='gray', ls=':', lw=1.5, label=f'KL at boundary ({kl_bnd_median:.2f})')

    y = med(sweep['bnd_orth'])
    sd = dir_spread(sweep['bnd_orth'])
    ax.plot(x, y[mask], color='#d62728', lw=2.5, label='From a_boundary, ⊥ m_hat')
    ax.fill_between(x, (y - sd)[mask], (y + sd)[mask], color='#d62728', alpha=0.15)

    y = med(sweep['bnd_random'])
    sd = dir_spread(sweep['bnd_random'])
    ax.plot(x, y[mask], color='#d62728', lw=2, ls='--', label='From a_boundary, unrestricted')
    ax.fill_between(x, (y - sd)[mask], (y + sd)[mask], color='#d62728', alpha=0.05)

    y = med(sweep['orig_orth'])
    sd = dir_spread(sweep['orig_orth'])
    ax.plot(x, y[mask], color='#1f77b4', lw=2.5, label='From a_original, ⊥ m_hat')
    ax.fill_between(x, (y - sd)[mask], (y + sd)[mask], color='#1f77b4', alpha=0.15)

    y = med(sweep['orig_random'])
    sd = dir_spread(sweep['orig_random'])
    ax.plot(x, y[mask], color='#1f77b4', lw=2, ls='--', label='From a_original, unrestricted')
    ax.fill_between(x, (y - sd)[mask], (y + sd)[mask], color='#1f77b4', alpha=0.05)

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel('KL divergence (vs a_original)', fontsize=12)
    ax.set_title(f'Random directions: all KL relative to a_original\n'
                 f'{args.model}, layer {args.layer}', fontsize=13)
    ax.legend(fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_angle)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'fixedref_overlay.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ===== Summary =====
    idx_15 = np.argmin(np.abs(angles_deg - 15))
    idx_30 = np.argmin(np.abs(angles_deg - 30))

    print(f"\n{'='*60}")
    print(f"SUMMARY (all KL relative to a_original)")
    print(f"{'='*60}")
    print(f"KL at boundary point: {kl_bnd_median:.4f}")

    for label, idx in [('15 deg', idx_15), ('30 deg', idx_30)]:
        print(f"\nMedian KL at {label} perturbation:")
        print(f"  From boundary:")
        print(f"    m_hat:       {np.median(sweep['bnd_mhat'][:, idx]):.4f}")
        print(f"    m_perp:      {np.median(sweep['bnd_mperp'][:, idx]):.4f}")
        print(f"    random ⊥:    {np.median(sweep['bnd_orth'][:, idx, :].mean(axis=1)):.4f}")
        print(f"    random unr:  {np.median(sweep['bnd_random'][:, idx, :].mean(axis=1)):.4f}")
        print(f"  From original:")
        print(f"    m_hat:       {np.median(sweep['orig_mhat'][:, idx]):.4f}")
        print(f"    m_perp:      {np.median(sweep['orig_mperp'][:, idx]):.4f}")
        print(f"    random ⊥:    {np.median(sweep['orig_orth'][:, idx, :].mean(axis=1)):.4f}")
        print(f"    random unr:  {np.median(sweep['orig_random'][:, idx, :].mean(axis=1)):.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
