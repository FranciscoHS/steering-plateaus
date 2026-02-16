#!/usr/bin/env python3
"""
Follow-up to run_boundary_orthogonal.py:
Compare fully random directions (not restricted to ⊥ m_hat) from a_original vs a_boundary.

Usage:
    python scripts/run_boundary_random_unrestricted.py --model qwen3-0.6b --layer 14
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--n_act', type=int, default=30)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--boundary_results', default=None,
                        help='Path to boundary_orthogonal results.pkl (for alpha* and directions)')
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

    # ===== Load previous results for alpha* and m_hat =====
    print("Loading boundary results for alpha* and m_hat...")
    with open(args.boundary_results, 'rb') as f:
        prev = pickle.load(f)
    alpha_star = prev['alpha_star_deg']
    alpha_star_rad = np.deg2rad(alpha_star)
    m_hat = prev['m_hat']
    print(f"  alpha* = {alpha_star:.1f} deg")

    # ===== Sample unrestricted random directions =====
    # Use a different seed offset so these differ from the ⊥ m_hat directions
    torch.manual_seed(args.seed + 999)
    d_model = m_hat.shape[0]
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_dirs = F.normalize(torch.randn(args.n_random_dirs, d_model, device=device_str), dim=-1)

    # Check how much m_hat component they have (for reference)
    dots = (random_dirs @ m_hat.to(device_str)).abs()
    print(f"  |cos(random_dir, m_hat)|: mean={dots.mean():.4f}, max={dots.max():.4f}")

    # ===== Load model and data =====
    torch.manual_seed(args.seed)  # Reset seed for data loading
    np.random.seed(args.seed)

    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    random_dirs = random_dirs.to(device)
    m_hat_dev = m_hat.to(device)

    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Activations: {activations.shape}")

    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)

    # Boundary activations
    boundary_acts = compute_boundary_activations(activations_dev, m_hat_dev, alpha_star_rad)

    # References
    with torch.no_grad():
        ref_hidden_orig = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_final_orig, ref_logits_orig = forward_to_hidden_and_logits(
            model, ref_hidden_orig, args.layer)
        ref_lp_orig = F.log_softmax(ref_logits_orig, dim=-1)

        ref_hidden_bnd = torch.cat([contexts_dev, boundary_acts.unsqueeze(1)], dim=1)
        ref_final_bnd, ref_logits_bnd = forward_to_hidden_and_logits(
            model, ref_hidden_bnd, args.layer)
        ref_lp_bnd = F.log_softmax(ref_logits_bnd, dim=-1)

    # ===== Sweeps =====
    angles_deg = np.linspace(0, 180, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    print(f"\nSweeping {args.n_random_dirs} unrestricted random dirs...")

    kl_bnd = np.zeros((args.n_act, len(angles_deg), args.n_random_dirs))
    kl_orig = np.zeros((args.n_act, len(angles_deg), args.n_random_dirs))

    for d_idx in range(args.n_random_dirs):
        print(f"  dir {d_idx+1}/{args.n_random_dirs}...", end=' ', flush=True)
        t0 = time.time()

        kl_b, _ = sweep_single_direction(
            model, contexts_dev, boundary_acts, ref_final_bnd, ref_lp_bnd,
            random_dirs[d_idx], angles_rad, args.layer, device)
        kl_bnd[:, :, d_idx] = kl_b

        kl_o, _ = sweep_single_direction(
            model, contexts_dev, activations_dev, ref_final_orig, ref_lp_orig,
            random_dirs[d_idx], angles_rad, args.layer, device)
        kl_orig[:, :, d_idx] = kl_o

        print(f"{time.time()-t0:.1f}s")

    # Also load the ⊥ m_hat results for comparison
    kl_bnd_orth = prev['sweep_results']['boundary_orth']['kl']
    kl_orig_orth = prev['sweep_results']['original_orth']['kl']

    # ===== Save =====
    results = {
        'args': vars(args),
        'alpha_star_deg': alpha_star,
        'random_dirs': random_dirs.cpu(),
        'angles_deg': angles_deg,
        'kl_boundary_random': kl_bnd,
        'kl_original_random': kl_orig,
    }
    results_path = os.path.join(args.results_dir, 'random_unrestricted_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {results_path}")

    # ===== Plot =====
    max_angle = 60
    mask = angles_deg <= max_angle
    x = angles_deg[mask]

    def get_orth_stats(data):
        per_act_mean = data.mean(axis=2)
        per_act_std = data.std(axis=2)
        med_mean = np.median(per_act_mean, axis=0)
        med_std = np.median(per_act_std, axis=0)
        return med_mean, med_mean - med_std, med_mean + med_std

    # --- Plot: unrestricted random, boundary vs original ---
    fig, ax = plt.subplots(figsize=(10, 7))

    med_bnd, lo_bnd, hi_bnd = get_orth_stats(kl_bnd)
    med_orig, lo_orig, hi_orig = get_orth_stats(kl_orig)

    ax.plot(x, med_bnd[mask], color='#d62728', lw=2.5, label='From a_boundary')
    ax.fill_between(x, lo_bnd[mask], hi_bnd[mask], color='#d62728', alpha=0.15)
    ax.plot(x, med_orig[mask], color='#1f77b4', lw=2.5, label='From a_original')
    ax.fill_between(x, lo_orig[mask], hi_orig[mask], color='#1f77b4', alpha=0.15)

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel('KL divergence (vs base point output)', fontsize=12)
    ax.set_title(f'Unrestricted random directions: boundary vs original\n'
                 f'{args.model}, layer {args.layer}, {args.n_random_dirs} directions',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_angle)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'overlay_random_unrestricted.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # --- Plot: 4-way comparison (unrestricted vs ⊥ m_hat, from both base points) ---
    fig, ax = plt.subplots(figsize=(10, 7))

    med, lo, hi = get_orth_stats(kl_bnd)
    ax.plot(x, med[mask], color='#d62728', lw=2.5, ls='-',
            label='Boundary, unrestricted random')
    ax.fill_between(x, lo[mask], hi[mask], color='#d62728', alpha=0.1)

    med, lo, hi = get_orth_stats(kl_bnd_orth)
    ax.plot(x, med[mask], color='#d62728', lw=2.0, ls='--',
            label='Boundary, random ⊥ m_hat')
    ax.fill_between(x, lo[mask], hi[mask], color='#d62728', alpha=0.05)

    med, lo, hi = get_orth_stats(kl_orig)
    ax.plot(x, med[mask], color='#1f77b4', lw=2.5, ls='-',
            label='Original, unrestricted random')
    ax.fill_between(x, lo[mask], hi[mask], color='#1f77b4', alpha=0.1)

    med, lo, hi = get_orth_stats(kl_orig_orth)
    ax.plot(x, med[mask], color='#1f77b4', lw=2.0, ls='--',
            label='Original, random ⊥ m_hat')
    ax.fill_between(x, lo[mask], hi[mask], color='#1f77b4', alpha=0.05)

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel('KL divergence (vs base point output)', fontsize=12)
    ax.set_title(f'Random directions: unrestricted vs ⊥ m_hat\n'
                 f'{args.model}, layer {args.layer}', fontsize=13)
    ax.legend(fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_angle)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'overlay_4way.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # Summary
    idx_15 = np.argmin(np.abs(angles_deg - 15))
    idx_30 = np.argmin(np.abs(angles_deg - 30))
    print(f"\nMedian KL (unrestricted random):")
    for label, idx in [('15 deg', idx_15), ('30 deg', idx_30)]:
        bnd_val = np.median(kl_bnd[:, idx, :].mean(axis=1))
        orig_val = np.median(kl_orig[:, idx, :].mean(axis=1))
        print(f"  {label}: boundary={bnd_val:.4f}, original={orig_val:.4f}, ratio={bnd_val/orig_val:.2f}x")

    print("\nDone!")


if __name__ == '__main__':
    main()
