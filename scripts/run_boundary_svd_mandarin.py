#!/usr/bin/env python3
"""
Boundary perturbation experiment using SVD top direction on Mandarin activations.

Perturbs Mandarin activations along the top singular value from EN-ZH SVD,
finds the boundary (max KL derivative), then from that boundary point perturbs
along (a) random directions, (b) directions orthogonal to the SVD direction,
and (c) the SVD direction itself. Measures L2 (at penultimate layer) and KL-div
relative to the original (unperturbed) activations.

Usage:
    python scripts/run_boundary_svd_mandarin.py --model qwen3-0.6b --layer 14
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
    load_model, extract_activations,
    _get_blocks, _is_rope_model, _get_position_embeddings, _run_block,
    _get_final_norm, _get_lm_head,
)
from data.bilingual_prompts import CHINESE_PROMPTS


# ── Forward pass capturing penultimate layer ──

def forward_penultimate_and_logits(model, hidden_states, start_layer):
    """Forward pass returning penultimate-layer last-token hidden state and logits.

    The penultimate layer is the 2nd-to-last transformer block (block n-2).
    """
    hidden = hidden_states
    blocks = _get_blocks(model)
    n_blocks = len(blocks)
    penultimate_block = n_blocks - 2
    rope = _is_rope_model(model)
    pos_emb = _get_position_embeddings(model, hidden) if rope else None

    penultimate = None
    for block_idx in range(start_layer + 1, n_blocks):
        hidden = _run_block(blocks[block_idx], hidden, model, pos_emb)
        if block_idx == penultimate_block:
            penultimate = hidden[:, -1, :].clone()

    final_hidden = hidden[:, -1, :]
    normed = _get_final_norm(model)(final_hidden)
    logits = _get_lm_head(model)(normed)
    return penultimate, logits


# ── Spherical perturbation ──

def compute_perturbed_batch(activation, direction, angles_rad, device):
    """Spherically perturb a single activation by multiple angles along direction.

    Returns [n_angles, d_model] perturbed activations (same norm as original).
    """
    a_norm = activation.norm()
    a_hat = activation / a_norm

    proj = (direction * a_hat).sum()
    u_perp = direction - proj * a_hat
    u_perp = u_perp / (u_perp.norm() + 1e-10)

    cos_t = torch.tensor(np.cos(angles_rad), dtype=activation.dtype, device=device).unsqueeze(1)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=activation.dtype, device=device).unsqueeze(1)

    return a_norm * (cos_t * a_hat.unsqueeze(0) + sin_t * u_perp.unsqueeze(0))


def compute_boundary_activation(activation, direction, angle_rad):
    """Spherically perturb a single activation by angle_rad along direction.

    Returns [d_model] boundary activation (same norm as original).
    """
    a_norm = activation.norm()
    a_hat = activation / a_norm
    proj = (direction * a_hat).sum()
    u_perp = direction - proj * a_hat
    u_perp = u_perp / (u_perp.norm() + 1e-10)
    return a_norm * (np.cos(angle_rad) * a_hat + np.sin(angle_rad) * u_perp)


# ── Direction sampling ──

def sample_random_directions(n_dirs, d_model, device):
    """Sample n_dirs random unit vectors in R^d_model."""
    v = torch.randn(n_dirs, d_model, device=device)
    return F.normalize(v, dim=-1)


def sample_orthogonal_directions(direction, n_dirs, d_model, device):
    """Sample n_dirs random unit vectors orthogonal to direction."""
    v = torch.randn(n_dirs, d_model, device=device)
    d = direction.to(device)
    v = v - (v @ d).unsqueeze(1) * d.unsqueeze(0)
    return F.normalize(v, dim=-1)


# ── Sweep function ──

def sweep_from_base_vs_original(model, contexts, orig_activations, base_activations,
                                direction, angles_rad, layer, device, angle_batch=30):
    """Sweep from base_activations along direction, measuring vs original activations.

    For each activation:
      - Reference = original activation forwarded through the model
      - Perturbed = base_activation (e.g. boundary) perturbed by angle along direction
      - L2 = ||penultimate(perturbed) - penultimate(original)||
      - KL = KL(original_logits || perturbed_logits)

    contexts: list of [1, seq_len_i-1, d_model] tensors (variable length)
    orig_activations: [N, d_model] original activations (reference)
    base_activations: [N, d_model] starting points for perturbation

    Returns: kl [N, n_angles], l2 [N, n_angles]
    """
    N = len(contexts)
    n_angles = len(angles_rad)
    kl_divs = np.zeros((N, n_angles))
    l2_dists = np.zeros((N, n_angles))
    dir_dev = direction.to(device)

    for i in range(N):
        ctx = contexts[i].to(device)          # [1, seq_len_i-1, d_model]
        orig_act = orig_activations[i].to(device)
        base_act = base_activations[i].to(device)

        with torch.no_grad():
            # Reference: forward original activation
            ref_hidden = torch.cat([ctx, orig_act.unsqueeze(0).unsqueeze(0)], dim=1)
            ref_penult, ref_logits = forward_penultimate_and_logits(model, ref_hidden, layer)
            ref_lp = F.log_softmax(ref_logits, dim=-1)

            # Process angles in batches
            for batch_start in range(0, n_angles, angle_batch):
                batch_end = min(batch_start + angle_batch, n_angles)
                batch_angles = angles_rad[batch_start:batch_end]
                bs = len(batch_angles)

                pert_acts = compute_perturbed_batch(base_act, dir_dev, batch_angles, device)
                ctx_batch = ctx.expand(bs, -1, -1)
                hidden_batch = torch.cat([ctx_batch, pert_acts.unsqueeze(1)], dim=1)

                pert_penult, pert_logits = forward_penultimate_and_logits(
                    model, hidden_batch, layer)
                pert_lp = F.log_softmax(pert_logits, dim=-1)

                l2 = (pert_penult - ref_penult).norm(dim=-1)
                kl = (ref_lp.exp() * (ref_lp - pert_lp)).sum(dim=-1)

                l2_dists[i, batch_start:batch_end] = l2.cpu().numpy()
                kl_divs[i, batch_start:batch_end] = kl.cpu().numpy()

    return kl_divs, l2_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--n_act', type=int, default=30,
                        help='Number of Mandarin activations to use')
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--n_random_dirs', type=int, default=5)
    parser.add_argument('--n_orth_dirs', type=int, default=5)
    parser.add_argument('--svd_results_path', default=None)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.svd_results_path is None:
        args.svd_results_path = f'results/svd_steering/{args.model}/results.pkl'
    if args.results_dir is None:
        args.results_dir = f'results/boundary_svd_mandarin/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Phase 1: Load SVD direction =====
    print("=" * 60)
    print("PHASE 1: Load SVD direction")
    print("=" * 60)

    with open(args.svd_results_path, 'rb') as f:
        svd_results = pickle.load(f)

    layer_results = svd_results['results_by_layer'][args.layer]
    svd_dir = torch.tensor(layer_results['pc1'], dtype=torch.float32)
    mean_dir = torch.tensor(layer_results['mean_direction'], dtype=torch.float32)

    # Ensure SVD direction points from Chinese toward English (same as mean)
    if (svd_dir @ mean_dir).item() < 0:
        svd_dir = -svd_dir
        print("  (Flipped SVD direction to align with mean EN→ZH direction)")

    svd_dir = svd_dir / svd_dir.norm()

    print(f"Loaded SVD PC1 at layer {args.layer}")
    print(f"  Variance explained by PC1: {layer_results['variance_explained'][0]*100:.1f}%")
    print(f"  cos(PC1, mean): {layer_results['cosine_pc1_mean']:.4f}")

    # ===== Phase 2: Load model & extract Mandarin activations =====
    print(f"\n{'='*60}")
    print("PHASE 2: Load model & extract Mandarin activations")
    print("=" * 60)

    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_blocks = len(_get_blocks(model))
    print(f"Device: {device}, d_model: {d_model}, n_blocks: {n_blocks}")
    print(f"Penultimate layer: block {n_blocks - 2}")

    # Select random subset of Mandarin prompts
    rng = np.random.RandomState(args.seed)
    n_available = len(CHINESE_PROMPTS)
    indices = rng.choice(n_available, size=min(args.n_act, n_available), replace=False)
    selected_prompts = [CHINESE_PROMPTS[i] for i in indices]
    print(f"Selected {len(selected_prompts)} Mandarin prompts (from {n_available})")

    contexts, activations = extract_activations(
        model, tokenizer, selected_prompts, args.layer, device)
    N = activations.shape[0]
    print(f"Activations: [{N}, {d_model}], mean norm: {activations.norm(dim=-1).mean():.2f}")
    print(f"Context lengths: {[c.shape[1] for c in contexts[:5]]}...")

    # ===== Phase 3: Boundary finding sweep =====
    print(f"\n{'='*60}")
    print("PHASE 3: Boundary finding — sweep along SVD direction")
    print("=" * 60)

    bfind_angles_deg = np.linspace(0, 180, args.n_angles + 1)[1:]
    bfind_angles_rad = np.deg2rad(bfind_angles_deg)
    svd_dir_dev = svd_dir.to(device)

    # For boundary finding, KL is measured vs the activation's own (unperturbed) output
    boundary_kl = np.zeros((N, len(bfind_angles_deg)))

    print(f"Sweeping {len(bfind_angles_deg)} angles for {N} activations...")
    t0 = time.time()

    for i in range(N):
        ctx = contexts[i].to(device)
        act = activations[i].to(device)

        with torch.no_grad():
            ref_hidden = torch.cat([ctx, act.unsqueeze(0).unsqueeze(0)], dim=1)
            _, ref_logits = forward_penultimate_and_logits(model, ref_hidden, args.layer)
            ref_lp = F.log_softmax(ref_logits, dim=-1)

            pert_acts = compute_perturbed_batch(act, svd_dir_dev, bfind_angles_rad, device)
            ctx_batch = ctx.expand(len(bfind_angles_deg), -1, -1)
            hidden_batch = torch.cat([ctx_batch, pert_acts.unsqueeze(1)], dim=1)
            _, pert_logits = forward_penultimate_and_logits(model, hidden_batch, args.layer)
            pert_lp = F.log_softmax(pert_logits, dim=-1)

            kl = (ref_lp.exp() * (ref_lp - pert_lp)).sum(dim=-1)
            boundary_kl[i, :] = kl.cpu().numpy()

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N} activations... ({time.time()-t0:.1f}s)")

    print(f"  Done in {time.time()-t0:.1f}s")

    # Find boundary angle from median KL curve (max derivative)
    median_kl = np.median(boundary_kl, axis=0)
    dkl = np.diff(median_kl) / np.diff(bfind_angles_deg)
    max_idx = np.argmax(dkl)
    alpha_star_deg = (bfind_angles_deg[max_idx] + bfind_angles_deg[max_idx + 1]) / 2
    alpha_star_rad = np.deg2rad(alpha_star_deg)

    print(f"\nBoundary angle: {alpha_star_deg:.1f} deg")
    print(f"  Max KL derivative: {dkl[max_idx]:.4f}")
    print(f"  KL at {bfind_angles_deg[max_idx]:.0f}°: {median_kl[max_idx]:.4f}")
    print(f"  KL at {bfind_angles_deg[max_idx+1]:.0f}°: {median_kl[max_idx+1]:.4f}")

    # ===== Phase 4: Compute boundary activations =====
    print(f"\n{'='*60}")
    print(f"PHASE 4: Compute boundary activations (α* = {alpha_star_deg:.1f}°)")
    print("=" * 60)

    boundary_acts = torch.zeros_like(activations)
    for i in range(N):
        boundary_acts[i] = compute_boundary_activation(
            activations[i], svd_dir, alpha_star_rad)

    print(f"  Norm preservation: orig={activations.norm(dim=-1).mean():.2f}, "
          f"boundary={boundary_acts.norm(dim=-1).mean():.2f}")
    cos_sim = F.cosine_similarity(activations, boundary_acts, dim=-1)
    print(f"  Cos(original, boundary): mean={cos_sim.mean():.4f} "
          f"(expected ~{np.cos(alpha_star_rad):.4f})")

    # ===== Phase 5: Sample directions =====
    print(f"\n{'='*60}")
    print(f"PHASE 5: Sample {args.n_random_dirs} random + {args.n_orth_dirs} orthogonal directions")
    print("=" * 60)

    random_dirs = sample_random_directions(args.n_random_dirs, d_model, torch.device(device))
    orth_dirs = sample_orthogonal_directions(svd_dir, args.n_orth_dirs, d_model,
                                             torch.device(device))

    dots_orth = (orth_dirs @ svd_dir.to(device)).abs()
    print(f"  Max |cos(orth_dir, svd_dir)|: {dots_orth.max().item():.2e}")
    dots_rand = (random_dirs @ svd_dir.to(device)).abs()
    print(f"  |cos(random_dir, svd_dir)|: mean={dots_rand.mean():.4f}, "
          f"max={dots_rand.max():.4f}")

    # ===== Phase 6: Perturbation sweeps from boundary =====
    # Include angle 0 so we see the boundary-vs-original baseline
    sweep_angles_deg = np.linspace(0, 180, args.n_angles + 1)
    sweep_angles_rad = np.deg2rad(sweep_angles_deg)

    print(f"\n{'='*60}")
    print(f"PHASE 6: Perturbation sweeps from boundary "
          f"({len(sweep_angles_deg)} angles, 0-180°)")
    print("=" * 60)

    sweep_results = {}

    # (c) SVD direction itself
    print(f"\n  SVD direction...", end=' ', flush=True)
    t0 = time.time()
    kl_svd, l2_svd = sweep_from_base_vs_original(
        model, contexts, activations, boundary_acts,
        svd_dir, sweep_angles_rad, args.layer, device)
    sweep_results['svd'] = {'kl': kl_svd, 'l2': l2_svd}
    print(f"{time.time()-t0:.1f}s")

    # (a) Random directions
    print(f"\n  {args.n_random_dirs} random directions...")
    kl_random = np.zeros((N, len(sweep_angles_deg), args.n_random_dirs))
    l2_random = np.zeros((N, len(sweep_angles_deg), args.n_random_dirs))
    for d_idx in range(args.n_random_dirs):
        t0 = time.time()
        kl, l2 = sweep_from_base_vs_original(
            model, contexts, activations, boundary_acts,
            random_dirs[d_idx], sweep_angles_rad, args.layer, device)
        kl_random[:, :, d_idx] = kl
        l2_random[:, :, d_idx] = l2
        print(f"    dir {d_idx+1}/{args.n_random_dirs}: {time.time()-t0:.1f}s")
    sweep_results['random'] = {'kl': kl_random, 'l2': l2_random}

    # (b) Orthogonal directions
    print(f"\n  {args.n_orth_dirs} orthogonal directions...")
    kl_orth = np.zeros((N, len(sweep_angles_deg), args.n_orth_dirs))
    l2_orth = np.zeros((N, len(sweep_angles_deg), args.n_orth_dirs))
    for d_idx in range(args.n_orth_dirs):
        t0 = time.time()
        kl, l2 = sweep_from_base_vs_original(
            model, contexts, activations, boundary_acts,
            orth_dirs[d_idx], sweep_angles_rad, args.layer, device)
        kl_orth[:, :, d_idx] = kl
        l2_orth[:, :, d_idx] = l2
        print(f"    dir {d_idx+1}/{args.n_orth_dirs}: {time.time()-t0:.1f}s")
    sweep_results['orthogonal'] = {'kl': kl_orth, 'l2': l2_orth}

    # ===== Phase 7: Save =====
    all_results = {
        'args': vars(args),
        'svd_dir': svd_dir.cpu(),
        'mean_dir': mean_dir.cpu(),
        'alpha_star_deg': alpha_star_deg,
        'boundary_finding_kl': boundary_kl,
        'boundary_finding_angles_deg': bfind_angles_deg,
        'sweep_angles_deg': sweep_angles_deg,
        'random_dirs': random_dirs.cpu(),
        'orth_dirs': orth_dirs.cpu(),
        'sweep_results': sweep_results,
        'selected_prompt_indices': indices,
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved results to {results_path}")

    # ===== Phase 8: Plots =====
    print(f"\n{'='*60}")
    print("PHASE 8: Plotting")
    print("=" * 60)

    max_angle = 90
    mask = sweep_angles_deg <= max_angle
    x = sweep_angles_deg[mask]

    def get_multi_dir_stats(data):
        """For [N, n_angles, n_dirs]: mean over dirs, then median + IQR over activations."""
        per_act = data.mean(axis=2)  # [N, n_angles]
        return (np.median(per_act, axis=0),
                np.percentile(per_act, 25, axis=0),
                np.percentile(per_act, 75, axis=0))

    def get_single_dir_stats(data):
        """For [N, n_angles]: median + IQR over activations."""
        return (np.median(data, axis=0),
                np.percentile(data, 25, axis=0),
                np.percentile(data, 75, axis=0))

    colors = {
        'svd': '#2ca02c',         # green
        'random': '#d62728',      # red
        'orthogonal': '#1f77b4',  # blue
    }

    # Main figure: 2 panels (L2, KL)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, label in [
        (axes[0], 'l2', 'L² distance (penultimate layer)'),
        (axes[1], 'kl', 'KL divergence'),
    ]:
        # SVD direction
        med, q25, q75 = get_single_dir_stats(sweep_results['svd'][metric])
        ax.plot(x, med[mask], color=colors['svd'], lw=2.5, label='SVD top direction')
        ax.fill_between(x, q25[mask], q75[mask], color=colors['svd'], alpha=0.12)

        # Random directions
        med, q25, q75 = get_multi_dir_stats(sweep_results['random'][metric])
        ax.plot(x, med[mask], color=colors['random'], lw=2.5,
                label=f'Random ({args.n_random_dirs} dirs)')
        ax.fill_between(x, q25[mask], q75[mask], color=colors['random'], alpha=0.12)

        # Orthogonal directions
        med, q25, q75 = get_multi_dir_stats(sweep_results['orthogonal'][metric])
        ax.plot(x, med[mask], color=colors['orthogonal'], lw=2.5,
                label=f'Orthogonal to SVD ({args.n_orth_dirs} dirs)')
        ax.fill_between(x, q25[mask], q75[mask], color=colors['orthogonal'], alpha=0.12)

        ax.set_xlabel('Perturbation angle from boundary (degrees)', fontsize=12)
        ax.set_ylabel(f'{label} (vs original)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_angle)

    fig.suptitle(
        f'From boundary (α*={alpha_star_deg:.1f}° along SVD), perturb in different directions\n'
        f'{args.model}, layer {args.layer}, {N} Mandarin activations — '
        f'metrics vs original (unperturbed)',
        fontsize=12)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'boundary_perturbation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # Boundary-finding sweep plot
    fig, ax = plt.subplots(figsize=(8, 5))
    med_kl = np.median(boundary_kl, axis=0)
    q25 = np.percentile(boundary_kl, 25, axis=0)
    q75 = np.percentile(boundary_kl, 75, axis=0)
    ax.plot(bfind_angles_deg, med_kl, 'b-', lw=2)
    ax.fill_between(bfind_angles_deg, q25, q75, alpha=0.2)
    ax.axvline(alpha_star_deg, color='r', ls='--', lw=1.5,
               label=f'α* = {alpha_star_deg:.1f}°')
    ax.set_xlabel('Perturbation angle along SVD direction (degrees)')
    ax.set_ylabel('KL divergence (vs unperturbed)')
    ax.set_title(f'Boundary finding: SVD direction on Mandarin activations\n'
                 f'{args.model}, layer {args.layer}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'boundary_finding.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ===== Summary =====
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Boundary angle α* = {alpha_star_deg:.1f}°")
    print(f"SVD PC1 variance explained: {layer_results['variance_explained'][0]*100:.1f}%")

    for angle_label, idx in [('15°', np.argmin(np.abs(sweep_angles_deg - 15))),
                              ('30°', np.argmin(np.abs(sweep_angles_deg - 30))),
                              ('45°', np.argmin(np.abs(sweep_angles_deg - 45)))]:
        svd_kl = np.median(kl_svd[:, idx])
        svd_l2 = np.median(l2_svd[:, idx])
        rand_kl = np.median(kl_random[:, idx, :].mean(axis=1))
        rand_l2 = np.median(l2_random[:, idx, :].mean(axis=1))
        orth_kl = np.median(kl_orth[:, idx, :].mean(axis=1))
        orth_l2 = np.median(l2_orth[:, idx, :].mean(axis=1))

        print(f"\n  At {angle_label} from boundary (vs original):")
        print(f"    SVD dir:    KL={svd_kl:.4f}, L2={svd_l2:.2f}")
        print(f"    Random:     KL={rand_kl:.4f}, L2={rand_l2:.2f}")
        print(f"    Orthogonal: KL={orth_kl:.4f}, L2={orth_l2:.2f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
