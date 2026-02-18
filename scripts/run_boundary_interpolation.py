#!/usr/bin/env python3
"""
Boundary perturbation from interpolation between two real activations.

Interpolates between 'The house was big' and 'The house was in' activations,
finds the boundary (max KL derivative), then perturbs from the boundary along:
(a) random directions, (b) directions orthogonal to the interpolation direction,
(c) the interpolation direction itself.

Measures L2 (penultimate layer) and KL-div vs the original ('big') activation.

Usage:
    python scripts/run_boundary_interpolation.py --model qwen3-0.6b --layer 14
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
    """Spherically perturb activation by multiple angles along direction.

    Returns [n_angles, d_model] perturbed activations (same norm as original).
    """
    a_norm = activation.norm()
    a_hat = activation / a_norm

    proj = (direction * a_hat).sum()
    u_perp = direction - proj * a_hat
    u_perp = u_perp / (u_perp.norm() + 1e-10)

    cos_t = torch.tensor(np.cos(angles_rad), dtype=activation.dtype,
                         device=device).unsqueeze(1)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=activation.dtype,
                         device=device).unsqueeze(1)

    return a_norm * (cos_t * a_hat.unsqueeze(0) + sin_t * u_perp.unsqueeze(0))


def compute_boundary_activation(activation, direction, angle_rad):
    """Spherically perturb activation by angle_rad along direction."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--prompt_a', default='The house was big')
    parser.add_argument('--prompt_b', default='The house was in')
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--n_random_dirs', type=int, default=5)
    parser.add_argument('--n_orth_dirs', type=int, default=5)
    parser.add_argument('--boundary_angle', type=float, default=None,
                        help='Override boundary angle in degrees (default: auto from max KL derivative)')
    parser.add_argument('--max_search_angle', type=float, default=None,
                        help='Limit boundary search to angles below this (degrees)')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/boundary_interpolation/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Phase 1: Load model & extract activations =====
    print("=" * 60)
    print("PHASE 1: Load model & extract activations")
    print("=" * 60)

    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_blocks = len(_get_blocks(model))
    print(f"Device: {device}, d_model: {d_model}, n_blocks: {n_blocks}")
    print(f"Penultimate layer: block {n_blocks - 2}")

    # Tokenize and show token breakdown
    tok_a = tokenizer(args.prompt_a, return_tensors='pt').to(device)
    tok_b = tokenizer(args.prompt_b, return_tensors='pt').to(device)
    ids_a = tok_a['input_ids'][0]
    ids_b = tok_b['input_ids'][0]
    print(f"  A: '{args.prompt_a}' -> {len(ids_a)} tokens: "
          f"{[tokenizer.decode(t) for t in ids_a]}")
    print(f"  B: '{args.prompt_b}' -> {len(ids_b)} tokens: "
          f"{[tokenizer.decode(t) for t in ids_b]}")

    # Extract hidden states
    with torch.no_grad():
        out_a = model(**tok_a, output_hidden_states=True)
        out_b = model(**tok_b, output_hidden_states=True)

    hidden_a = out_a.hidden_states[args.layer + 1]  # [1, seq_len, d_model]
    hidden_b = out_b.hidden_states[args.layer + 1]

    # Context = prefix hidden states (should be identical due to causal mask)
    context_a = hidden_a[:, :-1, :]
    context_b = hidden_b[:, :-1, :]
    ctx_diff = (context_a - context_b).norm().item()
    print(f"  Context difference norm: {ctx_diff:.6f} (should be ~0)")
    context = context_a  # use A's context

    act_a = hidden_a[0, -1, :]  # [d_model] - last token of 'big'
    act_b = hidden_b[0, -1, :]  # [d_model] - last token of 'in'

    print(f"  ||act_A|| = {act_a.norm():.2f}, ||act_B|| = {act_b.norm():.2f}")
    cos_ab = F.cosine_similarity(act_a.unsqueeze(0), act_b.unsqueeze(0)).item()
    angle_ab = np.degrees(np.arccos(np.clip(cos_ab, -1, 1)))
    print(f"  cos(A, B) = {cos_ab:.4f}, angle(A, B) = {angle_ab:.1f}°")

    # Interpolation direction: component of B orthogonal to A, normalized
    a_hat = act_a / act_a.norm()
    interp_dir = act_b - (act_b @ a_hat) * a_hat
    interp_dir = interp_dir / interp_dir.norm()
    print(f"  Interpolation direction computed (A → B)")

    # ===== Phase 2: Boundary finding =====
    print(f"\n{'='*60}")
    print("PHASE 2: Boundary finding — sweep along interpolation direction")
    print("=" * 60)

    bfind_angles_deg = np.linspace(0, 180, args.n_angles + 1)[1:]
    bfind_angles_rad = np.deg2rad(bfind_angles_deg)

    with torch.no_grad():
        # Reference: activation A
        ref_hidden = torch.cat([context, act_a.unsqueeze(0).unsqueeze(0)], dim=1)
        ref_penult_a, ref_logits_a = forward_penultimate_and_logits(
            model, ref_hidden, args.layer)
        ref_lp_a = F.log_softmax(ref_logits_a, dim=-1)

        # Sweep along interpolation direction from A
        pert_acts = compute_perturbed_batch(act_a, interp_dir, bfind_angles_rad, device)
        ctx_batch = context.expand(len(bfind_angles_deg), -1, -1)
        hidden_batch = torch.cat([ctx_batch, pert_acts.unsqueeze(1)], dim=1)
        _, pert_logits = forward_penultimate_and_logits(
            model, hidden_batch, args.layer)
        pert_lp = F.log_softmax(pert_logits, dim=-1)
        boundary_kl = (ref_lp_a.exp() * (ref_lp_a - pert_lp)).sum(dim=-1).cpu().numpy()

    # Find boundary angle
    dkl = np.diff(boundary_kl) / np.diff(bfind_angles_deg)

    if args.boundary_angle is not None:
        alpha_star_deg = args.boundary_angle
        print(f"Using user-specified boundary angle: {alpha_star_deg:.1f}°")
    else:
        # Optionally limit search range to find first boundary
        if args.max_search_angle is not None:
            search_mask = bfind_angles_deg[:-1] <= args.max_search_angle
            max_idx = np.where(search_mask)[0][np.argmax(dkl[search_mask])]
            print(f"(Searching up to {args.max_search_angle}° for first boundary)")
        else:
            max_idx = np.argmax(dkl)
        alpha_star_deg = (bfind_angles_deg[max_idx] + bfind_angles_deg[max_idx + 1]) / 2
        print(f"  Max KL derivative: {dkl[max_idx]:.4f}")
        print(f"  KL at {bfind_angles_deg[max_idx]:.0f}°: {boundary_kl[max_idx]:.4f}")
        print(f"  KL at {bfind_angles_deg[max_idx+1]:.0f}°: {boundary_kl[max_idx+1]:.4f}")

    alpha_star_rad = np.deg2rad(alpha_star_deg)
    print(f"Boundary angle: {alpha_star_deg:.1f}°")
    print(f"  For reference, A→B angle = {angle_ab:.1f}°")

    # ===== Phase 3: Compute boundary activation =====
    print(f"\n{'='*60}")
    print(f"PHASE 3: Compute boundary activation (α* = {alpha_star_deg:.1f}°)")
    print("=" * 60)

    boundary_act = compute_boundary_activation(act_a, interp_dir, alpha_star_rad)
    cos_a_bnd = F.cosine_similarity(
        act_a.unsqueeze(0), boundary_act.unsqueeze(0)).item()
    print(f"  ||boundary|| = {boundary_act.norm():.2f} (||A|| = {act_a.norm():.2f})")
    print(f"  cos(A, boundary) = {cos_a_bnd:.4f} (expected ~{np.cos(alpha_star_rad):.4f})")

    # ===== Phase 4: Sample directions =====
    print(f"\n{'='*60}")
    print(f"PHASE 4: Sample {args.n_random_dirs} random + "
          f"{args.n_orth_dirs} orthogonal directions")
    print("=" * 60)

    random_dirs = sample_random_directions(
        args.n_random_dirs, d_model, torch.device(device))
    orth_dirs = sample_orthogonal_directions(
        interp_dir, args.n_orth_dirs, d_model, torch.device(device))

    dots_orth = (orth_dirs @ interp_dir.to(device)).abs()
    print(f"  Max |cos(orth, interp_dir)|: {dots_orth.max().item():.2e}")
    dots_rand = (random_dirs @ interp_dir.to(device)).abs()
    print(f"  |cos(random, interp_dir)|: mean={dots_rand.mean():.4f}, "
          f"max={dots_rand.max():.4f}")

    # ===== Phase 5: Perturbation sweeps from boundary =====
    # Include angle 0 to show baseline (boundary vs original)
    sweep_angles_deg = np.linspace(0, 180, args.n_angles + 1)
    sweep_angles_rad = np.deg2rad(sweep_angles_deg)

    print(f"\n{'='*60}")
    print(f"PHASE 5: Perturbation sweeps from boundary "
          f"({len(sweep_angles_deg)} angles, 0-180°)")
    print("=" * 60)

    # Reference is already computed: ref_penult_a, ref_lp_a

    def sweep_single_dir(direction):
        """Sweep from boundary along direction, measure L2 and KL vs A."""
        with torch.no_grad():
            pert_acts = compute_perturbed_batch(
                boundary_act, direction, sweep_angles_rad, device)
            ctx_batch = context.expand(len(sweep_angles_rad), -1, -1)
            hidden_batch = torch.cat([ctx_batch, pert_acts.unsqueeze(1)], dim=1)
            pert_penult, pert_logits = forward_penultimate_and_logits(
                model, hidden_batch, args.layer)
            pert_lp = F.log_softmax(pert_logits, dim=-1)

            l2 = (pert_penult - ref_penult_a).norm(dim=-1).cpu().numpy()
            kl = (ref_lp_a.exp() * (ref_lp_a - pert_lp)).sum(dim=-1).cpu().numpy()
        return kl, l2

    sweep_results = {}

    # (c) Interpolation direction
    print("  Interpolation direction...", end=' ', flush=True)
    t0 = time.time()
    kl_interp, l2_interp = sweep_single_dir(interp_dir)
    sweep_results['interp'] = {'kl': kl_interp, 'l2': l2_interp}
    print(f"{time.time()-t0:.1f}s")

    # (a) Random directions
    print(f"  {args.n_random_dirs} random directions...")
    kl_random = np.zeros((len(sweep_angles_deg), args.n_random_dirs))
    l2_random = np.zeros((len(sweep_angles_deg), args.n_random_dirs))
    for d_idx in range(args.n_random_dirs):
        t0 = time.time()
        kl, l2 = sweep_single_dir(random_dirs[d_idx])
        kl_random[:, d_idx] = kl
        l2_random[:, d_idx] = l2
        print(f"    dir {d_idx+1}/{args.n_random_dirs}: {time.time()-t0:.1f}s")
    sweep_results['random'] = {'kl': kl_random, 'l2': l2_random}

    # (b) Orthogonal directions
    print(f"  {args.n_orth_dirs} orthogonal directions...")
    kl_orth = np.zeros((len(sweep_angles_deg), args.n_orth_dirs))
    l2_orth = np.zeros((len(sweep_angles_deg), args.n_orth_dirs))
    for d_idx in range(args.n_orth_dirs):
        t0 = time.time()
        kl, l2 = sweep_single_dir(orth_dirs[d_idx])
        kl_orth[:, d_idx] = kl
        l2_orth[:, d_idx] = l2
        print(f"    dir {d_idx+1}/{args.n_orth_dirs}: {time.time()-t0:.1f}s")
    sweep_results['orthogonal'] = {'kl': kl_orth, 'l2': l2_orth}

    # ===== Phase 6: Save =====
    all_results = {
        'args': vars(args),
        'prompt_a': args.prompt_a,
        'prompt_b': args.prompt_b,
        'act_a': act_a.cpu(),
        'act_b': act_b.cpu(),
        'interp_dir': interp_dir.cpu(),
        'angle_ab_deg': angle_ab,
        'alpha_star_deg': alpha_star_deg,
        'boundary_finding_kl': boundary_kl,
        'boundary_finding_angles_deg': bfind_angles_deg,
        'sweep_angles_deg': sweep_angles_deg,
        'random_dirs': random_dirs.cpu(),
        'orth_dirs': orth_dirs.cpu(),
        'sweep_results': sweep_results,
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved results to {results_path}")

    # ===== Phase 7: Plots =====
    print(f"\n{'='*60}")
    print("PHASE 7: Plotting")
    print("=" * 60)

    max_angle = 90
    mask = sweep_angles_deg <= max_angle
    x = sweep_angles_deg[mask]

    colors = {
        'interp': '#2ca02c',       # green
        'random': '#d62728',       # red
        'orthogonal': '#1f77b4',   # blue
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, label in [
        (axes[0], 'l2', 'L² distance (penultimate layer)'),
        (axes[1], 'kl', 'KL divergence'),
    ]:
        # Interpolation direction
        data = sweep_results['interp'][metric]
        ax.plot(x, data[mask], color=colors['interp'], lw=2.5,
                label='Interpolation dir (A→B)')

        # Random: mean ± std over directions
        data_rand = sweep_results['random'][metric]
        mean_r = data_rand.mean(axis=1)
        std_r = data_rand.std(axis=1)
        ax.plot(x, mean_r[mask], color=colors['random'], lw=2.5,
                label=f'Random ({args.n_random_dirs} dirs)')
        ax.fill_between(x, (mean_r - std_r)[mask], (mean_r + std_r)[mask],
                        color=colors['random'], alpha=0.15)

        # Orthogonal: mean ± std over directions
        data_orth = sweep_results['orthogonal'][metric]
        mean_o = data_orth.mean(axis=1)
        std_o = data_orth.std(axis=1)
        ax.plot(x, mean_o[mask], color=colors['orthogonal'], lw=2.5,
                label=f'Orthogonal ({args.n_orth_dirs} dirs)')
        ax.fill_between(x, (mean_o - std_o)[mask], (mean_o + std_o)[mask],
                        color=colors['orthogonal'], alpha=0.15)

        ax.set_xlabel('Perturbation angle from boundary (degrees)', fontsize=12)
        ax.set_ylabel(f'{label} (vs original A)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_angle)

    fig.suptitle(
        f'From boundary (α*={alpha_star_deg:.1f}° along A→B), '
        f'perturb in different directions\n'
        f'{args.model}, layer {args.layer} — '
        f'A="{args.prompt_a}" vs B="{args.prompt_b}" '
        f'(angle={angle_ab:.1f}°)\n'
        f'Metrics vs original A', fontsize=11)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'boundary_perturbation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # Boundary-finding plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bfind_angles_deg, boundary_kl, 'b-', lw=2)
    ax.axvline(alpha_star_deg, color='r', ls='--', lw=1.5,
               label=f'boundary α* = {alpha_star_deg:.1f}°')
    ax.axvline(angle_ab, color='gray', ls=':', lw=1.5,
               label=f'A→B angle = {angle_ab:.1f}°')
    ax.set_xlabel('Perturbation angle along A→B direction (degrees)')
    ax.set_ylabel('KL divergence (vs A)')
    ax.set_title(f'Boundary finding: "{args.prompt_a}" → "{args.prompt_b}"\n'
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
    print(f"A: '{args.prompt_a}'")
    print(f"B: '{args.prompt_b}'")
    print(f"Angle A→B: {angle_ab:.1f}°")
    print(f"Boundary α*: {alpha_star_deg:.1f}°")

    for angle_label, idx in [
        ('0° (=boundary)', np.argmin(np.abs(sweep_angles_deg - 0))),
        ('15°', np.argmin(np.abs(sweep_angles_deg - 15))),
        ('30°', np.argmin(np.abs(sweep_angles_deg - 30))),
        ('45°', np.argmin(np.abs(sweep_angles_deg - 45))),
    ]:
        print(f"\n  At {angle_label} from boundary (vs A):")
        print(f"    Interp dir: KL={kl_interp[idx]:.4f}, L2={l2_interp[idx]:.2f}")
        print(f"    Random:     KL={kl_random[idx,:].mean():.4f} "
              f"+/- {kl_random[idx,:].std():.4f}, "
              f"L2={l2_random[idx,:].mean():.2f} +/- {l2_random[idx,:].std():.2f}")
        print(f"    Orthogonal: KL={kl_orth[idx,:].mean():.4f} "
              f"+/- {kl_orth[idx,:].std():.4f}, "
              f"L2={l2_orth[idx,:].mean():.2f} +/- {l2_orth[idx,:].std():.2f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
