#!/usr/bin/env python3
"""
Unconstrained MELBO on diverse activations: find k orthogonal directions in R^d_model
that maximize perturbation response (KL divergence) on random FineWeb inputs.

Unlike run_melbo_unconstrained.py, this is standalone — no dependency on bilingual
data or constrained results. The goal is to discover interpretable feature directions
purely from perturbation response on diverse data.

Checkpoints after each direction found, so long runs can be resumed.

Usage:
    python scripts/run_melbo_diverse.py --model qwen3-0.6b --layer 14 --n_directions 10
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


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device,
                                   batch_size=64):
    """Extract layer-L context (prefix positions) and final-token activation."""
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
    """Spherical perturbation: rotate activation toward direction by angle_rad."""
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms
    proj = (direction.unsqueeze(0) * a_unit).sum(dim=-1, keepdim=True)
    u_perp = direction.unsqueeze(0) - proj * a_unit
    u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)
    a_pert = a_norms * (cos_t * a_unit + sin_t * u_perp)
    return torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)


def optimize_direction_unconstrained(model, contexts, activations, ref_log_probs,
                                     angle_rad, layer, device, d_model,
                                     n_steps=300, lr=0.01, n_restarts=5,
                                     prev_directions=None):
    """Find direction in R^d_model maximizing mean KL divergence.

    Optimizes directly in full activation space with Gram-Schmidt orthogonalization
    against previously found directions.

    Args:
        prev_directions: list of [d_model] unit vectors to be orthogonal to

    Returns:
        best_direction, best_kl, best_history
    """
    best_kl = -float('inf')
    best_direction = None
    best_history = None

    for restart in range(n_restarts):
        # Random init, projected orthogonal to previous directions
        direction = torch.randn(d_model, device=device)
        if prev_directions:
            for pd in prev_directions:
                direction = direction - (direction @ pd) * pd
        direction = F.normalize(direction, dim=0)
        direction = direction.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([direction], lr=lr)
        history = []

        for step in range(n_steps):
            optimizer.zero_grad()

            # Normalize + orthogonalize (differentiable)
            d_normed = direction / direction.norm()
            if prev_directions:
                for pd in prev_directions:
                    d_normed = d_normed - (d_normed @ pd) * pd
                d_normed = d_normed / (d_normed.norm() + 1e-10)

            # Perturbed forward pass
            hidden = build_perturbed_hidden(contexts, activations, d_normed, angle_rad)
            logits = forward_from_layer_allpos(model, hidden, layer)
            pert_log_probs = F.log_softmax(logits, dim=-1)

            # KL(ref || pert)
            kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1).mean()

            loss = -kl
            loss.backward()
            optimizer.step()

            # Project to orthogonal subspace after step
            if prev_directions:
                with torch.no_grad():
                    for pd in prev_directions:
                        direction.data -= (direction.data @ pd) * pd

            kl_val = kl.item()
            history.append(kl_val)

            if step % 50 == 0 or step == n_steps - 1:
                print(f"    restart {restart+1}/{n_restarts}, step {step}: "
                      f"KL = {kl_val:.6f}")

        final_kl = history[-1]
        if final_kl > best_kl:
            best_kl = final_kl
            with torch.no_grad():
                d_final = direction / direction.norm()
                if prev_directions:
                    for pd in prev_directions:
                        d_final = d_final - (d_final @ pd) * pd
                    d_final = d_final / d_final.norm()
                best_direction = d_final.clone()
            best_history = history

    return best_direction, best_kl, best_history


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
                pert_final, pert_logits = forward_to_hidden_and_logits(model, hidden, layer)
                pert_log_probs = F.log_softmax(pert_logits, dim=-1)
                l2 = (pert_final - ref_final).norm(dim=-1)
                kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
                l2_dists[:, ang_idx, dir_idx] = l2.cpu().numpy()
                kl_divs[:, ang_idx, dir_idx] = kl.cpu().numpy()
    return kl_divs, l2_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--opt_angle', type=float, default=15.0,
                        help='Angle in degrees for optimization')
    parser.add_argument('--n_act', type=int, default=50,
                        help='Number of diverse FineWeb activations')
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_directions', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_angles', type=int, default=60,
                        help='Number of angles for evaluation sweep')
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if it exists')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/melbo_diverse/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.results_dir, 'checkpoint.pkl')

    # ===== Load model =====
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    # ===== Load diverse activations =====
    print(f"\nLoading {args.n_act} FineWeb sequences (seq_len={args.seq_len}, seed={args.seed})...")
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

    # ===== Optimization =====
    angle_rad = np.deg2rad(args.opt_angle)

    # Resume from checkpoint if requested
    directions = []
    kls = []
    histories = []
    start_idx = 0

    if args.resume and os.path.exists(checkpoint_path):
        print(f"\nResuming from {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        directions = [d.to(device) for d in ckpt['directions']]
        kls = ckpt['kls']
        histories = ckpt['histories']
        start_idx = len(directions)
        print(f"Loaded {start_idx} directions, continuing from direction {start_idx+1}")

    print(f"\n{'='*60}")
    print(f"UNCONSTRAINED MELBO on diverse activations")
    print(f"Finding top-{args.n_directions} directions in R^{d_model}")
    print(f"Angle: {args.opt_angle} deg, steps: {args.n_steps}, restarts: {args.n_restarts}")
    print(f"{'='*60}")

    for d_idx in range(start_idx, args.n_directions):
        print(f"\n--- Direction {d_idx+1}/{args.n_directions} ---")
        t0 = time.time()
        direction, best_kl, history = optimize_direction_unconstrained(
            model, contexts_dev, activations_dev, ref_log_probs,
            angle_rad, args.layer, device, d_model,
            n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts,
            prev_directions=directions if directions else None
        )
        elapsed = time.time() - t0

        directions.append(direction)
        kls.append(best_kl)
        histories.append(history)

        print(f"  Best KL: {best_kl:.6f}")
        print(f"  Time: {elapsed:.1f}s")

        # Orthogonality check
        for prev_idx in range(d_idx):
            cos_prev = (direction @ directions[prev_idx]).abs().item()
            print(f"  cos(dir_{d_idx+1}, dir_{prev_idx+1}): {cos_prev:.6f}")

        # Checkpoint after each direction
        ckpt = {
            'args': vars(args),
            'directions': [d.cpu() for d in directions],
            'kls': kls,
            'histories': histories,
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(ckpt, f)
        print(f"  Checkpoint saved ({d_idx+1}/{args.n_directions} directions)")

    # ===== Evaluation sweep =====
    angles_deg = np.linspace(0, 180, args.n_angles + 1)[1:]
    angles_rad_sweep = np.deg2rad(angles_deg)

    print(f"\n{'='*60}")
    print(f"EVALUATION SWEEP: {len(angles_deg)} angles, 0-180 deg")
    print(f"{'='*60}")

    sweep_results = {}

    # Sweep each MELBO direction
    for i, d in enumerate(directions):
        name = f'MELBO_{i+1}'
        print(f"Sweeping: {name}...", end=' ', flush=True)
        t0 = time.time()
        kl, l2 = sweep_single_direction(
            model, contexts_dev, activations_dev, ref_final, ref_log_probs,
            d.cpu(), angles_rad_sweep, args.layer, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}
        print(f"{time.time()-t0:.1f}s")

    # Random baseline
    print(f"Sweeping: random ({args.n_random_dirs} dirs/act)...", end=' ', flush=True)
    t0 = time.time()
    rand_kl, rand_l2 = sweep_random_directions(
        model, contexts_dev, activations_dev, ref_final, ref_log_probs,
        angles_rad_sweep, args.layer, device, n_dirs=args.n_random_dirs)
    sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}
    print(f"{time.time()-t0:.1f}s")

    # ===== Save =====
    all_results = {
        'args': vars(args),
        'directions': [d.cpu() for d in directions],
        'kls': kls,
        'histories': histories,
        'angles_deg': angles_deg,
        'sweep_results': sweep_results,
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved to {results_path}")

    # ===== Plots =====

    # Use a colormap that works for many directions
    cmap = plt.cm.tab10 if args.n_directions <= 10 else plt.cm.tab20
    dir_colors = [cmap(i / max(args.n_directions, 1)) for i in range(args.n_directions)]

    # 1. Sweep comparison (all directions + random)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
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

        # MELBO directions
        for i in range(args.n_directions):
            name = f'MELBO_{i+1}'
            data = sweep_results[name][key]
            median = np.median(data, axis=0)
            # Thicker lines for top directions, thinner for later ones
            lw = 2.5 if i < 3 else 1.5
            alpha = 1.0 if i < 5 else 0.7
            ax.plot(angles_deg, median, color=dir_colors[i], lw=lw, alpha=alpha,
                    label=f'{name} (KL={kls[i]:.4f})' if key == 'kl' else name)

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=7, ncol=2 if args.n_directions > 5 else 1)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)

    fig.suptitle(f'Unconstrained MELBO on diverse activations\n'
                 f'{args.model}, layer {args.layer}, {args.n_act} activations, '
                 f'opt at {args.opt_angle} deg', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'sweep_all.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: sweep_all.png")

    # 2. KL at optimization angle (bar chart — quick summary of direction quality)
    opt_ang_idx = np.argmin(np.abs(angles_deg - args.opt_angle))
    rand_kl_at_opt = np.median(sweep_results['random']['kl'][:, opt_ang_idx, :])

    fig, ax = plt.subplots(figsize=(max(8, args.n_directions * 0.8), 5))
    x = np.arange(args.n_directions)
    bars = ax.bar(x, kls, color=dir_colors, edgecolor='black', linewidth=0.5)
    ax.axhline(rand_kl_at_opt, color='gray', ls='--', lw=2, label=f'Random median')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{i+1}' for i in range(args.n_directions)])
    ax.set_xlabel('Direction')
    ax.set_ylabel(f'KL divergence at {args.opt_angle} deg')
    ax.set_title(f'MELBO direction quality (optimization objective)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'direction_kls.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: direction_kls.png")

    # 3. Optimization histories
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(args.n_directions):
        ax.plot(histories[i], color=dir_colors[i], label=f'D{i+1} (final={kls[i]:.4f})',
                alpha=0.8)
    ax.set_xlabel('Optimization step')
    ax.set_ylabel('KL divergence')
    ax.set_title(f'Optimization convergence ({args.opt_angle} deg)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'optimization_history.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: optimization_history.png")

    # 4. Pairwise cosine similarity matrix
    n = args.n_directions
    cos_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_matrix[i, j] = (directions[i] @ directions[j]).abs().item()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cos_matrix, vmin=0, vmax=1, cmap='Reds')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f'D{i+1}' for i in range(n)])
    ax.set_yticklabels([f'D{i+1}' for i in range(n)])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{cos_matrix[i,j]:.2f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax, label='|cosine similarity|')
    ax.set_title('Pairwise direction similarity')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'cosine_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: cosine_matrix.png")

    # ===== Summary =====
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Random median KL at {args.opt_angle} deg: {rand_kl_at_opt:.6f}")
    for i in range(args.n_directions):
        ratio = kls[i] / rand_kl_at_opt if rand_kl_at_opt > 0 else float('inf')
        print(f"  MELBO_{i+1}: KL={kls[i]:.6f} ({ratio:.1f}x random)")

    print("\nDone!")


if __name__ == '__main__':
    main()
