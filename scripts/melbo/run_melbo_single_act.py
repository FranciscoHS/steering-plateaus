#!/usr/bin/env python3
"""
Capped MELBO on a SINGLE activation: decompose one point in activation space
into its constituent features by iteratively finding orthogonal directions
that produce KL ≈ target.

Runs until quality drops (reward < 25% of first direction) or hits --max_directions.

Usage:
    python scripts/melbo/run_melbo_single_act.py --max_directions 50
    python scripts/melbo/run_melbo_single_act.py --max_directions 50 --resume
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


def optimize_direction_capped(model, contexts, activations, ref_log_probs,
                               angle_rad, layer, device, d_model,
                               target_kl=1.0,
                               n_steps=300, lr=0.01, n_restarts=5,
                               prev_directions=None):
    """Find direction maximizing capped reward KL * exp(-KL / target_kl).

    Returns:
        best_direction, best_reward, best_history (list of (reward, raw_kl) tuples)
    """
    best_reward = -float('inf')
    best_direction = None
    best_history = None

    for restart in range(n_restarts):
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

            d_normed = direction / direction.norm()
            if prev_directions:
                for pd in prev_directions:
                    d_normed = d_normed - (d_normed @ pd) * pd
                d_normed = d_normed / (d_normed.norm() + 1e-10)

            hidden = build_perturbed_hidden(contexts, activations, d_normed, angle_rad)
            logits = forward_from_layer_allpos(model, hidden, layer)
            pert_log_probs = F.log_softmax(logits, dim=-1)

            kl_per_sample = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            rewards_per_sample = kl_per_sample * torch.exp(-kl_per_sample / target_kl)
            mean_reward = rewards_per_sample.mean()
            mean_kl = kl_per_sample.mean().detach()

            loss = -mean_reward
            loss.backward()
            optimizer.step()

            if prev_directions:
                with torch.no_grad():
                    for pd in prev_directions:
                        direction.data -= (direction.data @ pd) * pd

            reward_val = mean_reward.item()
            kl_val = mean_kl.item()
            history.append((reward_val, kl_val))

            if step % 50 == 0 or step == n_steps - 1:
                print(f"    restart {restart+1}/{n_restarts}, step {step}: "
                      f"reward = {reward_val:.6f}, raw KL = {kl_val:.4f}")

        final_reward = history[-1][0]
        if final_reward > best_reward:
            best_reward = final_reward
            with torch.no_grad():
                d_final = direction / direction.norm()
                if prev_directions:
                    for pd in prev_directions:
                        d_final = d_final - (d_final @ pd) * pd
                    d_final = d_final / d_final.norm()
                best_direction = d_final.clone()
            best_history = history

    return best_direction, best_reward, best_history


def sweep_single_direction(model, contexts, activations, ref_final, ref_log_probs,
                           direction, angles_rad, layer, device):
    """Perturbation sweep for one direction. Returns kl [N, n_angles], l2 [N, n_angles]."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--opt_angle', type=float, default=15.0)
    parser.add_argument('--target_kl', type=float, default=1.0)
    parser.add_argument('--n_act', type=int, default=1,
                        help='Number of activations (1 for single-activation decomposition)')
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--max_directions', type=int, default=50)
    parser.add_argument('--stop_fraction', type=float, default=0.25,
                        help='Stop when reward drops below this fraction of direction 1')
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = (f'results/melbo_single_act/{args.model}'
                            f'_target{args.target_kl}_seed{args.seed}')
    os.makedirs(args.results_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.results_dir, 'checkpoint.pkl')

    # ===== Load model =====
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    # ===== Load activation(s) =====
    print(f"\nLoading {args.n_act} FineWeb sequence(s) (seq_len={args.seq_len}, seed={args.seed})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)

    # Show what text we're decomposing
    for i, tids in enumerate(token_ids_list):
        text = tokenizer.decode(tids)
        print(f"  Activation {i}: '{text}'")

    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Activation shape: {activations.shape}, "
          f"norm: {activations.norm(dim=-1).mean():.2f}")

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
    rewards = []
    histories = []
    start_idx = 0

    if args.resume and os.path.exists(checkpoint_path):
        print(f"\nResuming from {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        directions = [d.to(device) for d in ckpt['directions']]
        rewards = ckpt['rewards']
        histories = ckpt['histories']
        start_idx = len(directions)
        print(f"Loaded {start_idx} directions, continuing from direction {start_idx+1}")

    theoretical_max = args.target_kl * np.exp(-1)
    print(f"\n{'='*60}")
    print(f"SINGLE-ACTIVATION CAPPED MELBO")
    print(f"  {args.n_act} activation(s), d_model={d_model}")
    print(f"  Angle: {args.opt_angle} deg, target KL: {args.target_kl}")
    print(f"  Theoretical max reward: {theoretical_max:.4f}")
    print(f"  Stop: reward < {args.stop_fraction:.0%} of direction 1")
    print(f"  Max directions: {args.max_directions}")
    print(f"  Steps: {args.n_steps}, restarts: {args.n_restarts}")
    print(f"{'='*60}")

    stop_threshold = None  # Set after first direction
    stopped_early = False

    for d_idx in range(start_idx, args.max_directions):
        print(f"\n--- Direction {d_idx+1}/{args.max_directions} ---")
        t0 = time.time()
        direction, best_reward, history = optimize_direction_capped(
            model, contexts_dev, activations_dev, ref_log_probs,
            angle_rad, args.layer, device, d_model,
            target_kl=args.target_kl,
            n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts,
            prev_directions=directions if directions else None
        )
        elapsed = time.time() - t0

        directions.append(direction)
        rewards.append(best_reward)
        histories.append(history)

        final_raw_kl = history[-1][1]
        print(f"  >> reward: {best_reward:.6f} (raw KL: {final_raw_kl:.4f}), {elapsed:.1f}s")

        # Set stop threshold from first direction
        if d_idx == 0 or (start_idx > 0 and stop_threshold is None):
            first_reward = rewards[0]
            stop_threshold = first_reward * args.stop_fraction
            print(f"  Stop threshold set: {stop_threshold:.6f} "
                  f"({args.stop_fraction:.0%} of D1 reward {first_reward:.6f})")

        # Check stopping criterion
        if best_reward < stop_threshold:
            print(f"\n  *** STOPPING: reward {best_reward:.6f} < threshold {stop_threshold:.6f} ***")
            stopped_early = True
            # Keep the direction but mark where we stopped
            break

        # Orthogonality spot-check (last 3 directions)
        check_range = range(max(0, d_idx - 2), d_idx)
        for prev_idx in check_range:
            cos_prev = (direction @ directions[prev_idx]).abs().item()
            if cos_prev > 0.01:
                print(f"  cos(D_{d_idx+1}, D_{prev_idx+1}): {cos_prev:.4f}")

        # Checkpoint
        ckpt = {
            'args': vars(args),
            'directions': [d.cpu() for d in directions],
            'rewards': rewards,
            'histories': histories,
            'stopped_early': False,
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(ckpt, f)

    n_found = len(directions)
    print(f"\n{'='*60}")
    print(f"FOUND {n_found} DIRECTIONS" +
          (" (stopped early)" if stopped_early else " (hit max)"))
    print(f"{'='*60}")

    # Quick summary table
    print(f"\n{'Dir':>4} {'Reward':>10} {'Raw KL':>10} {'% of D1':>10}")
    print("-" * 40)
    for i in range(n_found):
        raw_kl = histories[i][-1][1]
        pct = rewards[i] / rewards[0] * 100
        marker = " <<< STOP" if (stopped_early and i == n_found - 1) else ""
        print(f"  D{i+1:>2} {rewards[i]:>10.6f} {raw_kl:>10.4f} {pct:>9.1f}%{marker}")

    # ===== Evaluation sweep =====
    # Only sweep directions that met threshold (exclude last if stopped early)
    n_good = n_found - 1 if stopped_early else n_found
    angles_deg = np.linspace(0, 180, args.n_angles + 1)[1:]
    angles_rad_sweep = np.deg2rad(angles_deg)

    print(f"\n{'='*60}")
    print(f"EVALUATION SWEEP: {n_good} good directions, {len(angles_deg)} angles")
    print(f"{'='*60}")

    sweep_results = {}
    for i in range(n_good):
        name = f'MELBO_{i+1}'
        print(f"Sweeping: {name}...", end=' ', flush=True)
        t0 = time.time()
        kl, l2 = sweep_single_direction(
            model, contexts_dev, activations_dev, ref_final, ref_log_probs,
            directions[i].cpu(), angles_rad_sweep, args.layer, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}
        print(f"{time.time()-t0:.1f}s")

    # ===== Save =====
    all_results = {
        'args': vars(args),
        'directions': [d.cpu() for d in directions],
        'rewards': rewards,
        'histories': histories,
        'n_good': n_good,
        'stopped_early': stopped_early,
        'angles_deg': angles_deg,
        'sweep_results': sweep_results,
        'token_ids': [t.cpu() for t in token_ids_list],
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved to {results_path}")

    # Save checkpoint with stopped_early flag
    ckpt = {
        'args': vars(args),
        'directions': [d.cpu() for d in directions],
        'rewards': rewards,
        'histories': histories,
        'stopped_early': stopped_early,
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(ckpt, f)

    # ===== Plots =====
    cmap = plt.cm.viridis
    dir_colors = [cmap(i / max(n_good, 1)) for i in range(n_good)]

    # 1. Reward decay curve (the key plot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(range(1, n_found + 1), rewards, 'o-', color='steelblue', markersize=4)
    if stop_threshold is not None:
        ax.axhline(stop_threshold, color='red', ls='--', lw=1.5,
                    label=f'Stop threshold ({args.stop_fraction:.0%} of D1)')
    ax.axhline(theoretical_max, color='gray', ls=':', lw=1,
                label=f'Theoretical max ({theoretical_max:.4f})')
    ax.set_xlabel('Direction index')
    ax.set_ylabel('Capped reward')
    ax.set_title('Reward decay (feature exhaustion)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    raw_kls = [h[-1][1] for h in histories]
    ax.plot(range(1, n_found + 1), raw_kls, 'o-', color='darkorange', markersize=4)
    ax.axhline(args.target_kl, color='black', ls=':', lw=1.5, label=f'target={args.target_kl}')
    ax.set_xlabel('Direction index')
    ax.set_ylabel('Raw KL at optimization angle')
    ax.set_title('Raw KL per direction')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Single-activation MELBO: {n_found} directions '
                 f'({n_good} good)\n{args.model}, layer {args.layer}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'reward_decay.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: reward_decay.png")

    # 2. Sweep curves for good directions
    if n_good > 0 and sweep_results:
        fig, ax = plt.subplots(figsize=(12, 7))
        for i in range(n_good):
            name = f'MELBO_{i+1}'
            data = sweep_results[name]['kl']
            # Single activation: data is [1, n_angles], just plot directly
            curve = data[0] if data.shape[0] == 1 else np.median(data, axis=0)
            raw_kl = histories[i][-1][1]
            alpha = max(0.3, 1.0 - i * 0.015)
            lw = 2.0 if i < 5 else 1.0
            ax.plot(angles_deg, curve, color=dir_colors[i], lw=lw, alpha=alpha,
                    label=f'{name} (KL={raw_kl:.2f})' if i < 10 or i % 5 == 0 else None)
        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel('KL divergence')
        ax.set_title(f'Angle sweep: {n_good} capped MELBO directions')
        ax.legend(fontsize=6, ncol=3, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)
        plt.tight_layout()
        plt.savefig(os.path.join(args.results_dir, 'sweep_all.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: sweep_all.png")

    print(f"\nDone! {n_good} good directions found for single activation.")


if __name__ == '__main__':
    main()
