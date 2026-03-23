#!/usr/bin/env python3
"""Iterative capped MELBO for HuggingFace models (Qwen, Gemma, GPT-2, etc.).

Adapted from run_iterative_melbo_othello.py. Uses src/model.py infrastructure.
Operates on a single activation from FineWeb data.

Each round at a fixed opt_angle:
  1. Find directions via capped MELBO (cosine distance reward) until reward drops
     below reward_fraction * best_reward_this_round
  2. Refine each new direction via cone-constrained local opt (max_drift degrees)

Stops when a round adds zero new directions.

Usage:
    python scripts/melbo/run_iterative_melbo_hf.py --model qwen3-1.7b --inject_layer 7 --measure_layer -2
    python scripts/melbo/run_iterative_melbo_hf.py --model qwen3-1.7b --inject_layer 7 --measure_layer -2 --opt_angle 5
"""

import argparse, os, pickle, sys, time
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer, forward_from_layer_to_layer,
    forward_from_layer_to_final, _get_blocks, _get_final_norm, _get_lm_head,
)
from src.data import load_fineweb_fixed_length


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device, batch_size=64):
    all_contexts = []
    all_activations = []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].float().cpu())
        all_activations.append(hidden[:, -1, :].float().cpu())
    return torch.cat(all_contexts, 0), torch.cat(all_activations, 0)


def forward_to_L(model, ctx, a_pert_batch, inject_layer, measure_layer):
    """Forward from inject_layer to measure_layer. a_pert_batch: [batch, d_model]"""
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(dtype=model_dtype)
    point = a_pert_batch.to(dtype=model_dtype).unsqueeze(1)  # [batch, 1, d]
    if measure_layer is not None:
        out = forward_from_layer_to_layer(model, ctx_d, point, inject_layer, measure_layer)
    else:
        out = forward_from_layer_to_final(model, ctx_d, point, inject_layer)
    return out.float()


def forward_to_logits(model, ctx, a_pert_batch, inject_layer):
    """Forward from inject_layer all the way to logits. a_pert_batch: [batch, d_model]"""
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(dtype=model_dtype)
    point = a_pert_batch.to(dtype=model_dtype).unsqueeze(1)  # [batch, 1, d]
    logits = forward_from_layer(model, ctx_d, point, inject_layer)
    return logits.float()


def compute_reward(reward_type, model, ctx, a_pert_batch, ref_L, ref_logits,
                   inject_layer, measure_layer):
    """Compute reward for a batch of perturbed activations.

    Args:
        reward_type: 'l2', 'cos_dist', or 'kl_div'
        ref_L: reference hidden state at measure_layer [1, d]
        ref_logits: reference log-probs [1, vocab] (only needed for kl_div)
    Returns:
        rewards: [batch] tensor (higher = more plateau-breaking)
    """
    if reward_type == 'kl_div':
        pert_logits = forward_to_logits(model, ctx, a_pert_batch, inject_layer)
        pert_log_probs = F.log_softmax(pert_logits, dim=-1)
        ref_probs = ref_logits.exp()
        kl = (ref_probs * (ref_logits - pert_log_probs)).sum(dim=-1)
        return kl
    else:
        pert_L = forward_to_L(model, ctx, a_pert_batch, inject_layer, measure_layer)
        if reward_type == 'cos_dist':
            return 1 - F.cosine_similarity(ref_L, pert_L, dim=-1)
        else:  # l2
            return (ref_L - pert_L).norm(dim=-1)


def capped_melbo_one_direction(model, ctx, act, ref_L, a_unit, a_norm,
                                angle_rad, target_cos, prev_directions,
                                inject_layer, measure_layer, device, d_model,
                                n_steps=300, lr=0.01, n_restarts=3,
                                uncapped=False, reward_type='l2',
                                ref_logits=None):
    """Find one direction via capped MELBO, orthogonalized to prev_directions.

    Optimizes n_restarts candidates in parallel (batched forward passes)
    with early stopping per candidate.

    reward_type: 'l2', 'cos_dist', or 'kl_div' (only used when uncapped=True)
    ref_logits: reference log-probs [1, vocab] (needed for kl_div)
    """
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    B = n_restarts

    # Initialize B random directions
    directions = torch.randn(B, d_model, device=device)
    for b in range(B):
        directions[b] -= (directions[b] @ a_unit) * a_unit
        for pd in prev_directions:
            directions[b] -= (directions[b] @ pd) * pd
        directions[b] = F.normalize(directions[b], dim=0)
    directions = directions.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([directions], lr=lr)

    # Expand ctx for batch: [1, seq-1, d] -> [B, seq-1, d]
    ctx_batch = ctx.expand(B, -1, -1)
    ref_L_batch = ref_L.expand(B, -1)
    ref_logits_batch = ref_logits.expand(B, -1) if ref_logits is not None else None

    # Early stopping state per candidate
    early_stop_patience = 50
    best_rewards_running = torch.full((B,), -float('inf'), device=device)
    steps_without_improvement = torch.zeros(B, dtype=torch.long, device=device)
    active = torch.ones(B, dtype=torch.bool, device=device)

    best_reward = -float('inf')
    best_direction = None
    best_cos_dist = 0.0

    for step in range(n_steps):
        if not active.any():
            break

        optimizer.zero_grad()

        # Normalize and orthogonalize all candidates
        d = directions / directions.norm(dim=1, keepdim=True)
        d = d - (d @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        for pd in prev_directions:
            d = d - (d @ pd).unsqueeze(1) * pd.unsqueeze(0)
        d = d / (d.norm(dim=1, keepdim=True) + 1e-10)

        # Batched perturbation: [B, d_model]
        a_pert = a_norm * (cos_theta * a_unit.unsqueeze(0) + sin_theta * d)

        if uncapped:
            rewards = compute_reward(reward_type, model, ctx_batch, a_pert,
                                     ref_L_batch, ref_logits_batch,
                                     inject_layer, measure_layer)
            # Also get cos_dist for logging
            if reward_type != 'cos_dist':
                with torch.no_grad():
                    pert_L = forward_to_L(model, ctx_batch, a_pert, inject_layer, measure_layer)
                    cos_dist = 1 - F.cosine_similarity(ref_L_batch, pert_L, dim=-1)
            else:
                cos_dist = rewards
        else:
            pert_L = forward_to_L(model, ctx_batch, a_pert, inject_layer, measure_layer)
            cos_dist = 1 - F.cosine_similarity(ref_L_batch, pert_L, dim=-1)  # [B]
            rewards = cos_dist * torch.exp(-cos_dist / target_cos)  # [B]

        # Only backprop active candidates
        loss = -(rewards * active.float()).sum()
        loss.backward()

        # Zero gradients for stopped candidates
        with torch.no_grad():
            if directions.grad is not None:
                directions.grad[~active] = 0

        optimizer.step()

        # Project back
        with torch.no_grad():
            directions.data -= (directions.data @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
            for pd in prev_directions:
                directions.data -= (directions.data @ pd).unsqueeze(1) * pd.unsqueeze(0)

        # Early stopping check
        with torch.no_grad():
            improved = rewards > best_rewards_running * 1.1
            best_rewards_running = torch.where(improved, rewards, best_rewards_running)
            steps_without_improvement = torch.where(improved,
                torch.zeros_like(steps_without_improvement),
                steps_without_improvement + 1)
            newly_stopped = (steps_without_improvement >= early_stop_patience) & active
            active = active & (steps_without_improvement < early_stop_patience)

            # Track global best
            for b in range(B):
                if rewards[b].item() > best_reward:
                    best_reward = rewards[b].item()
                    d_final = directions[b].data / directions[b].data.norm()
                    d_final = d_final - (d_final @ a_unit) * a_unit
                    for pd in prev_directions:
                        d_final = d_final - (d_final @ pd) * pd
                    d_final = d_final / d_final.norm()
                    best_direction = d_final.clone()
                    best_cos_dist = cos_dist[b].item()

        if step % 100 == 0 or step == n_steps - 1:
            best_b = rewards.argmax().item()
            n_active = active.sum().item()
            print(f"      step {step}: best_reward={rewards[best_b].item():.6f}, "
                  f"cos_dist={cos_dist[best_b].item():.4f}, active={n_active}/{B}")

    return best_direction, best_reward, [(best_reward, best_cos_dist)]


def refine_in_cone(model, ctx, act, ref_L, a_unit, a_norm,
                   d_start, max_drift_deg, angle_rad,
                   inject_layer, measure_layer, device, d_model,
                   n_steps=200, lr=0.01, n_restarts=3,
                   reward_type='cos_dist', ref_logits=None):
    """Refine a direction within a cone of max_drift_deg around d_start."""
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    cos_drift_min = np.cos(np.deg2rad(max_drift_deg))

    best_cd = -1.0
    best_dir = None

    for restart in range(n_restarts):
        if max_drift_deg < 1.0:
            direction = d_start.clone()
        else:
            noise = torch.randn(d_model, device=device)
            noise = noise - (noise @ d_start) * d_start
            noise = F.normalize(noise, dim=0)
            rand_angle = np.deg2rad(max_drift_deg * np.random.uniform(0, 0.5))
            direction = np.cos(rand_angle) * d_start + np.sin(rand_angle) * noise

        direction = F.normalize(direction, dim=0)
        direction = direction.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([direction], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()

            d = direction / direction.norm()
            d = d - (d @ a_unit) * a_unit
            d = d / (d.norm() + 1e-10)

            a_pert = a_norm * (cos_theta * a_unit + sin_theta * d)
            reward = compute_reward(reward_type, model, ctx, a_pert.unsqueeze(0),
                                    ref_L, ref_logits, inject_layer, measure_layer)
            reward = reward.squeeze()

            loss = -reward
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                direction.data = F.normalize(direction.data, dim=0)
                cos_sim = (direction.data @ d_start).item()
                if cos_sim < cos_drift_min:
                    d_parallel = (direction.data @ d_start) * d_start
                    d_perp = direction.data - d_parallel
                    if d_perp.norm() > 1e-8:
                        d_perp = F.normalize(d_perp, dim=0)
                    sin_max = np.sin(np.deg2rad(max_drift_deg))
                    direction.data = cos_drift_min * d_start + sin_max * d_perp
                    direction.data = F.normalize(direction.data, dim=0)

        final_cd = reward.item()
        if final_cd > best_cd:
            best_cd = final_cd
            best_dir = F.normalize(direction.data.clone().detach(), dim=0)

    drift = np.degrees(np.arccos(np.clip((best_dir @ d_start).item(), -1, 1)))
    return best_dir, best_cd, drift


def sweep_dir(model, ctx, act, ref_L, a_unit, a_norm, d_vec,
              inject_layer, measure_layer, device, angles_deg):
    """Sweep a direction across angles, return cosine distance and L2 arrays."""
    angles_rad = np.deg2rad(angles_deg)
    cos_t = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)

    d = d_vec.to(device).float()
    d = d - (d @ a_unit) * a_unit
    d = F.normalize(d, dim=0)
    perturbed = a_norm * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * d[None])
    with torch.no_grad():
        pert_L = forward_to_L(model, ctx, perturbed, inject_layer, measure_layer)
        cos_dist = (1 - F.cosine_similarity(ref_L, pert_L, dim=-1)).cpu().numpy()
        l2_dist = (pert_L - ref_L).norm(dim=-1).cpu().numpy()
    return cos_dist, l2_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--inject_layer', type=int, default=7)
    parser.add_argument('--measure_layer', type=int, default=-2)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--act_idx', type=int, default=0,
                        help='Which activation to use (index into FineWeb samples)')
    parser.add_argument('--n_load', type=int, default=100,
                        help='Number of FineWeb sequences to load')

    # MELBO config
    parser.add_argument('--opt_angle', type=float, default=5.0)
    parser.add_argument('--angle_step', type=float, default=0,
                        help='Increment opt_angle by this when a round finds 0 dirs (0=stop instead)')
    parser.add_argument('--max_opt_angle', type=float, default=40.0,
                        help='Stop escalating angle beyond this')
    parser.add_argument('--target_cos', type=float, default=0.1)
    parser.add_argument('--uncapped', action='store_true',
                        help='Use raw L2 distance as reward instead of capped cosine reward')
    parser.add_argument('--reward_type', type=str, default='l2',
                        choices=['l2', 'cos_dist', 'kl_div'],
                        help='Reward type for uncapped mode (default: l2)')
    parser.add_argument('--reward_fraction', type=float, default=0.1)
    parser.add_argument('--stop_multiplier', type=float, default=5.0,
                        help='For --uncapped: stop when reward < multiplier * median_random_l2 (default 5.0)')
    parser.add_argument('--n_random_baseline', type=int, default=30,
                        help='Number of random directions to sweep for baseline (used with --uncapped)')
    parser.add_argument('--seed_directions', type=str, default=None,
                        help='Path to results.pkl; loads directions as frozen seeds for orthogonality')

    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--n_restarts', type=int, default=8,
                        help='Number of parallel candidates per direction (batched)')
    parser.add_argument('--lr', type=float, default=0.01)

    # Local opt config
    parser.add_argument('--refine_steps', type=int, default=200)
    parser.add_argument('--refine_restarts', type=int, default=3)
    parser.add_argument('--max_drift', type=float, default=5.0)
    parser.add_argument('--refine_lr', type=float, default=0.01)

    # Sweep config
    parser.add_argument('--max_sweep_angle', type=float, default=60.0)
    parser.add_argument('--n_sweep_angles', type=int, default=60)
    parser.add_argument('--n_rand', type=int, default=20)

    parser.add_argument('--ckpt_every', type=int, default=10)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--opt_seed', type=int, default=None,
                        help='Separate seed for optimization randomness (default: same as --seed)')
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = (f'results/iterative_melbo_hf/{args.model}'
                           f'_L{args.inject_layer}_meas{args.measure_layer}'
                           f'_angle{args.opt_angle}')
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    print(f"Loading model: {args.model}" + (f" (dtype={args.dtype})" if args.dtype else ""))
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))

    # Resolve negative measure_layer
    if args.measure_layer < 0:
        args.measure_layer = n_layers + args.measure_layer
    print(f"Device: {device}, d_model: {d_model}, n_layers: {n_layers}, "
          f"measure_layer: {args.measure_layer}")

    # Load data and extract activation
    print(f"Loading {args.n_load} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(args.n_load, tokenizer,
                                                seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.inject_layer} activations...")
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.inject_layer, device)

    ctx = all_contexts[args.act_idx].unsqueeze(0).to(device)  # [1, seq-1, d]
    act = all_activations[args.act_idx].to(device)             # [d]
    a_norm = act.norm()
    a_unit = act / a_norm
    print(f"Using activation {args.act_idx}, norm: {a_norm:.2f}")

    # Reseed for optimization if opt_seed specified
    if args.opt_seed is not None:
        torch.manual_seed(args.opt_seed)
        np.random.seed(args.opt_seed)
        print(f"Optimization seed: {args.opt_seed}")

    # Reference at measure layer
    with torch.no_grad():
        ref_L = forward_to_L(model, ctx, act.unsqueeze(0),
                             args.inject_layer, args.measure_layer)

    # Reference logits (needed for kl_div reward)
    ref_logits = None
    if args.reward_type == 'kl_div':
        with torch.no_grad():
            ref_logits = F.log_softmax(
                forward_to_logits(model, ctx, act.unsqueeze(0), args.inject_layer), dim=-1)

    current_angle = args.opt_angle
    angle_rad = np.deg2rad(current_angle)

    # === Compute random baseline for uncapped stopping ===
    uncapped_stop_threshold = None
    if args.uncapped:
        print(f"\nSweeping {args.n_random_baseline} random directions for baseline "
              f"(reward_type={args.reward_type})...")
        rand_rewards_at_angle = []
        for i in range(args.n_random_baseline):
            d_rand = torch.randn(d_model, device=device)
            d_rand = d_rand - (d_rand @ a_unit) * a_unit
            d_rand = F.normalize(d_rand, dim=0)
            a_pert = a_norm * (np.cos(angle_rad) * a_unit + np.sin(angle_rad) * d_rand)
            with torch.no_grad():
                r = compute_reward(args.reward_type, model, ctx, a_pert.unsqueeze(0),
                                   ref_L, ref_logits, args.inject_layer, args.measure_layer)
            rand_rewards_at_angle.append(r.item())
        median_rand = np.median(rand_rewards_at_angle)
        uncapped_stop_threshold = args.stop_multiplier * median_rand
        print(f"  Median random {args.reward_type} at {current_angle:.1f}°: {median_rand:.4f}")
        print(f"  Stop threshold ({args.stop_multiplier}x): {uncapped_stop_threshold:.4f}")

    # === Iterative MELBO ===
    all_directions = []
    round_logs = []
    rnd = 0
    best_reward_ever = None

    # Load seed directions if provided (frozen — used for orthogonality only)
    n_seed = 0
    if args.seed_directions:
        seed_data = pickle.load(open(args.seed_directions, 'rb'))
        seed_dirs = seed_data['directions']
        for d in seed_dirs:
            all_directions.append(d.to(device).float())
        n_seed = len(all_directions)
        print(f"Loaded {n_seed} seed directions from {args.seed_directions}")

    while True:
        rnd += 1
        print(f"\n{'='*60}")
        print(f"ROUND {rnd} — {len(all_directions)} directions so far, "
              f"opt_angle={current_angle:.1f}°")
        print(f"{'='*60}")

        round_info = {'round': rnd, 'opt_angle': current_angle,
                      'pre_refine': [], 'post_refine': []}
        new_directions = []
        d_idx = 0

        while True:
            d_idx += 1
            print(f"\n  --- Direction {d_idx} "
                  f"(global #{len(all_directions)+len(new_directions)+1}) ---")
            prev = all_directions + new_directions
            direction, reward, history = capped_melbo_one_direction(
                model, ctx, act, ref_L, a_unit, a_norm,
                angle_rad, args.target_cos, prev,
                args.inject_layer, args.measure_layer, device, d_model,
                n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts,
                uncapped=args.uncapped, reward_type=args.reward_type,
                ref_logits=ref_logits)

            if best_reward_ever is None or reward > best_reward_ever:
                best_reward_ever = reward

            print(f"    reward={reward:.6f}, cos_dist={history[-1][1]:.4f}, "
                  f"ratio={reward/best_reward_ever:.3f}")

            if args.uncapped and uncapped_stop_threshold is not None:
                if reward < uncapped_stop_threshold:
                    print(f"    Reward {reward:.1f} below {args.stop_multiplier}x median random "
                          f"({uncapped_stop_threshold:.1f}), stopping round")
                    break
            elif reward < args.reward_fraction * best_reward_ever:
                print(f"    Reward dropped below {args.reward_fraction:.0%} of "
                      f"best ever ({best_reward_ever:.6f}), stopping round")
                break

            new_directions.append(direction)
            round_info['pre_refine'].append({
                'reward': reward, 'cos_dist': history[-1][1],
                'direction': direction.cpu(),
            })

            if len(new_directions) % args.ckpt_every == 0:
                mid_ckpt = {
                    'round': rnd,
                    'directions': [d.cpu() for d in all_directions + new_directions],
                    'round_logs': round_logs + [round_info],
                    'best_reward_ever': best_reward_ever,
                    'args': vars(args),
                    'mid_round': True,
                }
                mid_path = os.path.join(args.results_dir, f'checkpoint_round{rnd}_mid.pkl')
                with open(mid_path, 'wb') as f:
                    pickle.dump(mid_ckpt, f)
                print(f"    Mid-round checkpoint: {len(all_directions)+len(new_directions)} dirs")

        n_new = len(new_directions)
        print(f"\n  Round {rnd}: found {n_new} directions")

        if n_new == 0:
            if args.angle_step > 0 and current_angle + args.angle_step <= args.max_opt_angle:
                current_angle += args.angle_step
                angle_rad = np.deg2rad(current_angle)
                best_reward_ever = None  # reset for new angle
                print(f"  Escalating to opt_angle={current_angle:.1f}°")
                # Recompute baseline at new angle
                if args.uncapped:
                    rand_rewards_at_angle = []
                    for i in range(args.n_random_baseline):
                        d_rand = torch.randn(d_model, device=device)
                        d_rand = d_rand - (d_rand @ a_unit) * a_unit
                        d_rand = F.normalize(d_rand, dim=0)
                        a_pert = a_norm * (np.cos(angle_rad) * a_unit + np.sin(angle_rad) * d_rand)
                        with torch.no_grad():
                            r = compute_reward(args.reward_type, model, ctx, a_pert.unsqueeze(0),
                                               ref_L, ref_logits, args.inject_layer, args.measure_layer)
                        rand_rewards_at_angle.append(r.item())
                    median_rand = np.median(rand_rewards_at_angle)
                    uncapped_stop_threshold = args.stop_multiplier * median_rand
                    print(f"  New baseline: median={median_rand:.4f}, threshold={uncapped_stop_threshold:.4f}")
                round_logs.append(round_info)
                continue
            else:
                print("  No new directions found, stopping")
                round_logs.append(round_info)
                break

        # Phase 2: Refine
        print(f"\n  --- Local optimization (drift cap={args.max_drift}°) ---")
        refined_directions = []
        for i, d_start in enumerate(new_directions):
            refine_rt = args.reward_type if args.uncapped else 'cos_dist'
            refined, cd, drift = refine_in_cone(
                model, ctx, act, ref_L, a_unit, a_norm,
                d_start, args.max_drift, angle_rad,
                args.inject_layer, args.measure_layer, device, d_model,
                n_steps=args.refine_steps, lr=args.refine_lr,
                n_restarts=args.refine_restarts,
                reward_type=refine_rt, ref_logits=ref_logits)
            refined_directions.append(refined)
            print(f"    Dir {i+1}: cd {round_info['pre_refine'][i]['cos_dist']:.4f}"
                  f" → {cd:.4f}, drift={drift:.1f}°")
            round_info['post_refine'].append({
                'cd': cd, 'drift': drift,
                'direction': refined.cpu(),
            })

        all_directions.extend(refined_directions)
        round_logs.append(round_info)

        # Checkpoint
        ckpt = {
            'round': rnd,
            'directions': [d.cpu() for d in all_directions],
            'round_logs': round_logs,
            'best_reward_ever': best_reward_ever,
            'args': vars(args),
        }
        ckpt_path = os.path.join(args.results_dir, f'checkpoint_round{rnd}.pkl')
        with open(ckpt_path, 'wb') as f:
            pickle.dump(ckpt, f)
        print(f"  Checkpointed: {ckpt_path}")

    n_total = len(all_directions)
    print(f"\n{'='*60}")
    print(f"DONE: {n_total} directions across {rnd} rounds")
    print(f"{'='*60}")

    # Pairwise cosines
    if n_total > 0:
        D = torch.stack(all_directions)
        G = (D @ D.T).abs()
        G.fill_diagonal_(0)
        print(f"Max |cos_sim|: {G.max().item():.4f}")
        print(f"Mean |cos_sim|: {G.sum().item() / max(n_total * (n_total - 1), 1):.6f}")
    else:
        G = torch.zeros(0, 0)

    # === Sweep ===
    if n_total == 0:
        print("No directions found, skipping sweep")
        results = {'directions': [], 'round_logs': round_logs, 'args': vars(args)}
        with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        return

    angles_deg = np.linspace(0, args.max_sweep_angle, args.n_sweep_angles + 1)[1:]

    print("\nSweeping MELBO directions...")
    melbo_cos = np.zeros((n_total, len(angles_deg)))
    melbo_l2 = np.zeros((n_total, len(angles_deg)))
    for i, d in enumerate(all_directions):
        melbo_cos[i], melbo_l2[i] = sweep_dir(
            model, ctx, act, ref_L, a_unit, a_norm, d,
            args.inject_layer, args.measure_layer, device, angles_deg)

    print("Sweeping random directions...")
    rand_dirs = F.normalize(torch.randn(args.n_rand, d_model), dim=-1)
    rand_cos = np.zeros((args.n_rand, len(angles_deg)))
    rand_l2 = np.zeros((args.n_rand, len(angles_deg)))
    for k in range(args.n_rand):
        rand_cos[k], rand_l2[k] = sweep_dir(
            model, ctx, act, ref_L, a_unit, a_norm,
            rand_dirs[k], args.inject_layer,
            args.measure_layer, device, angles_deg)

    # Build round_boundaries
    round_boundaries = []
    cumulative = 0
    for rl in round_logs:
        n_in_round = len(rl['post_refine'])
        cumulative += n_in_round
        round_boundaries.append(cumulative)

    # Save
    results = {
        'directions': [d.cpu() for d in all_directions],
        'round_logs': round_logs,
        'round_boundaries': round_boundaries,
        'angles_deg': angles_deg,
        'melbo_cos': melbo_cos,
        'melbo_l2': melbo_l2,
        'rand_cos': rand_cos,
        'rand_l2': rand_l2,
        'gram_matrix': G.cpu().numpy(),
        'args': vars(args),
    }
    save_path = os.path.join(args.results_dir, 'results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    # Plot
    n_rounds_with_dirs = len([rl for rl in round_logs if len(rl['post_refine']) > 0])
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax_idx, (metric, metric_name, melbo_data, rand_data) in enumerate([
        ('cos', 'Cosine distance', melbo_cos, rand_cos),
        ('l2', 'L2 distance', melbo_l2, rand_l2),
    ]):
        ax = axes[ax_idx]
        rand_med = np.median(rand_data, axis=0)
        rand_q25 = np.percentile(rand_data, 25, axis=0)
        rand_q75 = np.percentile(rand_data, 75, axis=0)
        ax.fill_between(angles_deg, rand_q25, rand_q75, color='C0', alpha=0.15)
        ax.plot(angles_deg, rand_med, color='C0', lw=2, label='Random')

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, max(n_rounds_with_dirs, 1)))
        dir_idx = 0
        color_idx = 0
        for rl in round_logs:
            n_in_round = len(rl['post_refine'])
            if n_in_round == 0:
                continue
            for j in range(n_in_round):
                ax.plot(angles_deg, melbo_data[dir_idx], color=colors[color_idx],
                        lw=1.5, alpha=0.6,
                        label=f'Round {rl["round"]}' if j == 0 else None)
                dir_idx += 1
            color_idx += 1

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(metric_name)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{metric_name}')

    # Right: Gram matrix
    ax = axes[2]
    im = ax.imshow(G.cpu().numpy(), cmap='RdBu_r', vmin=0, vmax=0.3)
    plt.colorbar(im, ax=ax)
    for b in round_boundaries[:-1]:
        ax.axhline(b - 0.5, color='white', lw=1)
        ax.axvline(b - 0.5, color='white', lw=1)
    ax.set_title(f'|cos_sim| ({n_total} dirs)')

    plt.suptitle(f'{args.model} L{args.inject_layer}→L{args.measure_layer}: '
                 f'{n_total} dirs in {rnd} rounds, '
                 f'opt@{args.opt_angle}°, drift={args.max_drift}°',
                 fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'iterative_melbo.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {plot_path}")
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    main()
