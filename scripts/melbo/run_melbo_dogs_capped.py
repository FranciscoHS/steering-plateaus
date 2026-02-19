#!/usr/bin/env python3
"""
Capped MELBO on dog breed prompts: find feature-like directions.

Instead of maximizing raw KL (which finds degenerate "model-breaking" directions),
uses a capped objective: KL * exp(-KL / target). This peaks at KL = target,
actively penalizing directions that cause too much disruption. The result should
be directions that break the plateau gradually — more like actual feature directions
than the colon/parenthesis directions found by uncapped MELBO.

Usage:
    python scripts/run_melbo_dogs_capped.py [--model qwen3-0.6b] [--layer 14] [--target_kl 1.0]
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
)
from data.animal_prompts import DOGS


def extract_prompt_activations(model, tokenizer, prompts, layer, device):
    """Extract context and last-token activation for each prompt.

    Returns lists (not stacked tensors) because prompts have different lengths.
    """
    contexts = []
    activations = []

    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        contexts.append(hidden[:, :-1, :])   # [1, seq_len-1, d]
        activations.append(hidden[:, -1, :].squeeze(0))  # [d]

    return contexts, activations


def build_perturbed_hidden_single(context, activation, direction, angle_rad):
    """Build perturbed hidden state for a single prompt (spherical perturbation)."""
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    a_norm = activation.norm()
    a_unit = activation / a_norm

    proj = (direction * a_unit).sum()
    u_perp = direction - proj * a_unit
    u_perp = u_perp / (u_perp.norm() + 1e-10)

    a_pert = a_norm * (cos_t * a_unit + sin_t * u_perp)
    return torch.cat([context, a_pert.unsqueeze(0).unsqueeze(0)], dim=1)


def optimize_direction(model, contexts, activations, ref_log_probs_list,
                       angle_rad, layer, device, d_model,
                       target_kl=1.0,
                       n_steps=200, lr=0.01, n_restarts=3,
                       prev_directions=None):
    """Find direction in R^d_model that produces KL ≈ target_kl (feature-seeking).

    Uses capped objective: per-prompt reward = KL * exp(-KL / target_kl).
    This peaks at KL = target_kl and penalizes directions that overshoot,
    preventing degenerate "model-breaking" solutions.
    """
    N = len(activations)
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

            # Accumulate capped reward over all prompts
            total_reward = torch.tensor(0.0, device=device)
            total_kl = torch.tensor(0.0, device=device)
            for i in range(N):
                hidden = build_perturbed_hidden_single(
                    contexts[i], activations[i], d_normed, angle_rad)
                logits = forward_from_layer_allpos(model, hidden, layer)
                pert_log_probs = F.log_softmax(logits, dim=-1)
                kl = (ref_log_probs_list[i].exp() *
                      (ref_log_probs_list[i] - pert_log_probs)).sum(dim=-1).squeeze()
                # Capped reward: peaks at kl = target_kl, penalizes overshoot
                reward = kl * torch.exp(-kl / target_kl)
                total_reward = total_reward + reward
                total_kl = total_kl + kl.detach()

            mean_reward = total_reward / N
            mean_kl = total_kl / N
            loss = -mean_reward
            loss.backward()
            optimizer.step()

            # Project to orthogonal subspace after step
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


def interpret_direction(model, tokenizer, contexts, activations, ref_log_probs_list,
                        prompts, direction, angle_deg, layer, device, top_k=10):
    """Show what tokens shift when perturbing along direction.

    For each prompt, shows tokens most promoted and demoted by the perturbation.
    """
    angle_rad = np.deg2rad(angle_deg)

    for sign, label in [(1.0, "+"), (-1.0, "-")]:
        print(f"\n  Direction {label} at {angle_deg} deg:")
        effective_angle = sign * angle_rad

        for i, prompt in enumerate(prompts):
            with torch.no_grad():
                hidden = build_perturbed_hidden_single(
                    contexts[i], activations[i], direction, effective_angle)
                logits = forward_from_layer_allpos(model, hidden, layer)
                pert_lp = F.log_softmax(logits, dim=-1).squeeze()

            ref_lp = ref_log_probs_list[i].squeeze()
            diff = pert_lp - ref_lp  # positive = promoted, negative = demoted

            # Top promoted tokens
            top_vals, top_ids = diff.topk(5)
            promoted = [(tokenizer.decode([tid]).strip(), v.item())
                        for tid, v in zip(top_ids, top_vals)]

            # Top demoted tokens
            bot_vals, bot_ids = (-diff).topk(5)
            demoted = [(tokenizer.decode([tid]).strip(), -v.item())
                       for tid, v in zip(bot_ids, bot_vals)]

            kl = (ref_lp.exp() * (ref_lp - pert_lp)).sum().item()

            prom_str = ", ".join(f"'{t}' ({v:+.2f})" for t, v in promoted[:3])
            dem_str = ", ".join(f"'{t}' ({v:+.2f})" for t, v in demoted[:3])
            print(f"    '{prompt}' (KL={kl:.3f})")
            print(f"      promoted: {prom_str}")
            print(f"      demoted:  {dem_str}")


def sweep_kl(model, contexts, activations, ref_log_probs_list, direction,
             angles_rad, layer, device):
    """Sweep perturbation angles for one direction. Returns [N, n_angles] KL array."""
    N = len(activations)
    n_angles = len(angles_rad)
    kl_vals = np.zeros((N, n_angles))

    for ang_idx, theta in enumerate(angles_rad):
        for i in range(N):
            with torch.no_grad():
                hidden = build_perturbed_hidden_single(
                    contexts[i], activations[i], direction, theta)
                logits = forward_from_layer_allpos(model, hidden, layer)
                pert_lp = F.log_softmax(logits, dim=-1)
                kl = (ref_log_probs_list[i].exp() *
                      (ref_log_probs_list[i] - pert_lp)).sum().item()
                kl_vals[i, ang_idx] = kl

    return kl_vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--opt_angle', type=float, default=15.0)
    parser.add_argument('--target_kl', type=float, default=1.0,
                        help='Target KL for capped objective (peak of reward)')
    parser.add_argument('--n_directions', type=int, default=3)
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--n_restarts', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_sweep_angles', type=int, default=20)
    parser.add_argument('--n_random_dirs', type=int, default=5)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/melbo_dogs_capped/{args.model}_target{args.target_kl}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Phase 1: Setup =====
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    prompts = DOGS
    print(f"\nExtracting activations from {len(prompts)} dog breed prompts at layer {args.layer}...")
    contexts, activations = extract_prompt_activations(
        model, tokenizer, prompts, args.layer, device)

    for i, p in enumerate(prompts):
        toks = tokenizer(p, return_tensors='pt')
        n_toks = toks['input_ids'].shape[1]
        print(f"  '{p}': {n_toks} tokens, act norm={activations[i].norm():.2f}")

    # Reference logits
    print("\nComputing reference logits...")
    ref_log_probs_list = []
    for i in range(len(prompts)):
        with torch.no_grad():
            hidden = torch.cat(
                [contexts[i], activations[i].unsqueeze(0).unsqueeze(0)], dim=1)
            logits = forward_from_layer_allpos(model, hidden, args.layer)
            ref_log_probs_list.append(F.log_softmax(logits, dim=-1))

    # Show what model predicts for each prompt (baseline)
    print("\nBaseline predictions:")
    for i, p in enumerate(prompts):
        top_vals, top_ids = ref_log_probs_list[i].squeeze().topk(5)
        top_tokens = [tokenizer.decode([tid]).strip() for tid in top_ids]
        print(f"  '{p}' -> {top_tokens}")

    # ===== Phase 2: MELBO optimization =====
    angle_rad = np.deg2rad(args.opt_angle)
    print(f"\n{'='*60}")
    print(f"CAPPED MELBO on dog breed activations")
    print(f"  {len(prompts)} prompts, d_model={d_model}")
    print(f"  Angle: {args.opt_angle} deg, target KL: {args.target_kl}")
    print(f"  Steps: {args.n_steps}, restarts: {args.n_restarts}")
    print(f"  Objective: KL * exp(-KL / {args.target_kl}), peak at KL = {args.target_kl}")
    print(f"{'='*60}")

    melbo_directions = []
    melbo_rewards = []
    melbo_histories = []

    for d_idx in range(args.n_directions):
        print(f"\n--- Direction {d_idx+1}/{args.n_directions} ---")
        t0 = time.time()
        direction, best_reward, history = optimize_direction(
            model, contexts, activations, ref_log_probs_list,
            angle_rad, args.layer, device, d_model,
            target_kl=args.target_kl,
            n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts,
            prev_directions=melbo_directions if melbo_directions else None
        )
        elapsed = time.time() - t0

        melbo_directions.append(direction)
        melbo_rewards.append(best_reward)
        melbo_histories.append(history)

        final_raw_kl = history[-1][1]
        print(f"  Best reward: {best_reward:.6f} (raw KL: {final_raw_kl:.4f})")
        print(f"  Time: {elapsed:.1f}s")

        for prev_idx in range(d_idx):
            cos_prev = (direction @ melbo_directions[prev_idx]).abs().item()
            print(f"  cos(D_{d_idx+1}, D_{prev_idx+1}): {cos_prev:.4f}")

    # ===== Phase 3: Interpretation =====
    print(f"\n{'='*60}")
    print("DIRECTION INTERPRETATION")
    print(f"{'='*60}")

    for d_idx, direction in enumerate(melbo_directions):
        final_raw_kl = melbo_histories[d_idx][-1][1]
        print(f"\n=== D_{d_idx+1} (reward={melbo_rewards[d_idx]:.4f}, raw KL={final_raw_kl:.4f}) ===")
        interpret_direction(
            model, tokenizer, contexts, activations, ref_log_probs_list,
            prompts, direction, args.opt_angle, args.layer, device)

    # ===== Phase 4: Sweep evaluation =====
    print(f"\n{'='*60}")
    print("EVALUATION SWEEP")
    print(f"{'='*60}")

    angles_deg = np.linspace(0, 180, args.n_sweep_angles + 1)[1:]
    angles_rad_sweep = np.deg2rad(angles_deg)

    sweep_results = {}
    for d_idx, direction in enumerate(melbo_directions):
        name = f'D_{d_idx+1}'
        print(f"Sweeping {name}...", end=' ', flush=True)
        t0 = time.time()
        sweep_results[name] = sweep_kl(
            model, contexts, activations, ref_log_probs_list,
            direction, angles_rad_sweep, args.layer, device)
        print(f"{time.time()-t0:.1f}s")

    # Random baseline
    print(f"Sweeping random ({args.n_random_dirs} dirs)...", end=' ', flush=True)
    t0 = time.time()
    N = len(prompts)
    n_angles = len(angles_deg)
    rand_kl = np.zeros((N, n_angles, args.n_random_dirs))

    for r_idx in range(args.n_random_dirs):
        for i in range(N):
            v = torch.randn(d_model, device=device)
            a_unit = activations[i] / activations[i].norm()
            v = v - (v @ a_unit) * a_unit
            v = F.normalize(v, dim=0)

            for ang_idx, theta in enumerate(angles_rad_sweep):
                with torch.no_grad():
                    hidden = build_perturbed_hidden_single(
                        contexts[i], activations[i], v, theta)
                    logits = forward_from_layer_allpos(model, hidden, args.layer)
                    pert_lp = F.log_softmax(logits, dim=-1)
                    kl = (ref_log_probs_list[i].exp() *
                          (ref_log_probs_list[i] - pert_lp)).sum().item()
                    rand_kl[i, ang_idx, r_idx] = kl

    sweep_results['random'] = rand_kl
    print(f"{time.time()-t0:.1f}s")

    # ===== Phase 5: Save and plot =====
    all_results = {
        'args': vars(args),
        'prompts': prompts,
        'directions': [d.cpu() for d in melbo_directions],
        'rewards': melbo_rewards,
        'histories': melbo_histories,
        'angles_deg': angles_deg,
        'sweep_results': {k: v for k, v in sweep_results.items()},
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved to {args.results_dir}/results.pkl")

    # KL sweep plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'D_1': 'red', 'D_2': 'orange', 'D_3': 'purple'}

    rand_per_act = np.median(rand_kl, axis=2)
    ax.fill_between(angles_deg,
                    np.percentile(rand_per_act, 25, axis=0),
                    np.percentile(rand_per_act, 75, axis=0),
                    color='gray', alpha=0.2)
    ax.plot(angles_deg, np.median(rand_per_act, axis=0),
            color='gray', lw=2, label='Random', ls='--')

    for name in ['D_1', 'D_2', 'D_3']:
        if name not in sweep_results or name == 'random':
            continue
        data = sweep_results[name]
        ax.plot(angles_deg, np.median(data, axis=0),
                color=colors[name], lw=2, label=name)
        ax.fill_between(angles_deg,
                        np.percentile(data, 25, axis=0),
                        np.percentile(data, 75, axis=0),
                        color=colors[name], alpha=0.1)

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('KL divergence')
    ax.set_title(f'Capped MELBO (target={args.target_kl}) on dog breeds\n{args.model}, layer {args.layer}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 180)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'sweep.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: sweep.png")

    # Optimization history (both reward and raw KL)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, history in enumerate(melbo_histories):
        rewards = [h[0] for h in history]
        raw_kls = [h[1] for h in history]
        axes[0].plot(rewards, label=f'D_{i+1} (final={rewards[-1]:.4f})',
                     color=list(colors.values())[i])
        axes[1].plot(raw_kls, label=f'D_{i+1} (final={raw_kls[-1]:.4f})',
                     color=list(colors.values())[i])
    axes[0].set_xlabel('Optimization step')
    axes[0].set_ylabel('Capped reward')
    axes[0].set_title(f'Reward: KL·exp(-KL/{args.target_kl})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Optimization step')
    axes[1].set_ylabel('Raw KL divergence')
    axes[1].set_title('Raw KL during optimization')
    axes[1].axhline(y=args.target_kl, color='black', ls=':', label=f'target={args.target_kl}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'optimization_history.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: optimization_history.png")

    # ===== Summary =====
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for i in range(len(melbo_directions)):
        final_raw_kl = melbo_histories[i][-1][1]
        print(f"D_{i+1}: reward={melbo_rewards[i]:.6f}, raw KL={final_raw_kl:.4f}")

    opt_ang_idx = np.argmin(np.abs(angles_deg - args.opt_angle))
    rand_kl_at_opt = np.median(rand_kl[:, opt_ang_idx, :])
    print(f"\nAt {args.opt_angle} deg (sweep):")
    print(f"  Random median KL: {rand_kl_at_opt:.6f}")
    for i in range(len(melbo_directions)):
        kl_at_opt = np.median(sweep_results[f'D_{i+1}'][:, opt_ang_idx])
        print(f"  D_{i+1} median KL: {kl_at_opt:.6f}")

    # Per-prompt KL distribution
    print(f"\nPer-prompt KL at {args.opt_angle} deg:")
    for i in range(len(melbo_directions)):
        data = sweep_results[f'D_{i+1}'][:, opt_ang_idx]
        print(f"  D_{i+1}: mean={data.mean():.4f}, median={np.median(data):.4f}, "
              f"min={data.min():.4f}, max={data.max():.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
