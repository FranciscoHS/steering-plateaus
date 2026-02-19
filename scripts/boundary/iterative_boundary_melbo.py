#!/usr/bin/env python3
"""
Iterative boundary MELBO: repeatedly walk to boundary, find new directions.

Starting from English activations:
  Iteration 0: Walk along -MELBO_1 (toward Zh) → find boundary → MELBO there
  Iteration 1: Walk along best bnd_MELBO from iter 0 → find next boundary → MELBO
  Iteration 2: Walk along best bnd_MELBO from iter 1 → find next boundary → MELBO
  ...

At each iteration, MELBO is orthogonal to all previous walk directions.

Usage:
    python scripts/boundary/iterative_boundary_melbo.py --model qwen3-0.6b --layer 14 --n_iterations 3
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
from data.bilingual_prompts import ENGLISH_PROMPTS

# Import building blocks from the single-iteration script
from scripts.boundary.run_boundary_melbo import (
    forward_to_hidden_and_logits,
    build_perturbed_hidden,
    compute_boundary_activations,
    sweep_single_direction,
    sweep_random_directions,
    find_boundary_angles,
    optimize_direction_at_boundary,
    extract_prompts_fixed_length,
    extract_context_and_activation,
)


def run_one_iteration(model, contexts, activations, walk_dir, walk_label,
                      orth_constraints, layer, device, d_model,
                      n_boundary_angles=60, n_eval_angles=60,
                      opt_angle=15.0, n_directions=2, n_steps=300,
                      n_restarts=5, lr=0.01, n_random_dirs=10):
    """Run one iteration of boundary-finding + MELBO.

    Args:
        contexts: [N, seq_len-1, d] on device
        activations: [N, d] on device (current starting point)
        walk_dir: [d] unit vector on device (direction to walk)
        walk_label: str
        orth_constraints: list of [d] unit vectors on device (enforce orthogonality)

    Returns dict with all results for this iteration.
    """
    N = activations.shape[0]
    results = {'walk_label': walk_label}

    # --- Reference at current activations ---
    with torch.no_grad():
        ref_hidden = torch.cat([contexts, activations.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_hidden_and_logits(model, ref_hidden, layer)
        ref_lp = F.log_softmax(ref_logits, dim=-1)

    # --- Sweep walk_dir to find boundary ---
    boundary_angles_deg = np.linspace(0, 90, n_boundary_angles + 1)[1:]
    boundary_angles_rad = np.deg2rad(boundary_angles_deg)

    print(f"  Sweeping {len(boundary_angles_deg)} angles along {walk_label}...")
    t0 = time.time()
    kl_sweep, _ = sweep_single_direction(
        model, contexts, activations, ref_final, ref_lp,
        walk_dir.cpu(), boundary_angles_rad, layer, device)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Per-prompt boundary angles
    per_prompt_boundary_deg = find_boundary_angles(kl_sweep, boundary_angles_deg)
    per_prompt_boundary_rad = torch.tensor(
        np.deg2rad(per_prompt_boundary_deg), device=device, dtype=torch.float32)

    median_bnd = np.median(per_prompt_boundary_deg)
    print(f"  Boundary angles: median={median_bnd:.1f}, "
          f"mean={np.mean(per_prompt_boundary_deg):.1f}, "
          f"std={np.std(per_prompt_boundary_deg):.1f}, "
          f"range=[{np.min(per_prompt_boundary_deg):.1f}, {np.max(per_prompt_boundary_deg):.1f}]")
    results['boundary_angles_deg'] = per_prompt_boundary_deg
    results['kl_boundary_sweep'] = kl_sweep
    results['sweep_angles_deg'] = boundary_angles_deg

    # --- Boundary activations ---
    boundary_acts = compute_boundary_activations(
        activations, walk_dir, per_prompt_boundary_rad)
    cos_sim = F.cosine_similarity(activations, boundary_acts, dim=-1)
    print(f"  Norms: start={activations.norm(dim=-1).mean():.2f}, "
          f"boundary={boundary_acts.norm(dim=-1).mean():.2f}")
    print(f"  cos(start, boundary): {cos_sim.mean():.4f}")
    results['boundary_acts'] = boundary_acts.cpu()

    # Reference at boundary
    with torch.no_grad():
        ref_hidden_bnd = torch.cat([contexts, boundary_acts.unsqueeze(1)], dim=1)
        ref_final_bnd, ref_logits_bnd = forward_to_hidden_and_logits(
            model, ref_hidden_bnd, layer)
        ref_lp_bnd = F.log_softmax(ref_logits_bnd, dim=-1)

    kl_start_to_bnd = (ref_lp.exp() * (ref_lp - ref_lp_bnd)).sum(dim=-1)
    print(f"  KL(start || boundary): mean={kl_start_to_bnd.mean():.4f}")

    # --- MELBO at boundary ---
    opt_angle_rad = np.deg2rad(opt_angle)
    print(f"  Running MELBO at boundary ({n_directions} dirs, "
          f"angle={opt_angle} deg, {n_steps} steps, {n_restarts} restarts)")
    print(f"  Orthogonal to {len(orth_constraints)} previous directions")

    bnd_dirs = []
    bnd_kls = []
    bnd_histories = []
    current_orth = list(orth_constraints)  # copy

    for d_idx in range(n_directions):
        print(f"    Direction {d_idx+1}/{n_directions}...")
        t0 = time.time()

        direction, best_kl, history = optimize_direction_at_boundary(
            model, contexts, boundary_acts, ref_lp_bnd,
            opt_angle_rad, layer, device, d_model,
            n_steps=n_steps, lr=lr, n_restarts=n_restarts,
            prev_directions=current_orth
        )

        bnd_dirs.append(direction.cpu())
        bnd_kls.append(best_kl)
        bnd_histories.append(history)
        current_orth.append(direction)

        # Diagnostics
        cos_walk = (direction @ walk_dir).abs().item()
        print(f"    KL={best_kl:.4f}, |cos(walk_dir)|={cos_walk:.4f}, "
              f"time={time.time()-t0:.1f}s")
        for prev_idx in range(d_idx):
            cos_prev = (direction @ bnd_dirs[prev_idx].to(device)).abs().item()
            print(f"    |cos(dir_{d_idx+1}, dir_{prev_idx+1})|={cos_prev:.4f}")

    results['bnd_melbo_directions'] = bnd_dirs
    results['bnd_melbo_kls'] = bnd_kls
    results['bnd_melbo_histories'] = bnd_histories

    # --- Eval sweeps from boundary ---
    eval_angles_deg = np.linspace(0, 90, n_eval_angles + 1)[1:]
    eval_angles_rad = np.deg2rad(eval_angles_deg)

    sweep_results = {}
    print(f"  Eval sweeps from boundary...")

    # Walk direction
    kl, l2 = sweep_single_direction(
        model, contexts, boundary_acts, ref_final_bnd, ref_lp_bnd,
        walk_dir.cpu(), eval_angles_rad, layer, device)
    sweep_results[f'{walk_label} (walk)'] = {'kl': kl, 'l2': l2}

    # Found directions
    for i, d in enumerate(bnd_dirs):
        kl, l2 = sweep_single_direction(
            model, contexts, boundary_acts, ref_final_bnd, ref_lp_bnd,
            d, eval_angles_rad, layer, device)
        sweep_results[f'bnd_MELBO_{i+1}'] = {'kl': kl, 'l2': l2}

    # Random
    rand_kl, rand_l2 = sweep_random_directions(
        model, contexts, boundary_acts, ref_final_bnd, ref_lp_bnd,
        eval_angles_rad, layer, device, n_dirs=n_random_dirs)
    sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}

    results['eval_angles_deg'] = eval_angles_deg
    results['sweep_results'] = sweep_results

    # Print summary KLs
    for ref_angle in [15, 30]:
        idx = np.argmin(np.abs(eval_angles_deg - ref_angle))
        print(f"  Median KL at {ref_angle} deg from boundary:")
        for name, data in sweep_results.items():
            if name == 'random':
                med = np.median(data['kl'][:, idx, :].mean(axis=1))
            else:
                med = np.median(data['kl'][:, idx])
            print(f"    {name}: {med:.4f}")

    return results, boundary_acts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--n_iterations', type=int, default=3)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--opt_angle', type=float, default=15.0)
    parser.add_argument('--n_directions', type=int, default=2)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--melbo_path', default=None)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.melbo_path is None:
        args.melbo_path = f'results/melbo_constrained/{args.model}/optimization_results.pkl'
    if args.results_dir is None:
        args.results_dir = f'results/iterative_boundary_melbo/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Setup =====
    print("=" * 60)
    print("SETUP: Load model, directions, and activations")
    print("=" * 60)

    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']

    print(f"Loading MELBO_1 from {args.melbo_path}")
    with open(args.melbo_path, 'rb') as f:
        saved_melbo = pickle.load(f)
    melbo_1 = saved_melbo['melbo_directions'][0]
    mean_dir = saved_melbo['mean_dir']

    # Ensure MELBO_1 points English-ward
    if (melbo_1 @ mean_dir) < 0:
        melbo_1 = -melbo_1
    melbo_1_dev = melbo_1.to(device)

    # Extract English activations
    token_ids_list, used_prompts = extract_prompts_fixed_length(
        tokenizer, ENGLISH_PROMPTS, args.seq_len)
    print(f"English prompts with >= {args.seq_len} tokens: "
          f"{len(token_ids_list)}/{len(ENGLISH_PROMPTS)}")

    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    N = activations.shape[0]
    print(f"Extracted {N} activations, d_model={d_model}")

    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)

    # ===== Iterative loop =====
    all_iterations = []
    current_acts = activations_dev
    # First walk direction: toward Mandarin
    walk_dir = -melbo_1_dev
    walk_label = '-MELBO_1 (->Zh)'
    # Orthogonality constraints accumulate
    orth_constraints = [melbo_1_dev]  # always ⊥ to original MELBO_1

    for iteration in range(args.n_iterations):
        print(f"\n{'=' * 60}")
        print(f"ITERATION {iteration}: walk along {walk_label}")
        print(f"Orthogonality constraints: {len(orth_constraints)} directions")
        print("=" * 60)

        iter_results, boundary_acts = run_one_iteration(
            model, contexts_dev, current_acts, walk_dir, walk_label,
            orth_constraints, args.layer, device, d_model,
            opt_angle=args.opt_angle, n_directions=args.n_directions,
            n_steps=args.n_steps, n_restarts=args.n_restarts, lr=args.lr,
        )
        iter_results['iteration'] = iteration
        all_iterations.append(iter_results)

        # Save checkpoint after each iteration
        ckpt = {
            'args': vars(args),
            'iterations': all_iterations,
            'used_prompts': used_prompts,
            'melbo_1': melbo_1.cpu(),
            'mean_dir': mean_dir.cpu(),
        }
        ckpt_path = os.path.join(args.results_dir, 'results.pkl')
        with open(ckpt_path, 'wb') as f:
            pickle.dump(ckpt, f)
        print(f"  Saved checkpoint to {ckpt_path}")

        # Prepare next iteration: walk along the highest-KL direction found
        best_idx = np.argmax(iter_results['bnd_melbo_kls'])
        next_walk = iter_results['bnd_melbo_directions'][best_idx].to(device)

        # Add the walk direction we just used to orthogonality constraints
        orth_constraints.append(walk_dir)
        # Also add the found directions (so we don't rediscover them)
        for d in iter_results['bnd_melbo_directions']:
            orth_constraints.append(d.to(device))

        # Update for next iteration
        current_acts = boundary_acts  # start from this boundary
        walk_dir = next_walk
        walk_label = f'bnd_MELBO_{best_idx+1} (iter {iteration})'

    # ===== Summary =====
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    for it in all_iterations:
        i = it['iteration']
        med_bnd = np.median(it['boundary_angles_deg'])
        print(f"\nIteration {i}: walk={it['walk_label']}")
        print(f"  Boundary angle (median): {med_bnd:.1f} deg")
        for j, (d, kl) in enumerate(zip(it['bnd_melbo_directions'], it['bnd_melbo_kls'])):
            print(f"  bnd_MELBO_{j+1}: KL={kl:.4f}")

        # KL at 15 deg
        eval_deg = it['eval_angles_deg']
        idx_15 = np.argmin(np.abs(eval_deg - 15))
        print(f"  KL at 15 deg from boundary:")
        for name, data in it['sweep_results'].items():
            if name == 'random':
                med = np.median(data['kl'][:, idx_15, :].mean(axis=1))
            else:
                med = np.median(data['kl'][:, idx_15])
            print(f"    {name}: {med:.4f}")

    # ===== Plot =====
    fig, axes = plt.subplots(1, args.n_iterations, figsize=(6*args.n_iterations, 5))
    if args.n_iterations == 1:
        axes = [axes]

    for it in all_iterations:
        i = it['iteration']
        ax = axes[i]
        eval_deg = it['eval_angles_deg']

        # Random baseline
        rand_data = it['sweep_results']['random']['kl']
        rand_per_act = np.median(rand_data, axis=2)
        rand_median = np.median(rand_per_act, axis=0)
        ax.plot(eval_deg, rand_median, color='gray', lw=2, ls='--', label='random')

        # Named directions
        colors = ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4']
        for j, (name, data) in enumerate(it['sweep_results'].items()):
            if name == 'random':
                continue
            median = np.median(data['kl'], axis=0)
            color = colors[j % len(colors)]
            ax.plot(eval_deg, median, color=color, lw=2, label=name)

        ax.set_xlabel('Angle from boundary (deg)')
        ax.set_ylabel('KL divergence')
        ax.set_title(f'Iter {i}: walk={it["walk_label"]}\n'
                     f'bnd @ {np.median(it["boundary_angles_deg"]):.0f} deg')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 90)

    plt.suptitle(f'Iterative Boundary MELBO: {args.model}, layer {args.layer}',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'iterative_sweeps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")

    # Boundary finding plot
    fig, axes = plt.subplots(1, args.n_iterations, figsize=(6*args.n_iterations, 4))
    if args.n_iterations == 1:
        axes = [axes]
    for it in all_iterations:
        i = it['iteration']
        ax = axes[i]
        sweep_deg = it['sweep_angles_deg']
        kl_sweep = it['kl_boundary_sweep']
        for s in range(min(kl_sweep.shape[0], 10)):
            ax.plot(sweep_deg, kl_sweep[s], alpha=0.3, color='gray')
        ax.plot(sweep_deg, np.median(kl_sweep, axis=0), color='red', lw=2.5)
        med_bnd = np.median(it['boundary_angles_deg'])
        ax.axvline(med_bnd, color='blue', ls='--', lw=1.5,
                   label=f'boundary ({med_bnd:.0f} deg)')
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('KL divergence')
        ax.set_title(f'Iter {i}: {it["walk_label"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle('Boundary finding per iteration', fontsize=12)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'iterative_boundary_finding.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
