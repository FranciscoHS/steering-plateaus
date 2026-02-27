#!/usr/bin/env python3
"""
Per-prompt unconstrained MELBO on Python-themed prompts.

For each of 10 prompts where "Python" is expected to be active, independently
finds the top-3 directions in full R^d_model that maximize KL divergence.
Then analyzes whether directions converge across prompts — convergence would
validate that MELBO independently recovers shared structure without being told
to look for it.

Usage:
    python scripts/melbo/run_melbo_python_individual.py [--model gpt2-large] [--layer 14]
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

from src.model import load_model, forward_from_layer_allpos


# 10 prompts where "Python" (the programming language) should be active.
# Mix of: Python code, English about Python, English about programming.
PYTHON_PROMPTS = [
    # Python code
    'def fibonacci(n):\n    if n <= 1:\n        return n',
    'import numpy as np\ndata = np.array([1, 2, 3])',
    'for i in range(100):\n    if i % 2 == 0:',
    'x = [i**2 for i in range(10)]\nprint(x',
    # English about Python
    'Python is a high-level interpreted programming language known',
    'Many data scientists prefer using Python because of its',
    'List comprehensions in Python provide a concise way to',
    'Python uses indentation instead of braces to define code',
    'The def keyword in Python is used to define a',
    'In Python, you can iterate over a dictionary using',
]


def extract_prompt_activations(model, tokenizer, prompts, layer, device):
    """Extract context and last-token activation for each prompt."""
    contexts = []
    activations = []
    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        contexts.append(hidden[:, :-1, :])       # [1, seq_len-1, d]
        activations.append(hidden[:, -1, :].squeeze(0))  # [d]
    return contexts, activations


def build_perturbed_hidden_single(context, activation, direction, angle_rad):
    """Spherical perturbation of a single activation along direction."""
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    a_norm = activation.norm()
    a_unit = activation / a_norm

    proj = (direction * a_unit).sum()
    u_perp = direction - proj * a_unit
    u_perp = u_perp / (u_perp.norm() + 1e-10)

    a_pert = a_norm * (cos_t * a_unit + sin_t * u_perp)
    return torch.cat([context, a_pert.unsqueeze(0).unsqueeze(0)], dim=1)


def optimize_direction_single(model, context, activation, ref_log_probs,
                               angle_rad, layer, device, d_model,
                               target_kl=1.0,
                               n_steps=300, lr=0.01, n_restarts=5,
                               prev_directions=None):
    """Find direction in R^d_model that produces KL ≈ target_kl for a SINGLE prompt.

    Uses capped objective: reward = KL * exp(-KL / target_kl).
    Peaks at KL = target_kl, penalizes overshoot — finds feature-like directions.

    prev_directions: previously found unit vectors for orthogonality.
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

            hidden = build_perturbed_hidden_single(
                context, activation, d_normed, angle_rad)
            logits = forward_from_layer_allpos(model, hidden, layer)
            pert_log_probs = F.log_softmax(logits, dim=-1)

            kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1).mean()
            # Capped reward: peaks at kl = target_kl, penalizes overshoot
            reward = kl * torch.exp(-kl / target_kl)
            loss = -reward
            loss.backward()
            optimizer.step()

            if prev_directions:
                with torch.no_grad():
                    for pd in prev_directions:
                        direction.data -= (direction.data @ pd) * pd

            history.append((reward.item(), kl.item()))

            if step % 100 == 0 or step == n_steps - 1:
                print(f"      restart {restart+1}/{n_restarts}, step {step}: "
                      f"reward = {reward.item():.6f}, KL = {kl.item():.4f}")

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


def interpret_direction(model, tokenizer, context, activation, ref_log_probs,
                        prompt, direction, angle_deg, layer):
    """Show what tokens shift when perturbing along direction."""
    angle_rad = np.deg2rad(angle_deg)

    for sign, label in [(1.0, "+"), (-1.0, "-")]:
        with torch.no_grad():
            hidden = build_perturbed_hidden_single(
                context, activation, direction, sign * angle_rad)
            logits = forward_from_layer_allpos(model, hidden, layer)
            pert_lp = F.log_softmax(logits, dim=-1).squeeze()

        ref_lp = ref_log_probs.squeeze()
        diff = pert_lp - ref_lp

        top_vals, top_ids = diff.topk(5)
        promoted = [(tokenizer.decode([tid]).strip(), v.item())
                    for tid, v in zip(top_ids, top_vals)]

        bot_vals, bot_ids = (-diff).topk(5)
        demoted = [(tokenizer.decode([tid]).strip(), -v.item())
                   for tid, v in zip(bot_ids, bot_vals)]

        kl = (ref_lp.exp() * (ref_lp - pert_lp)).sum().item()

        prom_str = ", ".join(f"'{t}' ({v:+.2f})" for t, v in promoted[:3])
        dem_str = ", ".join(f"'{t}' ({v:+.2f})" for t, v in demoted[:3])
        print(f"    {label} (KL={kl:.3f}): promoted=[{prom_str}], demoted=[{dem_str}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2-large')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--opt_angle', type=float, default=15.0)
    parser.add_argument('--target_kl', type=float, default=1.0,
                        help='Target KL for capped objective (peak of reward)')
    parser.add_argument('--n_directions', type=int, default=3)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/melbo_python_individual/{args.model}_target{args.target_kl}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Phase 1: Setup =====
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    prompts = PYTHON_PROMPTS
    n_prompts = len(prompts)

    # ===== Phase 2: Extract activations =====
    print(f"\nExtracting activations at layer {args.layer} for {n_prompts} prompts...")
    contexts, activations = extract_prompt_activations(
        model, tokenizer, prompts, args.layer, device)

    for i, p in enumerate(prompts):
        toks = tokenizer(p, return_tensors='pt')
        n_toks = toks['input_ids'].shape[1]
        print(f"  [{i}] ({n_toks} tok, norm={activations[i].norm():.1f}) {p[:60]}")

    # Reference logprobs
    print("\nComputing reference logprobs...")
    ref_log_probs_list = []
    for i in range(n_prompts):
        with torch.no_grad():
            hidden = torch.cat(
                [contexts[i], activations[i].unsqueeze(0).unsqueeze(0)], dim=1)
            logits = forward_from_layer_allpos(model, hidden, args.layer)
            ref_log_probs_list.append(F.log_softmax(logits, dim=-1))

    # Baseline predictions
    print("\nBaseline predictions:")
    for i, p in enumerate(prompts):
        top_vals, top_ids = ref_log_probs_list[i].squeeze().topk(5)
        top_tokens = [tokenizer.decode([tid]).strip() for tid in top_ids]
        print(f"  [{i}] {p[:40]}... -> {top_tokens}")

    # ===== Phase 3: Per-prompt MELBO =====
    angle_rad = np.deg2rad(args.opt_angle)
    print(f"\n{'='*60}")
    print(f"PER-PROMPT CAPPED MELBO")
    print(f"  {n_prompts} prompts, d_model={d_model}")
    print(f"  {args.n_directions} directions per prompt")
    print(f"  Angle: {args.opt_angle} deg, target KL: {args.target_kl}")
    print(f"  Objective: KL * exp(-KL / {args.target_kl}), peak at KL = {args.target_kl}")
    print(f"  Steps: {args.n_steps}, restarts: {args.n_restarts}")
    print(f"{'='*60}")

    all_directions = []  # all_directions[prompt_idx][dir_idx]
    all_rewards = []
    all_histories = []

    for p_idx in range(n_prompts):
        print(f"\n--- Prompt {p_idx}: {prompts[p_idx][:50]}... ---")
        t0 = time.time()

        prompt_directions = []
        prompt_rewards = []
        prompt_histories = []

        for d_idx in range(args.n_directions):
            print(f"  Direction {d_idx+1}/{args.n_directions}:")
            direction, best_reward, history = optimize_direction_single(
                model, contexts[p_idx], activations[p_idx],
                ref_log_probs_list[p_idx],
                angle_rad, args.layer, device, d_model,
                target_kl=args.target_kl,
                n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts,
                prev_directions=prompt_directions if prompt_directions else None
            )

            prompt_directions.append(direction)
            prompt_rewards.append(best_reward)
            prompt_histories.append(history)
            final_kl = history[-1][1]
            print(f"    -> reward = {best_reward:.6f}, KL = {final_kl:.4f}")

        elapsed = time.time() - t0
        print(f"  Total time: {elapsed:.1f}s")

        all_directions.append([d.cpu() for d in prompt_directions])
        all_rewards.append(prompt_rewards)
        all_histories.append(prompt_histories)

    # ===== Phase 4: Interpretation =====
    print(f"\n{'='*60}")
    print("DIRECTION INTERPRETATION (top-1 per prompt)")
    print(f"{'='*60}")

    for p_idx in range(n_prompts):
        print(f"\n  Prompt {p_idx}: {prompts[p_idx][:50]}...")
        interpret_direction(
            model, tokenizer, contexts[p_idx], activations[p_idx],
            ref_log_probs_list[p_idx], prompts[p_idx],
            all_directions[p_idx][0].to(device), args.opt_angle, args.layer)

    # ===== Phase 5: Convergence analysis =====
    print(f"\n{'='*60}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*60}")

    # Flatten all directions
    flat_directions = []
    flat_labels = []
    for p_idx in range(n_prompts):
        for d_idx in range(args.n_directions):
            flat_directions.append(all_directions[p_idx][d_idx])
            flat_labels.append(f"P{p_idx}_D{d_idx+1}")

    n_total = len(flat_directions)

    # Cosine similarity matrix
    cos_sim = torch.zeros(n_total, n_total)
    for i in range(n_total):
        for j in range(n_total):
            cos_sim[i, j] = (flat_directions[i] @ flat_directions[j]).abs().item()

    # Cross-prompt similarity summary
    print("\nMax |cos| between direction sets of each prompt pair:")
    for i in range(n_prompts):
        sims = []
        for j in range(n_prompts):
            if i == j:
                continue
            max_sim = 0
            for di in range(args.n_directions):
                for dj in range(args.n_directions):
                    idx_i = i * args.n_directions + di
                    idx_j = j * args.n_directions + dj
                    max_sim = max(max_sim, cos_sim[idx_i, idx_j].item())
            sims.append(f"P{j}:{max_sim:.3f}")
        print(f"  P{i} vs: {', '.join(sims)}")

    # ===== Phase 6: Save and plot =====
    results = {
        'args': vars(args),
        'prompts': prompts,
        'all_directions': all_directions,
        'all_rewards': all_rewards,
        'all_histories': all_histories,
        'cos_sim': cos_sim,
        'flat_labels': flat_labels,
    }

    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {results_path}")

    # Plot 1: Cosine similarity heatmap
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cos_sim.numpy(), cmap='RdBu_r', vmin=0, vmax=1)
    ax.set_xticks(range(n_total))
    ax.set_xticklabels(flat_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n_total))
    ax.set_yticklabels(flat_labels, fontsize=7)
    ax.set_title(f'Per-prompt MELBO: direction |cos| similarity\n'
                 f'{args.model}, layer {args.layer}, {args.opt_angle} deg')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Grid lines separating prompts
    for k in range(1, n_prompts):
        pos = k * args.n_directions - 0.5
        ax.axhline(pos, color='black', linewidth=0.5)
        ax.axvline(pos, color='black', linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(args.results_dir, 'cosine_similarity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # Plot 2: Raw KL values per prompt (from history)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_prompts)
    width = 0.25
    colors = ['red', 'orange', 'purple']
    for d_idx in range(args.n_directions):
        kls = [all_histories[p][d_idx][-1][1] for p in range(n_prompts)]
        ax.bar(x + d_idx * width, kls, width, label=f'Dir {d_idx+1}',
               color=colors[d_idx], alpha=0.7)
    ax.axhline(y=args.target_kl, color='black', ls=':', lw=2, label=f'target={args.target_kl}')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'P{i}' for i in range(n_prompts)])
    ax.set_ylabel('Raw KL divergence')
    ax.set_title(f'Per-prompt capped MELBO: raw KL at {args.opt_angle} deg')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'kl_per_prompt.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # Plot 3: Optimization histories (reward and raw KL)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.cm.tab10
    for p_idx in range(n_prompts):
        color = cmap(p_idx / n_prompts)
        for d_idx in range(args.n_directions):
            ls = ['-', '--', ':'][d_idx]
            label = f'P{p_idx}' if d_idx == 0 else None
            r_vals = [h[0] for h in all_histories[p_idx][d_idx]]
            kl_vals = [h[1] for h in all_histories[p_idx][d_idx]]
            axes[0].plot(r_vals, color=color, ls=ls, alpha=0.7, label=label)
            axes[1].plot(kl_vals, color=color, ls=ls, alpha=0.7, label=label)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Capped reward')
    axes[0].set_title(f'Reward: KL·exp(-KL/{args.target_kl})')
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Raw KL')
    axes[1].set_title('Raw KL during optimization')
    axes[1].axhline(y=args.target_kl, color='black', ls=':', label=f'target={args.target_kl}')
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'optimization_histories.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Target KL: {args.target_kl}")
    for p_idx in range(n_prompts):
        kl_str = ', '.join(f'{all_histories[p_idx][d][-1][1]:.4f}'
                           for d in range(args.n_directions))
        r_str = ', '.join(f'{r:.4f}' for r in all_rewards[p_idx])
        print(f"P{p_idx}: KL=[{kl_str}] reward=[{r_str}]  {prompts[p_idx][:40]}...")

    # Top-1 cross-prompt similarity stats
    top1_sims = []
    for i in range(n_prompts):
        for j in range(i+1, n_prompts):
            idx_i = i * args.n_directions
            idx_j = j * args.n_directions
            top1_sims.append(cos_sim[idx_i, idx_j].item())
    print(f"\nTop-1 direction cross-prompt |cos|: "
          f"mean={np.mean(top1_sims):.3f}, std={np.std(top1_sims):.3f}")

    # Best-match cross-prompt similarity
    max_sims = []
    for i in range(n_prompts):
        for j in range(i+1, n_prompts):
            max_sim = 0
            for di in range(args.n_directions):
                for dj in range(args.n_directions):
                    idx_i = i * args.n_directions + di
                    idx_j = j * args.n_directions + dj
                    max_sim = max(max_sim, cos_sim[idx_i, idx_j].item())
            max_sims.append(max_sim)
    print(f"Best-match cross-prompt |cos|: "
          f"mean={np.mean(max_sims):.3f}, std={np.std(max_sims):.3f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
