#!/usr/bin/env python3
"""
Unconstrained MELBO: find the direction in full R^d_model that maximizes KL divergence
when used as a spherical perturbation.

This is the unconstrained version of run_melbo_constrained.py — instead of optimizing
within span(D) (the bilingual difference matrix), we optimize over all directions.
Comparison with the constrained version tells us whether constraining to the bilingual
subspace was actually a binding constraint.

Usage:
    python scripts/run_melbo_unconstrained.py --model qwen3-0.6b --layer 14
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


# ── Reused from run_melbo_constrained.py ──

def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device,
                                   batch_size=64):
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


def sweep_random_directions(model, contexts, activations, ref_final, ref_log_probs,
                            angles_rad, layer, device, n_dirs=10):
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


# ── Unconstrained optimization ──

def optimize_direction_unconstrained(model, contexts, activations, ref_log_probs,
                                     angle_rad, layer, device, d_model,
                                     n_steps=300, lr=0.01, n_restarts=5,
                                     prev_directions=None):
    """Find direction in R^d_model maximizing mean KL divergence.

    Unlike the constrained version, this optimizes directly in full activation space
    (no projection to span(V)).

    Args:
        prev_directions: list of [d_model] unit vectors for orthogonality

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

            # Normalize + orthogonalize
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--opt_angle', type=float, default=15.0)
    parser.add_argument('--n_act', type=int, default=50)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_directions', type=int, default=3)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--constrained_path', default=None,
                        help='Path to constrained results for comparison')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.constrained_path is None:
        args.constrained_path = f'results/melbo_constrained/{args.model}/optimization_results.pkl'
    if args.results_dir is None:
        args.results_dir = f'results/melbo_unconstrained/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # Load constrained results for comparison
    print(f"Loading constrained results from {args.constrained_path}")
    with open(args.constrained_path, 'rb') as f:
        constrained = pickle.load(f)
    constrained_dirs = constrained['melbo_directions']
    mean_dir = constrained['mean_dir']
    pc1 = constrained['pc1']

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    # Load activations (same seed as constrained for fair comparison)
    print(f"Loading {args.n_act} FineWeb sequences (seed={args.seed})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Activation shape: {activations.shape}")

    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)

    # Reference
    with torch.no_grad():
        ref_hidden = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_hidden_and_logits(model, ref_hidden, args.layer)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # ===== Optimization =====
    angle_rad = np.deg2rad(args.opt_angle)
    print(f"\n{'='*60}")
    print(f"UNCONSTRAINED OPTIMIZATION: top-{args.n_directions} directions in R^{d_model}")
    print(f"Angle: {args.opt_angle} deg, steps: {args.n_steps}, restarts: {args.n_restarts}")
    print(f"{'='*60}")

    unc_directions = []
    unc_kls = []
    unc_histories = []

    for d_idx in range(args.n_directions):
        print(f"\n--- Direction {d_idx+1}/{args.n_directions} ---")
        t0 = time.time()
        direction, best_kl, history = optimize_direction_unconstrained(
            model, contexts_dev, activations_dev, ref_log_probs,
            angle_rad, args.layer, device, d_model,
            n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts,
            prev_directions=unc_directions if unc_directions else None
        )
        elapsed = time.time() - t0

        unc_directions.append(direction)
        unc_kls.append(best_kl)
        unc_histories.append(history)

        print(f"  Best KL: {best_kl:.6f}")
        print(f"  Time: {elapsed:.1f}s")

        # Compare with constrained
        cos_con = (direction @ constrained_dirs[d_idx].to(device)).abs().item()
        cos_mean = (direction @ mean_dir.to(device)).abs().item()
        cos_pc1 = (direction @ pc1.to(device)).abs().item()
        print(f"  cos(unc_{d_idx+1}, constrained_{d_idx+1}): {cos_con:.4f}")
        print(f"  cos(unc_{d_idx+1}, mean): {cos_mean:.4f}")
        print(f"  cos(unc_{d_idx+1}, PC1): {cos_pc1:.4f}")

        for prev_idx in range(d_idx):
            cos_prev = (direction @ unc_directions[prev_idx]).abs().item()
            print(f"  cos(unc_{d_idx+1}, unc_{prev_idx+1}): {cos_prev:.4f}")

    # Save optimization checkpoint
    opt_results = {
        'args': vars(args),
        'unc_directions': [d.cpu() for d in unc_directions],
        'unc_kls': unc_kls,
        'unc_histories': unc_histories,
        'constrained_kls': constrained['melbo_kls'],
    }
    with open(os.path.join(args.results_dir, 'optimization_results.pkl'), 'wb') as f:
        pickle.dump(opt_results, f)

    # ===== Evaluation sweep =====
    angles_deg = np.linspace(0, 180, args.n_angles + 1)[1:]
    angles_rad_sweep = np.deg2rad(angles_deg)

    print(f"\n{'='*60}")
    print(f"EVALUATION SWEEP: {len(angles_deg)} angles")
    print(f"{'='*60}")

    sweep_results = {}

    # Constrained and unconstrained directions + baselines
    all_named = [('mean', mean_dir), ('PC1', pc1)]
    for i, d in enumerate(constrained_dirs):
        all_named.append((f'constrained_{i+1}', d))
    for i, d in enumerate(unc_directions):
        all_named.append((f'unconstrained_{i+1}', d.cpu()))

    for name, direction in all_named:
        print(f"Sweeping: {name}...", end=' ', flush=True)
        t0 = time.time()
        kl, l2 = sweep_single_direction(
            model, contexts_dev, activations_dev, ref_final, ref_log_probs,
            direction, angles_rad_sweep, args.layer, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}
        print(f"{time.time()-t0:.1f}s")

    # Random
    print(f"Sweeping: random ({args.n_random_dirs} dirs/act)...", end=' ', flush=True)
    t0 = time.time()
    rand_kl, rand_l2 = sweep_random_directions(
        model, contexts_dev, activations_dev, ref_final, ref_log_probs,
        angles_rad_sweep, args.layer, device, n_dirs=args.n_random_dirs)
    sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}
    print(f"{time.time()-t0:.1f}s")

    # Save everything
    all_results = {
        **opt_results,
        'angles_deg': angles_deg,
        'sweep_results': sweep_results,
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved to {args.results_dir}/results.pkl")

    # ===== Plots =====
    colors = {
        'random': 'gray', 'mean': 'green', 'PC1': 'blue',
        'constrained_1': 'red', 'constrained_2': 'orange', 'constrained_3': 'purple',
        'unconstrained_1': 'red', 'unconstrained_2': 'orange', 'unconstrained_3': 'purple',
    }
    styles = {}
    for k in ['constrained_1', 'constrained_2', 'constrained_3']:
        styles[k] = '--'  # dashed for constrained
    for k in ['unconstrained_1', 'unconstrained_2', 'unconstrained_3']:
        styles[k] = '-'   # solid for unconstrained

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
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

        for name in ['constrained_1', 'unconstrained_1',
                      'constrained_2', 'unconstrained_2',
                      'constrained_3', 'unconstrained_3']:
            if name not in sweep_results:
                continue
            data = sweep_results[name][key]
            median = np.median(data, axis=0)
            color = colors.get(name, 'black')
            ls = styles.get(name, '-')
            ax.plot(angles_deg, median, color=color, lw=2, label=name, ls=ls)

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)

    fig.suptitle(f'Unconstrained vs Constrained MELBO\n'
                 f'{args.model}, layer {args.layer}, opt at {args.opt_angle} deg')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'sweep_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: sweep_comparison.png")

    # Optimization history comparison
    fig, axes = plt.subplots(1, args.n_directions, figsize=(5*args.n_directions, 5))
    if args.n_directions == 1:
        axes = [axes]
    for i in range(args.n_directions):
        ax = axes[i]
        if i < len(constrained['melbo_histories']):
            ax.plot(constrained['melbo_histories'][i], label='Constrained', ls='--')
        ax.plot(unc_histories[i], label='Unconstrained')
        ax.set_xlabel('Step')
        ax.set_ylabel('KL divergence')
        ax.set_title(f'Direction {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'optimization_history.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: optimization_history.png")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for i in range(args.n_directions):
        unc_d = unc_directions[i]
        con_d = constrained_dirs[i].to(device)
        cos_sim = (unc_d @ con_d).abs().item()
        print(f"Direction {i+1}:")
        print(f"  Constrained KL:   {constrained['melbo_kls'][i]:.6f}")
        print(f"  Unconstrained KL: {unc_kls[i]:.6f}")
        print(f"  cos(unc, con):    {cos_sim:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
