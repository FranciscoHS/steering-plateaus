#!/usr/bin/env python3
"""
Compare orthogonal components:
  - melbo1_orth_to_mean: component of MELBO_1 orthogonal to mean (already known to break plateau)
  - mean_orth_to_melbo1: component of mean orthogonal to MELBO_1 (both signs)
  - neg_mean_orth_to_melbo1: component of -mean orthogonal to MELBO_1

Usage:
    python scripts/run_orth_comparison.py --model qwen3-0.6b --layer 14
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


# ── Reused infrastructure ──

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


def project_out(v, u):
    """Component of v orthogonal to u (both unit norm), re-normalized."""
    residual = v - (v @ u) * u
    return residual / residual.norm()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--n_act', type=int, default=50)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--directions_path', default=None)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.directions_path is None:
        args.directions_path = f'results/melbo_constrained/{args.model}/optimization_results.pkl'
    if args.results_dir is None:
        args.results_dir = f'results/orth_comparison/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Load directions =====
    print(f"Loading saved directions from {args.directions_path}")
    with open(args.directions_path, 'rb') as f:
        saved = pickle.load(f)

    mean_dir = saved['mean_dir']
    pc1 = saved['pc1']
    melbo1 = saved['melbo_directions'][0]

    # ===== Compute orthogonal components =====
    print("\n" + "="*60)
    print("ORTHOGONAL DECOMPOSITIONS")
    print("="*60)

    # MELBO_1 orthogonal to mean
    melbo1_orth_to_mean = project_out(melbo1, mean_dir)

    # mean orthogonal to MELBO_1 (both signs)
    mean_orth_to_melbo1 = project_out(mean_dir, melbo1)
    neg_mean_orth_to_melbo1 = project_out(-mean_dir, melbo1)

    # -mean itself
    neg_mean = -mean_dir / mean_dir.norm()  # should already be unit but be safe

    print(f"cos(mean, MELBO_1) = {(mean_dir @ melbo1).item():.4f}")
    print(f"cos(-mean, MELBO_1) = {(-mean_dir @ melbo1).item():.4f}")
    print()
    print(f"melbo1_orth_to_mean:    norm of orth component = {(melbo1 - (melbo1 @ mean_dir) * mean_dir).norm().item():.4f}")
    print(f"mean_orth_to_melbo1:    norm of orth component = {(mean_dir - (mean_dir @ melbo1) * melbo1).norm().item():.4f}")
    print(f"neg_mean_orth_to_melbo1: same direction as -mean_orth_to_melbo1? {torch.allclose(neg_mean_orth_to_melbo1, -mean_orth_to_melbo1)}")
    print()
    print(f"cos(melbo1_orth_to_mean, mean_orth_to_melbo1) = {(melbo1_orth_to_mean @ mean_orth_to_melbo1).item():.4f}")
    print(f"cos(melbo1_orth_to_mean, neg_mean_orth_to_melbo1) = {(melbo1_orth_to_mean @ neg_mean_orth_to_melbo1).item():.4f}")

    # ===== Load model + activations =====
    print(f"\n{'='*60}")
    print(f"LOADING MODEL AND ACTIVATIONS (seed={args.seed})")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Activations: {activations.shape}")

    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)

    with torch.no_grad():
        ref_hidden = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_hidden_and_logits(model, ref_hidden, args.layer)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # ===== Sweeps =====
    angles_deg = np.linspace(0, 180, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    print(f"\n{'='*60}")
    print("PERTURBATION SWEEPS")
    print(f"{'='*60}")

    sweep_results = {}

    named_dirs = {
        'MELBO_1':               melbo1,
        '-mean':                 neg_mean,
        'melbo1_orth_to_mean':   melbo1_orth_to_mean,
        'mean_orth_to_melbo1':   mean_orth_to_melbo1,
        '-mean_orth_to_melbo1':  neg_mean_orth_to_melbo1,
    }
    for name, direction in named_dirs.items():
        print(f"Sweeping: {name}...", end=' ', flush=True)
        t0 = time.time()
        kl, l2 = sweep_single_direction(
            model, contexts_dev, activations_dev, ref_final, ref_log_probs,
            direction, angles_rad, args.layer, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}
        print(f"{time.time()-t0:.1f}s")

    # Random
    print(f"Sweeping: random ({args.n_random_dirs} dirs/act)...", end=' ', flush=True)
    t0 = time.time()
    rand_kl, rand_l2 = sweep_random_directions(
        model, contexts_dev, activations_dev, ref_final, ref_log_probs,
        angles_rad, args.layer, device, n_dirs=args.n_random_dirs)
    sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}
    print(f"{time.time()-t0:.1f}s")

    # ===== Save =====
    all_results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'sweep_results': sweep_results,
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved results to {args.results_dir}/results.pkl")

    # ===== Plot =====
    max_angle = 60
    mask = angles_deg <= max_angle
    x = angles_deg[mask]

    def get_stats(name, key):
        data = sweep_results[name][key]
        if data.ndim == 3:
            per_act = np.median(data, axis=2)
        else:
            per_act = data
        return (np.median(per_act, axis=0),
                np.percentile(per_act, 25, axis=0),
                np.percentile(per_act, 75, axis=0))

    styles = {
        'MELBO_1':               ('#d62728', '-',  2.5, 'MELBO₁'),
        '-mean':                 ('#2ca02c', '-',  2.5, '−mean'),
        'melbo1_orth_to_mean':   ('#d62728', ':',  2.0, 'MELBO₁ ⊥ mean'),
        'mean_orth_to_melbo1':   ('#2ca02c', ':',  2.0, 'mean ⊥ MELBO₁'),
        '-mean_orth_to_melbo1':  ('#2ca02c', '--', 2.0, '−mean ⊥ MELBO₁'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for metric_idx, (key, ylabel) in enumerate([('kl', 'KL divergence'), ('l2', 'L2 distance')]):
        ax = axes[metric_idx]

        # Random
        rm, rq25, rq75 = get_stats('random', key)
        ax.fill_between(x, rq25[mask], rq75[mask], color='gray', alpha=0.12)
        ax.plot(x, rm[mask], color='gray', lw=2, ls='--', label='Random', zorder=1)

        for name, (color, ls, lw, label) in styles.items():
            med, q25, q75 = get_stats(name, key)
            ax.fill_between(x, q25[mask], q75[mask], color=color, alpha=0.08)
            ax.plot(x, med[mask], color=color, ls=ls, lw=lw, label=label, zorder=3)

        ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_angle)

    fig.suptitle(
        'Orthogonal component comparison: MELBO₁ vs mean\n'
        'Qwen3-0.6B, layer 14',
        fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'orth_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: orth_comparison.png")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Median KL at 30° perturbation")
    print(f"{'='*60}")
    idx_30 = np.argmin(np.abs(angles_deg - 30))
    for name in styles:
        kl = np.median(sweep_results[name]['kl'][:, idx_30])
        print(f"  {name:25s}: {kl:.4f}")
    rand_kl = np.median(np.median(sweep_results['random']['kl'][:, idx_30, :], axis=1))
    print(f"  {'random':25s}: {rand_kl:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
