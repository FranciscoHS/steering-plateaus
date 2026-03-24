#!/usr/bin/env python3
"""
Manifold hypothesis test: does constrained MELBO recover the same direction
when using different subsets of bilingual prompt pairs?

Split 50 English-Mandarin pairs into 5 non-overlapping groups of 10.
For each group:
  1. Compute difference matrix D_i (en - zh activations)
  2. Get its column span via SVD
  3. Run constrained MELBO to find the top direction within span(D_i)
Compare MELBO_1 directions across groups via cosine similarity.

High cosine similarity → direction is global (evidence against manifold hypothesis)
Low cosine similarity → direction varies with position (suggestive of manifold hypothesis)

Usage:
    python scripts/melbo/run_melbo_manifold_test.py --model qwen3-0.6b
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
    _get_final_norm, _get_lm_head, _is_cat_pos_emb_model, _get_cat_pos_emb,
)
from src.data import load_fineweb_fixed_length
from data.bilingual_prompts import PROMPT_PAIRS


def forward_allpos_to_penultimate(model, hidden_states, start_layer):
    """Forward from start_layer to second-to-last block. Returns last-token hidden state."""
    hidden = hidden_states
    blocks = _get_blocks(model)
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, hidden)
    elif _is_cat_pos_emb_model(model):
        pos_emb = _get_cat_pos_emb(model, hidden.shape[1])
    # Run all blocks except the last one
    for block in blocks[start_layer + 1:-1]:
        hidden = _run_block(block, hidden, model, pos_emb)
    return hidden[:, -1, :]


def extract_diffs_for_pairs(model, tokenizer, device, layer, pairs):
    """Extract difference matrix D = en_acts - zh_acts for a list of (en, zh) pairs."""
    def get_acts(prompts):
        acts = []
        for p in prompts:
            tokens = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            acts.append(outputs.hidden_states[layer + 1][:, -1, :].squeeze(0).cpu())
        return torch.stack(acts)

    en_prompts = [p[0] for p in pairs]
    zh_prompts = [p[1] for p in pairs]
    en_acts = get_acts(en_prompts)
    zh_acts = get_acts(zh_prompts)
    D = en_acts - zh_acts
    return D


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device,
                                   batch_size=64):
    """Extract layer-L context and final-token activation for fixed-length sequences."""
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


def build_perturbed_hidden(contexts, activations, direction, angle_rad):
    """Construct hidden states after spherical perturbation along direction."""
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms

    proj = (direction.unsqueeze(0) * a_unit).sum(dim=-1, keepdim=True)
    u_perp = direction.unsqueeze(0) - proj * a_unit
    u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)

    a_pert = a_norms * (cos_t * a_unit + sin_t * u_perp)
    return torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)


def optimize_direction(model, contexts, activations, ref_penultimate,
                       V, angle_rad, layer, device,
                       n_steps=300, lr=0.01, n_restarts=5):
    """Find direction in span(V) maximizing mean cosine distance at 2nd-to-last layer.

    Returns:
        best_direction, best_cos_dist, best_history
    """
    r = V.shape[1]
    best_obj = -float('inf')
    best_direction = None
    best_history = None

    for restart in range(n_restarts):
        alpha = torch.randn(r, device=device)
        alpha = alpha / alpha.norm()
        alpha = alpha.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([alpha], lr=lr)
        history = []

        for step in range(n_steps):
            optimizer.zero_grad()

            a_normed = alpha / alpha.norm()
            direction = V @ a_normed

            hidden = build_perturbed_hidden(contexts, activations, direction, angle_rad)
            pert_penultimate = forward_allpos_to_penultimate(model, hidden, layer)

            # Cosine distance = 1 - cos_similarity
            cos_sim = F.cosine_similarity(ref_penultimate, pert_penultimate, dim=-1).mean()
            cos_dist = 1.0 - cos_sim

            loss = -cos_dist
            loss.backward()
            optimizer.step()

            val = cos_dist.item()
            history.append(val)

            if step % 100 == 0 or step == n_steps - 1:
                print(f"    restart {restart+1}/{n_restarts}, step {step}: "
                      f"cos_dist = {val:.6f}")

        final_val = history[-1]
        if final_val > best_obj:
            best_obj = final_val
            with torch.no_grad():
                a_final = alpha / alpha.norm()
                best_direction = (V @ a_final).detach()
                best_direction = best_direction / best_direction.norm()
            best_history = history

    return best_direction, best_obj, best_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--opt_angle', type=float, default=15.0,
                        help='Angle in degrees for optimization')
    parser.add_argument('--n_act', type=int, default=50)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--group_size', type=int, default=10,
                        help='Number of prompt pairs per group')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/melbo_manifold_test_cosdist/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Split prompt pairs into groups =====
    n_pairs = len(PROMPT_PAIRS)
    n_groups = n_pairs // args.group_size
    # Shuffle pairs deterministically
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(n_pairs)
    groups = []
    for g in range(n_groups):
        group_indices = indices[g * args.group_size : (g + 1) * args.group_size]
        group_pairs = [PROMPT_PAIRS[i] for i in group_indices]
        groups.append(group_pairs)
    print(f"Split {n_pairs} prompt pairs into {n_groups} groups of {args.group_size}")

    # ===== Load model =====
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    # ===== FineWeb activations (shared across all groups) =====
    print(f"\nLoading {args.n_act} FineWeb sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.layer} activations...")
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Activation shape: {activations.shape}, "
          f"mean norm: {activations.norm(dim=-1).mean():.2f}")

    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)

    # Reference (unperturbed) hidden state at 2nd-to-last layer
    with torch.no_grad():
        ref_hidden = torch.cat(
            [contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_penultimate = forward_allpos_to_penultimate(model, ref_hidden, args.layer)

    angle_rad = np.deg2rad(args.opt_angle)

    # ===== Run constrained MELBO for each group =====
    group_results = []

    for g_idx, group_pairs in enumerate(groups):
        print(f"\n{'='*60}")
        print(f"GROUP {g_idx+1}/{n_groups}")
        print(f"{'='*60}")

        # Extract difference matrix for this group
        D = extract_diffs_for_pairs(model, tokenizer, device, args.layer, group_pairs)
        mean_vec = D.mean(dim=0)
        mean_dir = mean_vec / mean_vec.norm()

        # SVD
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        V = Vh.T.to(device)  # [d_model, r]
        r = V.shape[1]
        pc1 = Vh[0]
        print(f"  D shape: {D.shape}, rank r={r}")
        print(f"  Top SV: {S[0]:.2f}, PC1 var explained: {S[0]**2 / (S**2).sum():.3f}")
        print(f"  cos(PC1, mean): {(pc1 @ mean_dir).abs().item():.3f}")

        # Optimize
        print(f"\n  Optimizing cosine distance at {args.opt_angle} deg...")
        t0 = time.time()
        direction, best_cos_dist, history = optimize_direction(
            model, contexts_dev, activations_dev, ref_penultimate,
            V, angle_rad, args.layer, device,
            n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts
        )
        elapsed = time.time() - t0

        cos_mean = (direction @ mean_dir.to(device)).abs().item()
        cos_pc1 = (direction @ pc1.to(device)).abs().item()
        print(f"  Best cos_dist: {best_cos_dist:.6f}")
        print(f"  cos(MELBO, mean): {cos_mean:.3f}")
        print(f"  cos(MELBO, PC1): {cos_pc1:.3f}")
        print(f"  Time: {elapsed:.1f}s")

        group_results.append({
            'group_idx': g_idx,
            'pairs': group_pairs,
            'D': D,
            'singular_values': S,
            'mean_dir': mean_dir,
            'pc1': pc1,
            'melbo_direction': direction.cpu(),
            'melbo_cos_dist': best_cos_dist,
            'melbo_history': history,
        })

    # ===== Cross-group comparison =====
    print(f"\n{'='*60}")
    print("CROSS-GROUP COSINE SIMILARITIES")
    print(f"{'='*60}")

    n = len(group_results)
    melbo_dirs = [r['melbo_direction'] for r in group_results]
    mean_dirs = [r['mean_dir'] for r in group_results]
    pc1_dirs = [r['pc1'] for r in group_results]

    # MELBO direction cosine similarities
    print("\nMELBO directions:")
    melbo_cos = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            cos = (melbo_dirs[i] @ melbo_dirs[j]).abs().item()
            melbo_cos[i, j] = cos
            if j > i:
                print(f"  Group {i+1} vs Group {j+1}: {cos:.4f}")
    off_diag = []
    for i in range(n):
        for j in range(i+1, n):
            off_diag.append(melbo_cos[i, j].item())
    print(f"  Mean pairwise |cos|: {np.mean(off_diag):.4f} ± {np.std(off_diag):.4f}")

    # Mean direction cosine similarities
    print("\nMean (en-zh) directions:")
    mean_cos_vals = []
    for i in range(n):
        for j in range(i+1, n):
            cos = (mean_dirs[i] @ mean_dirs[j]).abs().item()
            mean_cos_vals.append(cos)
            print(f"  Group {i+1} vs Group {j+1}: {cos:.4f}")
    print(f"  Mean pairwise |cos|: {np.mean(mean_cos_vals):.4f} ± {np.std(mean_cos_vals):.4f}")

    # PC1 cosine similarities
    print("\nPC1 directions:")
    pc1_cos_vals = []
    for i in range(n):
        for j in range(i+1, n):
            cos = (pc1_dirs[i] @ pc1_dirs[j]).abs().item()
            pc1_cos_vals.append(cos)
            print(f"  Group {i+1} vs Group {j+1}: {cos:.4f}")
    print(f"  Mean pairwise |cos|: {np.mean(pc1_cos_vals):.4f} ± {np.std(pc1_cos_vals):.4f}")

    # Random baseline: expected |cos| for random unit vectors in d_model dimensions
    # E[|cos|] ≈ sqrt(2/(pi*d)) for d-dimensional random unit vectors
    expected_random = np.sqrt(2 / (np.pi * d_model))
    print(f"\nRandom baseline E[|cos|] in d={d_model}: {expected_random:.4f}")

    # ===== Save =====
    all_results = {
        'args': vars(args),
        'group_results': group_results,
        'melbo_cos_matrix': melbo_cos,
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved results to {results_path}")

    # ===== Plot: cosine similarity matrices =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (name, dirs) in zip(axes, [
        ('MELBO', melbo_dirs), ('Mean diff', mean_dirs), ('PC1', pc1_dirs)
    ]):
        cos_mat = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                cos_mat[i, j] = (dirs[i] @ dirs[j]).abs().item()
        im = ax.imshow(cos_mat.numpy(), vmin=0, vmax=1, cmap='viridis')
        ax.set_title(f'{name} |cos| similarity')
        ax.set_xlabel('Group')
        ax.set_ylabel('Group')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(range(1, n+1))
        ax.set_yticklabels(range(1, n+1))
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{cos_mat[i,j]:.2f}', ha='center', va='center',
                        color='white' if cos_mat[i,j] < 0.5 else 'black', fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f'Manifold test: {args.model}, layer {args.layer}\n'
                 f'{n_groups} groups of {args.group_size} pairs, '
                 f'optimized at {args.opt_angle}°', fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'cosine_similarity_matrices.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
