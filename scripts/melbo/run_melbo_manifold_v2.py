#!/usr/bin/env python3
"""
Manifold hypothesis test v2: does the optimal MELBO direction within a fixed
subspace change depending on which activation we optimize at?

Setup:
  - Fixed subspace: span of all 50 English-Mandarin difference vectors
  - Optimize cosine distance at 2nd-to-last layer, angle 20°
  - Repeat for 5 different individual FineWeb activations

If the manifold hypothesis is true, the optimal direction should vary with
position in activation space.

Usage:
    python scripts/melbo/run_melbo_manifold_v2.py --model qwen3-0.6b
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
    for block in blocks[start_layer + 1:-1]:
        hidden = _run_block(block, hidden, model, pos_emb)
    return hidden[:, -1, :]


def extract_bilingual_diffs(model, tokenizer, device, layer):
    """Extract difference matrix D = en_acts - zh_acts for all prompt pairs."""
    def get_acts(prompts):
        acts = []
        for p in prompts:
            tokens = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            acts.append(outputs.hidden_states[layer + 1][:, -1, :].squeeze(0).cpu())
        return torch.stack(acts)

    en_prompts = [p[0] for p in PROMPT_PAIRS]
    zh_prompts = [p[1] for p in PROMPT_PAIRS]
    print(f"Extracting bilingual activations at layer {layer} ({len(PROMPT_PAIRS)} pairs)...")
    en_acts = get_acts(en_prompts)
    zh_acts = get_acts(zh_prompts)
    D = en_acts - zh_acts
    mean_vec = D.mean(dim=0)
    mean_dir = mean_vec / mean_vec.norm()
    print(f"  D shape: {D.shape}, mean diff norm: {mean_vec.norm():.2f}")
    return D, mean_dir


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


def build_perturbed_hidden(context, activation, direction, angle_rad):
    """Construct hidden states after spherical perturbation along direction.

    Args:
        context: [1, seq_len-1, d] single activation's context
        activation: [1, d] single activation
        direction: [d] unit vector (may carry gradients)
        angle_rad: float
    """
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    a_norm = activation.norm(dim=-1, keepdim=True)
    a_unit = activation / a_norm

    proj = (direction * a_unit).sum(dim=-1, keepdim=True)
    u_perp = direction.unsqueeze(0) - proj * a_unit
    u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)

    a_pert = a_norm * (cos_t * a_unit + sin_t * u_perp)
    return torch.cat([context, a_pert.unsqueeze(1)], dim=1)


def optimize_direction_single(model, context, activation, ref_penultimate,
                               V, angle_rad, layer, device,
                               n_steps=300, lr=0.01, n_restarts=5):
    """Find direction in span(V) maximizing cosine distance for a single activation.

    Args:
        context: [1, seq_len-1, d] on device
        activation: [1, d] on device
        ref_penultimate: [1, d] reference hidden at penultimate layer
        V: [d_model, r] orthonormal basis for subspace
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

            hidden = build_perturbed_hidden(context, activation, direction, angle_rad)
            pert_penultimate = forward_allpos_to_penultimate(model, hidden, layer)

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
    parser.add_argument('--opt_angle', type=float, default=20.0,
                        help='Angle in degrees for optimization')
    parser.add_argument('--n_test_acts', type=int, default=5,
                        help='Number of individual FineWeb activations to optimize at')
    parser.add_argument('--n_act_pool', type=int, default=50,
                        help='Total FineWeb activations to load (pick from these)')
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/melbo_manifold_v2/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Load model =====
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    # ===== Fixed subspace: all 50 bilingual pairs =====
    D, mean_dir = extract_bilingual_diffs(model, tokenizer, device, args.layer)
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    V = Vh.T.to(device)  # [d_model, r]
    r = V.shape[1]
    pc1 = Vh[0]
    print(f"SVD: rank r={r}, top SV={S[0]:.2f}, "
          f"PC1 var explained={S[0]**2 / (S**2).sum():.3f}")

    # ===== FineWeb activations =====
    print(f"\nLoading {args.n_act_pool} FineWeb sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act_pool, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.layer} activations...")
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Activation shape: {activations.shape}, "
          f"mean norm: {activations.norm(dim=-1).mean():.2f}")

    # Pick n_test_acts evenly spaced from the pool
    test_indices = np.linspace(0, args.n_act_pool - 1, args.n_test_acts, dtype=int)
    print(f"\nOptimizing at {args.n_test_acts} activations (indices: {list(test_indices)})")

    angle_rad = np.deg2rad(args.opt_angle)

    # ===== Optimize at each activation =====
    act_results = []

    for i, act_idx in enumerate(test_indices):
        print(f"\n{'='*60}")
        print(f"ACTIVATION {i+1}/{args.n_test_acts} (index {act_idx})")
        print(f"{'='*60}")

        ctx_i = contexts[act_idx:act_idx+1].to(device)       # [1, seq-1, d]
        act_i = activations[act_idx:act_idx+1].to(device)     # [1, d]

        # Reference penultimate for this activation
        with torch.no_grad():
            ref_hidden = torch.cat([ctx_i, act_i.unsqueeze(1)], dim=1)
            ref_pen = forward_allpos_to_penultimate(model, ref_hidden, args.layer)

        t0 = time.time()
        direction, best_cos_dist, history = optimize_direction_single(
            model, ctx_i, act_i, ref_pen,
            V, angle_rad, args.layer, device,
            n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts
        )
        elapsed = time.time() - t0

        cos_mean = (direction @ mean_dir.to(device)).abs().item()
        cos_pc1 = (direction @ pc1.to(device)).abs().item()
        print(f"  Best cos_dist: {best_cos_dist:.6f}")
        print(f"  cos(MELBO, mean_diff): {cos_mean:.3f}")
        print(f"  cos(MELBO, PC1): {cos_pc1:.3f}")
        print(f"  Time: {elapsed:.1f}s")

        act_results.append({
            'act_idx': int(act_idx),
            'melbo_direction': direction.cpu(),
            'melbo_cos_dist': best_cos_dist,
            'melbo_history': history,
            'cos_with_mean': cos_mean,
            'cos_with_pc1': cos_pc1,
        })

    # ===== Cross-activation comparison =====
    print(f"\n{'='*60}")
    print("CROSS-ACTIVATION COSINE SIMILARITIES")
    print(f"{'='*60}")

    n = len(act_results)
    dirs = [r['melbo_direction'] for r in act_results]

    print("\nMELBO directions:")
    off_diag = []
    for i in range(n):
        for j in range(i+1, n):
            cos = (dirs[i] @ dirs[j]).abs().item()
            off_diag.append(cos)
            print(f"  Act {i+1} (idx {test_indices[i]}) vs "
                  f"Act {j+1} (idx {test_indices[j]}): {cos:.4f}")
    print(f"  Mean pairwise |cos|: {np.mean(off_diag):.4f} ± {np.std(off_diag):.4f}")

    expected_random = np.sqrt(2 / (np.pi * d_model))
    print(f"\nRandom baseline E[|cos|] in d={d_model}: {expected_random:.4f}")

    # ===== Save =====
    all_results = {
        'args': vars(args),
        'act_results': act_results,
        'mean_dir': mean_dir,
        'pc1': pc1,
        'singular_values': S,
        'test_indices': test_indices,
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved results to {results_path}")

    # ===== Plot: cosine similarity matrix =====
    cos_mat = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            cos_mat[i, j] = (dirs[i] @ dirs[j]).abs().item()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cos_mat.numpy(), vmin=0, vmax=1, cmap='viridis')
    ax.set_title(f'MELBO |cos| similarity across activations\n'
                 f'{args.model}, layer {args.layer}, angle {args.opt_angle}°')
    ax.set_xlabel('Activation')
    ax.set_ylabel('Activation')
    labels = [f'{i+1}\n(idx {test_indices[i]})' for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{cos_mat[i,j]:.2f}', ha='center', va='center',
                    color='white' if cos_mat[i,j] < 0.5 else 'black', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'cosine_similarity.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
