#!/usr/bin/env python3
"""
Multi-contrastive genre direction isolation via all-pairwise contrasts.

Implements Section 8 of multi_contrastive_projection.pdf:
  For k genres, construct ALL k(k-1)/2 pairwise contrastive datasets.
  Least-squares recovery: f̂_i = (1/k) * Σ_{j≠i} v_ij
  where v_ij = mean(activations_i - activations_j) for matched prompts.
  Error: O(εC) instead of O(C/√k) from star topology.

Also computes the star-topology mean estimator (Section 4) and SVD estimator
(Section 5) for comparison.

Genres: news, scientific, recipe, legal, children, review, wikipedia,
        dialogue, email, technical, advertising (k=11, 55 pairwise contrasts)

Usage:
    python scripts/directions/run_register_contrastive.py [--model qwen3-0.6b] [--layer 14]
"""

import sys
import argparse
import os
import pickle
import time
from itertools import combinations
import numpy as np

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import (
    load_model, forward_from_layer_allpos,
    _get_blocks, _is_rope_model, _get_position_embeddings, _run_block,
    _get_final_norm, _get_lm_head,
)
from src.data import load_fineweb_fixed_length
from data.genre_prompts import GENRES as REGISTERS, PROMPT_SETS, get_prompts


# === Reusable functions (from run_multi_contrastive.py) ===

def get_acts(model, tokenizer, device, prompts, layer):
    """Extract final-token activations at given layer for a list of prompts."""
    acts = []
    for p in prompts:
        tokens = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        acts.append(outputs.hidden_states[layer + 1][:, -1, :].squeeze(0).cpu())
    return torch.stack(acts)


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


def sweep_single_direction(model, contexts, activations, ref_final, ref_log_probs,
                           direction, angles_rad, layer, device):
    """Perturbation sweep for one fixed direction."""
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


def optimize_direction(model, contexts, activations, ref_log_probs,
                       V, angle_rad, layer, device,
                       n_steps=300, lr=0.01, n_restarts=5,
                       prev_alphas=None):
    """Find direction in span(V) maximizing mean KL divergence."""
    r = V.shape[1]
    best_kl = -float('inf')
    best_alpha = None
    best_history = None

    for restart in range(n_restarts):
        alpha = torch.randn(r, device=device)
        if prev_alphas:
            for pa in prev_alphas:
                alpha = alpha - (alpha @ pa) * pa
        alpha = alpha / alpha.norm()
        alpha = alpha.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([alpha], lr=lr)
        history = []

        for step in range(n_steps):
            optimizer.zero_grad()
            a_normed = alpha / alpha.norm()
            if prev_alphas:
                for pa in prev_alphas:
                    a_normed = a_normed - (a_normed @ pa) * pa
                a_normed = a_normed / (a_normed.norm() + 1e-10)
            direction = V @ a_normed
            hidden = build_perturbed_hidden(contexts, activations, direction, angle_rad)
            logits = forward_from_layer_allpos(model, hidden, layer)
            pert_log_probs = F.log_softmax(logits, dim=-1)
            kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1).mean()
            loss = -kl
            loss.backward()
            optimizer.step()
            if prev_alphas:
                with torch.no_grad():
                    for pa in prev_alphas:
                        alpha.data -= (alpha.data @ pa) * pa
            kl_val = kl.item()
            history.append(kl_val)
            if step % 50 == 0 or step == n_steps - 1:
                print(f"    restart {restart+1}/{n_restarts}, step {step}: KL = {kl_val:.6f}")

        final_kl = history[-1]
        if final_kl > best_kl:
            best_kl = final_kl
            with torch.no_grad():
                a_final = alpha / alpha.norm()
                if prev_alphas:
                    for pa in prev_alphas:
                        a_final = a_final - (a_final @ pa) * pa
                    a_final = a_final / a_final.norm()
                best_alpha = a_final.clone()
            best_history = history

    best_direction = (V @ best_alpha).detach()
    best_direction = best_direction / best_direction.norm()
    return best_alpha, best_direction, best_kl, best_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--n_act', type=int, default=50,
                        help='Number of FineWeb sequences for MELBO/sweeps')
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--opt_angle', type=float, default=15.0)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--n_melbo', type=int, default=3,
                        help='Number of MELBO directions to find on joint span')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_melbo', action='store_true',
                        help='Skip MELBO optimization (faster, just contrastive directions)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    k = len(REGISTERS)
    n_prompts = len(PROMPT_SETS)
    n_pairs = k * (k - 1) // 2
    print(f"Registers: {REGISTERS} (k={k})")
    print(f"Prompt sets: {n_prompts}")
    print(f"Pairwise contrasts: {n_pairs}")

    results_dir = f'results/genre_contrastive/{args.model}'
    os.makedirs(results_dir, exist_ok=True)

    # ===== Phase A: Extract activations per register =====
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    print("\nExtracting activations for each register...")
    register_acts = {}
    for reg in REGISTERS:
        prompts = get_prompts(reg)
        acts = get_acts(model, tokenizer, device, prompts, args.layer)
        register_acts[reg] = acts
        print(f"  {reg}: {acts.shape}")

    # ===== Phase B: All pairwise difference matrices =====
    print("\nComputing all pairwise difference matrices...")
    D_pairs = {}  # (reg_i, reg_j) -> D_ij [n_prompts, d_model]
    v_pairs = {}  # (reg_i, reg_j) -> v_ij (mean diff, unnormalized)
    for reg_i, reg_j in combinations(REGISTERS, 2):
        D_ij = register_acts[reg_i] - register_acts[reg_j]
        D_pairs[(reg_i, reg_j)] = D_ij
        v_ij = D_ij.mean(dim=0)
        v_pairs[(reg_i, reg_j)] = v_ij
        # Also store reverse direction for convenience
        v_pairs[(reg_j, reg_i)] = -v_ij
        norm = v_ij.norm().item()
        print(f"  D({reg_i},{reg_j}): shape {D_ij.shape}, ||mean|| = {norm:.2f}")

    # ===== Phase C: Direction estimation =====

    # --- C1: Pairwise diff-of-means (normalized) ---
    print("\nPairwise diff-of-means directions:")
    d_pairwise = {}
    for (reg_i, reg_j), v in v_pairs.items():
        if reg_i < reg_j:  # only store canonical direction
            d = v / v.norm()
            d_pairwise[(reg_i, reg_j)] = d
            print(f"  d({reg_i},{reg_j}): ||v|| = {v.norm():.2f}")

    # --- C2: All-pairwise least-squares estimator (Section 8) ---
    # f̂_i = (1/k) * Σ_{j≠i} v_ij
    print("\nAll-pairwise least-squares estimator (Section 8):")
    f_allpair = {}
    for reg_i in REGISTERS:
        f_raw = torch.zeros(d_model)
        for reg_j in REGISTERS:
            if reg_j != reg_i:
                f_raw += v_pairs[(reg_i, reg_j)]
        f_raw /= k
        f_norm = f_raw / f_raw.norm()
        f_allpair[reg_i] = {'raw': f_raw, 'normalized': f_norm}
        print(f"  f̂_{reg_i}: ||raw|| = {f_raw.norm():.4f}")

    # Check balance constraint: Σ f̂_i should be ~0
    balance = sum(f_allpair[r]['raw'] for r in REGISTERS)
    print(f"  ||Σ f̂_i|| (should be ~0) = {balance.norm():.6f}")

    # --- C3: Star-topology mean estimator (Section 4) ---
    # For each register as target, use all others as contrasts
    print("\nStar-topology mean estimator (Section 4):")
    f_star = {}
    for reg_i in REGISTERS:
        f_raw = torch.zeros(d_model)
        for reg_j in REGISTERS:
            if reg_j != reg_i:
                f_raw += v_pairs[(reg_i, reg_j)]
        f_raw /= (k - 1)
        f_norm = f_raw / f_raw.norm()
        f_star[reg_i] = {'raw': f_raw, 'normalized': f_norm}
        print(f"  f_star_{reg_i}: ||raw|| = {f_raw.norm():.4f}")

    # --- C4: SVD of stacked difference matrix (Section 5) ---
    # Stack all individual differences as rows
    print("\nSVD of stacked difference matrix:")
    all_diffs = []
    pair_labels = []
    for (reg_i, reg_j), D_ij in D_pairs.items():
        all_diffs.append(D_ij)
        pair_labels.extend([(reg_i, reg_j)] * D_ij.shape[0])
    D_stacked = torch.cat(all_diffs, dim=0)  # [n_pairs * n_prompts, d_model]
    print(f"  Stacked matrix: {D_stacked.shape}")

    U_all, S_all, Vh_all = torch.linalg.svd(D_stacked, full_matrices=False)
    V_all = Vh_all.T

    total_var = (S_all ** 2).sum()
    cumvar = (S_all[:10] ** 2).cumsum(0) / total_var
    print(f"  Top 10 singular values: {S_all[:10].numpy().round(2)}")
    print(f"  Cumulative variance (top 10): {cumvar.numpy().round(3)}")
    print(f"  σ₂/σ₁ = {(S_all[1] / S_all[0]):.4f}")

    # Leading right singular vector
    svd_dir_1 = Vh_all[0]
    print(f"  Leading right SV direction alignment with all-pair estimators:")
    for reg in REGISTERS:
        cos = (svd_dir_1 @ f_allpair[reg]['normalized']).abs().item()
        print(f"    cos(u₁, f̂_{reg}) = {cos:.4f}")

    # Left singular vector weights (Section 7 diagnostics)
    u1_left = U_all[:, 0]
    print(f"\n  Left SV weights by pair (Section 7 diagnostics):")
    idx = 0
    for (reg_i, reg_j), D_ij in D_pairs.items():
        n = D_ij.shape[0]
        weights = u1_left[idx:idx+n]
        mean_w = weights.mean().item()
        std_w = weights.std().item()
        print(f"    ({reg_i},{reg_j}): mean={mean_w:.4f}, std={std_w:.4f}")
        idx += n

    # Also compute per-pair SVD for individual spans
    V_pair_svd = {}
    S_pair_svd = {}
    for (reg_i, reg_j), D_ij in D_pairs.items():
        _, S_ij, Vh_ij = torch.linalg.svd(D_ij, full_matrices=False)
        V_pair_svd[(reg_i, reg_j)] = Vh_ij.T
        S_pair_svd[(reg_i, reg_j)] = S_ij
        ratio = (S_ij[1] / S_ij[0]).item() if S_ij[0] > 0 else float('inf')
        print(f"  Per-pair SVD ({reg_i},{reg_j}): σ₂/σ₁ = {ratio:.4f}, "
              f"top SV = {S_ij[0]:.2f}")

    # ===== Phase D: Cosine similarity matrix =====
    print(f"\n{'='*60}")
    print("COSINE SIMILARITY: All-pairwise vs Star-topology vs Pairwise")
    print(f"{'='*60}")

    # Compare estimators
    print("\nAll-pair vs star-topology estimator alignment:")
    for reg in REGISTERS:
        cos_val = (f_allpair[reg]['normalized'] @ f_star[reg]['normalized']).item()
        print(f"  {reg}: cos = {cos_val:.6f}")

    # Mutual cosine similarity of all-pair estimators
    print("\nMutual cosines of all-pair estimators f̂_i:")
    header = f"{'':>12s}" + "".join(f"{r:>12s}" for r in REGISTERS)
    print(header)
    for reg_i in REGISTERS:
        row = f"{reg_i:>12s}"
        for reg_j in REGISTERS:
            cos_val = (f_allpair[reg_i]['normalized'] @ f_allpair[reg_j]['normalized']).item()
            row += f"{cos_val:>12.3f}"
        print(row)

    # Alignment of all-pair estimators with pairwise diffs
    print("\nAll-pair estimator alignment with pairwise diffs:")
    for reg in REGISTERS:
        alignments = []
        for (reg_i, reg_j), d in d_pairwise.items():
            if reg_i == reg or reg_j == reg:
                sign = 1.0 if reg_i == reg else -1.0
                cos_val = sign * (f_allpair[reg]['normalized'] @ d).item()
                alignments.append((f"d({reg_i},{reg_j})", cos_val))
        print(f"  f̂_{reg}:")
        for name, cos_val in alignments:
            print(f"    cos with {name}: {cos_val:.4f}")

    # ===== Phase E: FineWeb activations for sweeps =====
    print(f"\nLoading {args.n_act} FineWeb sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.layer} activations...")
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    contexts = contexts.to(device)
    activations = activations.to(device)

    # Reference logits
    with torch.no_grad():
        ref_hidden = torch.cat([contexts, activations.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_hidden_and_logits(model, ref_hidden, args.layer)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # ===== Phase F: MELBO on joint span (optional) =====
    melbo_results = {}
    if not args.skip_melbo:
        V_joint = V_all.to(device)
        opt_angle_rad = np.radians(args.opt_angle)

        print(f"\n{'='*60}")
        print(f"MELBO on joint span ({args.n_melbo} directions)")
        print(f"{'='*60}")

        joint_alphas = []
        joint_directions = []
        joint_kls = []
        for i in range(args.n_melbo):
            print(f"\n  Direction {i+1}/{args.n_melbo}:")
            alpha, direction, kl, history = optimize_direction(
                model, contexts, activations, ref_log_probs,
                V_joint, opt_angle_rad, args.layer, device,
                n_steps=args.n_steps, lr=0.01, n_restarts=args.n_restarts,
                prev_alphas=joint_alphas if joint_alphas else None)
            joint_alphas.append(alpha)
            joint_directions.append(direction.cpu())
            joint_kls.append(kl)
            print(f"  -> MELBO_joint_{i+1}: KL = {kl:.4f}")

        melbo_results = {
            'alphas': [a.cpu() for a in joint_alphas],
            'directions': joint_directions,
            'kls': joint_kls,
        }

    # ===== Phase G: Perturbation sweeps =====
    angles_deg = np.linspace(3, 180, args.n_angles)
    angles_rad = np.radians(angles_deg)

    # Collect all directions to sweep
    all_directions = {}

    # All-pairwise estimators
    for reg in REGISTERS:
        all_directions[f'f_allpair_{reg}'] = f_allpair[reg]['normalized']

    # Star-topology estimators
    for reg in REGISTERS:
        all_directions[f'f_star_{reg}'] = f_star[reg]['normalized']

    # A few pairwise diffs for reference
    for (reg_i, reg_j), d in d_pairwise.items():
        all_directions[f'd_{reg_i}_{reg_j}'] = d

    # MELBO directions
    if melbo_results:
        for i, d in enumerate(melbo_results['directions']):
            all_directions[f'MELBO_joint_{i+1}'] = d

    print(f"\n{'='*60}")
    print(f"Perturbation sweeps ({len(all_directions)} directions + random)")
    print(f"{'='*60}")

    sweep_results = {}
    for name, direction in all_directions.items():
        print(f"  Sweeping {name}...")
        kl, l2 = sweep_single_direction(
            model, contexts, activations, ref_final, ref_log_probs,
            direction, angles_rad, args.layer, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}

    print("  Sweeping random directions...")
    rand_kl, rand_l2 = sweep_random_directions(
        model, contexts, activations, ref_final, ref_log_probs,
        angles_rad, args.layer, device, n_dirs=args.n_random_dirs)
    sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}

    # ===== Phase H: Save =====
    save_data = {
        'args': vars(args),
        'registers': REGISTERS,
        'n_prompts': n_prompts,
        # All-pairwise estimators (Section 8)
        'f_allpair': {r: {'raw': f_allpair[r]['raw'], 'normalized': f_allpair[r]['normalized']}
                      for r in REGISTERS},
        # Star-topology estimators (Section 4)
        'f_star': {r: {'raw': f_star[r]['raw'], 'normalized': f_star[r]['normalized']}
                   for r in REGISTERS},
        # Pairwise diff-of-means
        'd_pairwise': {f'{ri}_{rj}': d for (ri, rj), d in d_pairwise.items()},
        # Pairwise difference matrices
        'D_pairs': {f'{ri}_{rj}': D for (ri, rj), D in D_pairs.items()},
        # SVD results
        'svd_joint': {'S': S_all, 'Vh': Vh_all, 'U_left_col0': U_all[:, 0]},
        'svd_per_pair': {f'{ri}_{rj}': S for (ri, rj), S in S_pair_svd.items()},
        # MELBO
        'melbo': melbo_results,
        # All directions (for convenience)
        'all_directions': {name: d.cpu() for name, d in all_directions.items()},
    }
    with open(os.path.join(results_dir, 'direction_results.pkl'), 'wb') as f:
        pickle.dump(save_data, f)

    sweep_save = {
        'angles_deg': angles_deg,
        'sweep_results': sweep_results,
    }
    with open(os.path.join(results_dir, 'sweep_results.pkl'), 'wb') as f:
        pickle.dump(sweep_save, f)

    print(f"\nSaved to {results_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
