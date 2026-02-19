#!/usr/bin/env python3
"""
Multi-contrastive programming language direction isolation experiment.

Implements the all-pairwise contrasts procedure from Section 8 of
multi_contrastive_projection.pdf:

1. Extract activations for all k(k-1)/2 language pairs
2. Compute pairwise mean difference vectors v̄_ij
3. Recover per-language features via least-squares: f̂_i = (1/k) Σ_{j≠i} v̄_ij
4. SVD of stacked difference matrix for comparison
5. Diagnostics: σ₂/σ₁, cosine similarities, feature sum check
6. Perturbation sweeps

Usage:
    python scripts/directions/run_programming_contrastive.py [--model qwen3-0.6b] [--layer 14]
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
from data.programming_prompts import PROMPT_SETS, LANGUAGES


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


# === New code for all-pairwise contrasts ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--n_act', type=int, default=50,
                        help='Number of FineWeb activations for perturbation sweeps')
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--skip_sweeps', action='store_true',
                        help='Skip perturbation sweeps (faster, for debugging)')
    parser.add_argument('--languages', nargs='+', default=None,
                        help='Subset of languages to use (default: all from programming_prompts.py)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve languages
    langs = args.languages if args.languages else LANGUAGES
    k = len(langs)
    print(f"Languages ({k}): {langs}")

    results_dir = f'results/programming_contrastive/{args.model}'
    os.makedirs(results_dir, exist_ok=True)

    # ===== Phase A: Extract activations for each language =====
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    # Gather prompts per language
    prompts_per_lang = {lang: [] for lang in langs}
    for pset in PROMPT_SETS:
        for lang in langs:
            prompts_per_lang[lang].append(pset[lang])
    n_prompts = len(PROMPT_SETS)
    print(f"\n{n_prompts} matched prompt sets across {k} languages")

    # Extract activations for each language
    print(f"\nExtracting activations (layer {args.layer})...")
    acts = {}  # lang -> [n_prompts, d_model]
    for lang in langs:
        acts[lang] = get_acts(model, tokenizer, device, prompts_per_lang[lang], args.layer)
        print(f"  {lang}: {acts[lang].shape}")

    # ===== Phase B: Compute all pairwise mean differences =====
    print(f"\n{'='*60}")
    print("PAIRWISE MEAN DIFFERENCES")
    print(f"{'='*60}")

    pairs = list(combinations(langs, 2))
    n_pairs = len(pairs)
    print(f"{n_pairs} pairs for {k} languages")

    # v_bar[(i,j)] = mean(a_i - a_j) across matched prompts
    v_bar = {}
    D_all = {}  # individual difference matrices for SVD
    for lang_i, lang_j in pairs:
        diff = acts[lang_i] - acts[lang_j]  # [n_prompts, d_model]
        v_bar[(lang_i, lang_j)] = diff.mean(dim=0)
        # Store anti-symmetric counterpart
        v_bar[(lang_j, lang_i)] = -v_bar[(lang_i, lang_j)]
        D_all[(lang_i, lang_j)] = diff
        norm = v_bar[(lang_i, lang_j)].norm().item()
        print(f"  v̄({lang_i}, {lang_j}): norm = {norm:.2f}")

    # ===== Phase C: Recover feature vectors (Section 8 least-squares) =====
    print(f"\n{'='*60}")
    print("ALL-PAIRWISE FEATURE RECOVERY (Section 8)")
    print(f"{'='*60}")

    # f̂_i = (1/k) Σ_{j≠i} v̄_ij
    f_hat = {}
    f_hat_raw = {}  # before normalization
    for lang_i in langs:
        total = torch.zeros(d_model)
        for lang_j in langs:
            if lang_j != lang_i:
                total += v_bar[(lang_i, lang_j)]
        f_raw = total / k
        f_hat_raw[lang_i] = f_raw
        f_hat[lang_i] = f_raw / f_raw.norm()
        print(f"  f̂_{lang_i}: norm(raw) = {f_raw.norm():.4f}")

    # Sanity check: features should sum to ~0 (translation ambiguity)
    feature_sum = sum(f_hat_raw.values())
    print(f"\n  ||Σ f̂_i|| (before normalization) = {feature_sum.norm():.6f}  (should be ~0)")

    # ===== Phase C2: Star-topology estimators for comparison =====
    print(f"\n{'='*60}")
    print("STAR-TOPOLOGY ESTIMATORS (Section 4, for comparison)")
    print(f"{'='*60}")

    # For each target language, use only the (k-1) contrasts through it
    f_star = {}
    f_star_raw = {}
    for lang_t in langs:
        total = torch.zeros(d_model)
        for lang_j in langs:
            if lang_j != lang_t:
                total += v_bar[(lang_t, lang_j)]
        f_raw = total / (k - 1)
        f_star_raw[lang_t] = f_raw
        f_star[lang_t] = f_raw / f_raw.norm()
        # How different from all-pairwise?
        cos = (f_hat[lang_t] @ f_star[lang_t]).item()
        print(f"  f*_{lang_t}: norm(raw) = {f_raw.norm():.4f}, cos(f̂, f*) = {cos:.4f}")

    # ===== Phase D: SVD estimator =====
    print(f"\n{'='*60}")
    print("SVD ESTIMATOR")
    print(f"{'='*60}")

    # Stack ALL individual difference vectors (not pre-averaged)
    all_diffs = []
    pair_labels = []  # track which pair each row belongs to
    for (lang_i, lang_j), diff_matrix in D_all.items():
        all_diffs.append(diff_matrix)
        pair_labels.extend([(lang_i, lang_j)] * diff_matrix.shape[0])
    V_stacked = torch.cat(all_diffs, dim=0)  # [N_total, d_model]
    print(f"  Stacked difference matrix: {V_stacked.shape}")

    U, S, Vh = torch.linalg.svd(V_stacked, full_matrices=False)
    print(f"  Top 10 singular values: {S[:10].numpy()}")
    print(f"  σ₂/σ₁ = {S[1]/S[0]:.4f}")
    print(f"  σ₃/σ₁ = {S[2]/S[0]:.4f}")

    # Variance explained by top-r
    total_var = (S**2).sum()
    for r in [1, 2, 3, 5, 8]:
        if r <= len(S):
            var_r = (S[:r]**2).sum() / total_var
            print(f"  Variance explained (top-{r}): {var_r:.4f}")

    # Leading right singular vector
    u1_right = Vh[0]  # leading right singular vector ≈ dominant direction
    u1_right = u1_right / u1_right.norm()
    print(f"\n  Leading right SV cosine with f̂_i:")
    for lang in langs:
        cos = (u1_right @ f_hat[lang]).item()
        print(f"    {lang}: {cos:.4f}")

    # ===== Phase E: Diagnostics =====
    print(f"\n{'='*60}")
    print("DIAGNOSTICS")
    print(f"{'='*60}")

    # Left singular vector weights (Section 7)
    u1_left = U[:, 0]  # [N_total]
    # Group by pair and compute average weight per pair
    print("\n  Left SV weights by pair (should be ~uniform):")
    pair_weights = {}
    idx = 0
    for (lang_i, lang_j), diff_matrix in D_all.items():
        n = diff_matrix.shape[0]
        pair_weight = u1_left[idx:idx+n].mean().item()
        pair_weights[(lang_i, lang_j)] = pair_weight
        idx += n
    # Sort by absolute weight
    sorted_pairs = sorted(pair_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    for (li, lj), w in sorted_pairs:
        print(f"    ({li}, {lj}): {w:.4f}")

    # Cosine similarity matrix between recovered features
    print(f"\n  COSINE SIMILARITY MATRIX (f̂ all-pairwise)")
    header = f"{'':>14s}" + "".join(f"{l[:10]:>12s}" for l in langs)
    print(header)
    cos_matrix = np.zeros((k, k))
    for i, li in enumerate(langs):
        row = f"{li:>14s}"
        for j, lj in enumerate(langs):
            cos_val = (f_hat[li] @ f_hat[lj]).item()
            cos_matrix[i, j] = cos_val
            row += f"{cos_val:>12.3f}"
        print(row)

    # Also show star vs all-pairwise cosine matrix
    print(f"\n  COSINE: f̂ (all-pairwise) vs f* (star-topology)")
    for lang in langs:
        cos = (f_hat[lang] @ f_star[lang]).item()
        print(f"    {lang}: {cos:.4f}")

    # ===== Phase F: Perturbation sweeps =====
    if not args.skip_sweeps:
        print(f"\n{'='*60}")
        print("PERTURBATION SWEEPS")
        print(f"{'='*60}")

        # Load FineWeb activations
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

        angles_deg = np.linspace(3, 180, args.n_angles)
        angles_rad = np.radians(angles_deg)

        # Collect all directions to sweep
        all_directions = {}
        for lang in langs:
            all_directions[f'f_hat_{lang}'] = f_hat[lang]
            all_directions[f'f_star_{lang}'] = f_star[lang]
        all_directions['svd_u1'] = u1_right.cpu()

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
    else:
        sweep_results = None
        angles_deg = None
        print("\n(Skipping perturbation sweeps)")

    # ===== Phase G: Save =====
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    save_data = {
        'args': vars(args),
        'languages': langs,
        'n_prompts': len(PROMPT_SETS),
        # Feature vectors
        'f_hat': {lang: f_hat[lang].cpu() for lang in langs},
        'f_hat_raw': {lang: f_hat_raw[lang].cpu() for lang in langs},
        'f_star': {lang: f_star[lang].cpu() for lang in langs},
        'f_star_raw': {lang: f_star_raw[lang].cpu() for lang in langs},
        # Pairwise data
        'v_bar': {f"{li}_{lj}": v.cpu() for (li, lj), v in v_bar.items()},
        'D_all': {f"{li}_{lj}": d.cpu() for (li, lj), d in D_all.items()},
        # SVD
        'singular_values': S.cpu(),
        'svd_u1_right': u1_right.cpu(),
        'svd_u1_left': u1_left.cpu() if not args.skip_sweeps else u1_left,
        # Diagnostics
        'cos_matrix': cos_matrix,
        'pair_weights': {f"{li}_{lj}": w for (li, lj), w in pair_weights.items()},
        # Per-language activations
        'activations': {lang: acts[lang].cpu() for lang in langs},
    }
    with open(os.path.join(results_dir, 'contrastive_results.pkl'), 'wb') as f:
        pickle.dump(save_data, f)

    if sweep_results is not None:
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
