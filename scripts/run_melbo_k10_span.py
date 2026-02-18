#!/usr/bin/env python3
"""
MELBO in the k=10 multi-contrastive span.

Find directions within the 9D subspace spanned by the k=10 pairwise
difference vectors that maximize KL divergence. Then check whether
these directions correspond to individual language features.

Usage:
    python scripts/run_melbo_k10_span.py
"""

import sys, os, pickle, time, numpy as np
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import load_model, forward_from_layer_allpos, _get_blocks
from src.data import load_fineweb_fixed_length
from data.multilingual_prompts import PROMPT_SETS, LANGUAGES

from scripts.run_multi_contrastive import (
    get_acts, extract_context_and_activation,
    forward_to_hidden_and_logits, build_perturbed_hidden,
    sweep_single_direction,
)

MODEL = 'qwen3-0.6b'
LAYER = 14
N_ACT = 50
SEQ_LEN = 5
SEED = 42

# Optimization params (beefed up for 9D search)
OPT_ANGLE = 15.0
N_STEPS = 500
N_RESTARTS = 20
LR = 0.01
N_DIRECTIONS = 5  # find top-5 orthogonal directions


def optimize_direction(model, contexts, activations, ref_log_probs,
                       V, angle_rad, layer, device,
                       n_steps=500, lr=0.01, n_restarts=20,
                       prev_alphas=None):
    """Find direction in span(V) maximizing mean KL divergence.

    Returns: best_alpha, best_direction, best_kl, all_final_kls, best_history
    """
    r = V.shape[1]
    best_kl = -float('inf')
    best_alpha = None
    best_history = None
    all_final_kls = []

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

            if step % 100 == 0 or step == n_steps - 1:
                print(f"    restart {restart+1}/{n_restarts}, step {step}: "
                      f"KL = {kl_val:.6f}")

        final_kl = history[-1]
        all_final_kls.append(final_kl)
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
    return best_alpha, best_direction, best_kl, all_final_kls, best_history


def main():
    results_dir = 'results/scaling_test/qwen3-0.6b'

    print("Loading model...")
    model, tokenizer, device = load_model(MODEL)

    # Load scaling results for use_indices and pairwise dirs
    with open(os.path.join(results_dir, 'scaling_results.pkl'), 'rb') as f:
        scaling = pickle.load(f)

    non_en = scaling['non_en']
    use_indices = scaling['use_indices']
    prompt_sets = [PROMPT_SETS[i] for i in use_indices]

    # Extract activations
    print("\nExtracting activations...")
    acts = {}
    for lang in LANGUAGES:
        prompts = [pset[lang] for pset in prompt_sets]
        acts[lang] = get_acts(model, tokenizer, device, prompts, LAYER)

    # Stack all EN-X difference matrices
    print("\nBuilding difference matrix...")
    diff_matrices = []
    for lang in non_en:
        diffs = acts['en'] - acts[lang]  # [30, d_model]
        diff_matrices.append(diffs)
    D_all = torch.cat(diff_matrices, dim=0)  # [270, d_model]
    print(f"  D_all shape: {D_all.shape}")

    # SVD
    U, S, Vh = torch.linalg.svd(D_all, full_matrices=False)
    V = Vh.T.to(device)  # [d_model, 270] but we only need the top components
    r_eff = min(V.shape[1], 9 * len(use_indices))  # 9 languages * 30 prompts max
    print(f"  Top singular values: {S[:15].tolist()}")
    var_explained = (S[:9]**2).sum() / (S**2).sum()
    print(f"  Top 9 components explain {var_explained:.1%} of variance")

    # Use top-9 singular vectors (matches the 9 language contrasts)
    V9 = V[:, :9]
    print(f"  Using top-9 SVD components (V9 shape: {V9.shape})")

    # Also compute the pairwise directions for comparison
    pairwise_dirs = {}
    for lang in non_en:
        d = F.normalize((acts['en'].mean(0) - acts[lang].mean(0)).unsqueeze(0), dim=-1).squeeze(0)
        pairwise_dirs[lang] = d

    # Reference direction (k=10 SVD = first singular vector, sign-corrected)
    svd_k10 = Vh[0].to(device)
    ref_mean = scaling['ref_direction'].to(device)
    if F.cosine_similarity(svd_k10.unsqueeze(0), ref_mean.unsqueeze(0)).item() < 0:
        svd_k10 = -svd_k10

    # === FineWeb activations ===
    print(f"\nLoading {N_ACT} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(N_ACT, tokenizer, seq_len=SEQ_LEN, seed=SEED)
    contexts, activations_fw = extract_context_and_activation(
        model, tokenizer, token_ids_list, LAYER, device)
    contexts = contexts.to(device)
    activations_fw = activations_fw.to(device)

    with torch.no_grad():
        ref_hidden = torch.cat([contexts, activations_fw.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_hidden_and_logits(model, ref_hidden, LAYER)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # === MELBO optimization ===
    angle_rad = np.deg2rad(OPT_ANGLE)
    print(f"\n{'='*60}")
    print(f"MELBO in 9D span: top-{N_DIRECTIONS} directions")
    print(f"Angle: {OPT_ANGLE}°, steps: {N_STEPS}, restarts: {N_RESTARTS}")
    print(f"{'='*60}")

    melbo_alphas = []
    melbo_directions = []
    melbo_kls = []
    melbo_all_restart_kls = []
    melbo_histories = []

    for d_idx in range(N_DIRECTIONS):
        print(f"\n--- Direction {d_idx+1}/{N_DIRECTIONS} ---")
        t0 = time.time()
        alpha, direction, best_kl, all_final_kls, history = optimize_direction(
            model, contexts, activations_fw, ref_log_probs,
            V9, angle_rad, LAYER, device,
            n_steps=N_STEPS, lr=LR, n_restarts=N_RESTARTS,
            prev_alphas=melbo_alphas if melbo_alphas else None
        )
        elapsed = time.time() - t0

        melbo_alphas.append(alpha)
        melbo_directions.append(direction.cpu())
        melbo_kls.append(best_kl)
        melbo_all_restart_kls.append(all_final_kls)
        melbo_histories.append(history)

        # Diagnostics
        print(f"\n  Best KL: {best_kl:.6f}")
        print(f"  All restart final KLs: {[f'{k:.4f}' for k in sorted(all_final_kls, reverse=True)]}")
        print(f"  Restart convergence: min={min(all_final_kls):.4f}, "
              f"max={max(all_final_kls):.4f}, "
              f"std={np.std(all_final_kls):.4f}")

        # Cosines with pairwise directions
        print(f"  Cosines with pairwise dirs:")
        dir_cpu = direction.cpu()
        for lang in non_en:
            cos = F.cosine_similarity(
                dir_cpu.unsqueeze(0),
                pairwise_dirs[lang].cpu().unsqueeze(0)).item()
            print(f"    cos(MELBO_{d_idx+1}, d_E_{lang}) = {cos:+.4f}")

        cos_svd = F.cosine_similarity(
            dir_cpu.unsqueeze(0), svd_k10.cpu().unsqueeze(0)).item()
        print(f"    cos(MELBO_{d_idx+1}, k=10 SVD) = {cos_svd:+.4f}")

        if d_idx > 0:
            for prev_idx in range(d_idx):
                cos_prev = F.cosine_similarity(
                    dir_cpu.unsqueeze(0),
                    melbo_directions[prev_idx].cpu().unsqueeze(0)).item()
                print(f"    cos(MELBO_{d_idx+1}, MELBO_{prev_idx+1}) = {cos_prev:+.4f}")

        print(f"  Time: {elapsed:.1f}s")

    # === Perturbation sweeps ===
    print(f"\n{'='*60}")
    print("Perturbation sweeps")
    print(f"{'='*60}")

    angles_deg = np.linspace(3, 180, 60)
    angles_rad_sweep = np.radians(angles_deg)

    sweep_results = {}

    # MELBO directions
    for i, d in enumerate(melbo_directions):
        name = f'MELBO_{i+1}'
        print(f"  Sweeping {name}...")
        kl, l2 = sweep_single_direction(
            model, contexts, activations_fw, ref_final, ref_log_probs,
            d, angles_rad_sweep, LAYER, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}

    # Reference directions
    for name, d in [('k=10 SVD', svd_k10.cpu()), ('k=10 mean', ref_mean.cpu())]:
        print(f"  Sweeping {name}...")
        kl, l2 = sweep_single_direction(
            model, contexts, activations_fw, ref_final, ref_log_probs,
            d, angles_rad_sweep, LAYER, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}

    # === Save ===
    save_data = {
        'melbo_alphas': [a.cpu() for a in melbo_alphas],
        'melbo_directions': melbo_directions,
        'melbo_kls': melbo_kls,
        'melbo_all_restart_kls': melbo_all_restart_kls,
        'melbo_histories': melbo_histories,
        'sweep_results': sweep_results,
        'angles_deg': angles_deg,
        'pairwise_dirs': {k: v.cpu() for k, v in pairwise_dirs.items()},
        'svd_k10': svd_k10.cpu(),
        'V9': V9.cpu(),
        'singular_values': S.cpu(),
        'opt_params': {
            'angle': OPT_ANGLE,
            'n_steps': N_STEPS,
            'n_restarts': N_RESTARTS,
            'lr': LR,
            'n_directions': N_DIRECTIONS,
        },
    }
    with open(os.path.join(results_dir, 'melbo_k10_span.pkl'), 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved: {results_dir}/melbo_k10_span.pkl")
    print("Done!")


if __name__ == '__main__':
    main()
