#!/usr/bin/env python3
"""
Multi-contrastive language direction isolation experiment.

Uses three language pairs (EN-ZH, EN-NL, ZH-NL) to compute:
1. Pairwise diff-of-means directions (d_EM, d_ED, d_MD)
2. Multi-contrastive mean estimators (f_E, f_M, f_D)
3. MELBO on joint span vs individual spans
4. Perturbation sweeps for all directions
5. Cosine similarity matrix

Usage:
    python scripts/run_multi_contrastive.py [--model qwen3-0.6b] [--layer 14]
"""

import sys
import argparse
import os
import pickle
import time
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
from data.bilingual_prompts import ENGLISH_PROMPTS as EN_PROMPTS_ZH, CHINESE_PROMPTS
from data.bilingual_prompts_dutch import ENGLISH_PROMPTS as EN_PROMPTS_NL, DUTCH_PROMPTS
from data.bilingual_prompts_mandarin_dutch import (
    CHINESE_PROMPTS as ZH_PROMPTS_NL, DUTCH_PROMPTS as NL_PROMPTS_ZH,
)


# === Reusable functions from run_melbo_constrained.py ===

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
            l2_dists[:, ang_idx] = l2.float().cpu().numpy()
            kl_divs[:, ang_idx] = kl.float().cpu().numpy()
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
    act_dtype = activations.dtype
    v_perps = torch.zeros(N, n_dirs, d, device=device, dtype=act_dtype)
    for i in range(N):
        v = torch.randn(n_dirs, d, device=device, dtype=act_dtype)
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
                l2_dists[:, ang_idx, dir_idx] = l2.float().cpu().numpy()
                kl_divs[:, ang_idx, dir_idx] = kl.float().cpu().numpy()
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
        alpha = torch.randn(r, device=device, dtype=V.dtype)
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
    parser.add_argument('--n_act', type=int, default=50)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--opt_angle', type=float, default=15.0)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bfloat16', action='store_true',
                        help='Load model in bfloat16 (needed for larger models on 24GB)')
    parser.add_argument('--no-melbo', action='store_true',
                        help='Skip MELBO optimization (just do directions + sweeps)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_dir = f'results/multi_contrastive/{args.model}'
    os.makedirs(results_dir, exist_ok=True)

    # ===== Phase A: Collect activation differences =====
    print(f"Loading model: {args.model}")
    load_dtype = torch.bfloat16 if args.bfloat16 else None
    model, tokenizer, device = load_model(args.model, dtype=load_dtype)
    if args.bfloat16:
        print("  Loaded in bfloat16")
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    print("\nExtracting activations for all language pairs...")
    en_acts_zh = get_acts(model, tokenizer, device, EN_PROMPTS_ZH, args.layer)
    zh_acts = get_acts(model, tokenizer, device, CHINESE_PROMPTS, args.layer)
    en_acts_nl = get_acts(model, tokenizer, device, EN_PROMPTS_NL, args.layer)
    nl_acts_en = get_acts(model, tokenizer, device, DUTCH_PROMPTS, args.layer)
    zh_acts_nl = get_acts(model, tokenizer, device, ZH_PROMPTS_NL, args.layer)
    nl_acts_zh = get_acts(model, tokenizer, device, NL_PROMPTS_ZH, args.layer)

    # Difference matrices
    D_EM = en_acts_zh - zh_acts       # [50, d_model]
    D_ED = en_acts_nl - nl_acts_en    # [50, d_model]
    D_MD = zh_acts_nl - nl_acts_zh    # [50, d_model]

    print(f"  D_EM: {D_EM.shape}, D_ED: {D_ED.shape}, D_MD: {D_MD.shape}")

    # Pairwise diff-of-means directions
    d_EM = D_EM.mean(dim=0)
    d_EM = d_EM / d_EM.norm()
    d_ED = D_ED.mean(dim=0)
    d_ED = d_ED / d_ED.norm()
    d_MD = D_MD.mean(dim=0)
    d_MD = d_MD / d_MD.norm()

    print(f"  d_EM norm (pre-normalize): {D_EM.mean(dim=0).norm():.2f}")
    print(f"  d_ED norm (pre-normalize): {D_ED.mean(dim=0).norm():.2f}")
    print(f"  d_MD norm (pre-normalize): {D_MD.mean(dim=0).norm():.2f}")

    # ===== Phase B: Multi-contrastive mean estimators =====
    print("\nComputing multi-contrastive estimators...")
    f_E_raw = (d_EM + d_ED) / 2
    f_E = f_E_raw / f_E_raw.norm()
    f_M_raw = (-d_EM + d_MD) / 2
    f_M = f_M_raw / f_M_raw.norm()
    f_D_raw = (-d_ED - d_MD) / 2
    f_D = f_D_raw / f_D_raw.norm()

    # Sanity check: should sum to ~0 (before normalization)
    residual = (f_E_raw + f_M_raw + f_D_raw).norm()
    print(f"  ||f_E + f_M + f_D|| (before normalization) = {residual:.6f}")

    # ===== Phase C: MELBO on joint and individual spans =====
    print("\nComputing SVDs...")
    # SVD requires float32 (not supported for bfloat16 on CPU)
    # Cast V matrices back to model dtype when moving to device for consistent forward passes
    model_dtype = next(model.parameters()).dtype
    D_joint = torch.cat([D_EM, D_ED, D_MD], dim=0).float()  # [150, d_model]
    U_j, S_j, Vh_j = torch.linalg.svd(D_joint, full_matrices=False)
    V_joint = Vh_j.T.to(dtype=model_dtype, device=device)
    print(f"  Joint SVD: shape {D_joint.shape}, rank={V_joint.shape[1]}, "
          f"top SV={S_j[0]:.2f}, var_explained_top5={((S_j[:5]**2).sum() / (S_j**2).sum()):.3f}")

    U_em, S_em, Vh_em = torch.linalg.svd(D_EM.float(), full_matrices=False)
    V_EM = Vh_em.T.to(dtype=model_dtype, device=device)
    U_ed, S_ed, Vh_ed = torch.linalg.svd(D_ED.float(), full_matrices=False)
    V_ED = Vh_ed.T.to(dtype=model_dtype, device=device)
    U_md, S_md, Vh_md = torch.linalg.svd(D_MD.float(), full_matrices=False)
    V_MD = Vh_md.T.to(dtype=model_dtype, device=device)
    print(f"  V_EM rank={V_EM.shape[1]}, V_ED rank={V_ED.shape[1]}, V_MD rank={V_MD.shape[1]}")

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

    opt_angle_rad = np.radians(args.opt_angle)

    if not args.no_melbo:
        # MELBO on joint span (3 directions)
        print(f"\n{'='*60}")
        print("MELBO on JOINT span (EN-ZH + EN-NL + ZH-NL)")
        print(f"{'='*60}")
        joint_alphas = []
        joint_directions = []
        joint_kls = []
        for i in range(3):
            print(f"\n  Direction {i+1}/3:")
            alpha, direction, kl, history = optimize_direction(
                model, contexts, activations, ref_log_probs,
                V_joint, opt_angle_rad, args.layer, device,
                n_steps=args.n_steps, lr=0.01, n_restarts=args.n_restarts,
                prev_alphas=joint_alphas if joint_alphas else None)
            joint_alphas.append(alpha)
            joint_directions.append(direction.cpu())
            joint_kls.append(kl)
            print(f"  -> MELBO_joint_{i+1}: KL = {kl:.4f}")

        # MELBO on individual spans (1 direction each)
        print(f"\n{'='*60}")
        print("MELBO on individual spans")
        print(f"{'='*60}")
        individual_melbo = {}
        for name, V_sub in [('EM', V_EM), ('ED', V_ED), ('MD', V_MD)]:
            print(f"\n  MELBO on {name} span:")
            _, direction, kl, _ = optimize_direction(
                model, contexts, activations, ref_log_probs,
                V_sub, opt_angle_rad, args.layer, device,
                n_steps=args.n_steps, lr=0.01, n_restarts=args.n_restarts)
            individual_melbo[name] = {'direction': direction.cpu(), 'kl': kl}
            print(f"  -> MELBO_{name}: KL = {kl:.4f}")
    else:
        joint_alphas = []
        joint_directions = []
        joint_kls = []
        individual_melbo = {}

    # ===== Phase E: Perturbation sweeps =====
    angles_deg = np.linspace(3, 180, args.n_angles)
    angles_rad = np.radians(angles_deg)

    # SVD leading directions (top right singular vectors)
    svd_dir_joint = Vh_j[0].to(dtype=model_dtype, device='cpu')
    svd_dir_EM = Vh_em[0].to(dtype=model_dtype, device='cpu')
    svd_dir_ED = Vh_ed[0].to(dtype=model_dtype, device='cpu')
    svd_dir_MD = Vh_md[0].to(dtype=model_dtype, device='cpu')

    all_directions = {
        'd_EM': d_EM,
        'd_ED': d_ED,
        'd_MD': d_MD,
        'f_E': f_E,
        'f_M': f_M,
        'f_D': f_D,
        'SVD_joint_1': svd_dir_joint,
        'SVD_EM_1': svd_dir_EM,
        'SVD_ED_1': svd_dir_ED,
        'SVD_MD_1': svd_dir_MD,
    }
    if not args.no_melbo:
        all_directions.update({
            'MELBO_joint_1': joint_directions[0],
            'MELBO_joint_2': joint_directions[1],
            'MELBO_joint_3': joint_directions[2],
            'MELBO_EM': individual_melbo['EM']['direction'],
            'MELBO_ED': individual_melbo['ED']['direction'],
            'MELBO_MD': individual_melbo['MD']['direction'],
        })

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

    # ===== Phase F: Cosine similarity matrix =====
    print(f"\n{'='*60}")
    print("COSINE SIMILARITY MATRIX")
    print(f"{'='*60}")

    dir_names = list(all_directions.keys())
    dir_tensors = [all_directions[n] for n in dir_names]

    # Also load existing MELBO_1 from constrained optimization for reference
    existing_path = f'results/melbo_constrained/{args.model}/optimization_results.pkl'
    if os.path.exists(existing_path):
        with open(existing_path, 'rb') as f:
            existing = pickle.load(f)
        old_mean = existing['mean_dir']
        old_melbo1 = existing['melbo_directions'][0]
        dir_names.extend(['old_mean', 'old_MELBO_1'])
        dir_tensors.extend([old_mean, old_melbo1])

    n = len(dir_names)
    cos_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_matrix[i, j] = (dir_tensors[i] @ dir_tensors[j]).item()

    # Print matrix
    header = f"{'':>16s}" + "".join(f"{n[:12]:>13s}" for n in dir_names)
    print(header)
    for i, name in enumerate(dir_names):
        row = f"{name:>16s}"
        for j in range(n):
            row += f"{cos_matrix[i, j]:>13.3f}"
        print(row)

    # ===== Phase G: Save =====
    save_data = {
        'args': vars(args),
        'directions': {name: d.cpu() for name, d in all_directions.items()},
        'cosine_matrix': cos_matrix,
        'cosine_names': dir_names,
        'D_EM': D_EM, 'D_ED': D_ED, 'D_MD': D_MD,
        'singular_values': {'joint': S_j, 'EM': S_em, 'ED': S_ed, 'MD': S_md},
    }
    if not args.no_melbo:
        save_data['joint_alphas'] = [a.cpu() for a in joint_alphas]
        save_data['joint_kls'] = joint_kls
        save_data['individual_melbo'] = individual_melbo
    with open(os.path.join(results_dir, 'optimization_results.pkl'), 'wb') as f:
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
