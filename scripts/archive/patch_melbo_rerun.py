#!/usr/bin/env python3
"""
Patch: Re-run MELBO optimizations with more restarts and add negative directions.

The original run_multi_contrastive.py MELBO got stuck in bad local optima (5 restarts).
This script:
1. Loads existing difference matrices and FineWeb data
2. Re-runs all MELBO optimizations with 20 restarts + explicit seed
3. Adds negative directions (-d_EM, -d_ED, -d_MD) to sweeps
4. Updates optimization_results.pkl and sweep_results.pkl

Usage:
    python scripts/patch_melbo_rerun.py
"""

import sys
import os
import pickle
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

# Reuse infrastructure from run_multi_contrastive
from scripts.run_multi_contrastive import (
    extract_context_and_activation,
    forward_to_hidden_and_logits,
    build_perturbed_hidden,
    sweep_single_direction,
    optimize_direction,
)

LAYER = 14
MODEL = 'qwen3-0.6b'
N_ACT = 50
SEQ_LEN = 5
SEED = 42
N_RESTARTS = 20
N_STEPS = 300
OPT_ANGLE = 15.0
N_ANGLES = 60

results_dir = f'results/multi_contrastive/{MODEL}'


def main():
    print("Loading model...")
    model, tokenizer, device = load_model(MODEL)

    # Load existing optimization results (has difference matrices, directions)
    with open(os.path.join(results_dir, 'optimization_results.pkl'), 'rb') as f:
        opt_data = pickle.load(f)

    D_EM = opt_data['D_EM']
    D_ED = opt_data['D_ED']
    D_MD = opt_data['D_MD']
    directions = opt_data['directions']

    # Recompute SVDs
    print("Computing SVDs...")
    D_joint = torch.cat([D_EM, D_ED, D_MD], dim=0)
    _, S_j, Vh_j = torch.linalg.svd(D_joint, full_matrices=False)
    V_joint = Vh_j.T.to(device)
    print(f"  Joint: rank={V_joint.shape[1]}, top SV={S_j[0]:.2f}")

    _, S_em, Vh_em = torch.linalg.svd(D_EM, full_matrices=False)
    V_EM = Vh_em.T.to(device)
    _, S_ed, Vh_ed = torch.linalg.svd(D_ED, full_matrices=False)
    V_ED = Vh_ed.T.to(device)
    _, S_md, Vh_md = torch.linalg.svd(D_MD, full_matrices=False)
    V_MD = Vh_md.T.to(device)
    print(f"  V_EM rank={V_EM.shape[1]}, V_ED rank={V_ED.shape[1]}, V_MD rank={V_MD.shape[1]}")

    # Load FineWeb activations (same seed as original)
    print(f"\nLoading {N_ACT} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(N_ACT, tokenizer, seq_len=SEQ_LEN, seed=SEED)
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, LAYER, device)
    contexts = contexts.to(device)
    activations = activations.to(device)

    # Reference logits
    with torch.no_grad():
        ref_hidden = torch.cat([contexts, activations.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_hidden_and_logits(model, ref_hidden, LAYER)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    opt_angle_rad = np.radians(OPT_ANGLE)

    # ===== Re-run MELBO with more restarts =====
    # Set explicit seed before MELBO
    torch.manual_seed(12345)

    print(f"\n{'='*60}")
    print(f"MELBO on JOINT span — {N_RESTARTS} restarts")
    print(f"{'='*60}")
    joint_alphas = []
    joint_directions = []
    joint_kls = []
    for i in range(3):
        print(f"\n  Direction {i+1}/3:")
        alpha, direction, kl, history = optimize_direction(
            model, contexts, activations, ref_log_probs,
            V_joint, opt_angle_rad, LAYER, device,
            n_steps=N_STEPS, lr=0.01, n_restarts=N_RESTARTS,
            prev_alphas=joint_alphas if joint_alphas else None)
        joint_alphas.append(alpha)
        joint_directions.append(direction.cpu())
        joint_kls.append(kl)
        print(f"  -> MELBO_joint_{i+1}: KL = {kl:.4f}")

    print(f"\n{'='*60}")
    print(f"MELBO on individual spans — {N_RESTARTS} restarts")
    print(f"{'='*60}")
    individual_melbo = {}
    for name, V_sub in [('EM', V_EM), ('ED', V_ED), ('MD', V_MD)]:
        print(f"\n  MELBO on {name} span:")
        _, direction, kl, _ = optimize_direction(
            model, contexts, activations, ref_log_probs,
            V_sub, opt_angle_rad, LAYER, device,
            n_steps=N_STEPS, lr=0.01, n_restarts=N_RESTARTS)
        individual_melbo[name] = {'direction': direction.cpu(), 'kl': kl}
        print(f"  -> MELBO_{name}: KL = {kl:.4f}")

    # Compare with old MELBO_1
    existing_path = f'results/melbo_constrained/{MODEL}/optimization_results.pkl'
    if os.path.exists(existing_path):
        with open(existing_path, 'rb') as f:
            old = pickle.load(f)
        old_melbo1 = old['melbo_directions'][0]
        cos_old = F.cosine_similarity(
            individual_melbo['EM']['direction'].unsqueeze(0),
            old_melbo1.unsqueeze(0)).item()
        print(f"\n  cos(new MELBO_EM, old MELBO_1) = {cos_old:.4f}")
        print(f"  old MELBO_1 KL was: {old['melbo_kls'][0]:.4f}")

    # ===== Update directions =====
    directions['MELBO_joint_1'] = joint_directions[0]
    directions['MELBO_joint_2'] = joint_directions[1]
    directions['MELBO_joint_3'] = joint_directions[2]
    directions['MELBO_EM'] = individual_melbo['EM']['direction']
    directions['MELBO_ED'] = individual_melbo['ED']['direction']
    directions['MELBO_MD'] = individual_melbo['MD']['direction']

    # ===== Sweep new/updated directions + negatives =====
    angles_deg = np.linspace(3, 180, N_ANGLES)
    angles_rad = np.radians(angles_deg)

    # Load existing sweep results
    with open(os.path.join(results_dir, 'sweep_results.pkl'), 'rb') as f:
        sweep_data = pickle.load(f)
    sweep_results = sweep_data['sweep_results']

    # Directions to (re-)sweep: updated MELBOs + new negatives
    dirs_to_sweep = {
        'MELBO_joint_1': joint_directions[0],
        'MELBO_joint_2': joint_directions[1],
        'MELBO_joint_3': joint_directions[2],
        'MELBO_EM': individual_melbo['EM']['direction'],
        'MELBO_ED': individual_melbo['ED']['direction'],
        'MELBO_MD': individual_melbo['MD']['direction'],
        '-d_EM': -directions['d_EM'],
        '-d_ED': -directions['d_ED'],
        '-d_MD': -directions['d_MD'],
    }

    print(f"\n{'='*60}")
    print(f"Sweeping {len(dirs_to_sweep)} directions")
    print(f"{'='*60}")
    for name, direction in dirs_to_sweep.items():
        print(f"  Sweeping {name}...")
        kl, l2 = sweep_single_direction(
            model, contexts, activations, ref_final, ref_log_probs,
            direction, angles_rad, LAYER, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}

    # ===== Update cosine similarity matrix =====
    print(f"\n{'='*60}")
    print("Updated cosine similarity matrix")
    print(f"{'='*60}")

    all_dir_names = ['d_EM', 'd_ED', 'd_MD', 'f_E', 'f_M', 'f_D',
                     'MELBO_joint_1', 'MELBO_joint_2', 'MELBO_joint_3',
                     'MELBO_EM', 'MELBO_ED', 'MELBO_MD']
    all_dir_tensors = [directions[n] for n in all_dir_names]

    # Add old directions for comparison
    if os.path.exists(existing_path):
        all_dir_names.extend(['old_mean', 'old_MELBO_1'])
        all_dir_tensors.extend([old['mean_dir'], old_melbo1])

    n = len(all_dir_names)
    cosine_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cosine_matrix[i, j] = F.cosine_similarity(
                all_dir_tensors[i].unsqueeze(0),
                all_dir_tensors[j].unsqueeze(0)).item()

    # Print key cosines
    for i, name in enumerate(all_dir_names):
        if 'MELBO' in name:
            print(f"  {name}: KL = "
                  f"{joint_kls[int(name[-1])-1] if 'joint' in name else individual_melbo.get(name.replace('MELBO_',''), {}).get('kl', '?')}")

    # ===== Save updated results =====
    opt_data['directions'] = directions
    opt_data['joint_alphas'] = [a.cpu() for a in joint_alphas]
    opt_data['joint_kls'] = joint_kls
    opt_data['individual_melbo'] = individual_melbo
    opt_data['cosine_matrix'] = cosine_matrix
    opt_data['cosine_names'] = all_dir_names

    with open(os.path.join(results_dir, 'optimization_results.pkl'), 'wb') as f:
        pickle.dump(opt_data, f)
    print(f"\nSaved: {results_dir}/optimization_results.pkl")

    sweep_data['sweep_results'] = sweep_results
    with open(os.path.join(results_dir, 'sweep_results.pkl'), 'wb') as f:
        pickle.dump(sweep_data, f)
    print(f"Saved: {results_dir}/sweep_results.pkl")

    print("\nDone!")


if __name__ == '__main__':
    main()
