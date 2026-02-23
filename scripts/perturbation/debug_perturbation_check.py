#!/usr/bin/env python3
"""Quick diagnostic: perturbation KL from unsteered activations, 0-120°."""

import sys
import os
import pickle
import numpy as np
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer
from src.data import load_fineweb_fixed_length

LAYER = 14
MODEL_NAME = 'qwen3-0.6b'
DIRECTIONS_PATH = 'results/allpairs_contrastive/qwen3-0.6b/directions.pkl'
N_RANDOM_DIRS = 10
N_ACT = 20
SEQ_LEN = 5
SEED = 123

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    angles_deg = np.linspace(0, 120, 61)[1:]  # 2° to 120°, 60 steps
    angles_rad = np.deg2rad(angles_deg)

    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer, device = load_model(MODEL_NAME)
    d_model = model._plateau_info['d_model']
    model_dtype = next(model.parameters()).dtype

    # Load zh direction
    with open(DIRECTIONS_PATH, 'rb') as f:
        dir_data = pickle.load(f)
    zh_dir = dir_data['features']['zh'].to(dtype=torch.float32)

    # Random directions
    random_dirs = F.normalize(torch.randn(N_RANDOM_DIRS, d_model), dim=-1)
    all_dirs = torch.cat([zh_dir.unsqueeze(0), random_dirs], dim=0)

    # Extract activations from FineWeb (fixed 5-token sequences)
    print(f"Loading {N_ACT} FineWeb sequences (seq_len={SEQ_LEN})...")
    token_ids_list = load_fineweb_fixed_length(N_ACT, tokenizer, seq_len=SEQ_LEN, seed=SEED)
    print(f"Extracting layer-{LAYER} activations...")
    contexts = []
    activations = []
    for i in range(0, len(token_ids_list), 64):
        batch = torch.stack(token_ids_list[i:i+64]).to(device)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
        hidden = outputs.hidden_states[LAYER + 1]
        for j in range(hidden.shape[0]):
            contexts.append(hidden[j:j+1, :-1, :].float().cpu())
            activations.append(hidden[j, -1, :].float().cpu())
    activations = torch.stack(activations)

    # Sweep
    n_prompts = N_ACT
    n_angles = len(angles_deg)
    n_dirs = all_dirs.shape[0]
    all_kl = np.zeros((n_prompts, n_angles, n_dirs))

    dirs_gpu = all_dirs.to(device=device, dtype=model_dtype)

    # Two runs: non-orthogonal and orthogonal
    all_kl_raw = np.zeros((n_prompts, n_angles, n_dirs))
    all_kl_orth = np.zeros((n_prompts, n_angles, n_dirs))

    for p_idx in range(n_prompts):
        ctx = contexts[p_idx].to(device=device, dtype=model_dtype)
        a = activations[p_idx].to(device=device, dtype=model_dtype)
        R = a.norm()
        a_unit = a / R

        # Orthogonalize directions to this activation
        proj = (dirs_gpu @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)  # [n_dirs, d]
        dirs_orth = dirs_gpu - proj
        dirs_orth = F.normalize(dirs_orth, dim=-1)

        # Reference
        with torch.no_grad():
            ref_logits = forward_from_layer(model, ctx, a.unsqueeze(0).unsqueeze(0), LAYER)
            ref_lp = F.log_softmax(ref_logits.float(), dim=-1)

        for ang_idx, theta in enumerate(angles_rad):
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # Non-orthogonal
            perturbed_raw = cos_t * a.unsqueeze(0) + sin_t * R * dirs_gpu
            perturbed_raw = perturbed_raw.unsqueeze(1)

            # Orthogonal
            perturbed_orth = cos_t * a.unsqueeze(0) + sin_t * R * dirs_orth
            perturbed_orth = perturbed_orth.unsqueeze(1)

            with torch.no_grad():
                raw_logits = forward_from_layer(model, ctx, perturbed_raw, LAYER)
                raw_lp = F.log_softmax(raw_logits.float(), dim=-1)
                ref_p = ref_lp.exp()
                kl_raw = (ref_p * (ref_lp - raw_lp)).sum(dim=-1)
                all_kl_raw[p_idx, ang_idx] = kl_raw.cpu().numpy()

                orth_logits = forward_from_layer(model, ctx, perturbed_orth, LAYER)
                orth_lp = F.log_softmax(orth_logits.float(), dim=-1)
                kl_orth = (ref_p * (ref_lp - orth_lp)).sum(dim=-1)
                all_kl_orth[p_idx, ang_idx] = kl_orth.cpu().numpy()

        if (p_idx + 1) % 5 == 0:
            print(f"  {p_idx+1}/{n_prompts}")

    all_kl = all_kl_orth  # use orthogonal for the rest of analysis

    # Print comparison with std and IQR
    for label, kl_data in [("NON-ORTHOGONAL", all_kl_raw), ("ORTHOGONAL", all_kl_orth)]:
        kl_zh = kl_data[:, :, 0]
        kl_rand = kl_data[:, :, 1:]
        print(f"\n{label} — Random dirs stats:")
        print(f"{'Angle':>8s}  {'median':>8s}  {'std':>8s}  {'IQR':>20s}")
        for deg in [10, 20, 30, 45, 60, 90, 120]:
            idx = np.argmin(np.abs(angles_deg - deg))
            flat = kl_rand[:, idx, :].flatten()
            med = np.median(flat)
            std = np.std(flat)
            q25, q75 = np.percentile(flat, [25, 75])
            print(f"{angles_deg[idx]:7.1f}°  {med:8.3f}  {std:8.3f}  [{q25:.3f}, {q75:.3f}]")

    # Plot: 2 panels comparing orthogonal vs non-orthogonal
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, label, kl_data in zip(axes, ['Non-orthogonal', 'Orthogonal'],
                                   [all_kl_raw, all_kl_orth]):
        kl_zh = kl_data[:, :, 0]
        kl_rand = kl_data[:, :, 1:]

        median_zh = np.median(kl_zh, axis=0)
        q25_zh = np.percentile(kl_zh, 25, axis=0)
        q75_zh = np.percentile(kl_zh, 75, axis=0)
        ax.plot(angles_deg, median_zh, 'r-', linewidth=2, label='zh')
        ax.fill_between(angles_deg, q25_zh, q75_zh, color='red', alpha=0.15)

        # Flatten prompts × dirs, keeping angle axis intact
        # [n_prompts, n_angles, n_rand_dirs] → [n_prompts * n_rand_dirs, n_angles]
        kl_rand_flat = kl_rand.transpose(0, 2, 1).reshape(-1, n_angles)
        median_rand = np.median(kl_rand_flat, axis=0)
        q25_rand = np.percentile(kl_rand_flat, 25, axis=0)
        q75_rand = np.percentile(kl_rand_flat, 75, axis=0)
        ax.plot(angles_deg, median_rand, 'b-', linewidth=2, label='Random (10 dirs)')
        ax.fill_between(angles_deg, q25_rand, q75_rand, color='blue', alpha=0.15)

        ax.axhline(0.1, color='gray', linestyle=':', alpha=0.7, label='KL=0.1')
        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel('KL divergence')
        ax.set_title(f'{label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 120)

    fig.suptitle(f'Perturbation from unsteered English — {MODEL_NAME}, layer {LAYER}', fontsize=13)
    os.makedirs('results/plateau_vs_steering/qwen3-0.6b', exist_ok=True)
    path = 'results/plateau_vs_steering/qwen3-0.6b/debug_perturbation.png'
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
