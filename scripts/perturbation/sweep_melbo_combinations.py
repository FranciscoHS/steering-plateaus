#!/usr/bin/env python3
"""
Sweep perturbations along combinations of MELBO-found directions.

Tests the error correction prediction: if MELBO directions are true features,
combining K directions should produce LONGER plateaus than individual directions
(because the perturbation energy is spread across K features).

Predicted ordering (if error correction + true features):
  1 direction < 2 combined < ... < K combined < random
"""

import argparse
import os
import pickle
import sys
from itertools import combinations
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_to_final,
    _get_final_norm, _get_lm_head,
)
from src.data import generate_chess_games


def sweep_direction(model, ctx, a, d, angles_rad, layer, device):
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = a.to(device=device, dtype=model_dtype)
    R = a_d.norm()
    a_unit = a_d / R
    d_d = d.to(device=device, dtype=model_dtype)

    d_orth = d_d - (d_d @ a_unit) * a_unit
    d_orth = F.normalize(d_orth, dim=-1)

    with torch.no_grad():
        ref_point = a_d.unsqueeze(0).unsqueeze(0)
        ref_final = forward_from_layer_to_final(model, ctx_d, ref_point, layer)
        ref_ln = _get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1)
        ref_logits = _get_lm_head(model)(ref_ln)
        ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)

    kl_arr = np.zeros(len(angles_rad))
    l2_arr = np.zeros(len(angles_rad))
    for j, theta in enumerate(angles_rad):
        perturbed = R * (np.cos(theta) * a_unit + np.sin(theta) * d_orth)
        with torch.no_grad():
            pert_final = forward_from_layer_to_final(
                model, ctx_d, perturbed.unsqueeze(0).unsqueeze(0), layer)
            l2_arr[j] = (pert_final - ref_final).float().norm().item()
            pert_ln = _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)
            pert_logits = _get_lm_head(model)(pert_ln)
            pert_lp = F.log_softmax(pert_logits.float(), dim=-1)
            kl_arr[j] = (ref_log_probs.exp() * (ref_log_probs - pert_lp)).sum().item()
    return kl_arr, l2_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=11)
    parser.add_argument('--act_idx', type=int, default=137)
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--n_random_dirs', type=int, default=20)
    parser.add_argument('--melbo_path', type=str, required=True)
    parser.add_argument('--n_top', type=int, default=6,
                        help='Use top N MELBO directions (by reward)')
    parser.add_argument('--max_combos_per_k', type=int, default=20,
                        help='Max combos to sample per k (0=all)')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load MELBO results
    with open(args.melbo_path, 'rb') as f:
        melbo = pickle.load(f)
    melbo_dirs = melbo['directions']  # list of tensors
    melbo_rewards = melbo['rewards']

    # Sort by reward, take top N
    order = np.argsort(melbo_rewards)[::-1]
    n_top = min(args.n_top, len(melbo_dirs))
    selected_indices = order[:n_top]
    selected_dirs = [melbo_dirs[i] for i in selected_indices]
    selected_rewards = [melbo_rewards[i] for i in selected_indices]

    print(f"Selected top {n_top} MELBO directions (by reward):")
    for i, (idx, rew) in enumerate(zip(selected_indices, selected_rewards)):
        print(f"  {i+1}. Dir {idx+1}: reward={rew:.6f}")

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(args.melbo_path), 'combo_sweep')
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    # Generate data
    n_games = max(args.act_idx + 1, 50)
    token_ids_list = generate_chess_games(n_games, args.seq_len, seed=args.seed)

    # Extract activation
    print(f"Extracting layer-{args.layer} activation for act_idx={args.act_idx}...")
    batch_size = 32
    all_ctx = []
    all_act = []
    for i in range(0, n_games, batch_size):
        bt = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            out = model(bt, output_hidden_states=True)
            h = out.hidden_states[args.layer + 1]
        all_ctx.append(h[:, :-1, :].float().cpu())
        all_act.append(h[:, -1, :].float().cpu())
    all_ctx = torch.cat(all_ctx, 0)
    all_act = torch.cat(all_act, 0)
    ctx = all_ctx[args.act_idx:args.act_idx+1].to(device)
    act = all_act[args.act_idx].to(device)
    print(f"Activation norm: {act.norm():.2f}")

    # Angles
    angles_deg = np.linspace(0, 180, 61)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Random baseline
    print(f"\nSweeping {args.n_random_dirs} random directions...")
    random_kl = np.zeros((args.n_random_dirs, len(angles_deg)))
    random_l2 = np.zeros((args.n_random_dirs, len(angles_deg)))
    for i in range(args.n_random_dirs):
        d = torch.randn(act.shape[-1])
        random_kl[i], random_l2[i] = sweep_direction(
            model, ctx, act, d, angles_rad, args.layer, device)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_random_dirs}")

    # Sweep all combinations
    combo_results = {}  # k -> list of (combo, kl, l2, labels)

    for k in range(1, n_top):  # 1 through n_top-1
        all_combos = list(combinations(range(n_top), k))
        if args.max_combos_per_k > 0 and len(all_combos) > args.max_combos_per_k:
            np.random.seed(args.seed + k)
            indices = np.random.choice(len(all_combos), args.max_combos_per_k, replace=False)
            combos = [all_combos[i] for i in indices]
            print(f"\nSweeping {len(combos)}/{len(all_combos)} sampled combinations of {k}...")
        else:
            combos = all_combos
            print(f"\nSweeping all C({n_top},{k}) = {len(combos)} combinations of {k}...")
        combo_results[k] = []
        for ci, combo in enumerate(combos):
            d_combined = torch.zeros_like(selected_dirs[0])
            for idx in combo:
                d_combined = d_combined + selected_dirs[idx]
            d_combined = F.normalize(d_combined, dim=-1)
            kl, l2 = sweep_direction(
                model, ctx, act, d_combined, angles_rad, args.layer, device)
            combo_results[k].append((combo, kl, l2))
            if (ci + 1) % 10 == 0:
                print(f"  {ci+1}/{len(combos)}")

    # Individual directions as k=1
    combo_results[1] = []
    for i in range(n_top):
        kl, l2 = sweep_direction(
            model, ctx, act, selected_dirs[i], angles_rad, args.layer, device)
        combo_results[1].append(((i,), kl, l2))
        print(f"  Dir {i+1}: KL@15°={kl[np.argmin(np.abs(angles_deg-15))]:.4f}, "
              f"KL@40°={kl[np.argmin(np.abs(angles_deg-40))]:.4f}")

    # Save
    results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'random_kl': random_kl,
        'random_l2': random_l2,
        'combo_results': {k: [(c, kl.tolist(), l2.tolist()) for c, kl, l2 in v]
                          for k, v in combo_results.items()},
        'selected_indices': selected_indices.tolist(),
        'selected_rewards': selected_rewards,
    }
    save_path = os.path.join(args.results_dir, 'melbo_combos.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    # Plot
    colors = {1: 'blue', 2: 'green', 3: 'orange', 4: 'red', 5: 'purple'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, (metric, ylabel, xlim) in enumerate([
        ('kl', 'KL divergence', (0, 180)),
        ('l2', 'L2 distance', (0, 40)),
    ]):
        ax = axes[ax_idx]
        rand_data = random_kl if metric == 'kl' else random_l2

        # Random baseline
        med = np.median(rand_data, axis=0)
        q25 = np.percentile(rand_data, 25, axis=0)
        q75 = np.percentile(rand_data, 75, axis=0)
        ax.fill_between(angles_deg, q25, q75, color='gray', alpha=0.15)
        ax.plot(angles_deg, med, color='gray', lw=2, ls='--', label='Random')

        for k in sorted(combo_results.keys()):
            data_idx = 1 if metric == 'kl' else 2
            all_data = np.array([np.array(item[data_idx]) for item in combo_results[k]])
            med_k = np.median(all_data, axis=0)
            q25_k = np.percentile(all_data, 25, axis=0)
            q75_k = np.percentile(all_data, 75, axis=0)
            c = colors.get(k, 'black')
            n_combos = len(combo_results[k])
            ax.fill_between(angles_deg, q25_k, q75_k, color=c, alpha=0.12)
            ax.plot(angles_deg, med_k, color=c, lw=2,
                    label=f'{k} MELBO dir{"s" if k > 1 else ""} (n={n_combos})')

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(*xlim)

    fig.suptitle(f'{args.model} layer {args.layer}, act #{args.act_idx}: '
                 f'MELBO direction combinations (error correction test)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'melbo_combos.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    main()
