#!/usr/bin/env python3
"""Sweep each MELBO direction individually + random baseline."""
import argparse, os, pickle, sys
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_final, _get_final_norm, _get_lm_head
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
    parser.add_argument('--melbo_path', type=str, required=True)
    parser.add_argument('--n_random_dirs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.melbo_path, 'rb') as f:
        melbo = pickle.load(f)
    dirs = torch.stack(melbo['directions'])
    rewards = melbo['rewards']
    n_dirs = dirs.shape[0]
    print(f"MELBO directions: {dirs.shape}, rewards: {[f'{r:.4f}' for r in rewards]}")

    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    n_games = max(args.act_idx + 1, 50)
    token_ids_list = generate_chess_games(n_games, args.seq_len, seed=args.seed)

    batch_size = 32
    all_ctx, all_act = [], []
    for i in range(0, n_games, batch_size):
        bt = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            out = model(bt, output_hidden_states=True)
            h = out.hidden_states[args.layer + 1]
        all_ctx.append(h[:, :-1, :].float().cpu())
        all_act.append(h[:, -1, :].float().cpu())
    ctx = torch.cat(all_ctx, 0)[args.act_idx:args.act_idx+1].to(device)
    act = torch.cat(all_act, 0)[args.act_idx].to(device)
    print(f"Activation norm: {act.norm():.2f}")

    angles_deg = np.linspace(0, 180, 61)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Random baseline
    print(f"Sweeping {args.n_random_dirs} random directions...")
    random_kl = np.zeros((args.n_random_dirs, len(angles_deg)))
    random_l2 = np.zeros((args.n_random_dirs, len(angles_deg)))
    for i in range(args.n_random_dirs):
        d = torch.randn(act.shape[-1])
        random_kl[i], random_l2[i] = sweep_direction(
            model, ctx, act, d, angles_rad, args.layer, device)

    # Individual MELBO dirs
    print("Sweeping individual MELBO directions...")
    indiv_kl = np.zeros((n_dirs, len(angles_deg)))
    indiv_l2 = np.zeros((n_dirs, len(angles_deg)))
    for i in range(n_dirs):
        indiv_kl[i], indiv_l2[i] = sweep_direction(
            model, ctx, act, dirs[i], angles_rad, args.layer, device)
        idx_15 = np.argmin(np.abs(angles_deg - 15))
        idx_40 = np.argmin(np.abs(angles_deg - 40))
        print(f"  Dir {i+1}: KL@15={indiv_kl[i][idx_15]:.4f}, KL@40={indiv_kl[i][idx_40]:.4f}, reward={rewards[i]:.4f}")

    # Plot
    cmap = plt.cm.tab10
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, (metric, ylabel, xlim) in enumerate([
        ('kl', 'KL divergence', (0, 180)),
        ('l2', 'L2 distance', (0, 40)),
    ]):
        ax = axes[ax_idx]
        rand = random_kl if metric == 'kl' else random_l2
        indiv = indiv_kl if metric == 'kl' else indiv_l2

        ax.fill_between(angles_deg,
                        np.percentile(rand, 25, axis=0),
                        np.percentile(rand, 75, axis=0),
                        color='gray', alpha=0.15)
        ax.plot(angles_deg, np.median(rand, axis=0),
                color='gray', lw=2, ls='--', label='Random')

        for i in range(n_dirs):
            ax.plot(angles_deg, indiv[i],
                    color=cmap(i % 10), lw=1.5, alpha=0.8,
                    label=f'Dir {i+1} (r={rewards[i]:.3f})')

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(*xlim)

    fig.suptitle(f'Individual MELBO directions perturbation sweep', fontsize=13, y=1.02)
    plt.tight_layout()
    out_dir = os.path.dirname(args.melbo_path)
    path = os.path.join(out_dir, 'individual_sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
