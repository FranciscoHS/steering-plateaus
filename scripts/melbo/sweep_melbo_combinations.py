#!/usr/bin/env python3
"""Sweep combinations of MELBO directions at various k values.

For k = 50, 100, ..., 450: sample combinations of k directions,
normalize their sum, sweep perturbation angles measuring cosine distance.

Usage:
    python scripts/melbo/sweep_melbo_combinations.py \
        --results_path results/melbo_single_act/qwen3-0.6b_target1.0_seed42/results_600.pkl \
        --n_dirs 500 --max_angle 60
"""
import argparse, os, sys, pickle, time
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model,
    _get_blocks, _is_rope_model, _get_position_embeddings, _run_block,
    _get_final_norm, _is_cat_pos_emb_model, _get_cat_pos_emb,
)
from src.data import load_fineweb_fixed_length


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device,
                                   batch_size=64):
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


def forward_to_final_hidden(model, hidden_states, start_layer):
    hidden = hidden_states
    blocks = _get_blocks(model)
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, hidden)
    elif _is_cat_pos_emb_model(model):
        pos_emb = _get_cat_pos_emb(model, hidden.shape[1])
    for block in blocks[start_layer + 1:]:
        hidden = _run_block(block, hidden, model, pos_emb)
    return hidden[:, -1, :]


def sweep_direction(model, contexts, activations, direction, angles_rad, layer, device, ref_final):
    """Sweep one direction, return cos_dist array (n_angles,)."""
    cos_t_vals = np.cos(angles_rad)
    sin_t_vals = np.sin(angles_rad)
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms
    d_dev = direction.to(device)
    proj = (d_dev.unsqueeze(0) * a_unit).sum(dim=-1, keepdim=True)
    u_perp = d_dev.unsqueeze(0) - proj * a_unit
    u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)

    cos_arr = np.zeros(len(angles_rad))
    for j, theta in enumerate(angles_rad):
        a_pert = a_norms * (cos_t_vals[j] * a_unit + sin_t_vals[j] * u_perp)
        hidden = torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)
        with torch.no_grad():
            pf = forward_to_final_hidden(model, hidden, layer)
            cos_arr[j] = (1.0 - F.cosine_similarity(
                ref_final.float(), pf.float(), dim=-1)).mean().item()
    return cos_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', required=True)
    parser.add_argument('--n_dirs', type=int, default=500)
    parser.add_argument('--k_values', type=str, default='50,100,150,200,250,300,350,400,450')
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Number of random combinations to sample per k')
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=120)
    parser.add_argument('--n_random_dirs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    k_values = [int(x) for x in args.k_values.split(',')]

    # Load MELBO results
    with open(args.results_path, 'rb') as f:
        melbo = pickle.load(f)
    orig_args = melbo['args']
    all_directions = [d.float() for d in melbo['directions'][:args.n_dirs]]
    n_dirs = len(all_directions)
    d_model = all_directions[0].shape[0]
    print(f"Loaded {n_dirs} directions (d_model={d_model})")

    # Stack for fast indexing
    dir_matrix = torch.stack(all_directions)  # (n_dirs, d_model)

    # Load model
    model_name = orig_args['model']
    layer = orig_args['layer']
    model, tokenizer, device = load_model(model_name)
    print(f"Model: {model_name}, layer={layer}, device={device}")

    # Reproduce activation
    token_ids_list = load_fineweb_fixed_length(
        orig_args['n_act'], tokenizer, seq_len=orig_args['seq_len'], seed=orig_args['seed'])
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, layer, device)
    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)

    # Reference
    with torch.no_grad():
        ref_hidden = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_final = forward_to_final_hidden(model, ref_hidden, layer)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Sweep combinations for each k
    results = {}
    t0 = time.time()
    for k in k_values:
        print(f"\nk={k}: sweeping {args.n_samples} combinations...")
        cos_curves = np.zeros((args.n_samples, len(angles_deg)))
        for s in range(args.n_samples):
            indices = np.random.choice(n_dirs, k, replace=False)
            d_combo = dir_matrix[indices].sum(dim=0)
            d_combo = F.normalize(d_combo, dim=0)
            cos_curves[s] = sweep_direction(
                model, contexts_dev, activations_dev, d_combo,
                angles_rad, layer, device, ref_final)
            if (s + 1) % 10 == 0:
                print(f"  {s+1}/{args.n_samples} ({time.time()-t0:.0f}s)")
        results[k] = cos_curves

    # Also sweep k=1 (individual directions, sample 50)
    print(f"\nk=1: sweeping {args.n_samples} individual directions...")
    cos_k1 = np.zeros((args.n_samples, len(angles_deg)))
    for s in range(args.n_samples):
        idx = np.random.randint(n_dirs)
        cos_k1[s] = sweep_direction(
            model, contexts_dev, activations_dev, all_directions[idx],
            angles_rad, layer, device, ref_final)
    results[1] = cos_k1

    # Random baseline
    print(f"\nSweeping {args.n_random_dirs} random directions...")
    rand_cos = np.zeros((args.n_random_dirs, len(angles_deg)))
    a_unit = activations_dev / activations_dev.norm(dim=-1, keepdim=True)
    for i in range(args.n_random_dirs):
        d_rand = torch.randn(d_model, device=device)
        d_rand = d_rand - (d_rand @ a_unit[0]) * a_unit[0]
        d_rand = F.normalize(d_rand, dim=0)
        rand_cos[i] = sweep_direction(
            model, contexts_dev, activations_dev, d_rand,
            angles_rad, layer, device, ref_final)
    results['random'] = rand_cos

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # Save
    out_dir = os.path.dirname(args.results_path)
    save_data = {
        'results': results,
        'angles_deg': angles_deg,
        'k_values': k_values,
        'args': vars(args),
        'orig_args': orig_args,
    }
    pkl_path = os.path.join(out_dir, 'combo_sweep_cosdist.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved: {pkl_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.cm.viridis

    # Random
    rc = results['random']
    ax.fill_between(angles_deg,
                    np.percentile(rc, 25, axis=0),
                    np.percentile(rc, 75, axis=0),
                    color='gray', alpha=0.15)
    ax.plot(angles_deg, np.median(rc, axis=0),
            color='gray', lw=2.5, ls='--', label='Random')

    # k=1
    cd1 = results[1]
    ax.fill_between(angles_deg,
                    np.percentile(cd1, 25, axis=0),
                    np.percentile(cd1, 75, axis=0),
                    color='tab:red', alpha=0.1)
    ax.plot(angles_deg, np.median(cd1, axis=0),
            color='tab:red', lw=2.5, label='k=1')

    # Combinations
    all_k = sorted(k_values)
    for i, k in enumerate(all_k):
        color = cmap(i / max(len(all_k) - 1, 1))
        cd = results[k]
        ax.fill_between(angles_deg,
                        np.percentile(cd, 25, axis=0),
                        np.percentile(cd, 75, axis=0),
                        color=color, alpha=0.08)
        ax.plot(angles_deg, np.median(cd, axis=0),
                color=color, lw=2, label=f'k={k}')

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel('Cosine distance', fontsize=12)
    ax.set_title(f'MELBO direction combinations: {model_name}, layer {layer}\n'
                 f'{n_dirs} orthogonal directions, {args.n_samples} samples/k',
                 fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, args.max_angle)

    plt.tight_layout()
    png_path = os.path.join(out_dir, 'combo_sweep_cosdist.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    main()
