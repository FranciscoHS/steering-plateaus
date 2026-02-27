#!/usr/bin/env python3
"""Sweep MELBO directions measuring cosine distance at final hidden layer.

Loads directions from a MELBO results pkl, reproduces the same activation(s),
and sweeps each direction measuring cosine distance (not KL).

Usage:
    python scripts/melbo/sweep_melbo_cosdist.py \
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
    """Forward pass returning final hidden state (before LM head), last token."""
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


def build_perturbed_hidden(contexts, activations, direction, angle_rad):
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms
    proj = (direction.unsqueeze(0) * a_unit).sum(dim=-1, keepdim=True)
    u_perp = direction.unsqueeze(0) - proj * a_unit
    u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)
    a_pert = a_norms * (cos_t * a_unit + sin_t * u_perp)
    return torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', required=True)
    parser.add_argument('--n_dirs', type=int, default=500)
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=120)
    parser.add_argument('--n_random_dirs', type=int, default=50)
    parser.add_argument('--batch_dirs', type=int, default=50,
                        help='Number of directions to sweep per batch (for progress)')
    args = parser.parse_args()

    # Load MELBO results
    with open(args.results_path, 'rb') as f:
        melbo = pickle.load(f)
    orig_args = melbo['args']
    all_directions = melbo['directions'][:args.n_dirs]
    n_dirs = len(all_directions)
    print(f"Loaded {n_dirs} directions from {args.results_path}")

    # Load model
    model_name = orig_args['model']
    layer = orig_args['layer']
    seed = orig_args['seed']
    seq_len = orig_args['seq_len']
    n_act = orig_args['n_act']

    model, tokenizer, device = load_model(model_name)
    d_model = model._plateau_info['d_model']
    print(f"Model: {model_name}, layer={layer}, d_model={d_model}, device={device}")

    # Reproduce activation
    token_ids_list = load_fineweb_fixed_length(n_act, tokenizer, seq_len=seq_len, seed=seed)
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, layer, device)
    print(f"Activation shape: {activations.shape}, norm: {activations.norm(dim=-1).mean():.2f}")

    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)

    # Reference
    with torch.no_grad():
        ref_hidden = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_final = forward_to_final_hidden(model, ref_hidden, layer)

    # Sweep angles
    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)
    N = activations.shape[0]

    # Sweep MELBO directions
    print(f"\nSweeping {n_dirs} MELBO directions at {len(angles_deg)} angles (0-{args.max_angle}°)...")
    cos_dists = np.zeros((n_dirs, N, len(angles_deg)))
    t0 = time.time()
    for i, d in enumerate(all_directions):
        d_dev = d.to(device)
        for j, theta in enumerate(angles_rad):
            with torch.no_grad():
                hidden = build_perturbed_hidden(contexts_dev, activations_dev, d_dev, theta)
                pf = forward_to_final_hidden(model, hidden, layer)
                cos_dists[i, :, j] = (1.0 - F.cosine_similarity(
                    ref_final.float(), pf.float(), dim=-1)).cpu().numpy()
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_dirs} ({elapsed:.0f}s)")

    # Sweep random directions
    print(f"\nSweeping {args.n_random_dirs} random directions...")
    rand_cos = np.zeros((args.n_random_dirs, N, len(angles_deg)))
    a_unit = (activations_dev / activations_dev.norm(dim=-1, keepdim=True))
    for i in range(args.n_random_dirs):
        d_rand = torch.randn(d_model, device=device)
        d_rand = d_rand - (d_rand @ a_unit[0]) * a_unit[0]  # orthogonalize to act
        d_rand = F.normalize(d_rand, dim=0)
        for j, theta in enumerate(angles_rad):
            with torch.no_grad():
                hidden = build_perturbed_hidden(contexts_dev, activations_dev, d_rand, theta)
                pf = forward_to_final_hidden(model, hidden, layer)
                rand_cos[i, :, j] = (1.0 - F.cosine_similarity(
                    ref_final.float(), pf.float(), dim=-1)).cpu().numpy()
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_random_dirs}")

    # Save
    out_dir = os.path.dirname(args.results_path)
    save_data = {
        'cos_dists': cos_dists,       # (n_dirs, N, n_angles)
        'rand_cos': rand_cos,          # (n_random, N, n_angles)
        'angles_deg': angles_deg,
        'n_dirs': n_dirs,
        'orig_args': orig_args,
        'args': vars(args),
    }
    pkl_path = os.path.join(out_dir, f'cosdist_sweep_{n_dirs}dirs.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved: {pkl_path}")

    # For single activation, squeeze out the N dimension
    # cos_dists shape: (n_dirs, 1, n_angles) -> (n_dirs, n_angles)
    cd = cos_dists[:, 0, :] if N == 1 else np.median(cos_dists, axis=1)
    rc = rand_cos[:, 0, :] if N == 1 else np.median(rand_cos, axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Random baseline
    ax.fill_between(angles_deg,
                    np.percentile(rc, 25, axis=0),
                    np.percentile(rc, 75, axis=0),
                    color='gray', alpha=0.15)
    ax.plot(angles_deg, np.median(rc, axis=0),
            color='gray', lw=2.5, ls='--', label=f'Random (n={args.n_random_dirs})')

    # MELBO directions: show median + spread
    ax.fill_between(angles_deg,
                    np.percentile(cd, 25, axis=0),
                    np.percentile(cd, 75, axis=0),
                    color='tab:blue', alpha=0.15)
    ax.plot(angles_deg, np.median(cd, axis=0),
            color='tab:blue', lw=2.5, label=f'MELBO median (n={n_dirs})')

    # Also show a few individual directions
    for i in range(min(5, n_dirs)):
        ax.plot(angles_deg, cd[i], color='tab:blue', alpha=0.3, lw=0.8)

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('Cosine distance')
    ax.set_title(f'MELBO directions cos_dist sweep: {model_name}, layer {layer}, '
                 f'{n_dirs} dirs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, args.max_angle)

    plt.tight_layout()
    png_path = os.path.join(out_dir, f'cosdist_sweep_{n_dirs}dirs.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path}")

    # Summary stats
    print(f"\n=== Summary at {args.max_angle/2:.0f}° ===")
    mid_idx = len(angles_deg) // 2
    print(f"MELBO median cos_dist: {np.median(cd[:, mid_idx]):.4f}")
    print(f"Random median cos_dist: {np.median(rc[:, mid_idx]):.4f}")
    print(f"MELBO range: [{cd[:, mid_idx].min():.4f}, {cd[:, mid_idx].max():.4f}]")


if __name__ == '__main__':
    main()
