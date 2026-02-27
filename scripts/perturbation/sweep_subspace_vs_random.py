#!/usr/bin/env python3
"""Sweep perturbation response for: individual features, random-in-subspace, fully random.

Loads sequential discovery directions, samples random vectors from their span
and from full R^d, then does perturbation sweeps for all three groups.

Usage:
    python scripts/perturbation/sweep_subspace_vs_random.py \
        --features_path results/sequential/chess_16L_L2_meas14_alpha8.0/features.pkl
"""
import argparse, os, sys, pickle
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_layer
from src.data import generate_chess_games


def sweep_direction(model, ctx, act, d, layer, measure_layer, device, angles_rad):
    """Sweep a single direction, return cosine distance array."""
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=model_dtype)
    R = a_d.float().norm()
    a_unit = a_d.float() / R
    d_d = d.to(device=device, dtype=torch.float32)
    d_orth = d_d - (d_d @ a_unit) * a_unit
    d_orth = F.normalize(d_orth, dim=0)

    cos_arr = np.zeros(len(angles_rad))
    with torch.no_grad():
        ref = forward_from_layer_to_layer(
            model, ctx_d, a_d[None, None], layer, measure_layer).float()
        for j, theta in enumerate(angles_rad):
            p = R * (np.cos(theta) * a_unit + np.sin(theta) * d_orth)
            p = p.to(model_dtype)
            pf = forward_from_layer_to_layer(
                model, ctx_d, p[None, None], layer, measure_layer).float()
            cos_arr[j] = 1.0 - F.cosine_similarity(ref, pf, dim=-1).item()
    return cos_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', required=True)
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--measure_layer', type=int, default=14)
    parser.add_argument('--act_idx', type=int, default=137)
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--n_subspace', type=int, default=50)
    parser.add_argument('--n_random', type=int, default=50)
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load directions
    data = pickle.load(open(args.features_path, 'rb'))
    if isinstance(data['directions'], list):
        all_dirs = torch.stack(data['directions']).float()
    else:
        all_dirs = data['directions'].float()

    # Filter to good directions (cd > 0.1)
    cos_dists = data['cos_dists']
    if isinstance(cos_dists, list):
        cos_dists = np.array(cos_dists)
    good_mask = cos_dists > 0.1
    dirs = all_dirs[good_mask]
    n_features = dirs.shape[0]
    d_model = dirs.shape[1]
    print(f"Loaded {n_features} good directions (cd>0.1) from {args.features_path}")

    # Load model
    model, tokenizer, device = load_model(args.model)
    model_dtype = next(model.parameters()).dtype

    # Extract activation
    n_games = max(args.act_idx + 1, 50)
    token_ids_list = generate_chess_games(n_games, args.seq_len, seed=args.seed)
    ctx = act = None
    batch_size = 32
    count = 0
    for i in range(0, n_games, batch_size):
        bt = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            out = model(bt, output_hidden_states=True)
            h = out.hidden_states[args.layer + 1]
        local_idx = args.act_idx - count
        if 0 <= local_idx < bt.shape[0]:
            ctx = h[local_idx:local_idx+1, :-1, :].to(device, dtype=model_dtype)
            act = h[local_idx, -1, :].to(device, dtype=model_dtype)
        count += bt.shape[0]
        del out, h, bt

    print(f"Activation {args.act_idx}, norm={act.float().norm():.2f}")

    angles_deg = np.linspace(0, args.max_angle, int(args.max_angle * 4) + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Orthonormal basis for the feature subspace via QR
    dirs_dev = dirs.to(device)
    Q, _ = torch.linalg.qr(dirs_dev.T, mode='reduced')  # (d_model, n_features)
    subspace_basis = Q  # columns are orthonormal basis vectors

    # Sample random-in-subspace directions
    print(f"Sampling {args.n_subspace} random-in-subspace directions...")
    subspace_dirs = []
    for _ in range(args.n_subspace):
        coeffs = torch.randn(n_features, device=device)
        d = (subspace_basis @ coeffs)
        d = F.normalize(d, dim=0)
        subspace_dirs.append(d)

    # Sample fully random directions
    print(f"Sampling {args.n_random} fully random directions...")
    random_dirs = [F.normalize(torch.randn(d_model, device=device), dim=0)
                   for _ in range(args.n_random)]

    # Sweep all three groups
    print(f"Sweeping {n_features} individual features...")
    feat_curves = np.zeros((n_features, len(angles_deg)))
    for i in range(n_features):
        feat_curves[i] = sweep_direction(
            model, ctx, act, dirs_dev[i], args.layer, args.measure_layer,
            device, angles_rad)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_features}")

    print(f"Sweeping {args.n_subspace} subspace-random directions...")
    sub_curves = np.zeros((args.n_subspace, len(angles_deg)))
    for i in range(args.n_subspace):
        sub_curves[i] = sweep_direction(
            model, ctx, act, subspace_dirs[i], args.layer, args.measure_layer,
            device, angles_rad)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_subspace}")

    print(f"Sweeping {args.n_random} fully random directions...")
    rand_curves = np.zeros((args.n_random, len(angles_deg)))
    for i in range(args.n_random):
        rand_curves[i] = sweep_direction(
            model, ctx, act, random_dirs[i], args.layer, args.measure_layer,
            device, angles_rad)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_random}")

    # Save results
    results_dir = os.path.dirname(args.features_path)
    save_data = {
        'angles_deg': angles_deg,
        'feat_curves': feat_curves,
        'sub_curves': sub_curves,
        'rand_curves': rand_curves,
        'n_features': n_features,
        'args': vars(args),
    }
    pickle.dump(save_data, open(os.path.join(results_dir, 'subspace_sweep.pkl'), 'wb'))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Individual features: thin lines + median
    for i in range(n_features):
        ax.plot(angles_deg, feat_curves[i], color='C0', alpha=0.15, lw=0.5)
    ax.plot(angles_deg, np.median(feat_curves, axis=0),
            color='C0', lw=2.5, label=f'Individual features (n={n_features})')

    # Random-in-subspace: IQR + median
    ax.fill_between(angles_deg,
                    np.percentile(sub_curves, 25, axis=0),
                    np.percentile(sub_curves, 75, axis=0),
                    color='C1', alpha=0.15)
    ax.plot(angles_deg, np.median(sub_curves, axis=0),
            color='C1', lw=2.5, label=f'Random in subspace (n={args.n_subspace})')

    # Fully random: IQR + median
    ax.fill_between(angles_deg,
                    np.percentile(rand_curves, 25, axis=0),
                    np.percentile(rand_curves, 75, axis=0),
                    color='gray', alpha=0.15)
    ax.plot(angles_deg, np.median(rand_curves, axis=0),
            color='gray', lw=2.5, ls='--', label=f'Fully random (n={args.n_random})')

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('Cosine distance')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, args.max_angle)
    ax.set_title(f'Subspace vs random: {n_features} features spanning {n_features}-dim subspace')
    plt.tight_layout()

    path = os.path.join(results_dir, 'subspace_vs_random.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # Numerical summary: angle where median crosses 0.1
    for name, curves in [('Features', feat_curves),
                          ('Subspace-random', sub_curves),
                          ('Fully-random', rand_curves)]:
        med = np.median(curves, axis=0)
        crossings = np.where(med > 0.1)[0]
        if len(crossings) > 0:
            break_angle = angles_deg[crossings[0]]
            print(f"{name}: breaks at {break_angle:.1f}°")
        else:
            print(f"{name}: never breaks 0.1")


if __name__ == '__main__':
    main()
