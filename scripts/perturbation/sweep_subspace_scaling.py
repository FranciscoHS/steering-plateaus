#!/usr/bin/env python3
"""Test subspace plateau-breaking scaling: observed vs theory for varying k.

For k in {5, 10, 15, 20, 25}, samples random subsets of k features, draws
random vectors from each subset's subspace, sweeps perturbation response,
and finds the median breaking angle. Compares against theory:

    θ_sub ≈ arcsin(sin(θ_1(k)) * sqrt(k / (2 ln k)))

where θ_1(k) is the median single-feature breaking angle of the sampled subset.

Usage:
    python scripts/perturbation/sweep_subspace_scaling.py \
        --features_path results/sequential/chess_16L_L2_meas14_alpha8.0/features.pkl \
        --act_cache results/activations/chess_16L_act137_L2_meas14.pkl
"""
import argparse, os, sys, pickle
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_layer


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


def crossing_angle(curves, angles_deg, threshold=0.1):
    """Find angle where median crosses threshold, with linear interpolation."""
    med = np.median(curves, axis=0)
    crossings = np.where(med > threshold)[0]
    if len(crossings) > 0:
        idx = crossings[0]
        if idx > 0:
            y0, y1 = med[idx-1], med[idx]
            x0, x1 = angles_deg[idx-1], angles_deg[idx]
            return x0 + (threshold - y0) / (y1 - y0) * (x1 - x0)
        return angles_deg[idx]
    return None


def crossing_angle_single(curve, angles_deg, threshold=0.1):
    """Find crossing angle for a single curve."""
    crossings = np.where(curve > threshold)[0]
    if len(crossings) > 0:
        idx = crossings[0]
        if idx > 0:
            y0, y1 = curve[idx-1], curve[idx]
            x0, x1 = angles_deg[idx-1], angles_deg[idx]
            return x0 + (threshold - y0) / (y1 - y0) * (x1 - x0)
        return angles_deg[idx]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', required=True)
    parser.add_argument('--act_cache', type=str, default=None)
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--measure_layer', type=int, default=14)
    parser.add_argument('--k_values', type=str, default='5,10,15,20,25',
                        help='Comma-separated subspace dimensions to test')
    parser.add_argument('--n_subsets', type=int, default=20,
                        help='Random subsets of features to sample per k')
    parser.add_argument('--n_dirs_per_subset', type=int, default=3,
                        help='Random directions per subset')
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(',')]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load directions
    data = pickle.load(open(args.features_path, 'rb'))
    if isinstance(data['directions'], list):
        all_dirs = torch.stack(data['directions']).float()
    else:
        all_dirs = data['directions'].float()

    cos_dists = data['cos_dists']
    if isinstance(cos_dists, list):
        cos_dists = np.array(cos_dists)
    good_mask = cos_dists > 0.1
    dirs = all_dirs[good_mask]
    n_features = dirs.shape[0]
    d_model = dirs.shape[1]
    print(f"Loaded {n_features} good directions from {args.features_path}")

    max_k = max(k_values)
    if n_features < max_k:
        print(f"Warning: only {n_features} features, reducing max k")
        k_values = [k for k in k_values if k <= n_features]

    # Load model
    model, tokenizer, device = load_model(args.model)
    model_dtype = next(model.parameters()).dtype

    # Load activation
    if args.act_cache:
        print(f"Loading cached activation from {args.act_cache}")
        act_data = pickle.load(open(args.act_cache, 'rb'))
        ctx = act_data['ctx'].to(device=device, dtype=model_dtype)
        act = act_data['act'].to(device=device, dtype=model_dtype)
    else:
        from src.data import generate_chess_games
        n_games = 138
        token_ids_list = generate_chess_games(n_games, 500, seed=args.seed)
        batch_size = 32
        count = 0
        for i in range(0, n_games, batch_size):
            bt = torch.stack(token_ids_list[i:i+batch_size]).to(device)
            with torch.no_grad():
                out = model(bt, output_hidden_states=True)
                h = out.hidden_states[args.layer + 1]
            local_idx = 137 - count
            if 0 <= local_idx < bt.shape[0]:
                ctx = h[local_idx:local_idx+1, :-1, :].to(device, dtype=model_dtype)
                act = h[local_idx, -1, :].to(device, dtype=model_dtype)
            count += bt.shape[0]
            del out, h, bt

    print(f"Activation norm={act.float().norm():.2f}")

    angles_deg = np.linspace(0, args.max_angle, int(args.max_angle * 4) + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    dirs_dev = dirs.to(device)

    # 1. Sweep all individual features for per-feature breaking angles
    print(f"\nSweeping {n_features} individual features...")
    feat_curves = np.zeros((n_features, len(angles_deg)))
    for i in range(n_features):
        feat_curves[i] = sweep_direction(
            model, ctx, act, dirs_dev[i], args.layer, args.measure_layer,
            device, angles_rad)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_features}")

    # Per-feature breaking angles
    feat_breaks = np.zeros(n_features)
    for i in range(n_features):
        b = crossing_angle_single(feat_curves[i], angles_deg)
        feat_breaks[i] = b if b is not None else args.max_angle

    theta_1_all = np.median(feat_breaks)
    print(f"Median single-feature breaking angle: {theta_1_all:.1f}°")
    print(f"Range: {feat_breaks.min():.1f}° - {feat_breaks.max():.1f}°")

    # 2. For each k: sample subsets, draw random dirs, sweep
    n_total_per_k = args.n_subsets * args.n_dirs_per_subset
    results = {}

    for k in k_values:
        print(f"\nk={k}: {args.n_subsets} subsets x {args.n_dirs_per_subset} dirs = {n_total_per_k} sweeps")
        all_curves = np.zeros((n_total_per_k, len(angles_deg)))
        subset_theta1s = []

        idx = 0
        for s in range(args.n_subsets):
            # Random subset of k features
            subset_indices = np.random.choice(n_features, size=k, replace=False)
            subset_dirs = dirs_dev[subset_indices]

            # θ₁ for this subset
            subset_theta1s.append(np.median(feat_breaks[subset_indices]))

            # QR basis
            Q, _ = torch.linalg.qr(subset_dirs.T, mode='reduced')

            for d in range(args.n_dirs_per_subset):
                coeffs = torch.randn(k, device=device)
                direction = Q @ coeffs
                direction = F.normalize(direction, dim=0)
                all_curves[idx] = sweep_direction(
                    model, ctx, act, direction, args.layer, args.measure_layer,
                    device, angles_rad)
                idx += 1

            if (s + 1) % 5 == 0:
                print(f"  {s+1}/{args.n_subsets} subsets")

        observed = crossing_angle(all_curves, angles_deg)

        # Theory prediction using median θ₁ across sampled subsets
        median_theta1_k = np.median(subset_theta1s)
        if k > 1:
            val = np.sin(np.deg2rad(median_theta1_k)) * np.sqrt(k / (2 * np.log(k)))
            predicted = np.degrees(np.arcsin(np.clip(val, -1, 1)))
        else:
            predicted = median_theta1_k

        results[k] = {
            'observed': observed,
            'predicted': predicted,
            'median_theta1_k': median_theta1_k,
            'curves': all_curves,
        }
        print(f"  k={k}: observed={observed:.1f}°, predicted={predicted:.1f}° (θ₁={median_theta1_k:.1f}°)")

    # Save
    results_dir = os.path.dirname(args.features_path)
    save_data = {
        'angles_deg': angles_deg,
        'feat_curves': feat_curves,
        'feat_breaks': feat_breaks,
        'theta_1_all': theta_1_all,
        'k_values': k_values,
        'results': {k: {'observed': v['observed'], 'predicted': v['predicted'],
                         'median_theta1_k': v['median_theta1_k']}
                    for k, v in results.items()},
        'args': vars(args),
    }
    save_path = os.path.join(results_dir, 'subspace_scaling.pkl')
    pickle.dump(save_data, open(save_path, 'wb'))

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    ks = sorted(results.keys())
    observed = [results[k]['observed'] for k in ks]
    predicted = [results[k]['predicted'] for k in ks]

    ax.plot(ks, observed, 'o-', color='C0', markersize=8, lw=2,
            label='Observed (median crossing)')
    ax.plot(ks, predicted, 's--', color='C3', markersize=8, lw=2,
            label=r'Theory: $\arcsin(\sin\theta_1(k)\sqrt{k/2\ln k})$')

    ax.axhline(theta_1_all, color='C0', ls=':', alpha=0.5,
               label=f'Single feature θ₁={theta_1_all:.1f}°')

    ax.set_xlabel('Subspace dimension k')
    ax.set_ylabel('Plateau-breaking angle (degrees)')
    ax.set_title('Subspace random: observed vs predicted breaking angle')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)
    plt.tight_layout()

    plot_path = os.path.join(results_dir, 'subspace_scaling.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {plot_path}")
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    main()
