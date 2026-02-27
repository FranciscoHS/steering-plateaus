#!/usr/bin/env python3
"""Test subspace prediction scaling: how does predicted vs empirical breaking angle
vary with subspace dimension k?

For each k, samples random subsets of k features, draws random vectors from each
subset's subspace, computes the predicted breaking angle via:

    theta_pred(v) = arcsin(1 / max_i(alpha_i / sin(theta_break_i)))

and also sweeps each direction empirically. Compares predicted vs empirical medians
as a function of k.

Usage:
    # With cd > 0.1 filter (62 features):
    python scripts/perturbation/sweep_subspace_prediction_scaling.py \
        --features_path results/progressive/chess_16L_L2_meas14/features.pkl \
        --k_values 10,20,30,40,50,60 --min_cd 0.1

    # With all features (135):
    python scripts/perturbation/sweep_subspace_prediction_scaling.py \
        --features_path results/progressive/chess_16L_L2_meas14/features.pkl \
        --k_values 20,40,60,80,100,120 --min_cd 0.0
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


def crossing_angle_single(curve, angles_deg, threshold=0.1):
    """Find crossing angle for a single curve with linear interpolation."""
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
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--measure_layer', type=int, default=14)
    parser.add_argument('--act_idx', type=int, default=137)
    parser.add_argument('--act_cache', type=str, default=None,
                        help='Path to cached activation .pkl (skips chess game generation)')
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--k_values', type=str, required=True,
                        help='Comma-separated subspace dimensions to test')
    parser.add_argument('--min_cd', type=float, default=0.0,
                        help='Minimum cosine distance to include feature (0.0 = all)')
    parser.add_argument('--n_subsets', type=int, default=10,
                        help='Random subsets per k')
    parser.add_argument('--n_dirs_per_subset', type=int, default=10,
                        help='Random directions per subset')
    parser.add_argument('--n_random', type=int, default=30,
                        help='Fully random directions (baseline)')
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--threshold', type=float, default=0.1)
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

    # Filter by min_cd
    cos_dists = data['cos_dists']
    if isinstance(cos_dists, list):
        cos_dists = np.array(cos_dists)
    if args.min_cd > 0:
        mask = cos_dists > args.min_cd
        dirs = all_dirs[mask]
        print(f"Filtered to {dirs.shape[0]} directions with cd > {args.min_cd}")
    else:
        dirs = all_dirs
        print(f"Using all {dirs.shape[0]} directions (no cd filter)")

    n_features = dirs.shape[0]
    d_model = dirs.shape[1]
    dirs = F.normalize(dirs, dim=1)

    # Cap k values to available features
    k_values = [k for k in k_values if k <= n_features]
    print(f"k values: {k_values} (max features: {n_features})")

    # Load model
    model, tokenizer, device = load_model(args.model)
    model_dtype = next(model.parameters()).dtype

    # Load activation (from cache or by generating chess games)
    if args.act_cache:
        print(f"Loading cached activation from {args.act_cache}")
        act_data = pickle.load(open(args.act_cache, 'rb'))
        ctx = act_data['ctx'].to(device=device, dtype=model_dtype)
        act = act_data['act'].to(device=device, dtype=model_dtype)
    else:
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

    a_unit = act.float() / act.float().norm()
    print(f"Activation {args.act_idx}, norm={act.float().norm():.2f}")

    angles_deg = np.linspace(0, args.max_angle, int(args.max_angle * 4) + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    dirs_dev = dirs.to(device)

    # Orthogonalize features to activation direction
    dirs_orth = dirs_dev.float() - (dirs_dev.float() @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
    dirs_orth = F.normalize(dirs_orth, dim=1)

    # ── Step 1: Sweep all individual features -> per-feature breaking angles ──
    print(f"\nStep 1: Sweeping {n_features} individual features...")
    feat_curves = np.zeros((n_features, len(angles_deg)))
    feat_breaks = np.zeros(n_features)
    for i in range(n_features):
        feat_curves[i] = sweep_direction(
            model, ctx, act, dirs_dev[i], args.layer, args.measure_layer,
            device, angles_rad)
        b = crossing_angle_single(feat_curves[i], angles_deg, args.threshold)
        feat_breaks[i] = b if b is not None else args.max_angle
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_features}")

    valid_mask = feat_breaks < args.max_angle
    n_valid = valid_mask.sum()
    print(f"  {n_valid}/{n_features} features break within {args.max_angle} deg")
    if n_valid > 0:
        print(f"  Median break: {np.median(feat_breaks[valid_mask]):.1f} deg")

    sin_breaks = np.sin(np.deg2rad(feat_breaks))

    # ── Step 2: Fully random baseline ──
    print(f"\nStep 2: Sweeping {args.n_random} fully random directions...")
    rand_angles = np.zeros(args.n_random)
    for i in range(args.n_random):
        d = F.normalize(torch.randn(d_model, device=device), dim=0)
        curve = sweep_direction(model, ctx, act, d, args.layer, args.measure_layer,
                                device, angles_rad)
        b = crossing_angle_single(curve, angles_deg, args.threshold)
        rand_angles[i] = b if b is not None else args.max_angle
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_random}")
    print(f"  Random median: {np.median(rand_angles):.1f} deg")

    # ── Step 3: For each k, subsample and compare predicted vs empirical ──
    results = {}
    for k in k_values:
        print(f"\nk={k}: {args.n_subsets} subsets x {args.n_dirs_per_subset} dirs")
        n_total = args.n_subsets * args.n_dirs_per_subset
        all_predicted = np.zeros(n_total)
        all_empirical = np.zeros(n_total)

        idx = 0
        for s in range(args.n_subsets):
            # Random subset of k features
            subset_indices = np.random.choice(n_features, size=k, replace=False)
            subset_dirs_orth = dirs_orth[subset_indices]
            subset_sin_breaks = sin_breaks[subset_indices]
            subset_valid = valid_mask[subset_indices]

            # QR basis for this subset's subspace
            Q, _ = torch.linalg.qr(subset_dirs_orth.T, mode='reduced')
            sub_rank = Q.shape[1]

            for d in range(args.n_dirs_per_subset):
                # Random direction in subset's subspace
                coeffs = torch.randn(sub_rank, device=device)
                v = Q @ coeffs
                v = F.normalize(v, dim=0)

                # Predicted breaking angle (max-ratio formula)
                alignments = torch.abs(subset_dirs_orth @ v).cpu().numpy()
                ratios = np.where(subset_valid, alignments / subset_sin_breaks, 0.0)
                max_ratio = ratios.max()
                if max_ratio < 1.0:
                    all_predicted[idx] = args.max_angle
                else:
                    all_predicted[idx] = np.degrees(np.arcsin(1.0 / max_ratio))

                # Empirical sweep
                curve = sweep_direction(
                    model, ctx, act, v, args.layer, args.measure_layer,
                    device, angles_rad)
                b = crossing_angle_single(curve, angles_deg, args.threshold)
                all_empirical[idx] = b if b is not None else args.max_angle
                idx += 1

            if (s + 1) % 5 == 0:
                print(f"  {s+1}/{args.n_subsets} subsets")

        pred_med = np.median(all_predicted)
        emp_med = np.median(all_empirical)
        corr = np.corrcoef(all_predicted, all_empirical)[0, 1]
        results[k] = {
            'predicted': all_predicted,
            'empirical': all_empirical,
            'pred_median': pred_med,
            'emp_median': emp_med,
            'correlation': corr,
        }
        print(f"  k={k}: predicted={pred_med:.1f} deg, empirical={emp_med:.1f} deg, "
              f"err={pred_med - emp_med:+.1f} deg, corr={corr:.3f}")

    # ── Save ──
    filter_tag = f"cd{args.min_cd}" if args.min_cd > 0 else "all"
    results_dir = os.path.dirname(args.features_path)
    save_data = {
        'k_values': k_values,
        'results': {k: {key: val for key, val in v.items()}
                    for k, v in results.items()},
        'feat_breaks': feat_breaks,
        'valid_mask': valid_mask,
        'rand_angles': rand_angles,
        'n_features': n_features,
        'args': vars(args),
    }
    save_path = os.path.join(results_dir, f'subspace_pred_scaling_{filter_tag}.pkl')
    pickle.dump(save_data, open(save_path, 'wb'))

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Predicted vs empirical median as function of k
    ax = axes[0]
    ks = sorted(results.keys())
    pred_meds = [results[k]['pred_median'] for k in ks]
    emp_meds = [results[k]['emp_median'] for k in ks]

    ax.plot(ks, pred_meds, 's--', color='C3', markersize=8, lw=2, label='Predicted (median)')
    ax.plot(ks, emp_meds, 'o-', color='C0', markersize=8, lw=2, label='Empirical (median)')

    # IQR bands
    for k in ks:
        r = results[k]
        ax.fill_between([k-1, k+1],
                        [np.percentile(r['predicted'], 25)] * 2,
                        [np.percentile(r['predicted'], 75)] * 2,
                        color='C3', alpha=0.1)
        ax.fill_between([k-1, k+1],
                        [np.percentile(r['empirical'], 25)] * 2,
                        [np.percentile(r['empirical'], 75)] * 2,
                        color='C0', alpha=0.1)

    ax.axhline(np.median(rand_angles), color='gray', ls='--', alpha=0.7,
               label=f'Fully random={np.median(rand_angles):.1f} deg')
    if n_valid > 0:
        ax.axhline(np.median(feat_breaks[valid_mask]), color='C2', ls=':', alpha=0.7,
                    label=f'Single feature={np.median(feat_breaks[valid_mask]):.1f} deg')

    ax.set_xlabel('Subspace dimension k')
    ax.set_ylabel('Breaking angle (deg)')
    ax.set_title(f'Scaling: predicted vs empirical ({n_features} features, {filter_tag})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)

    # Panel 2: Per-vector correlation as function of k
    ax = axes[1]
    corrs = [results[k]['correlation'] for k in ks]
    ax.bar(ks, corrs, width=max(ks)//len(ks)*0.6, color='C1', alpha=0.7)
    ax.axhline(0, color='k', ls='-', alpha=0.3)
    ax.set_xlabel('Subspace dimension k')
    ax.set_ylabel('Per-vector correlation (pred vs emp)')
    ax.set_title('Prediction quality per vector')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)
    ax.set_ylim(-0.3, 1.0)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'subspace_pred_scaling_{filter_tag}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY (filter: {filter_tag}, {n_features} features)")
    print(f"{'='*60}")
    print(f"Single feature median: {np.median(feat_breaks[valid_mask]):.1f} deg")
    for k in ks:
        r = results[k]
        print(f"k={k:3d}: pred={r['pred_median']:5.1f}, emp={r['emp_median']:5.1f}, "
              f"err={r['pred_median']-r['emp_median']:+5.1f}, corr={r['correlation']:.3f}")
    print(f"Random: {np.median(rand_angles):.1f} deg")
    print(f"\nSaved: {save_path}")
    print(f"Saved: {plot_path}")


if __name__ == '__main__':
    main()
