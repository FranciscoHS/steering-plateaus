#!/usr/bin/env python3
"""Test subspace plateau-breaking prediction: theoretical vs empirical.

For each random unit vector v sampled from the feature subspace, the error
correction model predicts the plateau-breaking angle as:

    theta_pred(v) = arcsin(1 / max_i(alpha_i / sin(theta_break_i)))

where:
    alpha_i = |v . d_i_orth|  (alignment of v with orthogonalized feature i)
    theta_break_i = individual plateau-breaking angle of feature i

The feature that "breaks first" is the one maximizing alpha_i / sin(theta_break_i).

This script:
1. Sweeps all individual features -> per-feature breaking angles
2. Samples random directions from the feature subspace
3. For each: computes predicted breaking angle AND sweeps empirically
4. Compares predicted vs empirical distributions

Usage:
    python scripts/perturbation/sweep_subspace_prediction.py \
        --features_path results/progressive/chess_16L_L2_meas14/features.pkl
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
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--n_subspace', type=int, default=100,
                        help='Number of random-in-subspace directions to sample')
    parser.add_argument('--n_random', type=int, default=30,
                        help='Number of fully random directions (sanity check)')
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--threshold', type=float, default=0.1)
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

    # Use all directions (no cd filter — per-feature breaking angles measured empirically)
    dirs = all_dirs
    n_features = dirs.shape[0]
    d_model = dirs.shape[1]
    print(f"Loaded {n_features} directions from {args.features_path}")

    # Normalize features
    dirs = F.normalize(dirs, dim=1)

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

    a_unit = act.float() / act.float().norm()
    print(f"Activation {args.act_idx}, norm={act.float().norm():.2f}")

    angles_deg = np.linspace(0, args.max_angle, int(args.max_angle * 4) + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Orthogonalize features to activation direction
    dirs_dev = dirs.to(device)
    dirs_orth = dirs_dev.float() - (dirs_dev.float() @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
    dirs_orth = F.normalize(dirs_orth, dim=1)

    # ── Step 1: Sweep individual features -> per-feature breaking angles ──
    print(f"\nStep 1: Sweeping {n_features} individual features...")
    feat_curves = np.zeros((n_features, len(angles_deg)))
    feat_breaks = np.zeros(n_features)
    for i in range(n_features):
        feat_curves[i] = sweep_direction(
            model, ctx, act, dirs_dev[i], args.layer, args.measure_layer,
            device, angles_rad)
        b = crossing_angle_single(feat_curves[i], angles_deg, args.threshold)
        feat_breaks[i] = b if b is not None else args.max_angle
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_features}")

    # Identify features that actually break within sweep range
    valid_mask = feat_breaks < args.max_angle
    n_valid = valid_mask.sum()
    print(f"\nPer-feature breaking angles:")
    print(f"  Valid (break < {args.max_angle} deg): {n_valid}/{n_features}")
    if n_valid > 0:
        print(f"  Median: {np.median(feat_breaks[valid_mask]):.1f} deg")
        print(f"  Range: {feat_breaks[valid_mask].min():.1f} - {feat_breaks[valid_mask].max():.1f} deg")

    sin_breaks = np.sin(np.deg2rad(feat_breaks))

    # ── Step 2: Build subspace basis from orthogonalized features ──
    Q, _ = torch.linalg.qr(dirs_orth.T, mode='reduced')  # (d_model, rank)
    rank = Q.shape[1]
    print(f"\nSubspace rank: {rank} (from {n_features} features)")

    print(f"\nStep 2: Sampling {args.n_subspace} random-in-subspace directions...")
    subspace_dirs = []
    for _ in range(args.n_subspace):
        coeffs = torch.randn(rank, device=device)
        d = Q @ coeffs
        d = F.normalize(d, dim=0)
        subspace_dirs.append(d)

    # ── Step 3: Compute predicted breaking angles ──
    print(f"\nStep 3: Computing predicted breaking angles...")
    predicted_angles = np.zeros(args.n_subspace)
    which_feature_breaks = np.zeros(args.n_subspace, dtype=int)

    for si, v in enumerate(subspace_dirs):
        # Alignment of v with each orthogonalized feature direction
        alignments = torch.abs(dirs_orth @ v).cpu().numpy()  # (n_features,)

        # Only consider features that actually break for the prediction
        if n_valid > 0:
            ratios = np.where(valid_mask, alignments / sin_breaks, 0.0)
        else:
            ratios = alignments / sin_breaks

        best_feat = np.argmax(ratios)
        max_ratio = ratios[best_feat]
        which_feature_breaks[si] = best_feat

        if max_ratio < 1.0:
            # No feature breaks: 1/max_ratio > 1, arcsin undefined
            predicted_angles[si] = args.max_angle
        else:
            predicted_angles[si] = np.degrees(np.arcsin(1.0 / max_ratio))

    print(f"  Predicted median: {np.median(predicted_angles):.1f} deg")
    print(f"  Predicted IQR: [{np.percentile(predicted_angles, 25):.1f}, "
          f"{np.percentile(predicted_angles, 75):.1f}] deg")

    # ── Step 4: Sweep subspace directions empirically ──
    print(f"\nStep 4: Sweeping {args.n_subspace} subspace directions empirically...")
    sub_curves = np.zeros((args.n_subspace, len(angles_deg)))
    empirical_angles = np.zeros(args.n_subspace)
    for i in range(args.n_subspace):
        sub_curves[i] = sweep_direction(
            model, ctx, act, subspace_dirs[i], args.layer, args.measure_layer,
            device, angles_rad)
        b = crossing_angle_single(sub_curves[i], angles_deg, args.threshold)
        empirical_angles[i] = b if b is not None else args.max_angle
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_subspace}")

    print(f"  Empirical median: {np.median(empirical_angles):.1f} deg")
    print(f"  Empirical IQR: [{np.percentile(empirical_angles, 25):.1f}, "
          f"{np.percentile(empirical_angles, 75):.1f}] deg")

    # ── Step 5: Fully random directions (sanity check) ──
    print(f"\nStep 5: Sweeping {args.n_random} fully random directions...")
    rand_curves = np.zeros((args.n_random, len(angles_deg)))
    random_angles = np.zeros(args.n_random)
    for i in range(args.n_random):
        d = F.normalize(torch.randn(d_model, device=device), dim=0)
        rand_curves[i] = sweep_direction(
            model, ctx, act, d, args.layer, args.measure_layer,
            device, angles_rad)
        b = crossing_angle_single(rand_curves[i], angles_deg, args.threshold)
        random_angles[i] = b if b is not None else args.max_angle
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_random}")

    print(f"  Random median: {np.median(random_angles):.1f} deg")

    # ── Save results ──
    results_dir = os.path.dirname(args.features_path)
    save_data = {
        'angles_deg': angles_deg,
        'feat_curves': feat_curves,
        'feat_breaks': feat_breaks,
        'sub_curves': sub_curves,
        'rand_curves': rand_curves,
        'predicted_angles': predicted_angles,
        'empirical_angles': empirical_angles,
        'random_angles': random_angles,
        'which_feature_breaks': which_feature_breaks,
        'n_features': n_features,
        'subspace_rank': rank,
        'valid_mask': valid_mask,
        'args': vars(args),
    }
    save_path = os.path.join(results_dir, 'subspace_prediction.pkl')
    pickle.dump(save_data, open(save_path, 'wb'))
    print(f"\nSaved: {save_path}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Scatter of predicted vs empirical per vector
    ax = axes[0]
    ax.scatter(predicted_angles, empirical_angles, alpha=0.4, s=20, c='C0')
    lim = max(predicted_angles.max(), empirical_angles.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.5, label='y=x')
    ax.axhline(np.median(empirical_angles), color='C0', ls=':', alpha=0.5,
               label=f'Emp. median={np.median(empirical_angles):.1f}')
    ax.axvline(np.median(predicted_angles), color='C3', ls=':', alpha=0.5,
               label=f'Pred. median={np.median(predicted_angles):.1f}')
    ax.set_xlabel('Predicted breaking angle (deg)')
    ax.set_ylabel('Empirical breaking angle (deg)')
    ax.set_title('Predicted vs empirical (per vector)')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Panel 2: Histograms of breaking angle distributions
    ax = axes[1]
    bins = np.linspace(0, args.max_angle, 30)
    ax.hist(feat_breaks[valid_mask], bins=bins, alpha=0.4,
            label=f'Individual features (n={n_valid})', color='C2')
    ax.hist(predicted_angles, bins=bins, alpha=0.4,
            label='Predicted subspace', color='C3')
    ax.hist(empirical_angles, bins=bins, alpha=0.4,
            label='Empirical subspace', color='C0')
    ax.hist(random_angles, bins=bins, alpha=0.3,
            label='Fully random', color='gray')
    ax.axvline(np.median(predicted_angles), color='C3', ls='--', lw=2)
    ax.axvline(np.median(empirical_angles), color='C0', ls='--', lw=2)
    ax.set_xlabel('Breaking angle (deg)')
    ax.set_ylabel('Count')
    ax.set_title('Breaking angle distributions')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Median response curves
    ax = axes[2]
    ax.plot(angles_deg, np.median(feat_curves, axis=0), color='C2', lw=2,
            label=f'Individual features (n={n_features})')
    ax.plot(angles_deg, np.median(sub_curves, axis=0), color='C0', lw=2,
            label=f'Subspace random (n={args.n_subspace})')
    ax.plot(angles_deg, np.median(rand_curves, axis=0), color='gray', lw=2, ls='--',
            label=f'Fully random (n={args.n_random})')
    ax.axhline(args.threshold, color='k', ls=':', alpha=0.3)
    # Mark predicted vs empirical medians
    pred_med = np.median(predicted_angles)
    emp_med = np.median(empirical_angles)
    ax.axvline(pred_med, color='C3', ls=':', alpha=0.7, label=f'Predicted median={pred_med:.1f}')
    ax.axvline(emp_med, color='C0', ls=':', alpha=0.7, label=f'Empirical median={emp_med:.1f}')
    ax.set_xlabel('Perturbation angle (deg)')
    ax.set_ylabel('Cosine distance')
    ax.set_title('Median response curves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, args.max_angle)

    plt.suptitle(f'Subspace prediction test ({n_features} features, rank-{rank} subspace)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'subspace_prediction.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Features: {n_features} (subspace rank {rank}, {n_valid} break within range)")
    if n_valid > 0:
        print(f"Individual features median break: {np.median(feat_breaks[valid_mask]):.1f} deg")
    print(f"Subspace random predicted (median): {np.median(predicted_angles):.1f} deg")
    print(f"Subspace random empirical (median): {np.median(empirical_angles):.1f} deg")
    print(f"Fully random (median): {np.median(random_angles):.1f} deg")
    print(f"Prediction error: {np.median(predicted_angles) - np.median(empirical_angles):.1f} deg")
    # Per-vector correlation
    corr = np.corrcoef(predicted_angles, empirical_angles)[0, 1]
    print(f"Per-vector correlation: {corr:.3f}")


if __name__ == '__main__':
    main()
