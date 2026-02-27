#!/usr/bin/env python3
"""Locally refine a discovered feature to test if it's 'dirty'.

For a given feature direction, optimizes within increasing angular neighborhoods
to find the direction that breaks the plateau earliest. A clean feature shows
flat θ_break; a dirty feature shows θ_break decreasing as drift allowance grows.

Usage:
    python scripts/perturbation/refine_feature.py \
        --features_path results/sequential/chess_16L_L2_meas14_alpha8.0/features.pkl \
        --act_cache results/activations/chess_16L_act137_L2_meas14.pkl \
        --feature_idx 0
"""
import argparse, os, sys, pickle
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (load_model, forward_from_layer_to_layer,
                       _get_blocks, _run_block, _is_rope_model,
                       _get_position_embeddings, _is_cat_pos_emb_model,
                       _get_cat_pos_emb)


def forward_checkpointed(model, ctx, points, start_layer, end_layer):
    """Forward with gradient checkpointing."""
    context_batch = ctx.expand(points.shape[0], -1, -1)
    hidden = torch.cat([context_batch, points], dim=1)
    blocks = _get_blocks(model)
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, hidden)
    elif _is_cat_pos_emb_model(model):
        pos_emb = _get_cat_pos_emb(model, hidden.shape[1])
    for block in blocks[start_layer + 1:end_layer + 1]:
        hidden = torch.utils.checkpoint.checkpoint(
            _run_block, block, hidden, model, pos_emb, use_reentrant=False)
    return hidden[:, -1, :]


def sweep_direction(model, ctx, act, d, layer, measure_layer, device, angles_rad):
    """Sweep a single direction, return cosine distance array."""
    model_dtype = next(model.parameters()).dtype
    R = act.float().norm()
    a_unit = act.float() / R
    d_d = d.to(device=device, dtype=torch.float32)
    d_orth = d_d - (d_d @ a_unit) * a_unit
    d_orth = F.normalize(d_orth, dim=0)

    cos_arr = np.zeros(len(angles_rad))
    with torch.no_grad():
        ref = forward_from_layer_to_layer(
            model, ctx, act[None, None], layer, measure_layer).float()
        for j, theta in enumerate(angles_rad):
            p = R * (np.cos(theta) * a_unit + np.sin(theta) * d_orth)
            p = p.to(model_dtype)
            pf = forward_from_layer_to_layer(
                model, ctx, p[None, None], layer, measure_layer).float()
            cos_arr[j] = 1.0 - F.cosine_similarity(ref, pf, dim=-1).item()
    return cos_arr


def crossing_angle_single(curve, angles_deg, threshold=0.1):
    """Find crossing angle for a single curve with interpolation."""
    crossings = np.where(curve > threshold)[0]
    if len(crossings) > 0:
        idx = crossings[0]
        if idx > 0:
            y0, y1 = curve[idx-1], curve[idx]
            x0, x1 = angles_deg[idx-1], angles_deg[idx]
            return x0 + (threshold - y0) / (y1 - y0) * (x1 - x0)
        return angles_deg[idx]
    return None


def optimize_in_cone(model, ctx, act, ref_out, opt_angle_deg, layer,
                     measure_layer, device, d_model, d_start, max_drift_deg,
                     n_steps=300, lr=0.01, n_restarts=3):
    """Optimize direction to maximize cd, constrained to cone around d_start.

    Returns: best_direction, best_cd, drift_angle
    """
    model_dtype = next(model.parameters()).dtype
    R = act.float().norm()
    a_unit = act.float() / R
    theta = np.deg2rad(opt_angle_deg)
    cos_drift_min = np.cos(np.deg2rad(max_drift_deg))

    best_cd = -1.0
    best_dir = None

    for restart in range(n_restarts):
        # Initialize: random perturbation within the cone
        if max_drift_deg < 1.0:
            direction = d_start.clone()
        else:
            noise = torch.randn(d_model, device=device, dtype=torch.float32)
            # Project out d_start component, add scaled noise
            noise = noise - (noise @ d_start) * d_start
            noise = F.normalize(noise, dim=0)
            # Random angle within cone
            rand_angle = np.deg2rad(max_drift_deg * np.random.uniform(0, 0.5))
            direction = np.cos(rand_angle) * d_start + np.sin(rand_angle) * noise

        direction = F.normalize(direction, dim=0)
        direction = torch.nn.Parameter(direction)
        optimizer = torch.optim.Adam([direction], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()

            d_normed = direction / direction.norm()

            # Orthogonalize w.r.t. activation direction
            proj = (d_normed @ a_unit)
            d_orth = d_normed - proj * a_unit
            d_orth = F.normalize(d_orth, dim=0)

            perturbed = R * (np.cos(theta) * a_unit + np.sin(theta) * d_orth)
            perturbed = perturbed.to(model_dtype)

            p_out = forward_checkpointed(
                model, ctx, perturbed[None, None], layer, measure_layer)

            cd = 1.0 - F.cosine_similarity(ref_out, p_out.float(), dim=-1)
            loss = -cd
            loss.backward()
            optimizer.step()

            # Project back to cone if needed
            with torch.no_grad():
                direction.data = F.normalize(direction.data, dim=0)
                cos_sim = (direction.data @ d_start).item()
                if cos_sim < cos_drift_min:
                    # Project back to cone boundary
                    d_parallel = (direction.data @ d_start) * d_start
                    d_perp = direction.data - d_parallel
                    d_perp = F.normalize(d_perp, dim=0) if d_perp.norm() > 1e-8 else d_perp
                    sin_max = np.sin(np.deg2rad(max_drift_deg))
                    direction.data = cos_drift_min * d_start + sin_max * d_perp
                    direction.data = F.normalize(direction.data, dim=0)

        final_cd = cd.item()
        if final_cd > best_cd:
            best_cd = final_cd
            best_dir = direction.data.clone().detach()

    drift = np.degrees(np.arccos(np.clip((best_dir @ d_start).item(), -1, 1)))
    return best_dir, best_cd, drift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', required=True)
    parser.add_argument('--act_cache', type=str, default=None)
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--measure_layer', type=int, default=14)
    parser.add_argument('--feature_idx', type=int, default=0,
                        help='Which feature to refine (index into good features)')
    parser.add_argument('--opt_angle', type=float, default=5.0,
                        help='Perturbation angle for optimization')
    parser.add_argument('--drift_values', type=str, default='0,2,4,6,8,10,15,20,25,30',
                        help='Comma-separated max drift angles to test')
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--max_sweep_angle', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    drift_values = [float(d) for d in args.drift_values.split(',')]

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

    feature_idx = args.feature_idx
    d_start = dirs[feature_idx]
    print(f"Feature {feature_idx}, original cd={cos_dists[good_mask][feature_idx]:.4f}")

    # Load model
    model, tokenizer, device = load_model(args.model)
    model_dtype = next(model.parameters()).dtype

    # Load activation
    if args.act_cache:
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

    d_start = d_start.to(device)

    # Reference output
    with torch.no_grad():
        ref_out = forward_from_layer_to_layer(
            model, ctx, act.unsqueeze(0).unsqueeze(0),
            args.layer, args.measure_layer).float()

    # Sweep angles for measuring breaking angle
    sweep_angles_deg = np.linspace(0, args.max_sweep_angle,
                                    int(args.max_sweep_angle * 4) + 1)[1:]
    sweep_angles_rad = np.deg2rad(sweep_angles_deg)

    # Baseline: sweep the original feature
    print(f"\nBaseline sweep of original feature...")
    baseline_curve = sweep_direction(
        model, ctx, act, d_start, args.layer, args.measure_layer,
        device, sweep_angles_rad)
    baseline_break = crossing_angle_single(baseline_curve, sweep_angles_deg)
    print(f"  Baseline θ_break = {baseline_break:.2f}°")

    # For each drift cap: optimize, then sweep the result
    results = []
    for max_drift in drift_values:
        print(f"\nDrift cap = {max_drift}°:")
        if max_drift < 0.5:
            # No drift allowed, use original
            refined_dir = d_start
            opt_cd = 0.0
            actual_drift = 0.0
        else:
            refined_dir, opt_cd, actual_drift = optimize_in_cone(
                model, ctx, act, ref_out, args.opt_angle, args.layer,
                args.measure_layer, device, d_model, d_start, max_drift,
                n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts)

        # Sweep refined direction
        refined_curve = sweep_direction(
            model, ctx, act, refined_dir, args.layer, args.measure_layer,
            device, sweep_angles_rad)
        refined_break = crossing_angle_single(refined_curve, sweep_angles_deg)

        results.append({
            'max_drift': max_drift,
            'actual_drift': actual_drift,
            'theta_break': refined_break,
            'opt_cd': opt_cd,
            'curve': refined_curve,
            'direction': refined_dir.cpu(),
        })
        print(f"  actual drift={actual_drift:.1f}°, θ_break={refined_break:.2f}°, opt_cd={opt_cd:.4f}")

    # Save
    results_dir = os.path.dirname(args.features_path)
    save_data = {
        'feature_idx': feature_idx,
        'baseline_break': baseline_break,
        'baseline_curve': baseline_curve,
        'sweep_angles_deg': sweep_angles_deg,
        'results': results,
        'args': vars(args),
    }
    save_path = os.path.join(results_dir, f'refine_feature_{feature_idx}.pkl')
    pickle.dump(save_data, open(save_path, 'wb'))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: θ_break vs drift cap
    ax = axes[0]
    drifts = [r['max_drift'] for r in results]
    breaks = [r['theta_break'] for r in results]
    ax.plot(drifts, breaks, 'o-', color='C0', markersize=6, lw=2)
    ax.axhline(baseline_break, color='gray', ls=':', alpha=0.5,
               label=f'Original θ_break={baseline_break:.1f}°')
    ax.set_xlabel('Max allowed drift (degrees)')
    ax.set_ylabel('Plateau-breaking angle (degrees)')
    ax.set_title(f'Feature {feature_idx}: refinement')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: perturbation curves at selected drift caps
    ax = axes[1]
    ax.plot(sweep_angles_deg, baseline_curve, color='black', lw=2,
            label=f'Original (drift=0°)')
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(results)))
    for i, r in enumerate(results):
        if r['max_drift'] in [0, 4, 10, 20, 30]:
            if r['max_drift'] == 0:
                continue
            ax.plot(sweep_angles_deg, r['curve'], color=cmap[i], lw=1.5,
                    label=f'drift≤{r["max_drift"]:.0f}° (actual={r["actual_drift"]:.1f}°)')
    ax.axhline(0.1, color='black', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('Cosine distance')
    ax.set_title(f'Feature {feature_idx}: perturbation curves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'refine_feature_{feature_idx}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {plot_path}")
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    main()
