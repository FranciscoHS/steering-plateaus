#!/usr/bin/env python3
"""Reconstruct novel chess activations using progressive MELBO directions.

Projects layer-2 activations from new chess games onto the 135 orthogonal
directions discovered for act_idx=137, then sweeps residual directions to
check if they look random (long plateau) or feature-like (short plateau).

Usage:
    python scripts/perturbation/reconstruct_activations.py \
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
    parser.add_argument('--features_path', required=True,
                        help='Path to features.pkl from progressive discovery')
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--measure_layer', type=int, default=14)
    parser.add_argument('--n_games', type=int, default=50,
                        help='Number of chess games to generate')
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--cd_threshold', type=float, default=0.1,
                        help='Cosine distance threshold for directions (min quality)')
    parser.add_argument('--sweep_threshold', type=float, default=0.1,
                        help='Threshold for crossing angle detection')
    parser.add_argument('--n_random_baseline', type=int, default=5,
                        help='Number of random directions to sweep per activation for baseline')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Load directions ----
    data = pickle.load(open(args.features_path, 'rb'))
    if isinstance(data['directions'], list):
        all_dirs = torch.stack(data['directions']).float()
    else:
        all_dirs = data['directions'].float()

    cos_dists = data['cos_dists']
    if isinstance(cos_dists, list):
        cos_dists = np.array(cos_dists)
    good_mask = cos_dists > args.cd_threshold
    dirs = all_dirs[good_mask]
    n_features = dirs.shape[0]
    d_model = dirs.shape[1]
    print(f"Loaded {n_features} directions (cd > {args.cd_threshold}) from {args.features_path}")
    print(f"  d_model={d_model}")

    # ---- Load model ----
    model, tokenizer, device = load_model(args.model)
    model_dtype = next(model.parameters()).dtype
    print(f"Model loaded on {device}, dtype={model_dtype}")

    # ---- Generate chess games and extract activations ----
    print(f"\nGenerating {args.n_games} chess games (seq_len={args.seq_len})...")
    token_ids_list = generate_chess_games(args.n_games, args.seq_len, seed=args.seed)
    n_actual = len(token_ids_list)
    print(f"  Got {n_actual} valid games")

    # Extract (ctx, act) for each game
    print("Extracting activations...")
    contexts = []
    activations = []
    batch_size = 32
    for i in range(0, n_actual, batch_size):
        batch = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            out = model(batch, output_hidden_states=True)
            h = out.hidden_states[args.layer + 1]  # residual after layer
        for j in range(batch.shape[0]):
            ctx = h[j:j+1, :-1, :].cpu()    # [1, seq_len-1, d_model]
            act = h[j, -1, :].cpu()           # [d_model]
            contexts.append(ctx)
            activations.append(act)
        del out, h, batch
        torch.cuda.empty_cache()
    print(f"  Extracted {len(activations)} (ctx, act) pairs")

    acts_tensor = torch.stack(activations).float()  # [n_actual, d_model]
    norms = acts_tensor.norm(dim=1)
    print(f"  Activation norms: mean={norms.mean():.1f}, std={norms.std():.1f}")

    # ---- Step 2: Project and compute residuals ----
    print("\n=== PROJECTION ===")
    dirs_matrix = dirs.float()  # [n_features, d_model]
    # coeffs[i, j] = dirs[j] . act[i]
    coeffs = acts_tensor @ dirs_matrix.T  # [n_actual, n_features]
    recon = coeffs @ dirs_matrix           # [n_actual, d_model]
    residuals = acts_tensor - recon        # [n_actual, d_model]

    resid_norms = residuals.norm(dim=1)
    act_norms = acts_tensor.norm(dim=1)
    resid_fracs = resid_norms / act_norms  # fraction unexplained

    print(f"Residual fraction ||resid||/||act||:")
    print(f"  mean={resid_fracs.mean():.4f}, std={resid_fracs.std():.4f}")
    print(f"  min={resid_fracs.min():.4f}, max={resid_fracs.max():.4f}")
    print(f"  median={resid_fracs.median():.4f}")

    # Variance explained
    var_explained = 1.0 - (resid_norms**2 / act_norms**2)
    print(f"Variance explained (1 - ||resid||²/||act||²):")
    print(f"  mean={var_explained.mean():.4f}, std={var_explained.std():.4f}")

    # Top coefficients per activation
    print(f"\nTop-5 coefficients (magnitude) per activation:")
    abs_coeffs = coeffs.abs()
    topk = torch.topk(abs_coeffs, k=min(5, n_features), dim=1)
    for i in range(min(10, n_actual)):
        top_indices = topk.indices[i].tolist()
        top_vals = [coeffs[i, idx].item() for idx in top_indices]
        desc = ", ".join([f"d{idx}={val:+.2f}" for idx, val in zip(top_indices, top_vals)])
        print(f"  act {i}: resid_frac={resid_fracs[i]:.4f}, top=[{desc}]")

    # ---- Step 3: Sweep residual directions ----
    print("\n=== RESIDUAL SWEEPS ===")
    angles_deg = np.linspace(0, args.max_angle, int(args.max_angle * 4) + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    dirs_dev = dirs.to(device)

    resid_results = []
    random_baselines = []

    # Sort by residual fraction to sweep a mix of small and large residuals
    sorted_indices = torch.argsort(resid_fracs).tolist()
    # Pick ~10 spread across the distribution: 5 smallest, 5 largest residuals
    n_sweep = min(10, n_actual)
    if n_actual >= 10:
        sweep_indices = sorted_indices[:5] + sorted_indices[-5:]
    else:
        sweep_indices = sorted_indices

    for count, act_idx in enumerate(sweep_indices):
        ctx = contexts[act_idx].to(device=device, dtype=model_dtype)
        act = activations[act_idx].to(device=device, dtype=model_dtype)
        resid = residuals[act_idx].to(device)
        resid_frac = resid_fracs[act_idx].item()

        resid_norm = resid.float().norm().item()
        if resid_norm < 1e-6:
            print(f"  act {act_idx}: residual ~0, skipping")
            continue

        # Sweep residual direction
        resid_dir = F.normalize(resid.float(), dim=0)
        resid_curve = sweep_direction(
            model, ctx, act, resid_dir, args.layer, args.measure_layer,
            device, angles_rad)
        resid_break = crossing_angle_single(resid_curve, angles_deg, args.sweep_threshold)

        # Random baseline: sweep a few random directions for this activation
        rand_breaks = []
        for r in range(args.n_random_baseline):
            rand_dir = F.normalize(torch.randn(d_model, device=device), dim=0)
            rand_curve = sweep_direction(
                model, ctx, act, rand_dir, args.layer, args.measure_layer,
                device, angles_rad)
            rb = crossing_angle_single(rand_curve, angles_deg, args.sweep_threshold)
            if rb is not None:
                rand_breaks.append(rb)

        rand_median = np.median(rand_breaks) if rand_breaks else None

        resid_results.append({
            'act_idx': act_idx,
            'resid_frac': resid_frac,
            'resid_curve': resid_curve,
            'resid_break': resid_break,
            'rand_breaks': rand_breaks,
            'rand_median': rand_median,
        })
        random_baselines.extend(rand_breaks)

        resid_str = f"{resid_break:.1f}°" if resid_break is not None else ">60°"
        rand_str = f"{rand_median:.1f}°" if rand_median is not None else ">60°"
        print(f"  [{count+1}/{len(sweep_indices)}] act {act_idx}: "
              f"resid_frac={resid_frac:.4f}, resid_break={resid_str}, "
              f"rand_baseline={rand_str}")

    # Summary
    print("\n=== SUMMARY ===")
    resid_breaks_all = [r['resid_break'] for r in resid_results if r['resid_break'] is not None]
    if resid_breaks_all:
        print(f"Residual breaking angles ({len(resid_breaks_all)} activations):")
        print(f"  mean={np.mean(resid_breaks_all):.1f}°, median={np.median(resid_breaks_all):.1f}°")
        print(f"  min={np.min(resid_breaks_all):.1f}°, max={np.max(resid_breaks_all):.1f}°")
    if random_baselines:
        print(f"Random baseline breaking angles ({len(random_baselines)} sweeps):")
        print(f"  mean={np.mean(random_baselines):.1f}°, median={np.median(random_baselines):.1f}°")

    # ---- Step 4: Feature interpretation ----
    print("\n=== FEATURE INTERPRETATION ===")
    print("(Showing top-5 features for first 5 games with PGN context)")
    for i in range(min(5, n_actual)):
        # Decode the game text
        tokens = token_ids_list[i]
        # Chess model uses character-level tokenizer
        try:
            game_text = tokenizer.decode(tokens.tolist())
        except Exception:
            game_text = "<decode failed>"

        # Show last ~80 chars of game for context
        game_snippet = game_text[-80:] if len(game_text) > 80 else game_text
        top_indices = topk.indices[i].tolist()
        top_vals = [coeffs[i, idx].item() for idx in top_indices]

        print(f"\nGame {i} (resid_frac={resid_fracs[i]:.4f}):")
        print(f"  ...{game_snippet}")
        for idx, val in zip(top_indices, top_vals):
            alpha = data['alphas_found'][idx] if 'alphas_found' in data else '?'
            cd = cos_dists[idx] if idx < len(cos_dists) else '?'
            print(f"  d{idx}: coeff={val:+.3f} (found at α={alpha}°, cd={cd:.3f})")

    # ---- Save results ----
    results_dir = os.path.dirname(args.features_path)
    save_data = {
        'coeffs': coeffs.cpu(),                    # [n_actual, n_features]
        'resid_fracs': resid_fracs.cpu().numpy(),   # [n_actual]
        'var_explained': var_explained.cpu().numpy(),
        'resid_norms': resid_norms.cpu().numpy(),
        'act_norms': act_norms.cpu().numpy(),
        'resid_results': resid_results,             # per-activation sweep data
        'random_baselines': random_baselines,
        'n_features': n_features,
        'n_activations': n_actual,
        'args': vars(args),
    }
    save_path = os.path.join(results_dir, 'reconstruction.pkl')
    pickle.dump(save_data, open(save_path, 'wb'))
    print(f"\nSaved: {save_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Residual fraction distribution
    ax = axes[0]
    ax.hist(resid_fracs.numpy(), bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(resid_fracs.mean().item(), color='red', ls='--',
               label=f'mean={resid_fracs.mean():.3f}')
    ax.set_xlabel('||residual|| / ||activation||')
    ax.set_ylabel('Count')
    ax.set_title(f'Residual fraction ({n_features} directions)')
    ax.legend()

    # Panel 2: Residual sweep curves
    ax = axes[1]
    for r in resid_results:
        label = f"act {r['act_idx']} (frac={r['resid_frac']:.3f})"
        ax.plot(angles_deg, r['resid_curve'], alpha=0.6, label=label)
    ax.axhline(args.sweep_threshold, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Cosine distance')
    ax.set_title('Residual direction sweeps')
    ax.legend(fontsize=6, ncol=2)

    # Panel 3: Breaking angle comparison (residual vs random)
    ax = axes[2]
    for r in resid_results:
        if r['resid_break'] is not None:
            color = 'C0' if r['resid_frac'] < resid_fracs.median().item() else 'C3'
            ax.scatter(r['resid_frac'], r['resid_break'], c=color, s=50, zorder=3)
    if random_baselines:
        ax.axhline(np.median(random_baselines), color='gray', ls='--',
                    label=f'Random baseline median={np.median(random_baselines):.1f}°')
    ax.set_xlabel('Residual fraction')
    ax.set_ylabel('Breaking angle (degrees)')
    ax.set_title('Residual breaking angle vs fraction')
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'reconstruction.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


if __name__ == '__main__':
    main()
