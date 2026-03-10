#!/usr/bin/env python3
"""Perturbation sweep on randomly sampled points in ChessGPT activation space.

Samples random points (Gaussian-matched to real layer-2 activations),
perturbs them along random directions on the sphere, and measures
cosine distance at the 2nd-to-last layer.

This tests whether the plateau phenomenon is specific to real activations
or also appears for arbitrary points in activation space.

Usage:
    python scripts/perturbation/sweep_random_points_chess.py
    python scripts/perturbation/sweep_random_points_chess.py --n_act 50 --n_dirs 20
"""
import argparse, os, sys, pickle, time
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from src.data import generate_chess_games, sample_random_activations_v2


def sweep_direction(model, ctx, act, direction, layer, measure_layer, device, angles_rad):
    """Sweep a single direction, return cosine distance array."""
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=model_dtype)
    R = a_d.float().norm()
    a_unit = a_d.float() / R
    d_d = direction.to(device=device, dtype=torch.float32)
    # Project out component along activation to get orthogonal direction
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
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=2,
                        help='Layer to inject activations at')
    parser.add_argument('--measure_layer', type=int, default=-2,
                        help='Layer to measure cosine distance at (default: 2nd-to-last)')
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--n_act', type=int, default=50,
                        help='Number of random points to sweep')
    parser.add_argument('--n_dirs', type=int, default=20,
                        help='Random directions per point')
    parser.add_argument('--n_fit', type=int, default=200,
                        help='Number of real activations for covariance fitting')
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_real', type=int, default=50,
                        help='Number of real activations to sweep for comparison')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/random_points_sweep/{args.model}_L{args.layer}'
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    n_layers = len(_get_blocks(model))
    d_model = model._plateau_info['d_model']

    # Resolve negative measure_layer
    if args.measure_layer < 0:
        args.measure_layer = n_layers + args.measure_layer
    print(f"d_model={d_model}, n_layers={n_layers}, "
          f"inject@layer {args.layer}, measure@layer {args.measure_layer}")

    # Generate chess games and extract real activations for covariance fitting
    n_games = max(args.n_fit, args.n_real)
    print(f"Generating {n_games} chess games (seq_len={args.seq_len})...")
    token_ids_list = generate_chess_games(n_games, args.seq_len, seed=args.seed)

    model_dtype = next(model.parameters()).dtype
    print(f"Extracting layer-{args.layer} activations from {n_games} games...")
    all_contexts = []
    all_activations = []
    batch_size = 32
    for i in range(0, n_games, batch_size):
        bt = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            out = model(bt, output_hidden_states=True)
            h = out.hidden_states[args.layer + 1]
        all_contexts.append(h[:, :-1, :].float().cpu())
        all_activations.append(h[:, -1, :].float().cpu())
        del out, h, bt
    all_contexts = torch.cat(all_contexts, 0)
    all_activations = torch.cat(all_activations, 0)
    print(f"Real activation norms: mean={all_activations.norm(dim=-1).mean():.2f}, "
          f"std={all_activations.norm(dim=-1).std():.2f}")

    # Sample random points from fitted Gaussian
    print(f"\nFitting covariance from {all_activations.shape[0]} activations...")
    random_acts, cov_stats = sample_random_activations_v2(
        all_activations, args.n_act, mode='full', seed=args.seed, device=device
    )
    print(f"Random point norms: mean={random_acts.norm(dim=-1).mean():.2f}, "
          f"std={random_acts.norm(dim=-1).std():.2f}")

    # Use real contexts as stand-ins for the random points
    # (we need some context to feed the earlier layers — reuse real ones)
    real_contexts = all_contexts[:max(args.n_act, args.n_real)]
    real_activations = all_activations[:args.n_real]

    # Angle grid
    angles_deg = np.linspace(0, args.max_angle, int(args.max_angle * 4) + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Sweep random points
    print(f"\n=== Random points ({args.n_act} points x {args.n_dirs} dirs) ===")
    rand_curves = np.zeros((args.n_act, args.n_dirs, len(angles_deg)))
    t0 = time.time()
    for i in range(args.n_act):
        ctx = real_contexts[i % len(real_contexts)]
        for j in range(args.n_dirs):
            d = torch.randn(d_model)
            rand_curves[i, j] = sweep_direction(
                model, ctx, random_acts[i], d,
                args.layer, args.measure_layer, device, angles_rad)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
    print(f"Random points done in {time.time()-t0:.1f}s")

    # Sweep real activations for comparison
    print(f"\n=== Real activations ({args.n_real} points x {args.n_dirs} dirs) ===")
    real_curves = np.zeros((args.n_real, args.n_dirs, len(angles_deg)))
    t0 = time.time()
    for i in range(args.n_real):
        for j in range(args.n_dirs):
            d = torch.randn(d_model)
            real_curves[i, j] = sweep_direction(
                model, real_contexts[i], real_activations[i], d,
                args.layer, args.measure_layer, device, angles_rad)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_real}  ({(i+1)/elapsed:.1f} act/s)")
    print(f"Real activations done in {time.time()-t0:.1f}s")

    # Save results
    results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'rand_curves': rand_curves,   # [n_act, n_dirs, n_angles]
        'real_curves': real_curves,    # [n_real, n_dirs, n_angles]
        'rand_norms': random_acts.norm(dim=-1).numpy(),
        'real_norms': real_activations.norm(dim=-1).numpy(),
    }
    pkl_path = os.path.join(args.results_dir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # Plot: median + IQR for both groups
    # Average over directions first, then show distribution over activations
    rand_per_act = np.median(rand_curves, axis=1)  # [n_act, n_angles]
    real_per_act = np.median(real_curves, axis=1)   # [n_real, n_angles]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Real activations
    ax.fill_between(angles_deg,
                    np.percentile(real_per_act, 25, axis=0),
                    np.percentile(real_per_act, 75, axis=0),
                    color='C0', alpha=0.15)
    ax.plot(angles_deg, np.median(real_per_act, axis=0),
            color='C0', lw=2.5, label=f'Real activations (n={args.n_real})')

    # Random points
    ax.fill_between(angles_deg,
                    np.percentile(rand_per_act, 25, axis=0),
                    np.percentile(rand_per_act, 75, axis=0),
                    color='C1', alpha=0.15)
    ax.plot(angles_deg, np.median(rand_per_act, axis=0),
            color='C1', lw=2.5, label=f'Random points (n={args.n_act})')

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('Cosine distance (at layer {})'.format(args.measure_layer))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, args.max_angle)
    ax.set_title(f'{args.model}: real vs random points, '
                 f'inject@L{args.layer} measure@L{args.measure_layer}')
    plt.tight_layout()

    plot_path = os.path.join(args.results_dir, 'real_vs_random_points.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Numerical summary
    for name, curves in [('Real', real_per_act), ('Random', rand_per_act)]:
        med = np.median(curves, axis=0)
        crossings = np.where(med > 0.1)[0]
        if len(crossings) > 0:
            print(f"{name}: median crosses 0.1 at {angles_deg[crossings[0]]:.1f}°")
        else:
            print(f"{name}: median never crosses 0.1")


if __name__ == '__main__':
    main()
