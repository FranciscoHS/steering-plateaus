#!/usr/bin/env python3
"""Pointwise test of collective threshold model.

For a subspace spanned by k MELBO directions with known individual breaking
angles θ_i, predict the breaking angle of random directions u from:

    α_pred = 1 / (Σ_i |u·d_i|^p / θ_i^p)^(1/p)

This has no free parameters beyond p. Test by scatter-plotting predicted
vs actual breaking angles for many random directions.

Usage:
    python scripts/perturbation/sweep_collective_threshold.py \
        --model qwen3-1.7b --inject_layer 7 --measure_layer -2 \
        --directions results/iterative_melbo_hf/qwen3-1.7b_L7_meas-2_escalation_v6/checkpoint_round3_mid.pkl \
        --outdir results/sweep_control/collective_threshold_k100 \
        --k 100 --n_random 500 --l2_threshold 100
"""
import argparse, os, sys, pickle, time
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.melbo.run_iterative_melbo_hf import (
    extract_context_and_activation, forward_to_L,
)


def crossing_angle(curve, angles_deg, threshold):
    """Find angle where curve first crosses threshold (linear interpolation)."""
    crossings = np.where(curve > threshold)[0]
    if len(crossings) > 0:
        idx = crossings[0]
        if idx > 0:
            y0, y1 = curve[idx-1], curve[idx]
            x0, x1 = angles_deg[idx-1], angles_deg[idx]
            if y1 > y0:
                return x0 + (threshold - y0) / (y1 - y0) * (x1 - x0)
        return angles_deg[idx]
    return None


def sweep_dirs_batched(model, ctx, act, ref_L, a_unit, a_norm,
                       dirs, inject_layer, measure_layer, device,
                       angles_deg, batch_size=32):
    """Sweep multiple directions, batching across directions."""
    n_dirs = dirs.shape[0]
    n_angles = len(angles_deg)
    angles_rad = np.deg2rad(angles_deg)
    cos_t = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)

    l2_curves = np.zeros((n_dirs, n_angles))

    for start in range(0, n_dirs, batch_size):
        end = min(start + batch_size, n_dirs)
        batch_dirs = dirs[start:end]
        B = batch_dirs.shape[0]

        perturbed = a_norm * (
            cos_t[None, :, None] * a_unit[None, None, :]
            + sin_t[None, :, None] * batch_dirs[:, None, :]
        )
        perturbed = perturbed.reshape(B * n_angles, -1)

        with torch.no_grad():
            pert_L = forward_to_L(model, ctx, perturbed, inject_layer, measure_layer)
            l2 = (pert_L - ref_L).norm(dim=-1).cpu().numpy()

        l2_curves[start:end] = l2.reshape(B, n_angles)

    return l2_curves


def main():
    parser = argparse.ArgumentParser(
        description='Pointwise test of collective threshold model')
    parser.add_argument('--model', required=True)
    parser.add_argument('--inject_layer', type=int, required=True)
    parser.add_argument('--measure_layer', type=int, required=True)
    parser.add_argument('--directions', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--k', type=int, default=100,
                        help='Subspace dimension (total basis directions)')
    parser.add_argument('--n_melbo', type=int, default=None,
                        help='Number of MELBO directions in basis (default: all k). '
                             'Remaining k - n_melbo are random.')
    parser.add_argument('--random_select', action='store_true',
                        help='Randomly select MELBO directions from pool instead of first k')
    parser.add_argument('--n_random', type=int, default=500,
                        help='Number of random directions to test')
    parser.add_argument('--l2_threshold', type=float, default=100)
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load directions
    print(f"Loading directions from {args.directions}")
    with open(args.directions, 'rb') as f:
        dir_data = pickle.load(f)
    if isinstance(dir_data['directions'], list):
        all_dirs = torch.stack(dir_data['directions']).float()
    else:
        all_dirs = dir_data['directions'].float()
    all_dirs = F.normalize(all_dirs, dim=1)
    d_model = all_dirs.shape[1]
    n_melbo = args.n_melbo if args.n_melbo is not None else args.k
    n_melbo = min(n_melbo, args.k)

    n_available = all_dirs.shape[0]
    if n_melbo == args.k and not args.random_select:
        basis_dirs = all_dirs[:args.k]
        print(f"Using first {args.k} MELBO directions, d_model={d_model}")
    elif n_melbo == args.k and args.random_select:
        melbo_idx = np.random.choice(n_available, size=n_melbo, replace=False)
        basis_dirs = all_dirs[melbo_idx]
        print(f"Using {args.k} randomly-selected MELBO directions (from {n_available}), d_model={d_model}")
    else:
        # Randomly select n_melbo MELBO directions from the pool
        melbo_idx = np.random.choice(n_available, size=n_melbo, replace=False)
        melbo_dirs = all_dirs[melbo_idx]
        n_rand_basis = args.k - n_melbo
        rand_basis = torch.randn(n_rand_basis, d_model)
        rand_basis = F.normalize(rand_basis, dim=1)
        basis_dirs = torch.cat([melbo_dirs, rand_basis], dim=0)
        print(f"Using {n_melbo} random MELBO (from {n_available}) + {n_rand_basis} random basis directions, d_model={d_model}")

    # Load model
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)
    n_layers = len(_get_blocks(model))
    measure_layer = args.measure_layer
    if measure_layer < 0:
        measure_layer = n_layers + measure_layer
    print(f"Layer {args.inject_layer} -> measure {measure_layer}")

    # Get activation
    token_ids_list = load_fineweb_fixed_length(100, tokenizer, seq_len=5, seed=args.seed)
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.inject_layer, device)

    ctx = all_contexts[0].unsqueeze(0).to(device)
    act = all_activations[0].to(device)
    a_norm = act.norm()
    a_unit = act / a_norm

    with torch.no_grad():
        ref_L = forward_to_L(model, ctx, act.unsqueeze(0),
                             args.inject_layer, measure_layer)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    dirs_dev = basis_dirs.to(device)

    # Orthogonalize basis directions to activation
    dirs_orth = dirs_dev - (dirs_dev @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
    dirs_orth = F.normalize(dirs_orth, dim=1)

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    # Step 1: Sweep each basis direction individually
    print(f"\nSweeping {args.k} basis directions...")
    basis_l2 = sweep_dirs_batched(model, ctx, act, ref_L, a_unit, a_norm,
                                   dirs_orth, args.inject_layer, measure_layer,
                                   device, angles_deg, batch_size=args.batch_size)
    basis_breaks = np.array([crossing_angle(basis_l2[i], angles_deg, args.l2_threshold)
                             or args.max_angle for i in range(args.k)])
    print(f"  Basis breaking angles: min={basis_breaks.min():.2f}, "
          f"median={np.median(basis_breaks):.2f}, max={basis_breaks.max():.2f}")
    print(f"  Done ({time.time()-t0:.0f}s)")

    # Step 2: Draw random directions from the subspace
    print(f"\nDrawing {args.n_random} random directions from {args.k}-d subspace...")
    # QR for orthonormal basis of the subspace
    Q, _ = torch.linalg.qr(dirs_orth.T.cpu(), mode='reduced')  # [d_model, rank]
    Q = Q.to(device)
    rank = Q.shape[1]
    print(f"  Subspace rank: {rank}")

    # Random directions in subspace
    coeffs = torch.randn(args.n_random, rank, device=device)
    random_dirs = coeffs @ Q.T  # [n_random, d_model]
    random_dirs = F.normalize(random_dirs, dim=1)

    # Re-orthogonalize to activation
    random_dirs = random_dirs - (random_dirs @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
    random_dirs = F.normalize(random_dirs, dim=1)

    # Compute projections |u · d_i| for each random dir onto each basis dir
    # dirs_orth: [k, d], random_dirs: [n_random, d]
    projections = torch.abs(random_dirs @ dirs_orth.T).cpu().numpy()  # [n_random, k]
    print(f"  Projections shape: {projections.shape}")

    # Step 3: Sweep random directions
    print(f"Sweeping {args.n_random} random directions...")
    random_l2 = sweep_dirs_batched(model, ctx, act, ref_L, a_unit, a_norm,
                                    random_dirs, args.inject_layer, measure_layer,
                                    device, angles_deg, batch_size=args.batch_size)
    random_breaks = np.array([crossing_angle(random_l2[i], angles_deg, args.l2_threshold)
                              or args.max_angle for i in range(args.n_random)])
    print(f"  Random breaking angles: min={random_breaks.min():.2f}, "
          f"median={np.median(random_breaks):.2f}, max={random_breaks.max():.2f}")
    print(f"  Done ({time.time()-t0:.0f}s)")

    # Step 4: Predict breaking angles for various p values
    p_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    predictions = {}
    r2_values = {}

    for p in p_values:
        # α_pred = 1 / (Σ_i |u·d_i|^p / θ_i^p)^(1/p)
        # projections: [n_random, k], basis_breaks: [k]
        terms = (projections ** p) / (basis_breaks[None, :] ** p)  # [n_random, k]
        sum_terms = terms.sum(axis=1)  # [n_random]
        alpha_pred = 1.0 / (sum_terms ** (1.0 / p))

        predictions[p] = alpha_pred

        # R² (on non-capped values)
        valid = (random_breaks < args.max_angle) & np.isfinite(alpha_pred)
        if valid.sum() > 2:
            actual = random_breaks[valid]
            pred = alpha_pred[valid]
            ss_res = np.sum((actual - pred) ** 2)
            ss_tot = np.sum((actual - actual.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        else:
            r2 = float('nan')
        r2_values[p] = r2
        print(f"  p={p:.1f}: R² = {r2:.4f}, median pred = {np.median(alpha_pred):.2f}")

    # Save
    save_data = {
        'k': args.k,
        'n_random': args.n_random,
        'l2_threshold': args.l2_threshold,
        'angles_deg': angles_deg,
        'basis_breaks': basis_breaks,
        'random_breaks': random_breaks,
        'projections': projections,
        'predictions': predictions,
        'r2_values': r2_values,
        'p_values': p_values,
        'args': vars(args),
    }
    pkl_path = os.path.join(args.outdir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved: {pkl_path}")

    # Plot: scatter for each p
    n_p = len(p_values)
    n_cols = 4
    n_rows = (n_p + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    valid = random_breaks < args.max_angle

    for pi, p in enumerate(p_values):
        ax = axes[pi]
        pred = predictions[p]
        mask = valid & np.isfinite(pred)

        ax.scatter(pred[mask], random_breaks[mask], s=8, alpha=0.3, color='#EE6677')

        # Identity line
        lo = min(pred[mask].min(), random_breaks[mask].min())
        hi = max(pred[mask].max(), random_breaks[mask].max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5)

        ax.set_xlabel('Predicted breaking angle (°)', fontsize=10)
        ax.set_ylabel('Actual breaking angle (°)', fontsize=10)
        ax.set_title(f'p = {p:.1f}  (R² = {r2_values[p]:.3f})', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    for pi in range(len(p_values), len(axes)):
        axes[pi].set_visible(False)

    plt.suptitle(f'{args.model}: collective threshold model, k={args.k}\n'
                 f'L2 thresh={args.l2_threshold}, {args.n_random} random dirs',
                 fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, 'scatter_by_p.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # R² vs p plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(p_values, [r2_values[p] for p in p_values], 'o-', color='#228833', lw=2, ms=8)
    ax.set_xlabel('p', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title(f'Collective threshold model: R² vs p (k={args.k})', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    r2_path = os.path.join(args.outdir, 'r2_vs_p.png')
    plt.savefig(r2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {r2_path}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
