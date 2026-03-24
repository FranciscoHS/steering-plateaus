#!/usr/bin/env python3
"""Control condition replication: 1 MELBO + (k-1) random directions.

Measures breaking angle vs subspace dimension k for just the control condition,
then fits both log and power-law models.

Usage:
    python scripts/perturbation/sweep_control_replication.py \
        --model llama-3.1-8b --inject_layer 2 --measure_layer 30 \
        --directions results/iterative_melbo_hf/llama-3.1-8b_L2_fixed10deg/results.pkl \
        --outdir results/sweep_control/llama_control \
        --l2_threshold 3 --n_dirs 100 --dtype bfloat16
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
    """Sweep multiple directions, batching across directions for each angle.

    Args:
        dirs: [n_dirs, d_model] already orthogonalized and normalized
    Returns:
        l2_curves: [n_dirs, n_angles]
    """
    n_dirs = dirs.shape[0]
    n_angles = len(angles_deg)
    angles_rad = np.deg2rad(angles_deg)
    cos_t = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)

    l2_curves = np.zeros((n_dirs, n_angles))

    for start in range(0, n_dirs, batch_size):
        end = min(start + batch_size, n_dirs)
        batch_dirs = dirs[start:end]  # [B, d]
        B = batch_dirs.shape[0]

        # perturbed[i, j] = a_norm * (cos(angle_j) * a_unit + sin(angle_j) * dir_i)
        perturbed = a_norm * (
            cos_t[None, :, None] * a_unit[None, None, :]
            + sin_t[None, :, None] * batch_dirs[:, None, :]
        )  # [B, n_angles, d]
        perturbed = perturbed.reshape(B * n_angles, -1)

        with torch.no_grad():
            pert_L = forward_to_L(model, ctx, perturbed, inject_layer, measure_layer)
            l2 = (pert_L - ref_L).norm(dim=-1).cpu().numpy()

        l2_curves[start:end] = l2.reshape(B, n_angles)

    return l2_curves


def sweep_control_subspace(model, ctx, act, ref_L, a_unit, a_norm,
                           melbo_dir, k, d_model, inject_layer, measure_layer,
                           device, angles_deg, n_dirs=100, batch_size=32):
    """Build 1 MELBO + (k-1) random subspace, sample n_dirs from it, sweep."""
    if k == 1:
        # Just the single MELBO direction
        v = melbo_dir.unsqueeze(0)  # [1, d]
        v = v - (v @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        v = F.normalize(v, dim=1)
        # Repeat for n_dirs (all identical for k=1)
        vecs = v.expand(n_dirs, -1)
    else:
        # Build basis: 1st MELBO + (k-1) random
        random_dirs = torch.randn(k - 1, d_model, device=device)
        random_dirs = F.normalize(random_dirs, dim=1)
        basis = torch.cat([melbo_dir.unsqueeze(0), random_dirs], dim=0)  # [k, d]

        # Orthogonalize to activation
        basis_orth = basis - (basis @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        basis_orth = F.normalize(basis_orth, dim=1)

        # QR for orthonormal basis of subspace
        Q, _ = torch.linalg.qr(basis_orth.T, mode='reduced')  # [d, rank]

        # Sample random directions from subspace
        coeffs = torch.randn(n_dirs, Q.shape[1], device=device)
        vecs = coeffs @ Q.T  # [n_dirs, d]
        vecs = F.normalize(vecs, dim=1)

        # Re-orthogonalize to activation
        vecs = vecs - (vecs @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        vecs = F.normalize(vecs, dim=1)

    l2_curves = sweep_dirs_batched(model, ctx, act, ref_L, a_unit, a_norm,
                                   vecs, inject_layer, measure_layer, device,
                                   angles_deg, batch_size=batch_size)
    return l2_curves


def main():
    parser = argparse.ArgumentParser(
        description='Control condition: 1 MELBO + (k-1) random directions')
    parser.add_argument('--model', required=True)
    parser.add_argument('--inject_layer', type=int, required=True)
    parser.add_argument('--measure_layer', type=int, required=True)
    parser.add_argument('--directions', required=True,
                        help='Path to .pkl with MELBO directions')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--l2_threshold', type=float, required=True,
                        help='Single L2 threshold for breaking angle')
    parser.add_argument('--n_dirs', type=int, default=100,
                        help='Random directions per subspace')
    parser.add_argument('--k_values', type=int, nargs='+',
                        default=[1, 10, 25, 50, 100, 200, 300, 500, 750, 1000])
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load directions — only need the first one
    print(f"Loading directions from {args.directions}")
    with open(args.directions, 'rb') as f:
        dir_data = pickle.load(f)
    if isinstance(dir_data['directions'], list):
        all_dirs = torch.stack(dir_data['directions']).float()
    else:
        all_dirs = dir_data['directions'].float()
    n_features, d_model = all_dirs.shape
    first_dir = F.normalize(all_dirs[0:1], dim=1).squeeze(0)  # [d_model]
    print(f"{n_features} total MELBO directions, using 1st, d_model={d_model}")

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
    melbo_dir = first_dir.to(device)

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    # Sweep control condition for each k
    all_breaks = {}  # k -> list of breaking angles
    all_l2_curves = {}  # k -> [n_dirs, n_angles]

    for k in args.k_values:
        print(f"\nk={k}...")
        l2_curves = sweep_control_subspace(
            model, ctx, act, ref_L, a_unit, a_norm,
            melbo_dir, k, d_model, args.inject_layer, measure_layer,
            device, angles_deg, n_dirs=args.n_dirs, batch_size=args.batch_size)

        breaks = [crossing_angle(l2_curves[d], angles_deg, args.l2_threshold)
                  or args.max_angle for d in range(args.n_dirs)]
        all_breaks[k] = breaks
        all_l2_curves[k] = l2_curves

        med = np.median(breaks)
        q25, q75 = np.percentile(breaks, [25, 75])
        print(f"  median={med:.2f}°, IQR=[{q25:.2f}, {q75:.2f}] ({time.time()-t0:.0f}s)")

    # Save
    save_data = {
        'k_values': args.k_values,
        'l2_threshold': args.l2_threshold,
        'angles_deg': angles_deg,
        'n_dirs': args.n_dirs,
        'all_breaks': all_breaks,
        'all_l2_curves': all_l2_curves,
        'args': vars(args),
    }
    pkl_path = os.path.join(args.outdir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved: {pkl_path}")

    # Fit and plot
    ks = np.array(args.k_values, dtype=float)
    meds = np.array([np.median(all_breaks[k]) for k in args.k_values])
    q25s = np.array([np.percentile(all_breaks[k], 25) for k in args.k_values])
    q75s = np.array([np.percentile(all_breaks[k], 75) for k in args.k_values])

    log_k = np.log(ks)
    log_m = np.log(meds)

    # Power law fit: log(y) = log_a + b * log(k)
    b_pow, log_a_pow = np.polyfit(log_k, log_m, 1)
    a_pow = np.exp(log_a_pow)
    pred_pow = log_a_pow + b_pow * log_k
    ss_res_pow = np.sum((log_m - pred_pow)**2)
    ss_tot = np.sum((log_m - log_m.mean())**2)
    r2_pow = 1 - ss_res_pow / ss_tot if ss_tot > 0 else 0

    # Log fit: y = a * log(k) + b
    a_log, b_log = np.polyfit(log_k, meds, 1)
    pred_log = a_log * log_k + b_log
    ss_res_log = np.sum((meds - pred_log)**2)
    ss_tot_lin = np.sum((meds - meds.mean())**2)
    r2_log = 1 - ss_res_log / ss_tot_lin if ss_tot_lin > 0 else 0

    print(f"\nFits (L2 threshold = {args.l2_threshold}):")
    print(f"  Power law: {a_pow:.3f} * k^{b_pow:.3f} (R²={r2_pow:.4f})")
    print(f"  Log:       {a_log:.3f} * log(k) + {b_log:.3f} (R²={r2_log:.4f})")

    # Plot 1: log-log with power law
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.errorbar(ks, meds, yerr=[meds - q25s, q75s - meds],
                fmt='none', ecolor='#EE6677', capsize=3, alpha=0.6)
    ax.scatter(ks, meds, s=50, color='#EE6677', marker='s', zorder=5)
    k_smooth = np.logspace(np.log10(ks.min()*0.8), np.log10(ks.max()*1.1), 200)
    ax.plot(k_smooth, a_pow * k_smooth**b_pow, '#EE6677', lw=2,
            label=f'Power law: k^{{{b_pow:.3f}}} (R²={r2_pow:.3f})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Subspace dimension k', fontsize=11)
    ax.set_ylabel('Breaking angle (degrees)', fontsize=11)
    ax.set_title('Log-log (power law fit)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: semilog-x with log fit
    ax = axes[1]
    ax.errorbar(ks, meds, yerr=[meds - q25s, q75s - meds],
                fmt='none', ecolor='#EE6677', capsize=3, alpha=0.6)
    ax.scatter(ks, meds, s=50, color='#EE6677', marker='s', zorder=5)
    ax.plot(k_smooth, a_log * np.log(k_smooth) + b_log, '#EE6677', lw=2,
            label=f'Log fit: {a_log:.2f}·log(k) + {b_log:.2f} (R²={r2_log:.3f})')
    ax.set_xscale('log')
    ax.set_xlabel('Subspace dimension k', fontsize=11)
    ax.set_ylabel('Breaking angle (degrees)', fontsize=11)
    ax.set_title('Semilog-x (log fit)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'{args.model}: 1 MELBO + (k-1) random (L2 thresh={args.l2_threshold})\n'
                 f'{args.n_dirs} dirs/subspace, 1 activation',
                 fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, 'control_fits.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
