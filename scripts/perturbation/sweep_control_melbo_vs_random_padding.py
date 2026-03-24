#!/usr/bin/env python3
"""Control experiment: does subspace scaling require multiple sensitive directions?

Compares two conditions for breaking angle vs subspace dimension k:
  (a) Ordered MELBO: span of first k MELBO directions (all sensitive)
  (b) Control: span of 1st MELBO direction + (k-1) random directions

If the sqrt(k) scaling comes from spreading energy across multiple sensitive
directions, (a) should show scaling but (b) should be flat — the random
directions don't contribute sensitivity, so the breaking angle is set by
the single MELBO direction's ~1/sqrt(k) share of the perturbation energy.

If instead it's purely geometric dilution (the sensitive direction gets
less energy regardless of what the other directions are), both should scale
similarly.

Usage:
    python scripts/perturbation/sweep_control_melbo_vs_random_padding.py \
        --model qwen3-1.7b --inject_layer 7 --measure_layer 26 \
        --directions results/iterative_melbo_hf/qwen3-1.7b_L7_meas-2_escalation_v6/checkpoint_round3_mid.pkl \
        --outdir results/sweep_control/qwen_melbo_vs_random_padding \
        --k_values 10 100 500 1000 1500 2000 3000 4000
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
    extract_context_and_activation, forward_to_L, sweep_dir,
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


def sweep_subspace(model, ctx, act, ref_L, a_unit, a_norm, basis_dirs,
                   inject_layer, measure_layer, device, angles_deg, n_dirs=10):
    """Sample random directions from a subspace and sweep each one.

    Args:
        basis_dirs: [k, d_model] directions spanning the subspace (on device)
        n_dirs: number of random directions to sample from the subspace

    Returns:
        cos_curves: [n_dirs, n_angles]
        l2_curves: [n_dirs, n_angles]
    """
    # Orthogonalize basis to activation, then QR for orthonormal basis
    dirs_orth = basis_dirs - (basis_dirs @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
    dirs_orth = F.normalize(dirs_orth, dim=1)
    Q, _ = torch.linalg.qr(dirs_orth.T, mode='reduced')
    rank = Q.shape[1]

    cos_curves = np.zeros((n_dirs, len(angles_deg)))
    l2_curves = np.zeros((n_dirs, len(angles_deg)))

    for d in range(n_dirs):
        coeffs = torch.randn(rank, device=device)
        v = Q @ coeffs
        v = F.normalize(v, dim=0)
        cos_c, l2_c = sweep_dir(model, ctx, act, ref_L, a_unit, a_norm,
                                v, inject_layer, measure_layer, device, angles_deg)
        cos_curves[d] = cos_c
        l2_curves[d] = l2_c

    return cos_curves, l2_curves


def main():
    parser = argparse.ArgumentParser(
        description='Control: ordered MELBO subspace vs 1 MELBO + random padding')
    parser.add_argument('--model', required=True)
    parser.add_argument('--inject_layer', type=int, required=True)
    parser.add_argument('--measure_layer', type=int, required=True)
    parser.add_argument('--directions', required=True,
                        help='Path to .pkl with MELBO directions')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--n_activations', type=int, default=10,
                        help='Average over this many activations')
    parser.add_argument('--n_dirs_per_k', type=int, default=10,
                        help='Random directions per subspace per activation')
    parser.add_argument('--k_values', type=int, nargs='+',
                        default=[10, 100, 500, 1000, 1500, 2000, 3000, 4000])
    parser.add_argument('--l2_thresholds', type=float, nargs='+',
                        default=[100, 200, 400, 600])
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=120)
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
    n_features, d_model = all_dirs.shape
    all_dirs = F.normalize(all_dirs, dim=1)
    print(f"{n_features} MELBO directions, d_model={d_model}")

    k_values = [k for k in args.k_values if k <= n_features]
    print(f"k_values: {k_values}")

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

    token_ids_list = load_fineweb_fixed_length(100, tokenizer, seq_len=5, seed=args.seed)
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.inject_layer, device)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    dirs_dev = all_dirs.to(device)

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    # Results storage: for each condition and k, store averaged L2 curves
    ordered_curves = {}  # k -> averaged [n_dirs, n_angles] L2 curves
    control_curves = {}  # k -> averaged [n_dirs, n_angles] L2 curves

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"k = {k}")
        print(f"{'='*60}")

        ordered_avg = np.zeros((args.n_dirs_per_k, len(angles_deg)))
        control_avg = np.zeros((args.n_dirs_per_k, len(angles_deg)))

        for ai in range(args.n_activations):
            ctx = all_contexts[ai].unsqueeze(0).to(device)
            act = all_activations[ai].to(device)
            a_norm = act.norm()
            a_unit = act / a_norm

            with torch.no_grad():
                ref_L = forward_to_L(model, ctx, act.unsqueeze(0),
                                     args.inject_layer, measure_layer)

            # (a) Ordered: first k MELBO directions
            _, l2_ordered = sweep_subspace(
                model, ctx, act, ref_L, a_unit, a_norm,
                dirs_dev[:k], args.inject_layer, measure_layer,
                device, angles_deg, n_dirs=args.n_dirs_per_k)
            ordered_avg += l2_ordered / args.n_activations

            # (b) Control: 1st MELBO + (k-1) random directions
            random_dirs = torch.randn(k - 1, d_model, device=device)
            random_dirs = F.normalize(random_dirs, dim=1)
            control_basis = torch.cat([dirs_dev[:1], random_dirs], dim=0)
            _, l2_control = sweep_subspace(
                model, ctx, act, ref_L, a_unit, a_norm,
                control_basis, args.inject_layer, measure_layer,
                device, angles_deg, n_dirs=args.n_dirs_per_k)
            control_avg += l2_control / args.n_activations

            print(f"  Activation {ai+1}/{args.n_activations} ({time.time()-t0:.0f}s)")

        ordered_curves[k] = ordered_avg
        control_curves[k] = control_avg

        # Quick summary
        for thresh in args.l2_thresholds[:1]:
            ord_breaks = [crossing_angle(ordered_avg[d], angles_deg, thresh)
                          or args.max_angle for d in range(args.n_dirs_per_k)]
            ctrl_breaks = [crossing_angle(control_avg[d], angles_deg, thresh)
                           or args.max_angle for d in range(args.n_dirs_per_k)]
            print(f"  L2>{thresh}: ordered median={np.median(ord_breaks):.1f}°, "
                  f"control median={np.median(ctrl_breaks):.1f}°")

    # Compute breaking angles for all thresholds
    ordered_breaks = {}  # thresh -> {k -> list of breaking angles}
    control_breaks = {}
    for thresh in args.l2_thresholds:
        ordered_breaks[thresh] = {}
        control_breaks[thresh] = {}
        for k in k_values:
            ordered_breaks[thresh][k] = [
                crossing_angle(ordered_curves[k][d], angles_deg, thresh)
                or args.max_angle for d in range(args.n_dirs_per_k)]
            control_breaks[thresh][k] = [
                crossing_angle(control_curves[k][d], angles_deg, thresh)
                or args.max_angle for d in range(args.n_dirs_per_k)]

    # Save
    save_data = {
        'k_values': k_values,
        'l2_thresholds': args.l2_thresholds,
        'angles_deg': angles_deg,
        'n_features': n_features,
        'n_activations': args.n_activations,
        'n_dirs_per_k': args.n_dirs_per_k,
        'ordered_curves': ordered_curves,
        'control_curves': control_curves,
        'ordered_breaks': ordered_breaks,
        'control_breaks': control_breaks,
        'args': vars(args),
    }
    pkl_path = os.path.join(args.outdir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved: {pkl_path}")

    # Plot: breaking angle vs k for both conditions, one panel per threshold
    n_thresh = len(args.l2_thresholds)
    fig, axes = plt.subplots(1, n_thresh, figsize=(7 * n_thresh, 5))
    if n_thresh == 1:
        axes = [axes]
    for ti, thresh in enumerate(args.l2_thresholds):
        ax = axes[ti]

        # Ordered MELBO
        ord_meds = [np.median(ordered_breaks[thresh][k]) for k in k_values]
        ord_q25 = [np.percentile(ordered_breaks[thresh][k], 25) for k in k_values]
        ord_q75 = [np.percentile(ordered_breaks[thresh][k], 75) for k in k_values]

        # Control
        ctrl_meds = [np.median(control_breaks[thresh][k]) for k in k_values]
        ctrl_q25 = [np.percentile(control_breaks[thresh][k], 25) for k in k_values]
        ctrl_q75 = [np.percentile(control_breaks[thresh][k], 75) for k in k_values]

        ks = np.array(k_values, dtype=float)

        # Power law fits with error bars
        for label, meds, q25, q75, color, marker in [
            ('Ordered MELBO', ord_meds, ord_q25, ord_q75, '#228833', 'o'),
            ('1 MELBO + random', ctrl_meds, ctrl_q25, ctrl_q75, '#EE6677', 's'),
        ]:
            meds_arr = np.array(meds)
            q25_arr = np.array(q25)
            q75_arr = np.array(q75)
            log_k = np.log(ks)
            log_m = np.log(meds_arr)
            b_exp, log_a = np.polyfit(log_k, log_m, 1)
            a_coeff = np.exp(log_a)
            pred = log_a + b_exp * log_k
            ss_res = np.sum((log_m - pred)**2)
            ss_tot = np.sum((log_m - log_m.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            ax.errorbar(ks, meds_arr,
                        yerr=[meds_arr - q25_arr, q75_arr - meds_arr],
                        fmt='none', ecolor=color, capsize=3, alpha=0.6)
            ax.scatter(ks, meds_arr, s=50, color=color, marker=marker, zorder=5)
            k_smooth = np.linspace(ks.min() * 0.8, ks.max() * 1.1, 200)
            ax.plot(k_smooth, a_coeff * k_smooth**b_exp, color=color, lw=2,
                    label=f'{label}: k^{{{b_exp:.2f}}} (R²={r2:.2f})')

        ax.set_xlabel('Subspace dimension k', fontsize=11)
        ax.set_ylabel('Breaking angle (degrees)', fontsize=11)
        ax.set_title(f'L2 threshold = {thresh}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')

    plt.suptitle(f'{args.model}: ordered MELBO vs 1 MELBO + random padding\n'
                 f'{args.n_activations} activations, {args.n_dirs_per_k} dirs/k',
                 fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, 'control_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: power law exponents (breaking angle ~ k^exponent)")
    print(f"{'='*60}")
    for thresh in args.l2_thresholds:
        ord_meds = np.array([np.median(ordered_breaks[thresh][k]) for k in k_values])
        ctrl_meds = np.array([np.median(control_breaks[thresh][k]) for k in k_values])
        log_k = np.log(np.array(k_values, dtype=float))
        b_ord, _ = np.polyfit(log_k, np.log(ord_meds), 1)
        b_ctrl, _ = np.polyfit(log_k, np.log(ctrl_meds), 1)
        print(f"  L2>{thresh}: ordered={b_ord:.2f}, control={b_ctrl:.2f}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
