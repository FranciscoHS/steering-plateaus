#!/usr/bin/env python3
"""Three-condition control experiment for subspace scaling.

Compares breaking angle vs subspace dimension k for three conditions:
  (a) Ordered MELBO: span of first k MELBO directions
  (b) Random MELBO subsets: span of k randomly-chosen MELBO directions (normalized
      by min individual breaking angle)
  (c) Control: span of 1st MELBO direction + (k-1) random directions

Usage:
    python scripts/perturbation/sweep_3condition_control.py \
        --model qwen3-1.7b --inject_layer 7 --measure_layer -2 \
        --directions results/iterative_melbo_hf/qwen3-1.7b_L7_meas-2_escalation_v6/checkpoint_round3_mid.pkl \
        --outdir results/sweep_control/qwen_3condition_v3 \
        --n_activations 1 --n_dirs_per_k 50 \
        --k_values 1 10 25 50 100 200 300 500 750 1000 1250 1500 1750 2000 2048 \
        --l2_thresholds 100 \
        --indiv_checkpoint results/sweep_control/qwen_3condition_v2/indiv_checkpoint.pkl
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

N_POOL = 2048  # Only use first 2048 MELBO directions


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
                       angles_deg, batch_size=64):
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


def sweep_subspace(model, ctx, act, ref_L, a_unit, a_norm, basis_dirs,
                   inject_layer, measure_layer, device, angles_deg,
                   n_dirs=10, batch_size=64):
    """Sample random directions from a subspace and sweep each one."""
    dirs_orth = basis_dirs - (basis_dirs @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
    dirs_orth = F.normalize(dirs_orth, dim=1)
    Q, _ = torch.linalg.qr(dirs_orth.T, mode='reduced')
    rank = Q.shape[1]

    # Sample random directions in the subspace
    coeffs = torch.randn(n_dirs, rank, device=device)
    vecs = coeffs @ Q.T  # [n_dirs, d]
    vecs = F.normalize(vecs, dim=1)

    # Re-orthogonalize to activation (should already be, but be safe)
    vecs = vecs - (vecs @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
    vecs = F.normalize(vecs, dim=1)

    l2_curves = sweep_dirs_batched(model, ctx, act, ref_L, a_unit, a_norm,
                                   vecs, inject_layer, measure_layer, device,
                                   angles_deg, batch_size=batch_size)
    return l2_curves


def main():
    parser = argparse.ArgumentParser(
        description='3-condition control: ordered vs random-subset vs 1+random')
    parser.add_argument('--model', required=True)
    parser.add_argument('--inject_layer', type=int, required=True)
    parser.add_argument('--measure_layer', type=int, required=True)
    parser.add_argument('--directions', required=True,
                        help='Path to .pkl with MELBO directions')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--indiv_checkpoint', type=str, default=None,
                        help='Path to cached indiv_checkpoint.pkl to skip individual sweeps')
    parser.add_argument('--n_activations', type=int, default=1)
    parser.add_argument('--n_dirs_per_k', type=int, default=50,
                        help='Random directions per subspace')
    parser.add_argument('--n_random_subsets', type=int, default=20,
                        help='Number of random MELBO subsets per k')
    parser.add_argument('--n_control_subspaces', type=int, default=1,
                        help='Number of independent subspace draws for control condition')
    parser.add_argument('--k_values', type=int, nargs='+',
                        default=[1, 10, 25, 50, 100, 200, 300, 500, 750,
                                 1000, 1250, 1500, 1750, 2000, 2048])
    parser.add_argument('--l2_thresholds', type=float, nargs='+',
                        default=[100])
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for direction sweeps')
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
    n_features_total = all_dirs.shape[0]
    d_model = all_dirs.shape[1]
    all_dirs = F.normalize(all_dirs, dim=1)

    n_pool = min(N_POOL, n_features_total)
    all_dirs = all_dirs[:n_pool]
    print(f"{n_features_total} total MELBO directions, using first {n_pool}, d_model={d_model}")

    k_values = [k for k in args.k_values if k <= n_pool]
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

    # Step 1: Individual direction sweeps (load cached or compute)
    if args.indiv_checkpoint and os.path.exists(args.indiv_checkpoint):
        print(f"\nLoading cached individual breaks from {args.indiv_checkpoint}")
        with open(args.indiv_checkpoint, 'rb') as f:
            indiv_data = pickle.load(f)
        indiv_breaks = indiv_data['indiv_breaks']
        # Check thresholds match — recompute any missing from cached curves
        for thresh in args.l2_thresholds:
            if thresh not in indiv_breaks:
                print(f"  Threshold {thresh} not in cache, recomputing from curves...")
                indiv_l2_curves = indiv_data['indiv_l2_curves']
                cached_angles = indiv_data.get('angles_deg', angles_deg)
                breaks = np.array([crossing_angle(indiv_l2_curves[i], cached_angles, thresh)
                                   or args.max_angle for i in range(indiv_l2_curves.shape[0])])
                indiv_breaks[thresh] = breaks
            print(f"  L2>{thresh}: median={np.median(indiv_breaks[thresh]):.2f}, "
                  f"min={indiv_breaks[thresh].min():.2f}, max={indiv_breaks[thresh].max():.2f}")
        print(f"Loaded ({time.time()-t0:.0f}s)")
    else:
        print(f"\nSweeping {n_pool} individual directions over {args.n_activations} activations (batched)...")
        indiv_l2_curves = np.zeros((n_pool, len(angles_deg)))

        for ai in range(args.n_activations):
            ctx = all_contexts[ai].unsqueeze(0).to(device)
            act = all_activations[ai].to(device)
            a_norm = act.norm()
            a_unit = act / a_norm

            with torch.no_grad():
                ref_L = forward_to_L(model, ctx, act.unsqueeze(0),
                                     args.inject_layer, measure_layer)

            dirs_orth = dirs_dev - (dirs_dev @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
            dirs_orth = F.normalize(dirs_orth, dim=1)

            l2_batch = sweep_dirs_batched(model, ctx, act, ref_L, a_unit, a_norm,
                                          dirs_orth, args.inject_layer, measure_layer,
                                          device, angles_deg, batch_size=args.batch_size)
            indiv_l2_curves += l2_batch / args.n_activations

            print(f"  Activation {ai+1}/{args.n_activations} ({time.time()-t0:.0f}s)")

        indiv_breaks = {}
        for thresh in args.l2_thresholds:
            breaks = np.array([crossing_angle(indiv_l2_curves[i], angles_deg, thresh) or args.max_angle
                              for i in range(n_pool)])
            indiv_breaks[thresh] = breaks
            print(f"  L2>{thresh}: median={np.median(breaks):.2f}, min={breaks.min():.2f}, max={breaks.max():.2f}")
        print(f"Individual sweeps done ({time.time()-t0:.0f}s)")

        with open(os.path.join(args.outdir, 'indiv_checkpoint.pkl'), 'wb') as f:
            pickle.dump({'indiv_l2_curves': indiv_l2_curves, 'indiv_breaks': indiv_breaks,
                          'angles_deg': angles_deg, 'n_pool': n_pool}, f)

    # Step 2: Three conditions
    ordered_breaks = {t: {} for t in args.l2_thresholds}
    control_breaks = {t: {} for t in args.l2_thresholds}
    random_subset_breaks = {t: {} for t in args.l2_thresholds}

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"k = {k}")
        print(f"{'='*60}")

        for ai in range(args.n_activations):
            ctx = all_contexts[ai].unsqueeze(0).to(device)
            act = all_activations[ai].to(device)
            a_norm = act.norm()
            a_unit = act / a_norm

            with torch.no_grad():
                ref_L = forward_to_L(model, ctx, act.unsqueeze(0),
                                     args.inject_layer, measure_layer)

            # (a) Ordered: first k MELBO directions
            l2_ordered = sweep_subspace(
                model, ctx, act, ref_L, a_unit, a_norm,
                dirs_dev[:k], args.inject_layer, measure_layer,
                device, angles_deg, n_dirs=args.n_dirs_per_k,
                batch_size=args.batch_size)

            for thresh in args.l2_thresholds:
                breaks = [crossing_angle(l2_ordered[d], angles_deg, thresh)
                          or args.max_angle for d in range(args.n_dirs_per_k)]
                ordered_breaks[thresh].setdefault(k, []).extend(breaks)

            # (c) Control: 1st MELBO + (k-1) random directions
            #     Multiple subspace draws to reduce noise
            if k == 1:
                # k=1: control is identical to ordered (just the 1st MELBO dir)
                for thresh in args.l2_thresholds:
                    control_breaks[thresh].setdefault(k, []).extend(
                        ordered_breaks[thresh][k])
            else:
                for si in range(args.n_control_subspaces):
                    random_dirs = torch.randn(k - 1, d_model, device=device)
                    random_dirs = F.normalize(random_dirs, dim=1)
                    control_basis = torch.cat([dirs_dev[:1], random_dirs], dim=0)
                    l2_control = sweep_subspace(
                        model, ctx, act, ref_L, a_unit, a_norm,
                        control_basis, args.inject_layer, measure_layer,
                        device, angles_deg, n_dirs=args.n_dirs_per_k,
                        batch_size=args.batch_size)

                    for thresh in args.l2_thresholds:
                        breaks = [crossing_angle(l2_control[d], angles_deg, thresh)
                                  or args.max_angle for d in range(args.n_dirs_per_k)]
                        control_breaks[thresh].setdefault(k, []).extend(breaks)

        # (b) Random MELBO subsets
        for s in range(args.n_random_subsets):
            subset_idx = np.random.choice(n_pool, size=k, replace=False)

            for ai in range(args.n_activations):
                ctx = all_contexts[ai].unsqueeze(0).to(device)
                act = all_activations[ai].to(device)
                a_norm = act.norm()
                a_unit = act / a_norm

                with torch.no_grad():
                    ref_L = forward_to_L(model, ctx, act.unsqueeze(0),
                                         args.inject_layer, measure_layer)

                subset_dirs = dirs_dev[subset_idx]
                dirs_orth = subset_dirs - (subset_dirs @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
                dirs_orth = F.normalize(dirs_orth, dim=1)
                Q, _ = torch.linalg.qr(dirs_orth.T, mode='reduced')

                coeffs = torch.randn(1, Q.shape[1], device=device)
                v = (coeffs @ Q.T).squeeze(0)
                v = F.normalize(v, dim=0)

                l2_curve = sweep_dirs_batched(
                    model, ctx, act, ref_L, a_unit, a_norm,
                    v.unsqueeze(0), args.inject_layer, measure_layer,
                    device, angles_deg)[0]

                for thresh in args.l2_thresholds:
                    ba = crossing_angle(l2_curve, angles_deg, thresh) or args.max_angle
                    min_indiv = indiv_breaks[thresh][subset_idx].min()
                    random_subset_breaks[thresh].setdefault(k, []).append({
                        'raw': ba,
                        'normalized': ba / min_indiv if min_indiv > 0 else 0,
                        'min_indiv': min_indiv,
                    })

        print(f"  Done ({time.time()-t0:.0f}s)")

    # Save
    save_data = {
        'k_values': k_values,
        'l2_thresholds': args.l2_thresholds,
        'angles_deg': angles_deg,
        'n_pool': n_pool,
        'n_activations': args.n_activations,
        'n_dirs_per_k': args.n_dirs_per_k,
        'n_random_subsets': args.n_random_subsets,
        'n_control_subspaces': args.n_control_subspaces,
        'indiv_breaks': indiv_breaks,
        'ordered_breaks': ordered_breaks,
        'control_breaks': control_breaks,
        'random_subset_breaks': random_subset_breaks,
        'args': vars(args),
    }
    pkl_path = os.path.join(args.outdir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved: {pkl_path}")

    # === Plotting ===
    n_thresh = len(args.l2_thresholds)

    # Plot: 3 conditions (ordered, normalized random subsets, control)
    fig, axes = plt.subplots(1, n_thresh, figsize=(7 * n_thresh, 5))
    if n_thresh == 1:
        axes = [axes]

    for ti, thresh in enumerate(args.l2_thresholds):
        ax = axes[ti]
        ks = np.array(k_values, dtype=float)
        log_k = np.log(ks)

        conditions = []

        # Ordered
        ord_vals = ordered_breaks[thresh]
        ord_meds = np.array([np.median(ord_vals[k]) for k in k_values])
        ord_q25 = np.array([np.percentile(ord_vals[k], 25) for k in k_values])
        ord_q75 = np.array([np.percentile(ord_vals[k], 75) for k in k_values])
        conditions.append(('First k MELBO dirs', ord_meds, ord_q25, ord_q75, '#228833', 'o'))

        # Random subsets (normalized)
        rsub_norm = {k: [x['normalized'] for x in random_subset_breaks[thresh][k]] for k in k_values}
        rsub_meds = np.array([np.median(rsub_norm[k]) for k in k_values])
        rsub_q25 = np.array([np.percentile(rsub_norm[k], 25) for k in k_values])
        rsub_q75 = np.array([np.percentile(rsub_norm[k], 75) for k in k_values])
        conditions.append(('Random k MELBO (normalized)', rsub_meds, rsub_q25, rsub_q75, '#4477AA', '^'))

        # Control
        ctrl_vals = control_breaks[thresh]
        ctrl_meds = np.array([np.median(ctrl_vals[k]) for k in k_values])
        ctrl_q25 = np.array([np.percentile(ctrl_vals[k], 25) for k in k_values])
        ctrl_q75 = np.array([np.percentile(ctrl_vals[k], 75) for k in k_values])
        conditions.append(('1 MELBO + (k-1) random', ctrl_meds, ctrl_q25, ctrl_q75, '#EE6677', 's'))

        for label, meds, q25, q75, color, marker in conditions:
            log_m = np.log(meds)
            b_exp, log_a = np.polyfit(log_k, log_m, 1)
            a_coeff = np.exp(log_a)
            pred = log_a + b_exp * log_k
            ss_res = np.sum((log_m - pred)**2)
            ss_tot = np.sum((log_m - log_m.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            ax.errorbar(ks, meds,
                        yerr=[meds - q25, q75 - meds],
                        fmt='none', ecolor=color, capsize=3, alpha=0.6)
            ax.scatter(ks, meds, s=50, color=color, marker=marker, zorder=5)
            k_smooth = np.linspace(ks.min() * 0.8, ks.max() * 1.1, 200)
            ax.plot(k_smooth, a_coeff * k_smooth**b_exp, color=color, lw=2,
                    label=f'{label}: k^{{{b_exp:.2f}}} (R\u00b2={r2:.2f})')

        ax.set_xlabel('Subspace dimension k', fontsize=11)
        ax.set_ylabel('Breaking angle (degrees)', fontsize=11)
        ax.set_title(f'L2 threshold = {thresh}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')

    plt.suptitle(f'{args.model}: 3-condition subspace scaling\n'
                 f'{args.n_activations} act, {args.n_dirs_per_k} dirs/k, '
                 f'{args.n_control_subspaces} ctrl subspaces, '
                 f'{args.n_random_subsets} rand subsets/k',
                 fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, 'control_3condition.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: power law exponents (breaking angle ~ k^exponent)")
    print(f"{'='*60}")
    for thresh in args.l2_thresholds:
        ord_meds = np.array([np.median(ordered_breaks[thresh][k]) for k in k_values])
        rsub_norm = {k: [x['normalized'] for x in random_subset_breaks[thresh][k]] for k in k_values}
        norm_meds = np.array([np.median(rsub_norm[k]) for k in k_values])
        ctrl_meds = np.array([np.median(control_breaks[thresh][k]) for k in k_values])
        log_k_arr = np.log(np.array(k_values, dtype=float))
        b_ord, _ = np.polyfit(log_k_arr, np.log(ord_meds), 1)
        b_norm, _ = np.polyfit(log_k_arr, np.log(norm_meds), 1)
        b_ctrl, _ = np.polyfit(log_k_arr, np.log(ctrl_meds), 1)
        print(f"  L2>{thresh}: ordered={b_ord:.2f}, random_subset(norm)={b_norm:.2f}, "
              f"control={b_ctrl:.2f}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
