"""Dominant direction test — generic version for any model.

Usage:
  python sweep_dominant_generic.py --model gpt2-xl --inject_layer 2 --measure_layer 46 \
    --directions results/iterative_melbo_hf/gpt2-xl_L2_fixed10deg_meas46/results.pkl \
    --outdir results/sweep_dominant/gpt2-xl_L2 --n_activations 10
"""
import os, sys, pickle, time, argparse
sys.path.insert(0, '/workspace/steering-plateaus')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model
from src.data import load_fineweb_fixed_length
from scripts.melbo.run_iterative_melbo_hf import (
    extract_context_and_activation, forward_to_L, sweep_dir,
)

def crossing_angle(curve, angles_deg, threshold):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--inject_layer', type=int, required=True)
    parser.add_argument('--measure_layer', type=int, required=True)
    parser.add_argument('--directions', required=True, help='Path to results.pkl with directions')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--n_activations', type=int, default=10)
    parser.add_argument('--dtype', type=str, default=None, choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=120)
    parser.add_argument('--l2_thresholds', type=float, nargs='+', default=[100, 200, 400, 600],
                        help='L2 thresholds for breaking angle computation')
    parser.add_argument('--k_values', type=int, nargs='+', default=None,
                        help='Subspace dimensions to test (default: auto from 10 to n_features)')
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    l2_thresholds = args.l2_thresholds
    k_values = [50, 250, 500]  # Will be filtered to <= n_features
    n_subsets_per_k = 20
    n_dirs_per_subset = 3

    # Load directions
    print(f"Loading directions from {args.directions}...")
    with open(args.directions, 'rb') as f:
        dir_data = pickle.load(f)
    all_dirs = torch.stack(dir_data['directions']).float()
    n_features, d_model = all_dirs.shape
    all_dirs = F.normalize(all_dirs, dim=1)
    print(f"{n_features} directions, d={d_model}")

    # Use provided k_values or auto-generate
    if args.k_values is not None:
        k_values = [k for k in args.k_values if k <= n_features]
    else:
        k_values = [10] + list(range(50, n_features + 1, 50))
    print(f"k_values: {k_values}")

    # Load model
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)

    token_ids_list = load_fineweb_fixed_length(100, tokenizer, seq_len=5, seed=42)
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.inject_layer, device)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    dirs_dev = all_dirs.to(device)

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    # Step 1: Sweep all directions individually, averaged over activations
    print(f"\nSweeping all {n_features} individual directions over {args.n_activations} activations...")
    indiv_l2_curves = np.zeros((n_features, len(angles_deg)))

    for ai in range(args.n_activations):
        ctx = all_contexts[ai].unsqueeze(0).to(device)
        act = all_activations[ai].to(device)
        a_norm = act.norm()
        a_unit = act / a_norm

        with torch.no_grad():
            ref_L = forward_to_L(model, ctx, act.unsqueeze(0), args.inject_layer, args.measure_layer)

        dirs_orth = dirs_dev - (dirs_dev @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        dirs_orth = F.normalize(dirs_orth, dim=1)

        for i in range(n_features):
            _, l2_c = sweep_dir(model, ctx, act, ref_L, a_unit, a_norm,
                                dirs_orth[i], args.inject_layer, args.measure_layer, device, angles_deg)
            indiv_l2_curves[i] += l2_c / args.n_activations

        print(f"  Activation {ai+1}/{args.n_activations} done ({time.time()-t0:.0f}s)")

    # Compute per-direction breaking angles
    indiv_breaks = {}
    for thresh in l2_thresholds:
        breaks = np.array([crossing_angle(indiv_l2_curves[i], angles_deg, thresh) or args.max_angle
                          for i in range(n_features)])
        indiv_breaks[thresh] = breaks
        print(f"  L2>{thresh}: median={np.median(breaks):.1f}, min={breaks.min():.1f}, max={breaks.max():.1f}")

    # Save checkpoint
    with open(f'{args.outdir}/indiv_checkpoint.pkl', 'wb') as f:
        pickle.dump({'indiv_l2_curves': indiv_l2_curves, 'indiv_breaks': indiv_breaks,
                      'angles_deg': angles_deg, 'n_features': n_features}, f)
    print(f"Individual sweeps done ({time.time()-t0:.0f}s)")

    # Step 2: Ordered first-k sweep (averaged)
    print(f"\nSweeping ordered first-k subsets...")
    ordered_data = {}
    for thresh in l2_thresholds:
        ordered_data[thresh] = {'k': [], 'breaking_angle': []}

    for k in k_values:
        print(f"  k={k}...")
        # Use first k directions, get QR basis
        avg_l2_curve = np.zeros(len(angles_deg))
        n_dirs_ordered = 6

        for d in range(n_dirs_ordered):
            dir_avg = np.zeros(len(angles_deg))
            for ai in range(args.n_activations):
                ctx = all_contexts[ai].unsqueeze(0).to(device)
                act = all_activations[ai].to(device)
                a_norm = act.norm()
                a_unit = act / a_norm

                with torch.no_grad():
                    ref_L = forward_to_L(model, ctx, act.unsqueeze(0), args.inject_layer, args.measure_layer)

                dirs_orth = dirs_dev[:k] - (dirs_dev[:k] @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
                dirs_orth = F.normalize(dirs_orth, dim=1)
                Q, _ = torch.linalg.qr(dirs_orth.T, mode='reduced')

                torch.manual_seed(42 + d)
                coeffs = torch.randn(Q.shape[1], device=device)
                v = Q @ coeffs
                v = F.normalize(v, dim=0)

                _, l2_c = sweep_dir(model, ctx, act, ref_L, a_unit, a_norm,
                                    v, args.inject_layer, args.measure_layer, device, angles_deg)
                dir_avg += l2_c / args.n_activations
            avg_l2_curve += dir_avg / n_dirs_ordered

        for thresh in l2_thresholds:
            ba = crossing_angle(avg_l2_curve, angles_deg, thresh) or args.max_angle
            ordered_data[thresh]['k'].append(k)
            ordered_data[thresh]['breaking_angle'].append(ba)

    print(f"  Ordered done ({time.time()-t0:.0f}s)")

    # Step 3: Random subsets (averaged)
    print(f"\nSweeping random subsets...")
    subset_data = {}
    for thresh in l2_thresholds:
        subset_data[thresh] = {}

    for k in k_values:
        n_total = n_subsets_per_k * n_dirs_per_subset
        print(f"  k={k}: {n_subsets_per_k} subsets x {n_dirs_per_subset} dirs = {n_total}")

        for thresh in l2_thresholds:
            subset_data[thresh][k] = {'measured': [], 'predicted': [], 'min_indiv': []}

        for s in range(n_subsets_per_k):
            subset_idx = np.random.choice(n_features, size=k, replace=False)

            for d in range(n_dirs_per_subset):
                avg_l2_curve = np.zeros(len(angles_deg))

                for ai in range(args.n_activations):
                    ctx = all_contexts[ai].unsqueeze(0).to(device)
                    act = all_activations[ai].to(device)
                    a_norm = act.norm()
                    a_unit = act / a_norm

                    with torch.no_grad():
                        ref_L = forward_to_L(model, ctx, act.unsqueeze(0), args.inject_layer, args.measure_layer)

                    dirs_orth = dirs_dev - (dirs_dev @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
                    dirs_orth_normed = F.normalize(dirs_orth, dim=1)
                    subset_dirs_orth = dirs_orth_normed[subset_idx]
                    Q, _ = torch.linalg.qr(subset_dirs_orth.T, mode='reduced')

                    torch.manual_seed(42 + s * n_dirs_per_subset + d)
                    coeffs = torch.randn(Q.shape[1], device=device)
                    v = Q @ coeffs
                    v = F.normalize(v, dim=0)

                    _, l2_c = sweep_dir(model, ctx, act, ref_L, a_unit, a_norm,
                                        v, args.inject_layer, args.measure_layer, device, angles_deg)
                    avg_l2_curve += l2_c / args.n_activations

                for thresh in l2_thresholds:
                    measured = crossing_angle(avg_l2_curve, angles_deg, thresh) or args.max_angle
                    min_indiv = indiv_breaks[thresh][subset_idx].min()
                    predicted = np.sqrt(k) * min_indiv
                    subset_data[thresh][k]['measured'].append(measured)
                    subset_data[thresh][k]['predicted'].append(predicted)
                    subset_data[thresh][k]['min_indiv'].append(min_indiv)

        print(f"    Done ({time.time()-t0:.0f}s)")

    # Save all results
    save_data = {
        'model': args.model,
        'inject_layer': args.inject_layer,
        'measure_layer': args.measure_layer,
        'k_values': k_values,
        'l2_thresholds': l2_thresholds,
        'subset_data': subset_data,
        'ordered_data': ordered_data,
        'indiv_breaks': indiv_breaks,
        'indiv_l2_curves': indiv_l2_curves,
        'angles_deg': angles_deg,
        'n_features': n_features,
        'n_activations': args.n_activations,
    }
    with open(f'{args.outdir}/results.pkl', 'wb') as f:
        pickle.dump(save_data, f)

    # Plot 1: Ordered first-k scaling
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ti, thresh in enumerate(l2_thresholds):
        ax = axes[ti // 2][ti % 2]
        ks = np.array(ordered_data[thresh]['k'], dtype=float)
        bas = np.array(ordered_data[thresh]['breaking_angle'])

        log_k = np.log(ks)
        log_b = np.log(bas)
        b_exp, log_a = np.polyfit(log_k, log_b, 1)
        a_coeff = np.exp(log_a)
        pred = log_a + b_exp * log_k
        ss_res = np.sum((log_b - pred)**2)
        ss_tot = np.sum((log_b - log_b.mean())**2)
        r2 = 1 - ss_res / ss_tot

        ax.scatter(ks, bas, s=50, color='red', zorder=5)
        k_smooth = np.linspace(0, ks.max() * 1.05, 200)
        ax.plot(k_smooth, a_coeff * k_smooth**b_exp, 'r-', lw=2,
                label=f'{a_coeff:.2f} k^{{{b_exp:.2f}}} (R²={r2:.3f})')
        ax.set_xlabel('Subspace dimension k')
        ax.set_ylabel('Breaking angle (deg)')
        ax.set_title(f'L2 threshold = {thresh}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{args.model} — Ordered first-k breaking angle', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{args.outdir}/ordered_scaling.png', dpi=150)
    print(f"Saved ordered_scaling.png")

    # Plot 2: Normalized breaking angle (random subsets)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ti, thresh in enumerate(l2_thresholds):
        ax = axes[ti // 2][ti % 2]
        medians = []
        for k in k_values:
            m = np.array(subset_data[thresh][k]['measured'])
            mi = np.array(subset_data[thresh][k]['min_indiv'])
            valid = mi > 0
            medians.append(np.median(m[valid] / mi[valid]))

        ks = np.array(k_values, dtype=float)
        meds = np.array(medians)
        log_k = np.log(ks)
        log_m = np.log(meds)
        b_exp, log_a = np.polyfit(log_k, log_m, 1)
        a_coeff = np.exp(log_a)
        pred_log = log_a + b_exp * log_k
        ss_res = np.sum((log_m - pred_log)**2)
        ss_tot = np.sum((log_m - log_m.mean())**2)
        r2 = 1 - ss_res / ss_tot

        ax.scatter(ks, meds, s=50, color='red', zorder=5)
        k_smooth = np.linspace(0, ks.max() * 1.05, 200)
        ax.plot(k_smooth, a_coeff * k_smooth**b_exp, 'r-', lw=2,
                label=f'{a_coeff:.2f} k^{{{b_exp:.2f}}} (R²={r2:.3f})')
        ax.set_xlabel('Subspace dimension k')
        ax.set_ylabel('Breaking angle / min individual breaking angle')
        ax.set_title(f'L2 threshold = {thresh}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{args.model} — Normalized breaking angle (random subsets, {args.n_activations} activations)', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{args.outdir}/normalized_breaking_angle.png', dpi=150)
    print(f"Saved normalized_breaking_angle.png")

    print(f"\nTotal time: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
