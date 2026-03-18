#!/usr/bin/env python3
"""Test EC prediction: plateau-breaking angle scales with subspace dimension k.

For each k, samples random directions from the span of k MELBO features,
sweeps perturbation angles, measures breaking angle. Compares to fully random.

EC predicts: breaking angle increases monotonically with k, approaching
the random baseline as k -> d_model.

Usage:
    python scripts/perturbation/sweep_subspace_scaling_qwen.py \
        --directions results/iterative_melbo_hf/qwen3-1.7b_L7_meas-2_escalation_v6/checkpoint_round3_mid.pkl \
        --k_values 10,20,50,100,200,500,1000,2048 \
        --n_dirs_per_k 30 --n_random 30
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


def crossing_angle(curve, angles_deg, threshold=0.1):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directions', required=True,
                        help='Path to .pkl with MELBO directions')
    parser.add_argument('--model', default=None,
                        help='Override model name (default: from directions file)')
    parser.add_argument('--inject_layer', type=int, default=None)
    parser.add_argument('--measure_layer', type=int, default=None)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--act_idx', type=int, default=0)
    parser.add_argument('--n_load', type=int, default=100)
    parser.add_argument('--k_values', type=str, required=True,
                        help='Comma-separated subspace dimensions (e.g. 10,20,50,100,200,500,1000,2048)')
    parser.add_argument('--n_dirs_per_k', type=int, default=30,
                        help='Random directions to sample per k value')
    parser.add_argument('--n_random', type=int, default=30,
                        help='Fully random baseline directions')
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=120)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Cosine distance threshold for breaking angle')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--random_subsets', action='store_true',
                        help='Sample random subsets of k features instead of first k')
    parser.add_argument('--n_subsets', type=int, default=5,
                        help='Number of random subsets per k (only with --random_subsets)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    k_values = [int(k) for k in args.k_values.split(',')]

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
    print(f"Loaded {n_features} directions (d_model={d_model})")

    # Get model/layer config from directions file or args
    dir_args = dir_data.get('args', {})
    model_name = args.model or dir_args.get('model', 'qwen3-1.7b')
    inject_layer = args.inject_layer if args.inject_layer is not None else dir_args.get('inject_layer', 7)
    measure_layer = args.measure_layer if args.measure_layer is not None else dir_args.get('measure_layer', -2)

    # Cap k values
    k_values = [k for k in k_values if k <= n_features]
    print(f"k values: {k_values} (max features: {n_features})")

    # Load model
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    print(f"Loading model: {model_name}")
    model, tokenizer, device = load_model(model_name, dtype=model_dtype)
    n_layers = len(_get_blocks(model))

    if measure_layer < 0:
        measure_layer = n_layers + measure_layer
    print(f"Layer {inject_layer} -> measure {measure_layer}, device={device}")

    # Load activation
    token_ids_list = load_fineweb_fixed_length(args.n_load, tokenizer,
                                                seq_len=args.seq_len, seed=args.seed)
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, inject_layer, device)

    ctx = all_contexts[args.act_idx].unsqueeze(0).to(device)
    act = all_activations[args.act_idx].to(device)
    a_norm = act.norm()
    a_unit = act / a_norm
    print(f"Activation {args.act_idx}, norm={a_norm:.2f}")

    # Reference output
    with torch.no_grad():
        ref_L = forward_to_L(model, ctx, act.unsqueeze(0), inject_layer, measure_layer)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]

    # Move dirs to device and orthogonalize to activation
    dirs_dev = all_dirs.to(device)
    dirs_orth = dirs_dev - (dirs_dev @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
    dirs_orth = F.normalize(dirs_orth, dim=1)

    # Output dir
    if args.results_dir is None:
        suffix = 'subspace_scaling_random_subsets' if args.random_subsets else 'subspace_scaling'
        args.results_dir = os.path.join(os.path.dirname(args.directions), suffix)
    os.makedirs(args.results_dir, exist_ok=True)

    t0 = time.time()

    # ── Sweep individual features (first N, since iterative MELBO finds best first) ──
    n_indiv = min(50, n_features)
    print(f"\nSweeping first {n_indiv} individual features...")
    indiv_breaks = np.zeros(n_indiv)
    indiv_cos_curves = np.zeros((n_indiv, len(angles_deg)))
    indiv_l2_curves = np.zeros((n_indiv, len(angles_deg)))
    for i in range(n_indiv):
        fi = i
        cos_curve, l2_curve = sweep_dir(model, ctx, act, ref_L, a_unit, a_norm,
                                  dirs_dev[fi], inject_layer, measure_layer,
                                  device, angles_deg)
        indiv_cos_curves[i] = cos_curve
        indiv_l2_curves[i] = l2_curve
        b = crossing_angle(cos_curve, angles_deg, args.threshold)
        indiv_breaks[i] = b if b is not None else args.max_angle
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_indiv} ({time.time()-t0:.0f}s)")
    print(f"  Individual median: {np.median(indiv_breaks):.1f} deg")

    # ── Fully random baseline ──
    print(f"\nSweeping {args.n_random} fully random directions...")
    rand_breaks = np.zeros(args.n_random)
    rand_cos_curves = np.zeros((args.n_random, len(angles_deg)))
    rand_l2_curves = np.zeros((args.n_random, len(angles_deg)))
    for i in range(args.n_random):
        d_rand = torch.randn(d_model, device=device)
        cos_curve, l2_curve = sweep_dir(model, ctx, act, ref_L, a_unit, a_norm,
                                  d_rand, inject_layer, measure_layer,
                                  device, angles_deg)
        rand_cos_curves[i] = cos_curve
        rand_l2_curves[i] = l2_curve
        b = crossing_angle(cos_curve, angles_deg, args.threshold)
        rand_breaks[i] = b if b is not None else args.max_angle
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_random} ({time.time()-t0:.0f}s)")
    print(f"  Random median: {np.median(rand_breaks):.1f} deg")

    # ── For each k: random directions from span of k features ──
    results = {}

    for k in k_values:
        if args.random_subsets:
            # Sample multiple random subsets of k features from the full pool
            n_per_subset = max(1, args.n_dirs_per_k // args.n_subsets)
            n_total = n_per_subset * args.n_subsets
            print(f"\nk={k}: {args.n_subsets} random subsets x {n_per_subset} dirs = {n_total}")
        else:
            # Use first k features (iterative MELBO finds highest-quality first)
            n_total = args.n_dirs_per_k
            print(f"\nk={k}: span of first {k} features, {n_total} random directions")

        all_breaks = np.zeros(n_total)
        all_cos_curves = np.zeros((n_total, len(angles_deg)))
        all_l2_curves = np.zeros((n_total, len(angles_deg)))

        idx = 0
        n_batches = args.n_subsets if args.random_subsets else 1
        dirs_per_batch = n_per_subset if args.random_subsets else n_total

        for batch in range(n_batches):
            if args.random_subsets:
                subset_indices = np.random.choice(n_features, size=k, replace=False)
                subset_dirs = dirs_orth[subset_indices]
            else:
                subset_dirs = dirs_orth[:k]

            Q, _ = torch.linalg.qr(subset_dirs.T, mode='reduced')
            sub_rank = Q.shape[1]
            if batch == 0:
                print(f"  QR rank: {sub_rank} (of {k})")

            for d in range(dirs_per_batch):
                coeffs = torch.randn(sub_rank, device=device)
                v = Q @ coeffs
                v = F.normalize(v, dim=0)

                cos_curve, l2_curve = sweep_dir(model, ctx, act, ref_L, a_unit, a_norm,
                                          v, inject_layer, measure_layer,
                                          device, angles_deg)
                b = crossing_angle(cos_curve, angles_deg, args.threshold)
                all_breaks[idx] = b if b is not None else args.max_angle
                all_cos_curves[idx] = cos_curve
                all_l2_curves[idx] = l2_curve
                idx += 1

            if (batch + 1) % 2 == 0 or batch == n_batches - 1:
                print(f"  {idx}/{n_total} ({time.time()-t0:.0f}s)")

        med = np.median(all_breaks)
        q25 = np.percentile(all_breaks, 25)
        q75 = np.percentile(all_breaks, 75)
        results[k] = {
            'breaks': all_breaks,
            'cos_curves': all_cos_curves,
            'l2_curves': all_l2_curves,
            'median': med,
            'q25': q25,
            'q75': q75,
        }
        print(f"  k={k}: median={med:.1f} deg [{q25:.1f}, {q75:.1f}]")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # ── Save ──
    save_data = {
        'k_values': k_values,
        'results': results,
        'indiv_breaks': indiv_breaks,
        'indiv_cos_curves': indiv_cos_curves,
        'indiv_l2_curves': indiv_l2_curves,
        'rand_breaks': rand_breaks,
        'rand_cos_curves': rand_cos_curves,
        'rand_l2_curves': rand_l2_curves,
        'angles_deg': angles_deg,
        'args': vars(args),
        'dir_args': dir_args,
        'n_features': n_features,
        'd_model': d_model,
    }
    pkl_path = os.path.join(args.results_dir, 'subspace_scaling.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved: {pkl_path}")

    # ── Plot 1: Breaking angle vs k ──
    fig, ax = plt.subplots(figsize=(10, 6))

    ks = sorted(results.keys())
    meds = [results[k]['median'] for k in ks]
    q25s = [results[k]['q25'] for k in ks]
    q75s = [results[k]['q75'] for k in ks]

    ax.fill_between(ks, q25s, q75s, color='C0', alpha=0.15)
    ax.plot(ks, meds, 'o-', color='C0', lw=2.5, markersize=7, label='Random in span(k features)')

    # Individual features baseline
    indiv_med = np.median(indiv_breaks)
    ax.axhline(indiv_med, color='C2', ls=':', lw=2, alpha=0.8,
               label=f'Individual features (median={indiv_med:.1f}°)')

    # Random baseline
    rand_med = np.median(rand_breaks)
    rand_q25 = np.percentile(rand_breaks, 25)
    rand_q75 = np.percentile(rand_breaks, 75)
    ax.axhspan(rand_q25, rand_q75, color='gray', alpha=0.1)
    ax.axhline(rand_med, color='gray', ls='--', lw=2, alpha=0.8,
               label=f'Fully random (median={rand_med:.1f}°)')

    ax.set_xlabel('Subspace dimension k', fontsize=12)
    ax.set_ylabel('Breaking angle (degrees)', fontsize=12)
    ax.set_title(f'{model_name} L{inject_layer}→L{measure_layer}: '
                 f'breaking angle vs subspace dimension\n'
                 f'{n_features} MELBO features, threshold={args.threshold}',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])

    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'subspace_scaling.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # ── Plot 2: Median response curves for each k (cosine distance and L2) ──
    for metric, curve_key, ylabel, suffix in [
        ('Cosine distance', 'cos_curves', 'Cosine distance', 'cos'),
        ('L2 distance', 'l2_curves', 'L2 distance', 'l2'),
    ]:
        fig, ax = plt.subplots(figsize=(12, 7))
        cmap = plt.cm.viridis

        # Individual features
        if suffix == 'cos':
            indiv_data = indiv_cos_curves
        else:
            indiv_data = indiv_l2_curves
        ax.fill_between(angles_deg,
                        np.percentile(indiv_data, 25, axis=0),
                        np.percentile(indiv_data, 75, axis=0),
                        color='tab:red', alpha=0.08)
        ax.plot(angles_deg, np.median(indiv_data, axis=0),
                color='tab:red', lw=2.5, label='Individual features')

        # Per-k curves
        for i, k in enumerate(ks):
            color = cmap(i / max(len(ks) - 1, 1))
            curves = results[k][curve_key]
            med_curve = np.median(curves, axis=0)
            q25_curve = np.percentile(curves, 25, axis=0)
            q75_curve = np.percentile(curves, 75, axis=0)
            ax.fill_between(angles_deg, q25_curve, q75_curve, color=color, alpha=0.08)
            ax.plot(angles_deg, med_curve, color=color, lw=2, label=f'k={k}')

        # Random baseline
        if suffix == 'cos':
            rand_data = rand_cos_curves
        else:
            rand_data = rand_l2_curves
        ax.fill_between(angles_deg,
                        np.percentile(rand_data, 25, axis=0),
                        np.percentile(rand_data, 75, axis=0),
                        color='gray', alpha=0.1)
        ax.plot(angles_deg, np.median(rand_data, axis=0),
                color='gray', lw=2.5, ls='--', label='Random')

        if suffix == 'cos':
            ax.axhline(args.threshold, color='red', ls=':', lw=1, alpha=0.5,
                        label=f'Threshold={args.threshold}')
        ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{model_name} L{inject_layer}→L{measure_layer}: '
                     f'{metric} by subspace dimension', fontsize=12)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        curves_path = os.path.join(args.results_dir, f'subspace_curves_{suffix}.png')
        plt.savefig(curves_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {curves_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY ({n_features} MELBO features, d_model={d_model})")
    print(f"{'='*60}")
    print(f"Individual features: {indiv_med:.1f} deg (median of {n_indiv})")
    for k in ks:
        r = results[k]
        print(f"k={k:5d}: {r['median']:5.1f} deg [{r['q25']:5.1f}, {r['q75']:5.1f}]")
    print(f"Random:              {rand_med:.1f} deg [{rand_q25:.1f}, {rand_q75:.1f}]")


if __name__ == '__main__':
    main()
