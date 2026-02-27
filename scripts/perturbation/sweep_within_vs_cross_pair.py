#!/usr/bin/env python3
"""Compare perturbation response for:
  - Individual EC directions (100 total)
  - Within-pair combinations: normalize(d1+d2) for matched pairs (50 total)
  - Cross-pair combinations: normalize(d_i+d_j) from different pairs (sampled)

Tests whether the EC pair optimization found pair-specific cancellations
vs. genuine feature-level dilution.
"""
import argparse, os, sys, pickle
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_final
from src.data import generate_chess_games


def sweep_direction(model, ctx, act, d, angles_rad, layer, device, measure_layer=None):
    """Sweep a single direction, return cosine distance and L2 arrays."""
    from scripts.perturbation.sweep_ec_combinations import forward_to_layer
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=model_dtype)
    R = a_d.norm()
    a_unit = a_d / R
    d_d = d.to(device=device, dtype=model_dtype)
    d_orth = d_d - (d_d @ a_unit) * a_unit
    d_orth = F.normalize(d_orth, dim=-1)

    with torch.no_grad():
        if measure_layer is not None:
            ref = forward_to_layer(model, ctx_d, a_d[None, None], layer, measure_layer)
        else:
            ref = forward_from_layer_to_final(model, ctx_d, a_d[None, None], layer)

    cos_arr = np.zeros(len(angles_rad))
    l2_arr = np.zeros(len(angles_rad))
    for j, theta in enumerate(angles_rad):
        p = R * (np.cos(theta) * a_unit + np.sin(theta) * d_orth)
        with torch.no_grad():
            if measure_layer is not None:
                pf = forward_to_layer(model, ctx_d, p[None, None], layer, measure_layer)
            else:
                pf = forward_from_layer_to_final(model, ctx_d, p[None, None], layer)
            cos_arr[j] = 1.0 - F.cosine_similarity(ref.float(), pf.float(), dim=-1).item()
            l2_arr[j] = (pf - ref).float().norm().item()
    return cos_arr, l2_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--act_idx', type=int, default=137)
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--ec_pairs_path', type=str, required=True)
    parser.add_argument('--measure_layer', type=int, default=None)
    parser.add_argument('--n_cross_pairs', type=int, default=200,
                        help='Number of cross-pair combinations to sample')
    parser.add_argument('--n_random_dirs', type=int, default=50)
    parser.add_argument('--max_angle', type=float, default=15.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load pairs (keep pair structure)
    with open(args.ec_pairs_path, 'rb') as f:
        pairs_data = pickle.load(f)
    pairs = []
    for p in pairs_data:
        d1 = F.normalize(p['d1'].float(), dim=0)
        d2 = F.normalize(p['d2'].float(), dim=0)
        pairs.append((d1, d2))
    n_pairs = len(pairs)
    all_dirs = []
    for d1, d2 in pairs:
        all_dirs.append(d1)
        all_dirs.append(d2)
    print(f"Loaded {n_pairs} pairs ({len(all_dirs)} directions)")

    # Load model and extract activation
    model, tokenizer, device = load_model(args.model)
    n_games = max(args.act_idx + 1, 50)
    token_ids_list = generate_chess_games(n_games, args.seq_len, seed=args.seed)

    batch_size = 32
    ctx = act = None
    count = 0
    for i in range(0, n_games, batch_size):
        bt = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            out = model(bt, output_hidden_states=True)
            h = out.hidden_states[args.layer + 1]
        local_idx = args.act_idx - count
        if 0 <= local_idx < bt.shape[0]:
            ctx = h[local_idx:local_idx+1, :-1, :].float().to(device)
            act = h[local_idx, -1, :].float().to(device)
        count += bt.shape[0]
        del out, h, bt
    print(f"Activation {args.act_idx} (norm={act.norm():.2f})")

    angles_deg = np.linspace(0, args.max_angle, int(args.max_angle * 2) + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # 1) Individual directions (all 100)
    print(f"\nSweeping {len(all_dirs)} individual directions...")
    indiv_cos = np.zeros((len(all_dirs), len(angles_deg)))
    indiv_l2 = np.zeros((len(all_dirs), len(angles_deg)))
    for i, d in enumerate(all_dirs):
        indiv_cos[i], indiv_l2[i] = sweep_direction(
            model, ctx, act, d, angles_rad, args.layer, device, args.measure_layer)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(all_dirs)}")

    # 2) Within-pair combinations (50)
    print(f"\nSweeping {n_pairs} within-pair combinations...")
    within_cos = np.zeros((n_pairs, len(angles_deg)))
    within_l2 = np.zeros((n_pairs, len(angles_deg)))
    for i, (d1, d2) in enumerate(pairs):
        d_combo = F.normalize(d1 + d2, dim=0)
        within_cos[i], within_l2[i] = sweep_direction(
            model, ctx, act, d_combo, angles_rad, args.layer, device, args.measure_layer)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_pairs}")

    # 3) Cross-pair combinations (sampled)
    print(f"\nSweeping {args.n_cross_pairs} cross-pair combinations...")
    cross_cos = np.zeros((args.n_cross_pairs, len(angles_deg)))
    cross_l2 = np.zeros((args.n_cross_pairs, len(angles_deg)))
    seen = set()
    cross_idx = 0
    while cross_idx < args.n_cross_pairs:
        # Pick two directions from different pairs
        p1 = np.random.randint(n_pairs)
        p2 = np.random.randint(n_pairs)
        while p2 == p1:
            p2 = np.random.randint(n_pairs)
        # Pick which direction from each pair (d1 or d2)
        s1 = np.random.randint(2)
        s2 = np.random.randint(2)
        key = (min(p1*2+s1, p2*2+s2), max(p1*2+s1, p2*2+s2))
        if key in seen:
            continue
        seen.add(key)
        d_combo = F.normalize(all_dirs[p1*2+s1] + all_dirs[p2*2+s2], dim=0)
        cross_cos[cross_idx], cross_l2[cross_idx] = sweep_direction(
            model, ctx, act, d_combo, angles_rad, args.layer, device, args.measure_layer)
        cross_idx += 1
        if cross_idx % 50 == 0:
            print(f"  {cross_idx}/{args.n_cross_pairs}")

    # 4) Random baseline
    print(f"\nSweeping {args.n_random_dirs} random directions...")
    rand_cos = np.zeros((args.n_random_dirs, len(angles_deg)))
    rand_l2 = np.zeros((args.n_random_dirs, len(angles_deg)))
    d_model = act.shape[-1]
    for i in range(args.n_random_dirs):
        rand_cos[i], rand_l2[i] = sweep_direction(
            model, ctx, act, torch.randn(d_model, device=device),
            angles_rad, args.layer, device, args.measure_layer)

    # Save
    results_dir = os.path.dirname(args.ec_pairs_path)
    save_data = {
        'indiv_cos': indiv_cos, 'indiv_l2': indiv_l2,
        'within_cos': within_cos, 'within_l2': within_l2,
        'cross_cos': cross_cos, 'cross_l2': cross_l2,
        'rand_cos': rand_cos, 'rand_l2': rand_l2,
        'angles_deg': angles_deg, 'args': vars(args),
        'n_pairs': n_pairs, 'n_cross': args.n_cross_pairs,
    }
    pkl_path = os.path.join(results_dir, 'within_vs_cross_pair.pkl')
    pickle.dump(save_data, open(pkl_path, 'wb'))
    print(f"\nSaved: {pkl_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax_idx, (metric, ylabel) in enumerate([('cos', 'Cosine distance'), ('l2', 'L2 distance')]):
        ax = axes[ax_idx]
        suffix = '_cos' if metric == 'cos' else '_l2'

        indiv = save_data[f'indiv{suffix}']
        within = save_data[f'within{suffix}']
        cross = save_data[f'cross{suffix}']
        rand = save_data[f'rand{suffix}']

        # Medians
        ax.plot(angles_deg, np.median(indiv, axis=0),
                color='tab:blue', lw=2.5, label=f'Individual (n={indiv.shape[0]})')
        ax.plot(angles_deg, np.median(within, axis=0),
                color='tab:green', lw=2.5, label=f'Within-pair (n={within.shape[0]})')
        ax.plot(angles_deg, np.median(cross, axis=0),
                color='tab:red', lw=2.5, label=f'Cross-pair (n={cross.shape[0]})')
        ax.plot(angles_deg, np.median(rand, axis=0),
                color='gray', lw=2.5, ls='--', label=f'Random (n={rand.shape[0]})')

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, args.max_angle)

    fig.suptitle(f'Individual vs Within-pair vs Cross-pair (layer {args.layer}, act={args.act_idx})',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    png_path = os.path.join(results_dir, 'within_vs_cross_pair.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    main()
