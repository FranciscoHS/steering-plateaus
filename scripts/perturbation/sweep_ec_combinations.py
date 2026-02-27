#!/usr/bin/env python3
"""Sweep combinations of EC pair directions to test error correction prediction.

Loads 10 directions from EC pairs, sweeps k=1..9 combinations,
plots median cosine distance and L2 for each k alongside random baseline.

Error correction predicts: 1 dir < 2 dirs < ... < K dirs < random (plateau length).
"""
import argparse, os, sys, pickle, itertools
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_final
from src.data import generate_chess_games


def forward_to_layer(model, ctx, point, start_layer, end_layer):
    """Forward from start_layer to end_layer (inclusive)."""
    from src.model import _get_blocks, _run_block, _is_rope_model, _get_position_embeddings, _is_cat_pos_emb_model, _get_cat_pos_emb
    context_batch = ctx.expand(point.shape[0], -1, -1)
    hidden = torch.cat([context_batch, point], dim=1)
    blocks = _get_blocks(model)
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, hidden)
    elif _is_cat_pos_emb_model(model):
        pos_emb = _get_cat_pos_emb(model, hidden.shape[1])
    for block in blocks[start_layer + 1:end_layer + 1]:
        hidden = _run_block(block, hidden, model, pos_emb)
    return hidden[:, -1, :]


def sweep_direction(model, ctx, act, d, angles_rad, layer, device, measure_layer=None):
    """Sweep a single direction, return cosine distance and L2 arrays."""
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


def run_sweep_for_activation(model, ctx, act, all_dirs, angles_rad, angles_deg,
                             layer, device, measure_layer, n_random_dirs, k_list,
                             n_samples_per_k):
    """Run combination sweep for a single activation. Returns results_by_k, rand_cos, rand_l2."""
    n_dirs = len(all_dirs)
    d_model = act.shape[-1]

    # Random baseline
    rand_cos = np.zeros((n_random_dirs, len(angles_deg)))
    rand_l2 = np.zeros((n_random_dirs, len(angles_deg)))
    for i in range(n_random_dirs):
        rand_cos[i], rand_l2[i] = sweep_direction(
            model, ctx, act, torch.randn(d_model, device=device),
            angles_rad, layer, device, measure_layer)

    # Sweep combinations
    results_by_k = {}
    for k in k_list:
        from math import comb
        n_total = comb(n_dirs, k)
        if n_total <= n_samples_per_k:
            combos = list(itertools.combinations(range(n_dirs), k))
        else:
            seen = set()
            combos = []
            while len(combos) < n_samples_per_k:
                c = tuple(sorted(np.random.choice(n_dirs, k, replace=False)))
                if c not in seen:
                    seen.add(c)
                    combos.append(c)

        print(f"  k={k}: {len(combos)} combos")
        cos_curves = np.zeros((len(combos), len(angles_deg)))
        l2_curves = np.zeros((len(combos), len(angles_deg)))

        for ci, combo in enumerate(combos):
            d_combo = sum(all_dirs[i] for i in combo)
            d_combo = F.normalize(d_combo, dim=0)
            cos_curves[ci], l2_curves[ci] = sweep_direction(
                model, ctx, act, d_combo, angles_rad, layer, device,
                measure_layer)

        results_by_k[k] = {
            'cos': cos_curves, 'l2': l2_curves,
            'n_combos': len(combos), 'n_total': n_total
        }

    return results_by_k, rand_cos, rand_l2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--act_indices', type=str, default='137',
                        help='Comma-separated activation indices to sweep')
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--ec_pairs_path', type=str, required=True,
                        help='Path to ec_pairs.pkl')
    parser.add_argument('--measure_layer', type=int, default=None)
    parser.add_argument('--n_random_dirs', type=int, default=20)
    parser.add_argument('--n_samples_per_k', type=int, default=20,
                        help='Max combinations to sample per k')
    parser.add_argument('--k_values', type=str, default=None,
                        help='Comma-separated k values to sweep (default: all 1..n-1)')
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    act_indices = [int(x) for x in args.act_indices.split(',')]

    # Load directions (auto-detect pairs vs triples format)
    with open(args.ec_pairs_path, 'rb') as f:
        data = pickle.load(f)
    all_dirs = []
    if isinstance(data, list) and len(data) > 0 and 'd1' in data[0]:
        # ec_pairs format: list of dicts with 'd1', 'd2'
        for p in data:
            all_dirs.append(F.normalize(p['d1'], dim=0))
            all_dirs.append(F.normalize(p['d2'], dim=0))
        print(f"Loaded {len(all_dirs)} directions from {len(data)} pairs")
    elif isinstance(data, dict) and 'runs' in data:
        # ec_triples format: dict with 'runs' list, each run has 'dirs'
        for run in data['runs']:
            for d in run['dirs']:
                all_dirs.append(F.normalize(d.float(), dim=0))
        print(f"Loaded {len(all_dirs)} directions from {len(data['runs'])} runs")
    elif isinstance(data, dict) and 'directions' in data:
        # progressive_feature_discovery format: dict with 'directions' tensor
        dirs_tensor = data['directions'].float()
        cos_dists = data.get('cos_dists', None)
        for i in range(dirs_tensor.shape[0]):
            all_dirs.append(F.normalize(dirs_tensor[i], dim=0))
        n_above = (cos_dists > 0.01).sum() if cos_dists is not None else '?'
        print(f"Loaded {len(all_dirs)} directions from progressive discovery "
              f"({n_above} with cos_dist > 0.01)")
    else:
        raise ValueError(f"Unknown directions format in {args.ec_pairs_path}")
    n_dirs = len(all_dirs)

    if args.k_values:
        k_list = [int(x) for x in args.k_values.split(',')]
        k_list = [k for k in k_list if 1 <= k < n_dirs]
    else:
        k_list = list(range(1, n_dirs))

    # Load model and extract all needed activations
    model, tokenizer, device = load_model(args.model)
    max_idx = max(act_indices)
    n_games = max(max_idx + 1, 50)
    token_ids_list = generate_chess_games(n_games, args.seq_len, seed=args.seed)

    # Extract activations one batch at a time, only keep what we need
    act_data = {}  # act_idx -> (ctx, act)
    batch_size = 32
    count = 0
    for i in range(0, n_games, batch_size):
        bt = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            out = model(bt, output_hidden_states=True)
            h = out.hidden_states[args.layer + 1]
        for aidx in act_indices:
            local_idx = aidx - count
            if 0 <= local_idx < bt.shape[0]:
                act_data[aidx] = (
                    h[local_idx:local_idx+1, :-1, :].float().to(device),
                    h[local_idx, -1, :].float().to(device)
                )
        count += bt.shape[0]
        del out, h, bt

    angles_deg = np.linspace(0, args.max_angle, int(args.max_angle * 2) + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Run sweep for each activation
    all_results = {}
    for aidx in act_indices:
        ctx, act = act_data[aidx]
        print(f"\nActivation {aidx} (norm={act.norm():.2f})")
        results_by_k, rand_cos, rand_l2 = run_sweep_for_activation(
            model, ctx, act, all_dirs, angles_rad, angles_deg,
            args.layer, device, args.measure_layer, args.n_random_dirs,
            k_list, args.n_samples_per_k)
        all_results[aidx] = {
            'results_by_k': results_by_k,
            'rand_cos': rand_cos, 'rand_l2': rand_l2
        }

    # Save raw results
    results_dir = os.path.dirname(args.ec_pairs_path)
    save_data = {
        'all_results': {aidx: {
            'results_by_k': {k: {'cos': v['cos'], 'l2': v['l2'],
                                  'n_combos': v['n_combos'], 'n_total': v['n_total']}
                             for k, v in r['results_by_k'].items()},
            'rand_cos': r['rand_cos'], 'rand_l2': r['rand_l2']
        } for aidx, r in all_results.items()},
        'angles_deg': angles_deg, 'args': vars(args)
    }
    suffix = '_multi' if len(act_indices) > 1 else ''
    pickle.dump(save_data, open(os.path.join(results_dir, f'combo_sweep{suffix}.pkl'), 'wb'))

    # Plot: one row per activation, 2 columns
    n_acts = len(act_indices)
    cmap = plt.cm.viridis
    fig, axes = plt.subplots(n_acts, 2, figsize=(16, 5 * n_acts))
    if n_acts == 1:
        axes = axes[np.newaxis, :]

    for row, aidx in enumerate(act_indices):
        r = all_results[aidx]
        results_by_k = r['results_by_k']

        for ax_idx, (metric, ylabel) in enumerate([
            ('cos', 'Cosine distance'), ('l2', 'L2 distance')
        ]):
            ax = axes[row, ax_idx]
            rd = r['rand_cos'] if metric == 'cos' else r['rand_l2']

            ax.fill_between(angles_deg,
                            np.percentile(rd, 25, axis=0),
                            np.percentile(rd, 75, axis=0),
                            color='gray', alpha=0.15)
            ax.plot(angles_deg, np.median(rd, axis=0),
                    color='gray', lw=2.5, ls='--', label='Random')

            k_values = sorted(results_by_k.keys())
            for k in k_values:
                data = results_by_k[k][metric]
                color = cmap(k / n_dirs)
                median = np.median(data, axis=0)
                ax.plot(angles_deg, median, color=color, lw=1.5, alpha=0.8,
                        label=f'k={k} (n={results_by_k[k]["n_combos"]})')

            ax.set_xlabel('Perturbation angle (degrees)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'act={aidx}: {ylabel}')
            ax.legend(fontsize=6, ncol=2, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, args.max_angle)

    fig.suptitle(f'EC direction combinations (layer {args.layer}, {n_dirs} dirs, {n_acts} activations)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(results_dir, f'combo_sweep{suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
