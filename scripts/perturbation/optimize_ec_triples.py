#!/usr/bin/env python3
"""
Optimize K=3 directions that individually break the plateau but whose
combinations do not — directly encoding the error correction prediction.

Objective: Σᵢ c(dᵢ, θ) - λ · Σ_{|s|≥2} c(normalize(Σᵢ∈s dᵢ), θ)

where c(d, θ) = cosine distance at measure_layer when perturbing at layer L
by angle θ along d.

Run N=4 independent replications with cross-run diversity penalty.
"""
import argparse, os, sys, time, pickle
from itertools import combinations
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_layer, forward_from_layer_to_final, \
    _get_blocks, _get_final_norm, _get_lm_head
from src.data import generate_chess_games


def cos_dist(model, ctx, a, d, angle_rad, layer, measure_layer, device):
    """Cosine distance at measure_layer for perturbation along d at given angle.

    All inputs should already be on device and in model dtype.
    d does not need to be pre-orthogonalized — this function handles it.
    """
    a_unit = a / a.norm()
    R = a.norm()

    d_orth = d - (d @ a_unit) * a_unit
    d_orth = d_orth / d_orth.norm()

    a_pert = R * (np.cos(angle_rad) * a_unit + np.sin(angle_rad) * d_orth)

    ref = forward_from_layer_to_layer(model, ctx, a.unsqueeze(0).unsqueeze(0), layer, measure_layer)
    pert = forward_from_layer_to_layer(model, ctx, a_pert.unsqueeze(0).unsqueeze(0), layer, measure_layer)

    return 1.0 - F.cosine_similarity(ref.float(), pert.float(), dim=-1).squeeze()


def optimize_triple(model, ctx, a, angle_rad, layer, measure_layer, device, d_model,
                    lam=1.0, n_steps=500, lr=0.01, n_restarts=3,
                    prev_dirs=None, diversity_weight=0.5):
    """Optimize 3 directions to maximize individual plateau-breaking minus combination penalty.

    Args:
        prev_dirs: list of unit vectors from previous runs (for cross-run diversity)
        diversity_weight: weight on the diversity penalty against prev_dirs
    """
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = a.to(device=device, dtype=model_dtype)
    a_unit = a_d / a_d.norm()

    K = 3
    # All subsets of size >= 2
    subset_indices = []
    for size in range(2, K + 1):
        subset_indices.extend(combinations(range(K), size))
    # subset_indices: [(0,1), (0,2), (1,2), (0,1,2)]

    best_obj = -float('inf')
    best_dirs = None
    best_history = None

    for restart in range(n_restarts):
        # Random init: K directions, each orthogonal to activation
        dirs = []
        for _ in range(K):
            d = torch.randn(d_model, device=device, dtype=model_dtype)
            d = d - (d @ a_unit) * a_unit
            d = F.normalize(d, dim=0).clone().detach().requires_grad_(True)
            dirs.append(d)

        optimizer = torch.optim.Adam(dirs, lr=lr)
        history = []

        for step in range(n_steps):
            optimizer.zero_grad()

            # Clean directions: project out activation component, normalize
            clean = []
            for d in dirs:
                dc = d - (d @ a_unit) * a_unit
                dc = dc / dc.norm()
                clean.append(dc)

            # Individual cosine distances
            c_individual = []
            for dc in clean:
                c_individual.append(cos_dist(model, ctx_d, a_d, dc, angle_rad,
                                             layer, measure_layer, device))

            # Combination cosine distances
            c_combos = []
            for subset in subset_indices:
                d_combo = sum(clean[i] for i in subset)
                d_combo = d_combo / d_combo.norm()
                c_combos.append(cos_dist(model, ctx_d, a_d, d_combo, angle_rad,
                                         layer, measure_layer, device))

            # Objective: maximize individuals, penalize combinations
            obj = sum(c_individual) - lam * sum(c_combos)

            # Cross-run diversity penalty
            if prev_dirs:
                cos2_vals = []
                for dc in clean:
                    for pd in prev_dirs:
                        cos2_vals.append((dc @ pd.to(device=device, dtype=model_dtype)).pow(2))
                max_cos2 = torch.stack(cos2_vals).max()
                diversity = 1.0 - max_cos2
                obj = obj * diversity

            loss = -obj
            loss.backward()
            optimizer.step()

            # Project out activation component after step
            with torch.no_grad():
                for d in dirs:
                    d.data -= (d.data @ a_unit) * a_unit

            # Logging
            c_ind_vals = [c.item() for c in c_individual]
            c_combo_vals = [c.item() for c in c_combos]

            history.append({
                'c_individual': c_ind_vals,
                'c_combos': c_combo_vals,
                'obj': obj.item(),
            })

            if step % 100 == 0 or step == n_steps - 1:
                ind_str = ', '.join(f'{c:.4f}' for c in c_ind_vals)
                combo_str = ', '.join(f'{c:.4f}' for c in c_combo_vals)
                div_str = f", div={diversity.item():.3f}" if prev_dirs else ""
                print(f"    restart {restart+1}/{n_restarts}, step {step}: "
                      f"ind=[{ind_str}], combos=[{combo_str}], "
                      f"obj={obj.item():.4f}{div_str}")

        if history[-1]['obj'] > best_obj:
            best_obj = history[-1]['obj']
            with torch.no_grad():
                best_dirs = []
                for d in dirs:
                    dc = d - (d @ a_unit) * a_unit
                    best_dirs.append(F.normalize(dc, dim=0).clone())
            best_history = history

    return best_dirs, best_obj, best_history


def sweep_direction(model, ctx, a, d, layer, measure_layer, device, angles_deg):
    """Sweep perturbation angles for a single direction. Returns cos_dist and KL arrays."""
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = a.to(device=device, dtype=model_dtype)
    R = a_d.norm()
    a_unit = a_d / R

    d_d = d.to(device=device, dtype=model_dtype)
    d_orth = d_d - (d_d @ a_unit) * a_unit
    d_orth = F.normalize(d_orth, dim=-1)

    cos_arr = np.zeros(len(angles_deg))
    kl_arr = np.zeros(len(angles_deg))

    with torch.no_grad():
        # Reference at measure layer
        ref_meas = forward_from_layer_to_layer(model, ctx_d, a_d[None, None], layer, measure_layer)
        # Reference logits (full forward)
        ref_final = forward_from_layer_to_final(model, ctx_d, a_d[None, None], layer)
        ref_lp = F.log_softmax(
            _get_lm_head(model)(_get_final_norm(model)(ref_final[:, None]).squeeze(1)).float(), dim=-1)

        for j, deg in enumerate(angles_deg):
            theta = np.deg2rad(deg)
            p = R * (np.cos(theta) * a_unit + np.sin(theta) * d_orth)

            pf_meas = forward_from_layer_to_layer(model, ctx_d, p[None, None], layer, measure_layer)
            cos_arr[j] = 1.0 - F.cosine_similarity(ref_meas.float(), pf_meas.float(), dim=-1).item()

            pf_final = forward_from_layer_to_final(model, ctx_d, p[None, None], layer)
            plp = F.log_softmax(
                _get_lm_head(model)(_get_final_norm(model)(pf_final[:, None]).squeeze(1)).float(), dim=-1)
            kl_arr[j] = (ref_lp.exp() * (ref_lp - plp)).sum().item()

    return cos_arr, kl_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--measure_layer', type=int, default=14)
    parser.add_argument('--act_idx', type=int, default=137)
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--angle', type=float, default=10.0)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--n_runs', type=int, default=4)
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--n_restarts', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_random_dirs', type=int, default=20)
    parser.add_argument('--diversity_weight', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optimize_only', action='store_true',
                        help='Skip per-run perturbation sweeps and plots (for large n_runs)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.output_dir:
        results_dir = args.output_dir
    else:
        results_dir = (f'results/ec_triples/{args.model}_L{args.layer}_meas{args.measure_layer}'
                       f'_act{args.act_idx}_angle{args.angle}_lam{args.lam}')
    os.makedirs(results_dir, exist_ok=True)

    print(f"Config: model={args.model}, layer={args.layer}, measure_layer={args.measure_layer}, "
          f"angle={args.angle}°, λ={args.lam}, n_runs={args.n_runs}")

    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    print(f"Model: {args.model}, d_model={d_model}, n_layers={n_layers}")
    assert args.measure_layer < n_layers, f"measure_layer={args.measure_layer} >= n_layers={n_layers}"

    # Generate data and extract activation
    n_games = max(args.act_idx + 1, 50)
    token_ids_list = generate_chess_games(n_games, args.seq_len, seed=args.seed)

    batch_size = 32
    all_ctx, all_act = [], []
    for i in range(0, n_games, batch_size):
        bt = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            out = model(bt, output_hidden_states=True)
            h = out.hidden_states[args.layer + 1]
        all_ctx.append(h[:, :-1, :].float().cpu())
        all_act.append(h[:, -1, :].float().cpu())

    ctx = torch.cat(all_ctx, 0)[args.act_idx:args.act_idx+1].to(device)
    act = torch.cat(all_act, 0)[args.act_idx].to(device)
    print(f"Activation norm: {act.norm():.2f}, shape: {act.shape}")

    angle_rad = np.deg2rad(args.angle)

    # Run optimizations with cross-run diversity
    all_runs = []
    prev_dirs = []  # accumulates across runs

    for run_idx in range(args.n_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx+1}/{args.n_runs} (prev_dirs: {len(prev_dirs)})")
        print(f"{'='*60}")
        torch.manual_seed(args.seed + run_idx * 1000)

        t0 = time.time()
        dirs, obj, history = optimize_triple(
            model, ctx, act, angle_rad, args.layer, args.measure_layer,
            device, d_model, lam=args.lam, n_steps=args.n_steps, lr=args.lr,
            n_restarts=args.n_restarts,
            prev_dirs=prev_dirs if prev_dirs else None,
            diversity_weight=args.diversity_weight)
        elapsed = time.time() - t0

        h = history[-1]
        print(f"  Final: ind={h['c_individual']}, combos={h['c_combos']}, "
              f"obj={h['obj']:.4f}, time={elapsed:.1f}s")

        all_runs.append({'dirs': dirs, 'obj': obj, 'history': history})

        # Add found directions to prev_dirs for cross-run diversity
        for d in dirs:
            prev_dirs.append(F.normalize(d, dim=0).detach().cpu())

        # Checkpoint after each run (spot instance safety)
        save_data = {
            'args': vars(args),
            'runs': [{'dirs': [d.cpu() for d in r['dirs']], 'obj': r['obj'], 'history': r['history']}
                     for r in all_runs],
        }
        pickle.dump(save_data, open(os.path.join(results_dir, 'ec_triples.pkl'), 'wb'))
        print(f"  Checkpointed {len(all_runs)} runs to {results_dir}/ec_triples.pkl")

    if args.optimize_only:
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION COMPLETE — {args.n_runs} runs, {len(prev_dirs)} directions total")
        print(f"Saved to: {os.path.join(results_dir, 'ec_triples.pkl')}")
        print(f"{'='*60}")
        return

    # Perturbation sweeps
    print(f"\n{'='*60}")
    print("PERTURBATION SWEEPS")
    print(f"{'='*60}")

    angles_deg = np.linspace(0, 60, 121)[1:]

    # Random baseline
    print("Sweeping random directions...")
    rand_cos = np.zeros((args.n_random_dirs, len(angles_deg)))
    rand_kl = np.zeros((args.n_random_dirs, len(angles_deg)))
    for i in range(args.n_random_dirs):
        d_rand = torch.randn(d_model, device=device)
        rand_cos[i], rand_kl[i] = sweep_direction(
            model, ctx, act, d_rand, args.layer, args.measure_layer, device, angles_deg)

    # Sweep each run's directions and their combinations
    K = 3
    subset_labels = {1: [], 2: [], 3: []}  # group by subset size

    for run_idx, run in enumerate(all_runs):
        dirs = run['dirs']
        print(f"\nSweeping run {run_idx+1} directions...")

        # Individual directions
        for i in range(K):
            cos_arr, kl_arr = sweep_direction(
                model, ctx, act, dirs[i], args.layer, args.measure_layer, device, angles_deg)
            subset_labels[1].append({
                'run': run_idx, 'indices': (i,),
                'cos': cos_arr, 'kl': kl_arr, 'label': f'R{run_idx+1} d{i+1}'})

        # Pairs
        for combo in combinations(range(K), 2):
            d_combo = F.normalize(sum(dirs[i] for i in combo), dim=0)
            cos_arr, kl_arr = sweep_direction(
                model, ctx, act, d_combo, args.layer, args.measure_layer, device, angles_deg)
            idx_str = '+'.join(str(i+1) for i in combo)
            subset_labels[2].append({
                'run': run_idx, 'indices': combo,
                'cos': cos_arr, 'kl': kl_arr, 'label': f'R{run_idx+1} d{idx_str}'})

        # Triple
        d_all = F.normalize(sum(dirs[i] for i in range(K)), dim=0)
        cos_arr, kl_arr = sweep_direction(
            model, ctx, act, d_all, args.layer, args.measure_layer, device, angles_deg)
        subset_labels[3].append({
            'run': run_idx, 'indices': tuple(range(K)),
            'cos': cos_arr, 'kl': kl_arr, 'label': f'R{run_idx+1} d1+2+3'})

    # Save sweep data
    sweep_data = {
        'angles_deg': angles_deg,
        'rand_cos': rand_cos, 'rand_kl': rand_kl,
        'subset_labels': subset_labels,
    }
    pickle.dump(sweep_data, open(os.path.join(results_dir, 'sweeps.pkl'), 'wb'))

    # Plot 1: Summary — average across runs by subset size
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {1: 'red', 2: 'orange', 3: 'green'}
    labels = {1: 'Single direction', 2: 'Pair combination', 3: 'Triple combination'}

    for ax, metric in zip(axes, ['cos', 'kl']):
        ylabel = 'Cosine distance' if metric == 'cos' else 'KL divergence'

        # Random baseline
        rd = rand_cos if metric == 'cos' else rand_kl
        ax.fill_between(angles_deg, np.percentile(rd, 25, axis=0),
                        np.percentile(rd, 75, axis=0), color='gray', alpha=0.15)
        ax.plot(angles_deg, np.median(rd, axis=0), color='gray', lw=2, ls='--', label='Random')

        # By subset size
        for size in [1, 2, 3]:
            all_curves = np.array([s[metric] for s in subset_labels[size]])
            median = np.median(all_curves, axis=0)
            q25, q75 = np.percentile(all_curves, [25, 75], axis=0)
            ax.fill_between(angles_deg, q25, q75, color=colors[size], alpha=0.15)
            ax.plot(angles_deg, median, color=colors[size], lw=2, label=labels[size])

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 60)

    fig.suptitle(f'EC triple optimization: {args.model} L{args.layer}→L{args.measure_layer}, '
                 f'θ={args.angle}°, λ={args.lam}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(results_dir, 'summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")

    # Plot 2: Per-run detail
    fig, axes = plt.subplots(args.n_runs, 2, figsize=(14, 5 * args.n_runs))
    if args.n_runs == 1:
        axes = axes[np.newaxis, :]

    for run_idx in range(args.n_runs):
        for ax_idx, metric in enumerate(['cos', 'kl']):
            ax = axes[run_idx, ax_idx]
            ylabel = 'Cosine distance' if metric == 'cos' else 'KL divergence'

            rd = rand_cos if metric == 'cos' else rand_kl
            ax.fill_between(angles_deg, np.percentile(rd, 25, axis=0),
                            np.percentile(rd, 75, axis=0), color='gray', alpha=0.15)
            ax.plot(angles_deg, np.median(rd, axis=0), color='gray', lw=2, ls='--', label='Random')

            # This run's individual directions
            for s in subset_labels[1]:
                if s['run'] == run_idx:
                    ax.plot(angles_deg, s[metric], color='red', lw=1.5, alpha=0.7, label=s['label'])

            # This run's pair combos
            for s in subset_labels[2]:
                if s['run'] == run_idx:
                    ax.plot(angles_deg, s[metric], color='orange', lw=1.5, alpha=0.7,
                            ls='--', label=s['label'])

            # This run's triple
            for s in subset_labels[3]:
                if s['run'] == run_idx:
                    ax.plot(angles_deg, s[metric], color='green', lw=2, label=s['label'])

            ax.set_xlabel('Perturbation angle (degrees)')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 60)
            ax.set_title(f'Run {run_idx+1}')

    fig.suptitle(f'EC triples per-run: {args.model} L{args.layer}→L{args.measure_layer}, '
                 f'θ={args.angle}°, λ={args.lam}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(results_dir, 'per_run.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY: median cosine distance at θ=10°")
    print(f"{'='*60}")
    target_idx = np.argmin(np.abs(angles_deg - args.angle))
    for size in [1, 2, 3]:
        vals = [s['cos'][target_idx] for s in subset_labels[size]]
        print(f"  {labels[size]}: median={np.median(vals):.4f}, "
              f"range=[{min(vals):.4f}, {max(vals):.4f}] (n={len(vals)})")
    rand_vals = rand_cos[:, target_idx]
    print(f"  Random: median={np.median(rand_vals):.4f}, "
          f"range=[{rand_vals.min():.4f}, {rand_vals.max():.4f}] (n={len(rand_vals)})")


if __name__ == '__main__':
    main()
