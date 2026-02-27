#!/usr/bin/env python3
"""
Optimize pairs of directions that individually break the plateau but whose
combination does not — directly encoding the error correction prediction.

Objective: maximize c(d1, θ) + c(d2, θ) - λ * c(normalize(d1+d2), θ)

where c(d, θ) = cosine distance at final layer when perturbing at layer L by angle θ along d.
"""
import argparse, os, sys, time, pickle
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_final
from src.data import generate_chess_games


def forward_to_layer(model, ctx, point, start_layer, end_layer):
    """Forward from start_layer to end_layer (inclusive). Like forward_from_layer_to_final
    but stops at a specific layer instead of running all remaining blocks."""
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


def cos_dist_at_final(model, ctx, a, d, angle_rad, layer, device, measure_layer=None):
    """Compute cosine distance at measure_layer for perturbation along d at given angle."""
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = a.to(device=device, dtype=model_dtype)
    R = a_d.norm()
    a_unit = a_d / R

    # Project d orthogonal to activation and normalize
    d_orth = d - (d @ a_unit) * a_unit
    d_orth = d_orth / d_orth.norm()

    # Perturbed activation (on sphere)
    a_pert = R * (np.cos(angle_rad) * a_unit + np.sin(angle_rad) * d_orth)

    # Forward both
    if measure_layer is not None:
        ref_final = forward_to_layer(model, ctx_d, a_d.unsqueeze(0).unsqueeze(0), layer, measure_layer)
        pert_final = forward_to_layer(model, ctx_d, a_pert.unsqueeze(0).unsqueeze(0), layer, measure_layer)
    else:
        ref_final = forward_from_layer_to_final(model, ctx_d, a_d.unsqueeze(0).unsqueeze(0), layer)
        pert_final = forward_from_layer_to_final(model, ctx_d, a_pert.unsqueeze(0).unsqueeze(0), layer)

    # Cosine distance
    cos_sim = F.cosine_similarity(ref_final.float(), pert_final.float(), dim=-1)
    return 1.0 - cos_sim.squeeze()


def optimize_pair(model, ctx, a, angle_rad, layer, device, d_model,
                  lam=1.0, n_steps=500, lr=0.01, n_restarts=3, measure_layer=None,
                  prev_dirs=None, diversity_power=1):
    """Optimize a pair (d1, d2) to maximize c(d1) + c(d2) - λ*c(combo)."""
    a_d = a.to(device)
    a_unit = a_d / a_d.norm()

    best_obj = -float('inf')
    best_d1 = best_d2 = None
    best_history = None

    for restart in range(n_restarts):
        # Random init
        d1 = torch.randn(d_model, device=device)
        d1 = d1 - (d1 @ a_unit) * a_unit
        d1 = F.normalize(d1, dim=0).clone().detach().requires_grad_(True)

        d2 = torch.randn(d_model, device=device)
        d2 = d2 - (d2 @ a_unit) * a_unit
        d2 = F.normalize(d2, dim=0).clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([d1, d2], lr=lr)
        history = []

        for step in range(n_steps):
            optimizer.zero_grad()

            # Project out activation component and normalize
            d1_clean = d1 - (d1 @ a_unit) * a_unit
            d1_clean = d1_clean / d1_clean.norm()
            d2_clean = d2 - (d2 @ a_unit) * a_unit
            d2_clean = d2_clean / d2_clean.norm()

            # Combo direction
            d_combo = d1_clean + d2_clean
            d_combo = d_combo / d_combo.norm()

            # Cosine distances
            c1 = cos_dist_at_final(model, ctx, a, d1_clean, angle_rad, layer, device, measure_layer)
            c2 = cos_dist_at_final(model, ctx, a, d2_clean, angle_rad, layer, device, measure_layer)
            c_combo = cos_dist_at_final(model, ctx, a, d_combo, angle_rad, layer, device, measure_layer)

            obj = c1 + c2 - lam * c_combo

            # Diversity penalty against previously found directions
            if prev_dirs:
                cos2_vals = []
                for pd in prev_dirs:
                    cos2_vals.append((d1_clean @ pd).pow(2))
                    cos2_vals.append((d2_clean @ pd).pow(2))
                max_cos2 = torch.stack(cos2_vals).max()
                diversity = (1.0 - max_cos2) ** diversity_power
                obj = obj * diversity

            loss = -obj
            loss.backward()
            optimizer.step()

            # Project out activation component after step
            with torch.no_grad():
                d1.data -= (d1.data @ a_unit) * a_unit
                d2.data -= (d2.data @ a_unit) * a_unit

            cos12 = abs(F.normalize(d1.data - (d1.data @ a_unit) * a_unit, dim=0) @
                       F.normalize(d2.data - (d2.data @ a_unit) * a_unit, dim=0)).item()

            history.append({
                'c1': c1.item(), 'c2': c2.item(), 'c_combo': c_combo.item(),
                'obj': obj.item(), 'cos12': cos12
            })

            if step % 100 == 0 or step == n_steps - 1:
                div_str = f", div={diversity.item():.3f}" if prev_dirs else ""
                print(f"    restart {restart+1}/{n_restarts}, step {step}: "
                      f"c1={c1.item():.4f}, c2={c2.item():.4f}, "
                      f"c_combo={c_combo.item():.4f}, obj={obj.item():.4f}, "
                      f"cos(d1,d2)={cos12:.3f}{div_str}")

        if history[-1]['obj'] > best_obj:
            best_obj = history[-1]['obj']
            with torch.no_grad():
                best_d1 = F.normalize(d1 - (d1 @ a_unit) * a_unit, dim=0).clone()
                best_d2 = F.normalize(d2 - (d2 @ a_unit) * a_unit, dim=0).clone()
            best_history = history

    return best_d1, best_d2, best_obj, best_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=11)
    parser.add_argument('--act_idx', type=int, default=137)
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--angle', type=float, default=10.0)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--n_pairs', type=int, default=3)
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--n_restarts', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--measure_layer', type=int, default=None,
                        help='Layer to measure cos/L2 at (default: final block)')
    parser.add_argument('--n_random_dirs', type=int, default=20)
    parser.add_argument('--diversity_power', type=int, default=1,
                        help='Exponent for diversity penalty against previous pairs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ml_str = f'_meas{args.measure_layer}' if args.measure_layer is not None else ''
    dp_str = f'_dp{args.diversity_power}' if args.diversity_power > 1 else ''
    results_dir = (f'results/ec_pairs/{args.model}_layer{args.layer}'
                   f'_act{args.act_idx}_angle{args.angle}_lam{args.lam}{ml_str}{dp_str}')
    os.makedirs(results_dir, exist_ok=True)

    print(f"Config: model={args.model}, layer={args.layer}, angle={args.angle}°, "
          f"λ={args.lam}, n_pairs={args.n_pairs}")

    model, tokenizer, device = load_model(args.model)
    d_model = model.config.n_embd

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
    print(f"Activation norm: {act.norm():.2f}")

    angle_rad = np.deg2rad(args.angle)

    # Optimize pairs
    pairs = []
    prev_dirs = []
    for p_idx in range(args.n_pairs):
        print(f"\n{'='*60}")
        print(f"PAIR {p_idx+1}/{args.n_pairs} (prev_dirs: {len(prev_dirs)})")
        print(f"{'='*60}")
        torch.manual_seed(args.seed + p_idx * 100)
        t0 = time.time()
        d1, d2, obj, history = optimize_pair(
            model, ctx, act, angle_rad, args.layer, device, d_model,
            lam=args.lam, n_steps=args.n_steps, lr=args.lr,
            n_restarts=args.n_restarts, measure_layer=args.measure_layer,
            prev_dirs=prev_dirs if prev_dirs else None,
            diversity_power=args.diversity_power)
        elapsed = time.time() - t0
        h = history[-1]
        print(f"  Final: c1={h['c1']:.4f}, c2={h['c2']:.4f}, "
              f"c_combo={h['c_combo']:.4f}, obj={h['obj']:.4f}, "
              f"cos(d1,d2)={h['cos12']:.3f}, time={elapsed:.1f}s")
        pairs.append({'d1': d1, 'd2': d2, 'obj': obj, 'history': history})
        # Add found directions to prev_dirs for diversity penalty
        prev_dirs.append(F.normalize(d1, dim=0).detach())
        prev_dirs.append(F.normalize(d2, dim=0).detach())

    # Save results
    pickle.dump(pairs, open(os.path.join(results_dir, 'ec_pairs.pkl'), 'wb'))

    # Now do perturbation sweep for each pair
    print(f"\n{'='*60}")
    print("PERTURBATION SWEEPS")
    print(f"{'='*60}")

    angles_deg = np.linspace(0, 60, 121)[1:]
    angles_sweep = np.deg2rad(angles_deg)

    def sweep(d):
        kl_arr = np.zeros(len(angles_deg))
        cos_arr = np.zeros(len(angles_deg))
        l2_arr = np.zeros(len(angles_deg))
        model_dtype = next(model.parameters()).dtype
        a_d = act.to(device=device, dtype=model_dtype)
        R = a_d.norm(); a_unit = a_d / R
        d_d = d.to(device=device, dtype=model_dtype)
        d_orth = d_d - (d_d @ a_unit) * a_unit
        d_orth = F.normalize(d_orth, dim=-1)
        ctx_d = ctx.to(device=device, dtype=model_dtype)
        from src.model import _get_final_norm, _get_lm_head
        with torch.no_grad():
            # KL always uses final layer
            ref_final = forward_from_layer_to_final(model, ctx_d, a_d[None, None], args.layer)
            ref_lp = F.log_softmax(
                _get_lm_head(model)(_get_final_norm(model)(ref_final[:, None]).squeeze(1)).float(), dim=-1)
            # Cos/L2 use measure_layer if specified
            if args.measure_layer is not None:
                ref_meas = forward_to_layer(model, ctx_d, a_d[None, None], args.layer, args.measure_layer)
            else:
                ref_meas = ref_final
        for j, theta in enumerate(angles_sweep):
            p = R * (np.cos(theta) * a_unit + np.sin(theta) * d_orth)
            with torch.no_grad():
                pf_final = forward_from_layer_to_final(model, ctx_d, p[None, None], args.layer)
                if args.measure_layer is not None:
                    pf_meas = forward_to_layer(model, ctx_d, p[None, None], args.layer, args.measure_layer)
                else:
                    pf_meas = pf_final
                cos_arr[j] = 1.0 - F.cosine_similarity(ref_meas.float(), pf_meas.float(), dim=-1).item()
                l2_arr[j] = (pf_meas - ref_meas).float().norm().item()
                plp = F.log_softmax(
                    _get_lm_head(model)(_get_final_norm(model)(pf_final[:, None]).squeeze(1)).float(), dim=-1)
                kl_arr[j] = (ref_lp.exp() * (ref_lp - plp)).sum().item()
        return kl_arr, cos_arr, l2_arr

    # Random baseline
    rand_kl = np.zeros((args.n_random_dirs, len(angles_deg)))
    rand_cos = np.zeros((args.n_random_dirs, len(angles_deg)))
    rand_l2 = np.zeros((args.n_random_dirs, len(angles_deg)))
    for i in range(args.n_random_dirs):
        rand_kl[i], rand_cos[i], rand_l2[i] = sweep(torch.randn(d_model))

    # Plot: one row per pair, 3 columns (KL, cosine distance, L2)
    fig, axes = plt.subplots(args.n_pairs, 3, figsize=(20, 5 * args.n_pairs))
    if args.n_pairs == 1:
        axes = axes[np.newaxis, :]

    for p_idx, pair in enumerate(pairs):
        d1, d2 = pair['d1'], pair['d2']
        d_combo = F.normalize(d1 + d2, dim=0)
        cos12 = abs(F.normalize(d1, dim=0) @ F.normalize(d2, dim=0)).item()

        kl1, cos1, l21 = sweep(d1)
        kl2, cos2, l22 = sweep(d2)
        kl_c, cos_c, l2_c = sweep(d_combo)

        h = pair['history'][-1]
        for ax_idx, (metric, ylabel) in enumerate([
            ('kl', 'KL divergence'), ('cos', 'Cosine distance'), ('l2', 'L2 distance')
        ]):
            ax = axes[p_idx, ax_idx]
            rd = rand_kl if metric == 'kl' else (rand_cos if metric == 'cos' else rand_l2)
            v1 = kl1 if metric == 'kl' else (cos1 if metric == 'cos' else l21)
            v2 = kl2 if metric == 'kl' else (cos2 if metric == 'cos' else l22)
            vc = kl_c if metric == 'kl' else (cos_c if metric == 'cos' else l2_c)

            ax.fill_between(angles_deg, np.percentile(rd, 25, axis=0),
                            np.percentile(rd, 75, axis=0), color='gray', alpha=0.15)
            ax.plot(angles_deg, np.median(rd, axis=0), color='gray', lw=2, ls='--', label='Random')
            ax.plot(angles_deg, v1, color='blue', lw=2, label='d1')
            ax.plot(angles_deg, v2, color='red', lw=2, label='d2')
            ax.plot(angles_deg, vc, color='green', lw=2, label='d1+d2 (combo)')
            ax.set_xlabel('Perturbation angle (degrees)')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 60)
            ax.set_title(f'Pair {p_idx+1}: cos(d1,d2)={cos12:.3f}, '
                        f'c1={h["c1"]:.3f}, c2={h["c2"]:.3f}, c_combo={h["c_combo"]:.3f}')

    fig.suptitle(f'Error correction pairs (θ={args.angle}°, λ={args.lam})', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(results_dir, 'ec_pairs_sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
