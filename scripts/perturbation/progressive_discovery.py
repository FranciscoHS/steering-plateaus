#!/usr/bin/env python3
"""Progressive feature discovery: sweep angles from start to end, full orthogonality.

Runs sequential MELBO at progressively larger perturbation angles. At each angle,
finds all features until discovery stalls, then bumps to the next angle. All
previously found directions are carried forward as hard orthogonality constraints.

Can seed from an existing features.pkl to avoid redoing earlier angles.

Usage:
    python scripts/perturbation/progressive_discovery.py \
        --model chess_16L --layer 2 --measure_layer 14 \
        --start_alpha 8 --end_alpha 20 --alpha_step 1

    # Seed from existing 8° results:
    python scripts/perturbation/progressive_discovery.py \
        --seed_bank results/sequential/chess_16L_L2_meas14_alpha8.0/features.pkl \
        --start_alpha 9 --end_alpha 20
"""
import argparse, os, sys, time, pickle
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (load_model, forward_from_layer_to_layer,
                       _get_blocks, _run_block, _is_rope_model,
                       _get_position_embeddings, _is_cat_pos_emb_model,
                       _get_cat_pos_emb)
from src.data import generate_chess_games


def forward_checkpointed(model, ctx, points, start_layer, end_layer):
    """Forward with gradient checkpointing, from start_layer to end_layer."""
    context_batch = ctx.expand(points.shape[0], -1, -1)
    hidden = torch.cat([context_batch, points], dim=1)

    blocks = _get_blocks(model)
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, hidden)
    elif _is_cat_pos_emb_model(model):
        pos_emb = _get_cat_pos_emb(model, hidden.shape[1])

    for block in blocks[start_layer + 1:end_layer + 1]:
        hidden = torch.utils.checkpoint.checkpoint(
            _run_block, block, hidden, model, pos_emb, use_reentrant=False)

    return hidden[:, -1, :]


def optimize_one_direction(model, ctx, act, ref_out, angle_deg, layer,
                           measure_layer, device, d_model, prev_directions,
                           n_steps=300, lr=0.01, n_restarts=3):
    """Find one direction maximizing cosine distance, orthogonal to all previous."""
    model_dtype = next(model.parameters()).dtype
    R = act.float().norm()
    a_unit = act.float() / R
    theta = np.deg2rad(angle_deg)

    best_cd = -1.0
    best_dir = None

    for restart in range(n_restarts):
        direction = torch.randn(d_model, device=device, dtype=torch.float32)
        if prev_directions:
            for pd in prev_directions:
                direction = direction - (direction @ pd) * pd
        direction = F.normalize(direction, dim=0)
        direction = torch.nn.Parameter(direction)

        optimizer = torch.optim.Adam([direction], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()

            d_normed = direction / direction.norm()
            if prev_directions:
                for pd in prev_directions:
                    d_normed = d_normed - (d_normed @ pd) * pd
                d_normed = d_normed / (d_normed.norm() + 1e-10)

            proj = (d_normed @ a_unit)
            d_orth = d_normed - proj * a_unit
            d_orth = F.normalize(d_orth, dim=0)

            perturbed = R * (np.cos(theta) * a_unit + np.sin(theta) * d_orth)
            perturbed = perturbed.to(model_dtype)

            p_out = forward_checkpointed(
                model, ctx, perturbed[None, None], layer, measure_layer)

            cd = 1.0 - F.cosine_similarity(ref_out, p_out.float(), dim=-1)
            loss = -cd
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if prev_directions:
                    for pd in prev_directions:
                        direction.data -= (direction.data @ pd) * pd
                direction.data = F.normalize(direction.data, dim=0)

        final_cd = cd.item()
        if final_cd > best_cd:
            best_cd = final_cd
            with torch.no_grad():
                d_final = direction.data.clone()
                if prev_directions:
                    for pd in prev_directions:
                        d_final -= (d_final @ pd) * pd
                    d_final = F.normalize(d_final, dim=0)
                best_dir = d_final

    return best_dir, best_cd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--measure_layer', type=int, default=14)
    parser.add_argument('--act_idx', type=int, default=137)
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--start_alpha', type=float, default=8.0)
    parser.add_argument('--end_alpha', type=float, default=20.0)
    parser.add_argument('--alpha_step', type=float, default=1.0)
    parser.add_argument('--max_per_angle', type=int, default=128,
                        help='Max directions to try per angle')
    parser.add_argument('--stop_cd', type=float, default=0.05,
                        help='Stop searching at an angle when cd drops below this')
    parser.add_argument('--n_stale', type=int, default=3,
                        help='Stop after this many consecutive directions below stop_cd')
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed_bank', type=str, default=None,
                        help='Path to features.pkl to seed directions from')
    parser.add_argument('--act_cache', type=str, default=None,
                        help='Path to cached activation pkl (skips game generation)')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_dir = (f"results/progressive/{args.model}_L{args.layer}"
                   f"_meas{args.measure_layer}")
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(results_dir, 'checkpoint.pkl')

    # Load model
    model, tokenizer, device = load_model(args.model)
    model_dtype = next(model.parameters()).dtype
    d_model = model._plateau_info['d_model']
    print(f"Model: {args.model}, d_model={d_model}, device={device}")

    # Load activation (from cache or by generating games)
    if args.act_cache:
        print(f"Loading cached activation from {args.act_cache}")
        act_data = pickle.load(open(args.act_cache, 'rb'))
        ctx = act_data['ctx'].to(device=device, dtype=model_dtype)
        act = act_data['act'].to(device=device, dtype=model_dtype)
        ref_out = act_data['ref_out'].to(device=device, dtype=torch.float32)
    else:
        n_games = max(args.act_idx + 1, 50)
        token_ids_list = generate_chess_games(n_games, args.seq_len, seed=args.seed)

        ctx = act = None
        batch_size = 32
        count = 0
        for i in range(0, n_games, batch_size):
            bt = torch.stack(token_ids_list[i:i+batch_size]).to(device)
            with torch.no_grad():
                out = model(bt, output_hidden_states=True)
                h = out.hidden_states[args.layer + 1]
            local_idx = args.act_idx - count
            if 0 <= local_idx < bt.shape[0]:
                ctx = h[local_idx:local_idx+1, :-1, :].to(device, dtype=model_dtype)
                act = h[local_idx, -1, :].to(device, dtype=model_dtype)
            count += bt.shape[0]
            del out, h, bt

        with torch.no_grad():
            ref_out = forward_from_layer_to_layer(
                model, ctx, act.unsqueeze(0).unsqueeze(0),
                args.layer, args.measure_layer).float()

    print(f"Activation norm={act.float().norm():.2f}")

    # Initialize from seed bank or checkpoint
    directions = []
    cos_dists = []
    alphas_found = []  # which angle each direction was found at
    completed_alphas = []

    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        ckpt = pickle.load(open(checkpoint_path, 'rb'))
        directions = [d.to(device) for d in ckpt['directions']]
        cos_dists = ckpt['cos_dists']
        alphas_found = ckpt['alphas_found']
        completed_alphas = ckpt.get('completed_alphas', [])
        print(f"  Loaded {len(directions)} directions, completed angles: {completed_alphas}")
    elif args.seed_bank:
        print(f"Seeding from {args.seed_bank}")
        seed_data = pickle.load(open(args.seed_bank, 'rb'))
        seed_dirs = seed_data['directions']
        seed_cds = seed_data['cos_dists']
        if isinstance(seed_dirs, list):
            seed_dirs = [d.to(device).float() for d in seed_dirs]
        else:
            seed_dirs = [seed_dirs[i].to(device).float() for i in range(seed_dirs.shape[0])]
        if isinstance(seed_cds, list):
            seed_cds = seed_cds
        else:
            seed_cds = seed_cds.tolist()

        # Only keep good directions (cd > stop_cd)
        for d, cd in zip(seed_dirs, seed_cds):
            if cd > args.stop_cd:
                directions.append(F.normalize(d, dim=0))
                cos_dists.append(cd)
                alphas_found.append(args.start_alpha - args.alpha_step)  # mark as "pre-existing"
        print(f"  Seeded {len(directions)} directions (cd > {args.stop_cd})")

    # Build angle schedule
    angles = []
    alpha = args.start_alpha
    while alpha <= args.end_alpha + 1e-6:
        if alpha not in completed_alphas:
            angles.append(alpha)
        alpha += args.alpha_step

    print(f"\n{'='*60}")
    print(f"PROGRESSIVE FEATURE DISCOVERY")
    print(f"  angles: {args.start_alpha}° to {args.end_alpha}° (step {args.alpha_step}°)")
    print(f"  stop_cd={args.stop_cd}, n_stale={args.n_stale}")
    print(f"  starting with {len(directions)} directions")
    print(f"{'='*60}")

    for alpha in angles:
        print(f"\n--- Alpha = {alpha}° (bank: {len(directions)} dirs) ---")
        n_before = len(directions)
        stale_count = 0

        for d_idx in range(args.max_per_angle):
            t0 = time.time()
            d, cd = optimize_one_direction(
                model, ctx, act, ref_out, alpha, args.layer,
                args.measure_layer, device, d_model,
                prev_directions=directions if directions else None,
                n_steps=args.n_steps, lr=args.lr,
                n_restarts=args.n_restarts)

            elapsed = time.time() - t0

            if cd < args.stop_cd:
                stale_count += 1
                print(f"  try {d_idx+1}: cd={cd:.4f} (stale {stale_count}/{args.n_stale}) ({elapsed:.1f}s)")
                if stale_count >= args.n_stale:
                    print(f"  => Stalled at alpha={alpha}°")
                    break
            else:
                stale_count = 0
                directions.append(d)
                cos_dists.append(cd)
                alphas_found.append(alpha)
                n_new = len(directions) - n_before
                print(f"  D{len(directions):>3} (new #{n_new}): cd={cd:.4f} ({elapsed:.1f}s)")

        n_new = len(directions) - n_before
        completed_alphas.append(alpha)
        print(f"  Alpha {alpha}°: found {n_new} new directions (total: {len(directions)})")

        # Checkpoint after each angle
        ckpt = {
            'directions': [d.cpu() for d in directions],
            'cos_dists': cos_dists,
            'alphas_found': alphas_found,
            'completed_alphas': completed_alphas,
            'args': vars(args),
        }
        pickle.dump(ckpt, open(checkpoint_path, 'wb'))
        print(f"  Checkpointed.")

    # Final save
    n_total = len(directions)
    print(f"\n{'='*60}")
    print(f"TOTAL: {n_total} DIRECTIONS")
    for alpha in sorted(set(alphas_found)):
        count = sum(1 for a in alphas_found if a == alpha)
        print(f"  alpha={alpha}°: {count} directions")
    print(f"{'='*60}")

    # Orthogonality check
    if n_total > 1:
        all_dirs = torch.stack(directions)
        G = (all_dirs @ all_dirs.T).abs()
        G.fill_diagonal_(0)
        print(f"Max |cos_sim|: {G.max().item():.6f}")
        print(f"Mean |cos_sim|: {G.sum().item() / (n_total * (n_total - 1)):.6f}")
    else:
        G = np.zeros((1, 1))

    save_data = {
        'directions': [d.cpu() for d in directions],
        'cos_dists': cos_dists,
        'alphas_found': alphas_found,
        'completed_alphas': completed_alphas,
        'gram_matrix': G.cpu().numpy() if isinstance(G, torch.Tensor) else G,
        'args': vars(args),
        'n_total': n_total,
    }
    pickle.dump(save_data, open(os.path.join(results_dir, 'features.pkl'), 'wb'))

    # Plot: cd by direction index, colored by alpha
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: cd decay colored by angle
    ax = axes[0]
    unique_alphas = sorted(set(alphas_found))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(unique_alphas)))
    for i, (cd, alpha) in enumerate(zip(cos_dists, alphas_found)):
        color_idx = unique_alphas.index(alpha)
        ax.plot(i + 1, cd, 'o', color=cmap[color_idx], markersize=4)
    ax.set_xlabel('Direction index')
    ax.set_ylabel('Cosine distance')
    ax.set_title('Progressive discovery: cd per direction')
    ax.grid(True, alpha=0.3)
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=cmap[i], markersize=6,
                              label=f'{a}°')
                       for i, a in enumerate(unique_alphas)]
    ax.legend(handles=legend_elements, title='Alpha', fontsize=8, ncol=2)

    # Right: count per angle
    ax = axes[1]
    angle_counts = [sum(1 for a in alphas_found if a == alpha) for alpha in unique_alphas]
    cumulative = np.cumsum(angle_counts)
    ax.bar(range(len(unique_alphas)), angle_counts, color='steelblue', alpha=0.7)
    ax.plot(range(len(unique_alphas)), cumulative, 'ro-', markersize=5)
    ax.set_xticks(range(len(unique_alphas)))
    ax.set_xticklabels([f'{a}°' for a in unique_alphas], rotation=45)
    ax.set_xlabel('Perturbation angle')
    ax.set_ylabel('Count')
    ax.set_title('Directions found per angle (cumulative in red)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'progressive_discovery.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved to {results_dir}/")


if __name__ == '__main__':
    main()
