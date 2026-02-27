#!/usr/bin/env python3
"""Sequential feature discovery: find directions one at a time with Gram-Schmidt.

Finds plateau-breaking directions sequentially. Each new direction is projected
away from all previously found directions during optimization, forcing the
optimizer to find genuinely new features.

Uses cosine distance at a measure layer (not KL at logits).

Usage:
    python scripts/perturbation/sequential_feature_discovery.py \
        --model chess_16L --layer 2 --measure_layer 14 \
        --alpha 10.0 --max_directions 128
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
                           n_steps=300, lr=0.01, n_restarts=3, cap=None):
    """Find one direction maximizing cosine distance, orthogonal to all previous.

    Returns: best_direction, best_cd, best_history
    """
    model_dtype = next(model.parameters()).dtype
    R = act.float().norm()
    a_unit = act.float() / R
    theta = np.deg2rad(angle_deg)

    best_cd = -1.0
    best_dir = None
    best_history = None

    for restart in range(n_restarts):
        # Initialize random, project away from previous
        direction = torch.randn(d_model, device=device, dtype=torch.float32)
        if prev_directions:
            for pd in prev_directions:
                direction = direction - (direction @ pd) * pd
        direction = F.normalize(direction, dim=0)
        direction = torch.nn.Parameter(direction)

        optimizer = torch.optim.Adam([direction], lr=lr)
        history = []

        for step in range(n_steps):
            optimizer.zero_grad()

            # Normalize and project away from previous directions
            d_normed = direction / direction.norm()
            if prev_directions:
                for pd in prev_directions:
                    d_normed = d_normed - (d_normed @ pd) * pd
                d_normed = d_normed / (d_normed.norm() + 1e-10)

            # Orthogonalize w.r.t. activation direction
            proj = (d_normed @ a_unit)
            d_orth = d_normed - proj * a_unit
            d_orth = F.normalize(d_orth, dim=0)

            # Spherical perturbation
            perturbed = R * (np.cos(theta) * a_unit + np.sin(theta) * d_orth)
            perturbed = perturbed.to(model_dtype)

            # Forward to measure layer
            p_out = forward_checkpointed(
                model, ctx, perturbed[None, None], layer, measure_layer)

            # Cosine distance
            cd = 1.0 - F.cosine_similarity(ref_out, p_out.float(), dim=-1)
            cd_val = cd.item()

            if cap is not None:
                loss = -torch.clamp(cd, max=cap)
            else:
                loss = -cd
            loss.backward()
            optimizer.step()

            # Hard projection after step
            with torch.no_grad():
                if prev_directions:
                    for pd in prev_directions:
                        direction.data -= (direction.data @ pd) * pd
                direction.data = F.normalize(direction.data, dim=0)

            history.append(cd_val)

        final_cd = history[-1]
        if final_cd > best_cd:
            best_cd = final_cd
            with torch.no_grad():
                d_final = direction.data.clone()
                if prev_directions:
                    for pd in prev_directions:
                        d_final -= (d_final @ pd) * pd
                    d_final = F.normalize(d_final, dim=0)
                best_dir = d_final
            best_history = history

    return best_dir, best_cd, best_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--measure_layer', type=int, default=14)
    parser.add_argument('--act_idx', type=int, default=137)
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=10.0,
                        help='Perturbation angle in degrees')
    parser.add_argument('--cap', type=float, default=None,
                        help='Cap cosine distance (None = no cap)')
    parser.add_argument('--max_directions', type=int, default=128)
    parser.add_argument('--stop_fraction', type=float, default=0.1,
                        help='Stop when cd drops below this fraction of direction 1')
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_dir = (f"results/sequential/{args.model}_L{args.layer}"
                   f"_meas{args.measure_layer}_alpha{args.alpha}")
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(results_dir, 'checkpoint.pkl')

    # Load model
    model, tokenizer, device = load_model(args.model)
    model_dtype = next(model.parameters()).dtype
    d_model = model._plateau_info['d_model']
    print(f"Model: {args.model}, d_model={d_model}, device={device}")

    # Extract activation
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

    print(f"Activation {args.act_idx}, norm={act.float().norm():.2f}")

    # Reference output
    with torch.no_grad():
        ref_out = forward_from_layer_to_layer(
            model, ctx, act.unsqueeze(0).unsqueeze(0),
            args.layer, args.measure_layer).float()

    # Resume from checkpoint
    directions = []
    cos_dists = []
    start_idx = 0

    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        ckpt = pickle.load(open(checkpoint_path, 'rb'))
        directions = [d.to(device) for d in ckpt['directions']]
        cos_dists = ckpt['cos_dists']
        start_idx = len(directions)
        print(f"  Loaded {start_idx} directions")

    # Sequential optimization
    print(f"\n{'='*60}")
    print(f"SEQUENTIAL FEATURE DISCOVERY")
    print(f"  alpha={args.alpha}°, max_directions={args.max_directions}")
    print(f"  steps={args.n_steps}, restarts={args.n_restarts}, cap={args.cap}")
    print(f"  stop_fraction={args.stop_fraction}")
    print(f"{'='*60}")

    stop_threshold = None
    stopped_early = False

    for d_idx in range(start_idx, args.max_directions):
        t0 = time.time()
        d, cd, history = optimize_one_direction(
            model, ctx, act, ref_out, args.alpha, args.layer,
            args.measure_layer, device, d_model,
            prev_directions=directions if directions else None,
            n_steps=args.n_steps, lr=args.lr,
            n_restarts=args.n_restarts, cap=args.cap)

        directions.append(d)
        cos_dists.append(cd)
        elapsed = time.time() - t0

        # Set stop threshold from first direction
        if d_idx == 0:
            stop_threshold = cos_dists[0] * args.stop_fraction

        print(f"  D{d_idx+1:>3}: cd={cd:.4f} ({elapsed:.1f}s)"
              + (f"  [stop_thresh={stop_threshold:.4f}]" if d_idx == 0 else ""))

        # Check stopping
        if cd < stop_threshold:
            print(f"\n  *** STOPPING: cd {cd:.4f} < threshold {stop_threshold:.4f} ***")
            stopped_early = True
            break

        # Checkpoint every 10 directions
        if (d_idx + 1) % 10 == 0:
            ckpt = {
                'directions': [d.cpu() for d in directions],
                'cos_dists': cos_dists,
                'args': vars(args),
                'stopped_early': False,
            }
            pickle.dump(ckpt, open(checkpoint_path, 'wb'))
            print(f"  Checkpointed {len(directions)} directions")

    n_found = len(directions)
    print(f"\n{'='*60}")
    print(f"FOUND {n_found} DIRECTIONS"
          + (" (stopped early)" if stopped_early else " (hit max)"))
    print(f"{'='*60}")

    # Orthogonality check
    all_dirs = torch.stack(directions)
    G = (all_dirs @ all_dirs.T).abs()
    G.fill_diagonal_(0)
    print(f"Max |cos_sim|: {G.max().item():.6f}")
    print(f"Mean |cos_sim|: {G.sum().item() / (n_found * (n_found - 1)):.6f}")
    print(f"cd > 0.1: {sum(1 for c in cos_dists if c > 0.1)}/{n_found}")
    print(f"cd > 0.01: {sum(1 for c in cos_dists if c > 0.01)}/{n_found}")

    # Save
    save_data = {
        'directions': [d.cpu() for d in directions],
        'cos_dists': cos_dists,
        'gram_matrix': G.cpu().numpy(),
        'args': vars(args),
        'stopped_early': stopped_early,
        'n_found': n_found,
    }
    pickle.dump(save_data, open(os.path.join(results_dir, 'features.pkl'), 'wb'))

    # Also save checkpoint
    ckpt = {
        'directions': [d.cpu() for d in directions],
        'cos_dists': cos_dists,
        'args': vars(args),
        'stopped_early': stopped_early,
    }
    pickle.dump(ckpt, open(checkpoint_path, 'wb'))

    # Plot reward decay
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, n_found + 1), cos_dists, 'o-', markersize=3, color='steelblue')
    if stop_threshold is not None:
        ax.axhline(stop_threshold, color='red', ls='--', lw=1.5,
                    label=f'Stop threshold ({args.stop_fraction:.0%} of D1)')
    ax.set_xlabel('Direction index')
    ax.set_ylabel('Cosine distance at optimization angle')
    ax.set_title(f'Sequential discovery: {args.model} L{args.layer}→L{args.measure_layer}, '
                 f'alpha={args.alpha}°')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cd_decay.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved to {results_dir}/")


if __name__ == '__main__':
    main()
