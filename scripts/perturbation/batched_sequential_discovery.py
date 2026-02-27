#!/usr/bin/env python3
"""Batched sequential feature discovery with soft bank penalty.

Batch 1: Sequential discovery with strict orthogonality (same as sequential_feature_discovery.py).
Batch 2+: Sequential discovery with:
  - Strict orthogonality WITHIN the batch (Gram-Schmidt)
  - Soft penalty against bank B: subtract lambda * max_j |<d_i, b_j>| from reward
  - After convergence, discard directions with max alignment to B above threshold
  - Add survivors to bank, repeat

Usage:
    python scripts/perturbation/batched_sequential_discovery.py \
        --model chess_16L --layer 2 --measure_layer 14 \
        --alpha 8.0 --batch_size 25 --max_batches 10 \
        --bank_lambda 2.0 --bank_threshold 0.2
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
    """Forward with gradient checkpointing."""
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
                           measure_layer, device, d_model,
                           batch_prev_dirs, bank_dirs, bank_lambda,
                           n_steps=300, lr=0.01, n_restarts=3, cap=None):
    """Find one direction maximizing cosine distance.

    Hard orthogonality to batch_prev_dirs (within-batch).
    Soft penalty against bank_dirs (cross-batch).
    """
    model_dtype = next(model.parameters()).dtype
    R = act.float().norm()
    a_unit = act.float() / R
    theta = np.deg2rad(angle_deg)

    best_cd = -1.0
    best_reward = -1.0
    best_dir = None
    best_history = None

    for restart in range(n_restarts):
        direction = torch.randn(d_model, device=device, dtype=torch.float32)
        # Project away from within-batch directions (hard)
        if batch_prev_dirs:
            for pd in batch_prev_dirs:
                direction = direction - (direction @ pd) * pd
        direction = F.normalize(direction, dim=0)
        direction = torch.nn.Parameter(direction)

        optimizer = torch.optim.Adam([direction], lr=lr)
        history = []

        for step in range(n_steps):
            optimizer.zero_grad()

            # Normalize and project away from batch directions (hard)
            d_normed = direction / direction.norm()
            if batch_prev_dirs:
                for pd in batch_prev_dirs:
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

            if cap is not None:
                cd_loss = torch.clamp(cd, max=cap)
            else:
                cd_loss = cd

            # Soft bank penalty
            bank_penalty = torch.tensor(0.0, device=device)
            if bank_dirs is not None and len(bank_dirs) > 0:
                bank_stack = torch.stack(bank_dirs)  # (n_bank, d)
                alignments = (d_normed @ bank_stack.T).abs()  # (n_bank,)
                bank_penalty = bank_lambda * alignments.max()

            loss = -(cd_loss - bank_penalty)
            loss.backward()
            optimizer.step()

            # Hard projection after step (within-batch only)
            with torch.no_grad():
                if batch_prev_dirs:
                    for pd in batch_prev_dirs:
                        direction.data -= (direction.data @ pd) * pd
                direction.data = F.normalize(direction.data, dim=0)

            cd_val = cd.item()
            reward_val = (cd_loss - bank_penalty).item()
            history.append((cd_val, reward_val, bank_penalty.item()))

        final_reward = history[-1][1]
        final_cd = history[-1][0]
        if final_reward > best_reward:
            best_reward = final_reward
            best_cd = final_cd
            with torch.no_grad():
                d_final = direction.data.clone()
                if batch_prev_dirs:
                    for pd in batch_prev_dirs:
                        d_final -= (d_final @ pd) * pd
                    d_final = F.normalize(d_final, dim=0)
                best_dir = d_final
            best_history = history

    return best_dir, best_cd, best_reward, best_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--measure_layer', type=int, default=14)
    parser.add_argument('--act_idx', type=int, default=137)
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=8.0)
    parser.add_argument('--cap', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--max_batches', type=int, default=10)
    parser.add_argument('--stop_fraction', type=float, default=0.1,
                        help='Stop batch when cd drops below this fraction of batch D1')
    parser.add_argument('--bank_lambda', type=float, default=2.0,
                        help='Soft penalty weight for bank alignment')
    parser.add_argument('--bank_threshold', type=float, default=0.2,
                        help='Discard directions with max bank alignment above this')
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed_bank', type=str, default=None,
                        help='Path to features.pkl to seed the bank with')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_dir = (f"results/batched_seq/{args.model}_L{args.layer}"
                   f"_meas{args.measure_layer}_alpha{args.alpha}")
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(results_dir, 'checkpoint.pkl')

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

    with torch.no_grad():
        ref_out = forward_from_layer_to_layer(
            model, ctx, act.unsqueeze(0).unsqueeze(0),
            args.layer, args.measure_layer).float()

    # Resume or seed bank
    bank = []  # list of (direction, cd, batch_idx) tuples
    batch_logs = []
    start_batch = 0

    if args.resume and os.path.exists(checkpoint_path):
        ckpt = pickle.load(open(checkpoint_path, 'rb'))
        bank = [(d.to(device), cd, bi) for d, cd, bi in ckpt['bank']]
        batch_logs = ckpt['batch_logs']
        start_batch = len(batch_logs)
        print(f"Resumed: {len(bank)} directions in bank, starting batch {start_batch+1}")
    elif args.seed_bank:
        seed_data = pickle.load(open(args.seed_bank, 'rb'))
        if isinstance(seed_data['directions'], list):
            seed_dirs = seed_data['directions']
        else:
            seed_dirs = [seed_data['directions'][i] for i in range(seed_data['directions'].shape[0])]
        seed_cds = seed_data['cos_dists']
        if not isinstance(seed_cds, list):
            seed_cds = seed_cds.tolist()
        for i, d in enumerate(seed_dirs):
            bank.append((d.to(device).float(), seed_cds[i], 0))
        batch_logs.append({
            'batch_idx': 0, 'n_found': len(bank), 'n_survived': len(bank),
            'cds': seed_cds, 'rewards': seed_cds,
        })
        start_batch = 1
        print(f"Seeded bank with {len(bank)} directions from {args.seed_bank}")

    print(f"\n{'='*60}")
    print(f"BATCHED SEQUENTIAL DISCOVERY")
    print(f"  alpha={args.alpha}°, batch_size={args.batch_size}, max_batches={args.max_batches}")
    print(f"  bank_lambda={args.bank_lambda}, bank_threshold={args.bank_threshold}")
    print(f"  stop_fraction={args.stop_fraction}")
    print(f"{'='*60}")

    for batch_idx in range(start_batch, args.max_batches):
        bank_dirs = [b[0] for b in bank] if bank else None
        n_bank = len(bank)

        # For batch 1 (no bank), use bank_lambda=0
        effective_lambda = args.bank_lambda if n_bank > 0 else 0.0

        print(f"\n--- BATCH {batch_idx+1} (bank={n_bank}) ---")

        batch_dirs = []
        batch_cds = []
        batch_rewards = []
        stop_threshold = None

        for d_idx in range(args.batch_size):
            t0 = time.time()
            d, cd, reward, history = optimize_one_direction(
                model, ctx, act, ref_out, args.alpha, args.layer,
                args.measure_layer, device, d_model,
                batch_prev_dirs=batch_dirs if batch_dirs else None,
                bank_dirs=bank_dirs,
                bank_lambda=effective_lambda,
                n_steps=args.n_steps, lr=args.lr,
                n_restarts=args.n_restarts, cap=args.cap)

            batch_dirs.append(d)
            batch_cds.append(cd)
            batch_rewards.append(reward)
            elapsed = time.time() - t0

            if d_idx == 0:
                stop_threshold = cd * args.stop_fraction

            # Bank alignment
            max_bank_align = 0.0
            if bank_dirs:
                bank_stack = torch.stack(bank_dirs)
                max_bank_align = (d @ bank_stack.T).abs().max().item()

            print(f"    D{d_idx+1:>2}: cd={cd:.4f}, reward={reward:.4f}, "
                  f"max_bank={max_bank_align:.3f} ({elapsed:.1f}s)")

            if cd < stop_threshold:
                print(f"    *** Stopping batch: cd {cd:.4f} < {stop_threshold:.4f} ***")
                break

        # Filter: discard directions too aligned with bank
        n_before = len(batch_dirs)
        survivors = []
        for i, d in enumerate(batch_dirs):
            if bank_dirs:
                bank_stack = torch.stack(bank_dirs)
                max_align = (d @ bank_stack.T).abs().max().item()
            else:
                max_align = 0.0

            if max_align <= args.bank_threshold:
                survivors.append((d, batch_cds[i], batch_idx))
            else:
                print(f"    Discarded D{i+1}: max_bank_align={max_align:.3f} > {args.bank_threshold}")

        n_survived = len(survivors)
        print(f"  Batch {batch_idx+1}: {n_before} found, {n_survived} survived "
              f"(threshold={args.bank_threshold})")

        # Add survivors to bank
        bank.extend(survivors)
        batch_logs.append({
            'batch_idx': batch_idx,
            'n_found': n_before,
            'n_survived': n_survived,
            'cds': batch_cds,
            'rewards': batch_rewards,
        })

        # Checkpoint
        ckpt = {
            'bank': [(d.cpu(), cd, bi) for d, cd, bi in bank],
            'batch_logs': batch_logs,
            'args': vars(args),
        }
        pickle.dump(ckpt, open(checkpoint_path, 'wb'))
        print(f"  Bank size: {len(bank)}")

        # Stop if no survivors
        if n_survived == 0:
            print(f"\n  *** No survivors in batch {batch_idx+1}, stopping ***")
            break

    # Final summary
    n_total = len(bank)
    print(f"\n{'='*60}")
    print(f"FINAL: {n_total} directions in bank")
    print(f"{'='*60}")

    if n_total > 0:
        all_dirs = torch.stack([b[0] for b in bank])
        all_cds = [b[1] for b in bank]
        G = (all_dirs @ all_dirs.T).abs()
        G.fill_diagonal_(0)
        print(f"Max |cos_sim|: {G.max().item():.4f}")
        print(f"Mean |cos_sim|: {G.sum().item() / max(n_total * (n_total - 1), 1):.4f}")
        print(f"cd > 0.1: {sum(1 for c in all_cds if c > 0.1)}/{n_total}")
        print(f"cd > 0.01: {sum(1 for c in all_cds if c > 0.01)}/{n_total}")

        for blog in batch_logs:
            print(f"  Batch {blog['batch_idx']+1}: {blog['n_found']} found, "
                  f"{blog['n_survived']} survived")

        # Save
        save_data = {
            'directions': [b[0].cpu() for b in bank],
            'cos_dists': [b[1] for b in bank],
            'batch_indices': [b[2] for b in bank],
            'gram_matrix': G.cpu().numpy(),
            'batch_logs': batch_logs,
            'args': vars(args),
        }
        pickle.dump(save_data, open(os.path.join(results_dir, 'features.pkl'), 'wb'))

        # Plot cd decay across batches
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(batch_logs)))
        offset = 0
        for blog in batch_logs:
            n = len(blog['cds'])
            x = range(offset + 1, offset + n + 1)
            ax.plot(x, blog['cds'], 'o-', markersize=3, color=colors[blog['batch_idx']],
                    label=f"Batch {blog['batch_idx']+1} ({blog['n_survived']} survived)")
            offset += n
        ax.set_xlabel('Direction index (global)')
        ax.set_ylabel('Cosine distance')
        ax.set_title(f'Batched sequential: {args.model} L{args.layer}→L{args.measure_layer}, '
                     f'alpha={args.alpha}°')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'cd_decay.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nSaved to {results_dir}/")


if __name__ == '__main__':
    main()
