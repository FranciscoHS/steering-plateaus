#!/usr/bin/env python3
"""
Capped MELBO v2: soft orthogonality, tuned for single activation.

Changes from v1:
- Soft orthogonality: reward *= (1 - max_j cos²(d, d_j)) instead of hard projection
- Configurable for specific activation and angle/target calibration
"""

import argparse
import os
import pickle
import re
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, _get_blocks, _get_final_norm, _get_lm_head,
)
from src.data import generate_chess_games


ITOS = {0: ' ', 1: '#', 2: '+', 3: '-', 4: '.', 5: '0', 6: '1', 7: '2',
        8: '3', 9: '4', 10: '5', 11: '6', 12: '7', 13: '8', 14: '9',
        15: ';', 16: '=', 17: 'B', 18: 'K', 19: 'N', 20: 'O', 21: 'Q',
        22: 'R', 23: 'a', 24: 'b', 25: 'c', 26: 'd', 27: 'e', 28: 'f',
        29: 'g', 30: 'h', 31: 'x'}

PIECE_NAMES = ['blank', 'K', 'Q', 'R', 'B', 'N', 'P', 'k', 'q', 'r', 'b', 'n', 'p']
FILES = 'abcdefgh'
RANKS = '12345678'
PIECE_TO_PROBE_CLASS = {
    'K': 1, 'Q': 2, 'R': 3, 'B': 4, 'N': 5, 'P': 6,
    'k': 7, 'q': 8, 'r': 9, 'b': 10, 'n': 11, 'p': 12,
}


def decode_board_state(token_ids_list, act_idx):
    import chess
    tokens = token_ids_list[act_idx]
    pgn = ''.join(ITOS[t.item()] for t in tokens)
    print(f"PGN: {pgn[:200]}...")
    pgn_clean = pgn.lstrip(';')
    moves_str = re.sub(r'\d+\.', '', pgn_clean).split()
    board = chess.Board()
    for move_san in moves_str:
        move_san = move_san.strip()
        if not move_san:
            continue
        try:
            board.push(board.parse_san(move_san))
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            break
    print(f"\nBoard after {len(board.move_stack)} moves:")
    print(board)
    return board


def forward_from_layer(model, ctx, a_pert, layer):
    """Forward pass from layer through remaining blocks to logits."""
    hidden = torch.cat([ctx, a_pert.unsqueeze(0).unsqueeze(0)], dim=1)
    blocks = _get_blocks(model)
    for block in blocks[layer + 1:]:
        result = block(hidden)
        hidden = result[0] if isinstance(result, tuple) else result
    final = hidden[:, -1, :]
    logits = _get_lm_head(model)(_get_final_norm(model)(final))
    return logits


def optimize_direction(model, ctx, activation, ref_log_probs,
                       angle_rad, layer, device, d_model,
                       target_kl=1.0, n_steps=500, lr=0.01,
                       n_restarts=5, prev_directions=None,
                       diversity_power=1):
    """Find direction producing KL ≈ target_kl with soft orthogonality."""
    a = activation.to(device)
    a_norm = a.norm()
    a_unit = a / a_norm
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    best_reward = -float('inf')
    best_direction = None
    best_history = None

    for restart in range(n_restarts):
        # Random init, project out activation direction
        direction = torch.randn(d_model, device=device)
        direction = direction - (direction @ a_unit) * a_unit
        direction = F.normalize(direction, dim=0)
        direction = direction.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([direction], lr=lr)
        history = []

        for step in range(n_steps):
            optimizer.zero_grad()

            # Normalize and project out activation component
            d = direction / direction.norm()
            d = d - (d @ a_unit) * a_unit
            d = d / (d.norm() + 1e-10)

            # Perturb
            a_pert = a_norm * (cos_t * a_unit + sin_t * d)
            logits = forward_from_layer(model, ctx, a_pert, layer)
            pert_log_probs = F.log_softmax(logits, dim=-1)

            kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum()

            # Capped reward: peaks at KL = target_kl
            reward = kl * torch.exp(-kl / target_kl)

            # Soft orthogonality penalty
            if prev_directions:
                d_detached = d.detach()
                max_cos2 = max((d_detached @ pd).pow(2).item() for pd in prev_directions)
                diversity = (1.0 - max_cos2) ** diversity_power
                reward = reward * diversity

            loss = -reward
            loss.backward()
            optimizer.step()

            # Project out activation component after step
            with torch.no_grad():
                direction.data -= (direction.data @ a_unit) * a_unit

            history.append((reward.item(), kl.item()))

            if step % 100 == 0 or step == n_steps - 1:
                div_str = f", div={diversity:.3f}" if prev_directions else ""
                print(f"    restart {restart+1}/{n_restarts}, step {step}: "
                      f"reward={reward.item():.6f}, KL={kl.item():.4f}{div_str}")

        final_reward = history[-1][0]
        if final_reward > best_reward:
            best_reward = final_reward
            with torch.no_grad():
                d_final = direction / direction.norm()
                d_final = d_final - (d_final @ a_unit) * a_unit
                d_final = d_final / d_final.norm()
                best_direction = d_final.clone()
            best_history = history

    return best_direction, best_reward, best_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=11)
    parser.add_argument('--opt_angle', type=float, default=15.0)
    parser.add_argument('--target_kl', type=float, default=1.0)
    parser.add_argument('--act_idx', type=int, default=137)
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--n_directions', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--probe_path', type=str, default=None)
    parser.add_argument('--diversity_power', type=int, default=1,
                        help='Exponent for diversity penalty: (1-cos²)^p')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        dp_str = f'_dp{args.diversity_power}' if args.diversity_power > 1 else ''
        args.results_dir = (f'results/capped_melbo_v2/{args.model}_layer{args.layer}'
                           f'_act{args.act_idx}_angle{args.opt_angle}_kl{args.target_kl}{dp_str}')
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Config: model={args.model}, layer={args.layer}, angle={args.opt_angle} deg, "
          f"target_kl={args.target_kl}, act_idx={args.act_idx}, "
          f"n_directions={args.n_directions}, n_steps={args.n_steps}, "
          f"n_restarts={args.n_restarts}, diversity_power={args.diversity_power}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']

    # Generate data
    n_games = max(args.act_idx + 1, 50)
    token_ids_list = generate_chess_games(n_games, args.seq_len, seed=args.seed)

    # Ground truth
    print(f"\n{'='*60}")
    board = decode_board_state(token_ids_list, args.act_idx)

    # Extract activation
    print(f"\nExtracting layer-{args.layer} activation...")
    batch_size = 32
    all_ctx = []
    all_act = []
    for i in range(0, n_games, batch_size):
        bt = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            out = model(bt, output_hidden_states=True)
            h = out.hidden_states[args.layer + 1]
        all_ctx.append(h[:, :-1, :].float().cpu())
        all_act.append(h[:, -1, :].float().cpu())
    all_ctx = torch.cat(all_ctx, 0)
    all_act = torch.cat(all_act, 0)
    ctx = all_ctx[args.act_idx:args.act_idx+1].to(device)
    act = all_act[args.act_idx].to(device)
    print(f"Activation norm: {act.norm():.2f}")

    # Reference logits
    with torch.no_grad():
        ref_logits = forward_from_layer(model, ctx, act, args.layer)
        ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)

    # Optimize
    angle_rad = np.deg2rad(args.opt_angle)
    directions = []
    rewards = []
    histories = []

    print(f"\n{'='*60}")
    print(f"CAPPED MELBO v2 (soft orthogonality)")
    print(f"  target_kl={args.target_kl}, angle={args.opt_angle} deg")
    print(f"{'='*60}")

    for d_idx in range(args.n_directions):
        print(f"\n--- Direction {d_idx+1}/{args.n_directions} ---")
        t0 = time.time()
        direction, best_reward, history = optimize_direction(
            model, ctx, act, ref_log_probs,
            angle_rad, args.layer, device, d_model,
            target_kl=args.target_kl,
            n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts,
            prev_directions=directions if directions else None,
            diversity_power=args.diversity_power
        )
        elapsed = time.time() - t0
        directions.append(direction)
        rewards.append(best_reward)
        histories.append(history)
        final_kl = history[-1][1]
        print(f"  Result: reward={best_reward:.6f}, KL={final_kl:.4f}, time={elapsed:.1f}s")

        # Show cosine with previous directions
        if len(directions) > 1:
            cosines = [abs((direction @ d).item()) for d in directions[:-1]]
            print(f"  Cosines with prev: {['%.3f' % c for c in cosines]}")

    # Save
    results = {
        'args': vars(args),
        'directions': [d.cpu() for d in directions],
        'rewards': rewards,
        'histories': histories,
        'board_fen': board.fen(),
    }
    save_path = os.path.join(args.results_dir, 'melbo_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved to {save_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for i, (rew, hist) in enumerate(zip(rewards, histories)):
        print(f"  Dir {i+1}: reward={rew:.6f}, final_KL={hist[-1][1]:.4f}")

    # Pairwise cosines
    print(f"\nPairwise |cosines|:")
    n = len(directions)
    for i in range(n):
        row = []
        for j in range(n):
            c = abs((directions[i] @ directions[j]).item())
            row.append(f"{c:.3f}")
        print(f"  {' '.join(row)}")


if __name__ == '__main__':
    main()
