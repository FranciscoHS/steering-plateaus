#!/usr/bin/env python3
"""
Capped MELBO on a single ChessGPT activation.

Finds feature directions for one specific board position, then compares
against the linear probe directions for pieces actually present on that board.

Objective: KL * exp(-KL / target), peaks at KL = target.
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

# Probe class indices: 0=blank, 1=K, 2=Q, 3=R, 4=B, 5=N, 6=P, 7=k, 8=q, 9=r, 10=b, 11=n, 12=p
PIECE_NAMES = ['blank', 'K', 'Q', 'R', 'B', 'N', 'P', 'k', 'q', 'r', 'b', 'n', 'p']
FILES = 'abcdefgh'
RANKS = '12345678'

# Map python-chess piece symbols to probe class indices
PIECE_TO_PROBE_CLASS = {
    'K': 1, 'Q': 2, 'R': 3, 'B': 4, 'N': 5, 'P': 6,
    'k': 7, 'q': 8, 'r': 9, 'b': 10, 'n': 11, 'p': 12,
}


def decode_board_state(token_ids_list, act_idx):
    """Decode a chess game and return the board."""
    import chess
    tokens = token_ids_list[act_idx]
    pgn = ''.join(ITOS[t.item()] for t in tokens)
    print(f"PGN: {pgn}")

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
    print(f"Turn: {'White' if board.turn else 'Black'}")
    return board


def get_board_probe_indices(board):
    """Get probe indices for all pieces on the board.

    Returns list of (probe_flat_index, description) for each piece present.
    Probe tensor is [1, 512, 8, 8, 13] -> flattened as [8*8*13, 512]
    with flat_index = rank * (8*13) + file * 13 + class
    """
    import chess
    pieces = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        rank = chess.square_rank(square)  # 0-7
        file = chess.square_file(square)  # 0-7
        piece_class = PIECE_TO_PROBE_CLASS[piece.symbol()]
        flat_idx = rank * (8 * 13) + file * 13 + piece_class
        square_name = f"{FILES[file]}{RANKS[rank]}"
        desc = f"{piece.symbol()} on {square_name}"
        pieces.append((flat_idx, desc))
    return pieces


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


def optimize_direction_capped(model, ctx, activation, ref_log_probs,
                               angle_rad, layer, device, d_model,
                               target_kl=0.5, n_steps=300, lr=0.01,
                               n_restarts=3, prev_directions=None):
    """Find direction producing KL ≈ target_kl for a single activation."""
    a = activation.to(device)
    a_norm = a.norm()
    a_unit = a / a_norm
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    best_reward = -float('inf')
    best_direction = None
    best_history = None

    for restart in range(n_restarts):
        direction = torch.randn(d_model, device=device)
        # Project out activation direction and previous directions
        direction = direction - (direction @ a_unit) * a_unit
        if prev_directions:
            for pd in prev_directions:
                direction = direction - (direction @ pd) * pd
        direction = F.normalize(direction, dim=0)
        direction = direction.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([direction], lr=lr)
        history = []

        for step in range(n_steps):
            optimizer.zero_grad()

            # Normalize and project
            d = direction / direction.norm()
            d = d - (d @ a_unit) * a_unit
            if prev_directions:
                for pd in prev_directions:
                    d = d - (d @ pd) * pd
            d = d / (d.norm() + 1e-10)

            # Perturb
            a_pert = a_norm * (cos_t * a_unit + sin_t * d)
            logits = forward_from_layer(model, ctx, a_pert, layer)
            pert_log_probs = F.log_softmax(logits, dim=-1)

            kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum()
            reward = kl * torch.exp(-kl / target_kl)

            loss = -reward
            loss.backward()
            optimizer.step()

            # Project after step
            with torch.no_grad():
                direction.data -= (direction.data @ a_unit) * a_unit
                if prev_directions:
                    for pd in prev_directions:
                        direction.data -= (direction.data @ pd) * pd

            history.append((reward.item(), kl.item()))

            if step % 100 == 0 or step == n_steps - 1:
                print(f"    restart {restart+1}/{n_restarts}, step {step}: "
                      f"reward={reward.item():.6f}, KL={kl.item():.4f}")

        final_reward = history[-1][0]
        if final_reward > best_reward:
            best_reward = final_reward
            with torch.no_grad():
                d_final = direction / direction.norm()
                d_final = d_final - (d_final @ a_unit) * a_unit
                if prev_directions:
                    for pd in prev_directions:
                        d_final = d_final - (d_final @ pd) * pd
                d_final = d_final / d_final.norm()
                best_direction = d_final.clone()
            best_history = history

    return best_direction, best_reward, best_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=11)
    parser.add_argument('--opt_angle', type=float, default=40.0)
    parser.add_argument('--target_kl', type=float, default=0.5)
    parser.add_argument('--act_idx', type=int, default=0)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--n_directions', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--probe_path', type=str, default=None)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = (f'results/capped_melbo/{args.model}_layer{args.layer}'
                           f'_act{args.act_idx}_target{args.target_kl}')
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Config: model={args.model}, layer={args.layer}, theta={args.opt_angle}°, "
          f"target_kl={args.target_kl}, act_idx={args.act_idx}, "
          f"n_directions={args.n_directions}, n_steps={args.n_steps}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']

    # Generate data
    n_games = max(args.act_idx + 1, 50)
    token_ids_list = generate_chess_games(n_games, args.seq_len, seed=args.seed)

    # Ground truth
    print(f"\n{'='*60}")
    print(f"GROUND TRUTH (activation {args.act_idx})")
    print(f"{'='*60}")
    board = decode_board_state(token_ids_list, args.act_idx)
    board_pieces = get_board_probe_indices(board)
    print(f"\nPieces on board ({len(board_pieces)}):")
    for _, desc in board_pieces:
        print(f"  {desc}")

    # Extract single activation
    print(f"\nExtracting layer-{args.layer} activation...")
    bt = torch.stack(token_ids_list[:n_games]).to(device)
    with torch.no_grad():
        out = model(bt[:64], output_hidden_states=True)
        h = out.hidden_states[args.layer + 1]
    ctx = h[args.act_idx:args.act_idx+1, :-1, :].float()
    act = h[args.act_idx, -1, :].float()
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
    print(f"CAPPED MELBO (single activation)")
    print(f"  target_kl={args.target_kl}, angle={args.opt_angle}°")
    print(f"{'='*60}")

    for d_idx in range(args.n_directions):
        print(f"\n--- Direction {d_idx+1}/{args.n_directions} ---")
        t0 = time.time()
        direction, best_reward, history = optimize_direction_capped(
            model, ctx, act, ref_log_probs,
            angle_rad, args.layer, device, d_model,
            target_kl=args.target_kl,
            n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts,
            prev_directions=directions if directions else None
        )
        elapsed = time.time() - t0
        directions.append(direction)
        rewards.append(best_reward)
        histories.append(history)
        print(f"  Result: reward={best_reward:.6f}, KL={history[-1][1]:.4f}, time={elapsed:.1f}s")

    # Compare against probe
    if args.probe_path:
        print(f"\n{'='*60}")
        print(f"PROBE COMPARISON")
        print(f"{'='*60}")
        probe_ckpt = torch.load(args.probe_path, map_location='cpu', weights_only=False)
        probe = probe_ckpt['linear_probe']  # [1, 512, 8, 8, 13]
        probe_dirs = probe[0].permute(1, 2, 3, 0).reshape(-1, probe.shape[1])
        probe_dirs_norm = F.normalize(probe_dirs, dim=-1)

        # Get probe indices for pieces on this board
        board_indices = set(idx for idx, _ in board_pieces)

        for i, d in enumerate(directions):
            d_cpu = d.cpu()
            d_norm = F.normalize(d_cpu.unsqueeze(0), dim=-1)
            cosines = (d_norm @ probe_dirs_norm.T).squeeze(0)

            # Best match among pieces ON the board
            best_board_cos = 0
            best_board_desc = ""
            for pidx, desc in board_pieces:
                c = cosines[pidx].abs().item()
                if c > best_board_cos:
                    best_board_cos = c
                    best_board_desc = desc
                    best_board_sign = '+' if cosines[pidx].item() > 0 else '-'

            # Best match overall
            top_cos, top_idx = cosines.abs().topk(5)

            kl = histories[i][-1][1]
            print(f"\n  Direction {i+1} (KL={kl:.4f}):")
            print(f"    Best on-board match: {best_board_sign}cos={best_board_cos:.3f} "
                  f"({best_board_desc})")
            print(f"    Top 5 overall:")
            for j in range(5):
                idx = top_idx[j].item()
                r = idx // (8 * 13)
                c_idx = (idx % (8 * 13)) // 13
                k = idx % 13
                square = f"{FILES[c_idx]}{RANKS[r]}"
                sign = '+' if cosines[top_idx[j]] > 0 else '-'
                on_board = " [ON BOARD]" if idx in board_indices else ""
                print(f"      {sign}cos={top_cos[j]:.3f}: "
                      f"{PIECE_NAMES[k]} on {square}{on_board}")

    # Save
    results = {
        'args': vars(args),
        'directions': [d.cpu() for d in directions],
        'rewards': rewards,
        'histories': histories,
        'board_fen': board.fen(),
        'board_pieces': board_pieces,
    }
    save_path = os.path.join(args.results_dir, 'melbo_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved to {save_path}")
    print("Done!")


if __name__ == '__main__':
    main()
