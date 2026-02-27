#!/usr/bin/env python3
"""
Perturbation sweep on a SINGLE activation: random dirs + probe dirs for on-board pieces.

Helps calibrate MELBO hyperparameters by showing what the plateau looks like
for this specific activation, and what probe-direction perturbations look like.
"""

import argparse
import os
import pickle
import re
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_to_final,
    _get_final_norm, _get_lm_head,
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
    return board


def get_board_probe_directions(board, probe_dirs, probe_norms):
    """Get probe directions for pieces on the board, sorted by norm (strongest first)."""
    import chess
    pieces = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        piece_class = PIECE_TO_PROBE_CLASS[piece.symbol()]
        flat_idx = rank * (8 * 13) + file * 13 + piece_class
        square_name = f"{FILES[file]}{RANKS[rank]}"
        desc = f"{piece.symbol()} on {square_name}"
        norm = probe_norms[flat_idx].item()
        pieces.append((flat_idx, desc, norm))
    # Sort by probe norm (strongest feature first)
    pieces.sort(key=lambda x: -x[2])
    return pieces


def sweep_direction(model, ctx, a, d, angles_rad, layer, device):
    """Sweep one direction across angles, return KL and L2 arrays."""
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = a.to(device=device, dtype=model_dtype)
    R = a_d.norm()
    a_unit = a_d / R
    d_d = d.to(device=device, dtype=model_dtype)

    # Orthogonalize
    d_orth = d_d - (d_d @ a_unit) * a_unit
    d_orth = F.normalize(d_orth, dim=-1)

    # Reference
    with torch.no_grad():
        ref_point = a_d.unsqueeze(0).unsqueeze(0)
        ref_final = forward_from_layer_to_final(model, ctx_d, ref_point, layer)
        ref_ln = _get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1)
        ref_logits = _get_lm_head(model)(ref_ln)
        ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)

    kl_arr = np.zeros(len(angles_rad))
    l2_arr = np.zeros(len(angles_rad))

    for j, theta in enumerate(angles_rad):
        perturbed = R * (np.cos(theta) * a_unit + np.sin(theta) * d_orth)
        with torch.no_grad():
            pert_final = forward_from_layer_to_final(
                model, ctx_d, perturbed.unsqueeze(0).unsqueeze(0), layer)
            l2_arr[j] = (pert_final - ref_final).float().norm().item()
            pert_ln = _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)
            pert_logits = _get_lm_head(model)(pert_ln)
            pert_lp = F.log_softmax(pert_logits.float(), dim=-1)
            kl_arr[j] = (ref_log_probs.exp() * (ref_log_probs - pert_lp)).sum().item()

    return kl_arr, l2_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=11)
    parser.add_argument('--act_idx', type=int, default=0)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--n_random_dirs', type=int, default=20)
    parser.add_argument('--n_probe_dirs', type=int, default=10,
                        help='Number of on-board probe directions (strongest by norm)')
    parser.add_argument('--probe_path', type=str, required=True)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = (f'results/single_act_sweep/{args.model}_layer{args.layer}'
                           f'_act{args.act_idx}')
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    # Generate data (same seed as MELBO)
    n_games = max(args.act_idx + 1, 50)
    token_ids_list = generate_chess_games(n_games, args.seq_len, seed=args.seed)

    # Decode board
    print(f"\n{'='*60}")
    board = decode_board_state(token_ids_list, args.act_idx)

    # Extract activation (handle act_idx > batch_size)
    print(f"\nExtracting layer-{args.layer} activation for act_idx={args.act_idx}...")
    batch_size = 32
    all_contexts = []
    all_activations = []
    for i in range(0, n_games, batch_size):
        bt = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            out = model(bt, output_hidden_states=True)
            h = out.hidden_states[args.layer + 1]
        all_contexts.append(h[:, :-1, :].float().cpu())
        all_activations.append(h[:, -1, :].float().cpu())
    all_contexts = torch.cat(all_contexts, 0)
    all_activations = torch.cat(all_activations, 0)
    ctx = all_contexts[args.act_idx:args.act_idx+1].to(device)
    act = all_activations[args.act_idx].to(device)
    print(f"Activation norm: {act.norm():.2f}")

    # Load probe
    probe_ckpt = torch.load(args.probe_path, map_location='cpu', weights_only=False)
    probe = probe_ckpt['linear_probe']
    probe_dirs = probe[0].permute(1, 2, 3, 0).reshape(-1, probe.shape[1])
    probe_norms = probe_dirs.norm(dim=-1)
    probe_dirs_normalized = F.normalize(probe_dirs, dim=-1)

    # Get on-board probe directions
    board_probes = get_board_probe_directions(board, probe_dirs, probe_norms)
    n_probe = min(args.n_probe_dirs, len(board_probes))
    print(f"\nTop {n_probe} on-board probe directions (by norm):")
    for _, desc, norm in board_probes[:n_probe]:
        print(f"  {desc} (norm={norm:.3f})")

    # Angles
    angles_deg = np.linspace(0, 180, 61)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Sweep random directions
    print(f"\nSweeping {args.n_random_dirs} random directions...")
    random_kl = np.zeros((args.n_random_dirs, len(angles_deg)))
    random_l2 = np.zeros((args.n_random_dirs, len(angles_deg)))
    for i in range(args.n_random_dirs):
        d = torch.randn(act.shape[-1])
        random_kl[i], random_l2[i] = sweep_direction(
            model, ctx, act, d, angles_rad, args.layer, device)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{args.n_random_dirs}")

    # Sweep probe directions (both +d and -d)
    print(f"\nSweeping {n_probe} on-board probe directions (+ and - signs)...")
    probe_kl_pos = np.zeros((n_probe, len(angles_deg)))
    probe_l2_pos = np.zeros((n_probe, len(angles_deg)))
    probe_kl_neg = np.zeros((n_probe, len(angles_deg)))
    probe_l2_neg = np.zeros((n_probe, len(angles_deg)))
    probe_labels = []
    idx_40 = np.argmin(np.abs(angles_deg - 40))
    for i in range(n_probe):
        flat_idx, desc, norm = board_probes[i]
        d = probe_dirs_normalized[flat_idx]
        probe_kl_pos[i], probe_l2_pos[i] = sweep_direction(
            model, ctx, act, d, angles_rad, args.layer, device)
        probe_kl_neg[i], probe_l2_neg[i] = sweep_direction(
            model, ctx, act, -d, angles_rad, args.layer, device)
        probe_labels.append(desc)
        print(f"  {desc}: KL@40° +d={probe_kl_pos[i, idx_40]:.4f}, -d={probe_kl_neg[i, idx_40]:.4f}")

    # Save
    results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'random_kl': random_kl,
        'random_l2': random_l2,
        'probe_kl_pos': probe_kl_pos,
        'probe_l2_pos': probe_l2_pos,
        'probe_kl_neg': probe_kl_neg,
        'probe_l2_neg': probe_l2_neg,
        'probe_labels': probe_labels,
        'board_fen': board.fen(),
        'act_norm': act.norm().item(),
    }
    save_path = os.path.join(args.results_dir, 'single_act_sweep.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    # Plot: +d vs -d comparison (KL only, cleaner)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for sign_idx, (sign_label, kl_data) in enumerate([('+d', probe_kl_pos), ('-d', probe_kl_neg)]):
        ax = axes[sign_idx]

        # Random baseline
        med = np.median(random_kl, axis=0)
        q25 = np.percentile(random_kl, 25, axis=0)
        q75 = np.percentile(random_kl, 75, axis=0)
        ax.fill_between(angles_deg, q25, q75, color='gray', alpha=0.2)
        ax.plot(angles_deg, med, color='gray', lw=2, ls='--', label='Random')

        # Probe directions
        colors = plt.cm.tab10(np.linspace(0, 1, min(n_probe, 10)))
        for i in range(n_probe):
            c = colors[i % len(colors)]
            ax.plot(angles_deg, kl_data[i], color=c, lw=1.2,
                    alpha=0.7, label=probe_labels[i])

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel('KL divergence')
        ax.set_title(f'Probe direction {sign_label}')
        ax.legend(fontsize=5.5, ncol=2, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)

    fig.suptitle(f'{args.model} layer {args.layer}, act #{args.act_idx}: +probe vs -probe',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'probe_pos_vs_neg.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")

    # Summary
    print(f"\nKL at 40° (+d / -d):")
    print(f"  Random median: {np.median(random_kl[:, idx_40]):.4f}")
    for i in range(n_probe):
        print(f"  {probe_labels[i]}: {probe_kl_pos[i, idx_40]:.4f} / {probe_kl_neg[i, idx_40]:.4f}")


if __name__ == '__main__':
    main()
