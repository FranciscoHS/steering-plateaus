#!/usr/bin/env python3
"""
Sweep perturbations along combinations of probe directions.

Takes the K most sensitive probes for a single activation, then sweeps
all C(K,1), C(K,2), ..., C(K,K-1) combinations. Each combination direction
is the normalized sum of the constituent probe directions.
"""

import argparse
import os
import pickle
import re
import sys
from itertools import combinations
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
    print(f"Board after {len(board.move_stack)} moves:")
    print(board)
    return board


def get_board_probe_directions(board, probe_dirs, probe_norms):
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
    pieces.sort(key=lambda x: -x[2])
    return pieces


def sweep_direction(model, ctx, a, d, angles_rad, layer, device):
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = a.to(device=device, dtype=model_dtype)
    R = a_d.norm()
    a_unit = a_d / R
    d_d = d.to(device=device, dtype=model_dtype)

    d_orth = d_d - (d_d @ a_unit) * a_unit
    d_orth = F.normalize(d_orth, dim=-1)

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
    parser.add_argument('--act_idx', type=int, default=137)
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--n_random_dirs', type=int, default=20)
    parser.add_argument('--n_top', type=int, default=6,
                        help='Number of most sensitive probe directions to use')
    parser.add_argument('--probe_path', type=str, required=True)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = (f'results/probe_combos/{args.model}_layer{args.layer}'
                           f'_act{args.act_idx}')
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    # Generate data
    n_games = max(args.act_idx + 1, 50)
    token_ids_list = generate_chess_games(n_games, args.seq_len, seed=args.seed)

    # Decode board
    board = decode_board_state(token_ids_list, args.act_idx)

    # Extract activation
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

    # Angles
    angles_deg = np.linspace(0, 180, 61)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Random baseline
    print(f"\nSweeping {args.n_random_dirs} random directions...")
    random_kl = np.zeros((args.n_random_dirs, len(angles_deg)))
    random_l2 = np.zeros((args.n_random_dirs, len(angles_deg)))
    for i in range(args.n_random_dirs):
        d = torch.randn(act.shape[-1])
        random_kl[i], random_l2[i] = sweep_direction(model, ctx, act, d, angles_rad, args.layer, device)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_random_dirs}")

    # First pass: sweep all on-board probes in both signs to find the most sensitive
    print(f"\nSweeping all {len(board_probes)} on-board probes (+d and -d) to rank sensitivity...")
    idx_40 = np.argmin(np.abs(angles_deg - 40))
    probe_sensitivity = []
    for flat_idx, desc, norm in board_probes:
        d = probe_dirs_normalized[flat_idx]
        kl_pos, l2_pos = sweep_direction(model, ctx, act, d, angles_rad, args.layer, device)
        kl_neg, l2_neg = sweep_direction(model, ctx, act, -d, angles_rad, args.layer, device)
        # Pick sign that breaks away more (higher KL at 40°)
        if kl_pos[idx_40] >= kl_neg[idx_40]:
            probe_sensitivity.append((kl_pos[idx_40], +1, flat_idx, desc, norm, kl_pos, l2_pos))
        else:
            probe_sensitivity.append((kl_neg[idx_40], -1, flat_idx, desc, norm, kl_neg, l2_neg))

    probe_sensitivity.sort(key=lambda x: -x[0])

    print(f"\nAll probes ranked by KL@40° (best sign):")
    for kl40, sign, fidx, desc, norm, _, _ in probe_sensitivity:
        sign_str = '+' if sign > 0 else '-'
        print(f"  {sign_str}{desc}: KL@40°={kl40:.4f}")

    # Select top N
    top_probes = probe_sensitivity[:args.n_top]
    print(f"\nTop {args.n_top} selected:")
    selected_dirs = []  # (direction_tensor, label, kl_curve, l2_curve)
    for kl40, sign, fidx, desc, norm, kl_curve, l2_curve in top_probes:
        sign_str = '+' if sign > 0 else '-'
        label = f"{sign_str}{desc}"
        d = sign * probe_dirs_normalized[fidx]
        selected_dirs.append((d, label, kl_curve, l2_curve))
        print(f"  {label}: KL@40°={kl40:.4f}")

    # Sweep all combinations
    n_top = len(selected_dirs)
    combo_results = {}  # k -> list of (combo_indices, kl_curve, l2_curve, labels)

    for k in range(1, n_top):  # 1 through n_top-1
        combos = list(combinations(range(n_top), k))
        print(f"\nSweeping C({n_top},{k}) = {len(combos)} combinations of {k}...")
        combo_results[k] = []
        for ci, combo in enumerate(combos):
            # Combined direction = normalized sum
            d_combined = torch.zeros_like(selected_dirs[0][0])
            for idx in combo:
                d_combined = d_combined + selected_dirs[idx][0]
            d_combined = F.normalize(d_combined, dim=-1)
            kl, l2 = sweep_direction(model, ctx, act, d_combined, angles_rad, args.layer, device)
            labels = [selected_dirs[idx][1] for idx in combo]
            combo_results[k].append((combo, kl, l2, labels))
            if (ci + 1) % 10 == 0:
                print(f"  {ci+1}/{len(combos)}")

    # Also store the individual results as k=1
    combo_results[1] = []
    for i, (d, label, kl_curve, l2_curve) in enumerate(selected_dirs):
        combo_results[1].append(((i,), kl_curve, l2_curve, [label]))

    # Save
    results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'random_kl': random_kl,
        'random_l2': random_l2,
        'selected_labels': [s[1] for s in selected_dirs],
        'combo_results': {k: [(c, kl.tolist(), l2.tolist(), labs)
                              for c, kl, l2, labs in v]
                          for k, v in combo_results.items()},
        'board_fen': board.fen(),
    }
    save_path = os.path.join(args.results_dir, 'probe_combos.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    # Color per combination size
    colors = {1: 'blue', 2: 'green', 3: 'orange', 4: 'red', 5: 'purple'}

    # Plot KL (full range) and L2 (0-40°)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, (metric, ylabel, xlim) in enumerate([
        ('kl', 'KL divergence', (0, 180)),
        ('l2', 'L2 distance', (0, 40)),
    ]):
        ax = axes[ax_idx]
        rand_data = random_kl if metric == 'kl' else random_l2

        # Random baseline
        med = np.median(rand_data, axis=0)
        q25 = np.percentile(rand_data, 25, axis=0)
        q75 = np.percentile(rand_data, 75, axis=0)
        ax.fill_between(angles_deg, q25, q75, color='gray', alpha=0.15)
        ax.plot(angles_deg, med, color='gray', lw=2, ls='--', label='Random')

        for k in sorted(combo_results.keys()):
            data_idx = 1 if metric == 'kl' else 2  # kl or l2 in tuple
            all_data = np.array([
                np.array(item[data_idx]) for item in combo_results[k]
            ])
            med_k = np.median(all_data, axis=0)
            q25_k = np.percentile(all_data, 25, axis=0)
            q75_k = np.percentile(all_data, 75, axis=0)
            c = colors.get(k, 'black')
            n_combos = len(combo_results[k])
            ax.fill_between(angles_deg, q25_k, q75_k, color=c, alpha=0.12)
            ax.plot(angles_deg, med_k, color=c, lw=2,
                    label=f'{k} probe{"s" if k > 1 else ""} (n={n_combos})')

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(*xlim)

    fig.suptitle(f'{args.model} layer {args.layer}, act #{args.act_idx}: '
                 f'combinations of top-{n_top} sensitive probes',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'probe_combos.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    main()
