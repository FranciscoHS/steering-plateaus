#!/usr/bin/env python3
"""Correlate d0/d1/d2 coefficients with game properties across varied sequence lengths.

Generates chess games truncated at different lengths (100-500) so game phase,
piece count, etc. actually vary. Projects activations onto the progressive
directions and correlates coefficients with board properties.

Usage:
    python scripts/perturbation/reconstruct_varied_length.py \
        --features_path results/progressive/chess_16L_L2_meas14/features.pkl
"""
import argparse, os, sys, pickle
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import chess, random

from src.model import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', required=True)
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--n_games', type=int, default=30,
                        help='Number of base games to generate')
    parser.add_argument('--seq_lens', type=str, default='100,200,300,400,500',
                        help='Comma-separated truncation lengths')
    parser.add_argument('--cd_threshold', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seq_lens = [int(s) for s in args.seq_lens.split(',')]
    max_seq_len = max(seq_lens)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Load directions ----
    data = pickle.load(open(args.features_path, 'rb'))
    if isinstance(data['directions'], list):
        all_dirs = torch.stack(data['directions']).float()
    else:
        all_dirs = data['directions'].float()
    cos_dists = data['cos_dists']
    if isinstance(cos_dists, list):
        cos_dists = np.array(cos_dists)
    good_mask = cos_dists > args.cd_threshold
    dirs = all_dirs[good_mask]
    n_features = dirs.shape[0]
    print(f"Loaded {n_features} directions")

    # ---- Generate long games ----
    random.seed(args.seed)
    stoi = {' ': 0, '#': 1, '+': 2, '-': 3, '.': 4, '0': 5, '1': 6, '2': 7,
            '3': 8, '4': 9, '5': 10, '6': 11, '7': 12, '8': 13, '9': 14,
            ';': 15, '=': 16, 'B': 17, 'K': 18, 'N': 19, 'O': 20, 'Q': 21,
            'R': 22, 'a': 23, 'b': 24, 'c': 25, 'd': 26, 'e': 27, 'f': 28,
            'g': 29, 'h': 30, 'x': 31}

    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

    def game_to_pgn(board):
        pgn = ";"
        temp = chess.Board()
        for i, move in enumerate(board.move_stack):
            if i % 2 == 0:
                pgn += f"{i//2 + 1}."
            pgn += temp.san(move) + " "
            temp.push(move)
        return pgn

    def board_at_truncation(board, pgn, trunc_len):
        """Replay moves up to those that fit in trunc_len characters."""
        replay = chess.Board()
        pgn_so_far = ";"
        for i, move in enumerate(board.move_stack):
            if i % 2 == 0:
                pgn_so_far += f"{i//2 + 1}."
            pgn_so_far += replay.san(move) + " "
            if len(pgn_so_far) > trunc_len:
                break
            replay.push(move)
        return replay

    def extract_props(b):
        n_hm = len(b.move_stack)
        w_pieces = sum(len(b.pieces(pt, chess.WHITE)) for pt in chess.PIECE_TYPES)
        b_pieces = sum(len(b.pieces(pt, chess.BLACK)) for pt in chess.PIECE_TYPES)
        w_mat = sum(piece_values[pt] * len(b.pieces(pt, chess.WHITE)) for pt in piece_values)
        b_mat = sum(piece_values[pt] * len(b.pieces(pt, chess.BLACK)) for pt in piece_values)
        wk = list(b.pieces(chess.KING, chess.WHITE))
        bk = list(b.pieces(chess.KING, chess.BLACK))
        return {
            'n_half_moves': n_hm,
            'n_full_moves': n_hm // 2,
            'white_to_move': 1 if b.turn == chess.WHITE else 0,
            'n_total_pieces': w_pieces + b_pieces,
            'material_balance': w_mat - b_mat,
            'n_pawns': len(b.pieces(chess.PAWN, chess.WHITE)) + len(b.pieces(chess.PAWN, chess.BLACK)),
            'has_queens': 1 if (len(b.pieces(chess.QUEEN, chess.WHITE)) > 0 and
                                len(b.pieces(chess.QUEEN, chess.BLACK)) > 0) else 0,
            'is_check': 1 if b.is_check() else 0,
            'white_king_rank': chess.square_rank(wk[0]) if wk else -1,
            'black_king_rank': chess.square_rank(bk[0]) if bk else -1,
        }

    # Generate full-length games
    print(f"Generating {args.n_games} games (max seq_len={max_seq_len})...")
    full_games = []
    attempts = 0
    while len(full_games) < args.n_games and attempts < args.n_games * 20:
        attempts += 1
        board = chess.Board()
        for _ in range(max_seq_len):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(random.choice(legal))
        pgn = game_to_pgn(board)
        if not all(c in stoi for c in pgn):
            continue
        if len(pgn) < max_seq_len:
            continue
        full_games.append((board, pgn))
    print(f"  Got {len(full_games)} valid games")

    # ---- Load model ----
    model, tokenizer, device = load_model(args.model)
    model_dtype = next(model.parameters()).dtype
    print(f"Model on {device}")

    # ---- For each (game, seq_len): extract activation, project, get properties ----
    all_coeffs = []
    all_props = []
    all_seq_lens = []

    dirs_matrix = dirs.float()

    for sl in seq_lens:
        print(f"\nseq_len={sl}:")
        tokens_batch = []
        props_batch = []
        for board, pgn in full_games:
            truncated = pgn[:sl]
            toks = torch.tensor([stoi[c] for c in truncated], dtype=torch.long)
            tokens_batch.append(toks)

            b = board_at_truncation(board, pgn, sl)
            props_batch.append(extract_props(b))

        # Extract activations
        batch = torch.stack(tokens_batch).to(device)
        with torch.no_grad():
            out = model(batch, output_hidden_states=True)
            h = out.hidden_states[args.layer + 1]
            acts = h[:, -1, :].cpu().float()  # [n_games, d_model]
        del out, h, batch
        torch.cuda.empty_cache()

        # Project
        coeffs = acts @ dirs_matrix.T  # [n_games, n_features]
        all_coeffs.append(coeffs.numpy())
        all_props.extend(props_batch)
        all_seq_lens.extend([sl] * len(props_batch))

        # Quick summary
        print(f"  d0: mean={coeffs[:, 0].mean():.3f}, std={coeffs[:, 0].std():.3f}")
        print(f"  d1: mean={coeffs[:, 1].mean():.3f}, std={coeffs[:, 1].std():.3f}")
        print(f"  d2: mean={coeffs[:, 2].mean():.3f}, std={coeffs[:, 2].std():.3f}")

    # ---- Aggregate and correlate ----
    all_coeffs = np.concatenate(all_coeffs, axis=0)  # [n_total, n_features]
    n_total = len(all_props)
    print(f"\n{'='*60}")
    print(f"Total samples: {n_total} ({len(full_games)} games x {len(seq_lens)} lengths)")

    # Build property arrays
    prop_names = list(all_props[0].keys())
    prop_arrays = {}
    for pn in prop_names:
        prop_arrays[pn] = np.array([p[pn] for p in all_props], dtype=float)
    prop_arrays['seq_len'] = np.array(all_seq_lens, dtype=float)
    prop_names.append('seq_len')

    # Correlations for d0-d10
    print(f"\nCorrelations (d0-d10 vs game properties, N={n_total}):")
    header = f"{'Property':<22s}" + "".join(f"{'d'+str(i):>8s}" for i in range(min(11, n_features)))
    print(header)
    print("-" * len(header))
    for pn in prop_names:
        pv = prop_arrays[pn]
        if pv.std() < 1e-10:
            continue
        row = f"{pn:<22s}"
        for di in range(min(11, n_features)):
            r = np.corrcoef(pv, all_coeffs[:, di])[0, 1]
            row += f"{r:+.3f}   "
        print(row)

    # Property stats with variation
    print(f"\nProperty stats (across all samples):")
    for pn in prop_names:
        pv = prop_arrays[pn]
        print(f"  {pn}: mean={pv.mean():.1f}, std={pv.std():.1f}, range=[{pv.min():.0f}, {pv.max():.0f}]")

    # ---- Save ----
    results_dir = os.path.dirname(args.features_path)
    save_data = {
        'all_coeffs': all_coeffs,
        'prop_arrays': prop_arrays,
        'prop_names': prop_names,
        'seq_lens': seq_lens,
        'n_games': len(full_games),
        'args': vars(args),
    }
    save_path = os.path.join(results_dir, 'reconstruction_varied.pkl')
    pickle.dump(save_data, open(save_path, 'wb'))
    print(f"\nSaved: {save_path}")


if __name__ == '__main__':
    main()
