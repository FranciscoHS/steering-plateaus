#!/usr/bin/env python3
"""Find chess game activations with few pieces (endgame positions)."""
import sys, re, torch
sys.path.insert(0, '.')
from src.data import generate_chess_games
import chess

ITOS = {0: ' ', 1: '#', 2: '+', 3: '-', 4: '.', 5: '0', 6: '1', 7: '2',
        8: '3', 9: '4', 10: '5', 11: '6', 12: '7', 13: '8', 14: '9',
        15: ';', 16: '=', 17: 'B', 18: 'K', 19: 'N', 20: 'O', 21: 'Q',
        22: 'R', 23: 'a', 24: 'b', 25: 'c', 26: 'd', 27: 'e', 28: 'f',
        29: 'g', 30: 'h', 31: 'x'}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=300)
parser.add_argument('--n_games', type=int, default=500)
parser.add_argument('--max_pieces', type=int, default=12)
args = parser.parse_args()

token_ids_list = generate_chess_games(args.n_games, args.seq_len, seed=42)

for i in range(len(token_ids_list)):
    tokens = token_ids_list[i]
    pgn = ''.join(ITOS[t.item()] for t in tokens)
    pgn_clean = pgn.lstrip(';')
    moves_str = re.sub(r'\d+\.', '', pgn_clean).split()
    board = chess.Board()
    for m in moves_str:
        m = m.strip()
        if not m:
            continue
        try:
            board.push(board.parse_san(m))
        except:
            break
    n_pieces = len(board.piece_map())
    n_moves = len(board.move_stack)
    if n_pieces <= args.max_pieces:
        pieces = []
        for sq, p in board.piece_map().items():
            pieces.append(f"{p.symbol()}{chess.square_name(sq)}")
        print(f"act_idx={i}: {n_moves} moves, {n_pieces} pieces: {' '.join(pieces)}")
        print(f"  FEN: {board.fen()}")
        print(board)
        print()
