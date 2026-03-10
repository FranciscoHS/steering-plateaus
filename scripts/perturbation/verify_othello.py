#!/usr/bin/env python3
"""Verify OthelloGPT tokenization and probe convention."""

import sys, torch, numpy as np
from transformer_lens import HookedTransformer

FLIP_DIRS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# Token mapping: matches OthelloGPT's actual tokenization (1-indexed, gaps at center)
def _build_token_maps():
    stoi = {}
    for sq in range(27):        stoi[sq] = sq + 1
    for sq in range(29, 35):    stoi[sq] = sq - 1
    for sq in range(37, 64):    stoi[sq] = sq - 3
    itos = {v: k for k, v in stoi.items()}
    return stoi, itos
SQ2TOK, TOK2SQ = _build_token_maps()

def play_random_game(rng):
    board = [0]*64; board[27]=2; board[28]=1; board[35]=1; board[36]=2
    player = 1; moves = []; board_states = []
    def wf(sq,p):
        r,c=sq//8,sq%8; opp=3-p
        for dr,dc in FLIP_DIRS_8:
            nr,nc=r+dr,c+dc; found=False
            while 0<=nr<8 and 0<=nc<8:
                idx=nr*8+nc
                if board[idx]==opp: found=True
                elif board[idx]==p and found: return True
                else: break
                nr+=dr; nc+=dc
        return False
    def dm(sq,p):
        r,c=sq//8,sq%8; opp=3-p; board[sq]=p
        for dr,dc in FLIP_DIRS_8:
            tf=[]; nr,nc=r+dr,c+dc
            while 0<=nr<8 and 0<=nc<8:
                idx=nr*8+nc
                if board[idx]==opp: tf.append(idx)
                elif board[idx]==p:
                    for f in tf: board[f]=p
                    break
                else: break
                nr+=dr; nc+=dc
    while True:
        legal=[sq for sq in range(64) if sq not in {27,28,35,36} and board[sq]==0 and wf(sq,player)]
        if not legal:
            player=3-player
            legal=[sq for sq in range(64) if sq not in {27,28,35,36} and board[sq]==0 and wf(sq,player)]
            if not legal: break
        sq=legal[rng.integers(len(legal))]; dm(sq,player)
        moves.append(SQ2TOK[sq]); board_states.append(list(board)); player=3-player
    return moves, board_states

def get_legal_moves(board, player):
    legal = []
    for sq in range(64):
        if sq in {27,28,35,36} or board[sq] != 0:
            continue
        r, c = sq // 8, sq % 8
        opp = 3 - player
        for dr, dc in FLIP_DIRS_8:
            nr, nc = r + dr, c + dc
            found = False
            while 0 <= nr < 8 and 0 <= nc < 8:
                idx = nr * 8 + nc
                if board[idx] == opp:
                    found = True
                elif board[idx] == player and found:
                    legal.append(sq)
                    break
                else:
                    break
                nr += dr; nc += dc
            if legal and legal[-1] == sq:
                break
    return legal


rng = np.random.default_rng(42)
games = []
for _ in range(20):
    m, b = play_random_game(rng)
    if len(m) >= 20:
        games.append((m[:20], b[:20]))

model = HookedTransformer.from_pretrained('othello-gpt')
model.to('cuda').eval()

# Check 1: Does model predict legal next moves?
print("=== Check 1: Model predicts legal moves? ===")
n_legal_in_top5 = 0
n_total = 0
for moves, boards in games[:10]:
    tokens = torch.tensor(moves, dtype=torch.long).unsqueeze(0).cuda()
    with torch.no_grad():
        logits = model(tokens)
    for pos in range(19):
        top5 = logits[0, pos].topk(5).indices.cpu().tolist()
        board = boards[pos]
        next_player = 2 if pos % 2 == 0 else 1
        legal_sqs = get_legal_moves(board, next_player)
        legal_tokens = set(SQ2TOK[sq] for sq in legal_sqs)
        hits = sum(1 for t in top5 if t in legal_tokens)
        n_legal_in_top5 += hits
        n_total += 1

pct = 100 * n_legal_in_top5 / (n_total * 5)
print(f"Top-5 predictions that are legal: {n_legal_in_top5}/{n_total*5} = {pct:.1f}%")

# Check 2: Compare board states with original OthelloBoardState
print("\n=== Check 2: Board state comparison with original code ===")
try:
    sys.path.insert(0, '/workspace/mech_int_othelloGPT')
    from data.othello import OthelloBoardState

    rng2 = np.random.default_rng(42)
    moves_ours, boards_ours = play_random_game(rng2)

    orig_board = OthelloBoardState()
    mismatches = 0
    for i, tok in enumerate(moves_ours[:20]):
        raw_sq = TOK2SQ[tok]
        orig_board.umpire(raw_sq)
        orig_state = orig_board.state.flatten()
        our_state = boards_ours[i]
        for j in range(64):
            o, u = orig_state[j], our_state[j]
            ok = (o == 0 and u == 0) or (o == 1 and u == 1) or (o == -1 and u == 2)
            if not ok:
                print(f"  MISMATCH move {i}, sq {j}: orig={o}, ours={u}")
                mismatches += 1
    if mismatches == 0:
        print("All board states match!")
except ImportError as e:
    print(f"  Skipped (missing dependency: {e})")

# Check 3: Probe accuracy with both conventions, layer 6 (should be high)
print("\n=== Check 3: Probe accuracy ===")
for layer in [2, 6]:
    probe = torch.load(
        f'/workspace/mech_int_othelloGPT/mech_int/probes/linear/resid_{layer}_linear.pth',
        map_location='cpu', weights_only=True)
    hook_name = f'blocks.{layer}.hook_resid_post'

    for convention in ['moved', 'to_move']:
        correct = 0; total = 0
        for moves, boards in games[:10]:
            tokens = torch.tensor(moves, dtype=torch.long).unsqueeze(0).cuda()
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            resid = cache[hook_name].float()

            for pos in range(5, 20):
                act = resid[0, pos]
                pred = torch.einsum('d, d r c o -> r c o', act.cpu(), probe[0])
                pred_class = pred.argmax(dim=-1)

                board = boards[pos]
                if convention == 'moved':
                    mine_color = 1 if pos % 2 == 0 else 2
                else:
                    mine_color = 2 if pos % 2 == 0 else 1

                for r in range(8):
                    for c in range(8):
                        actual = board[r * 8 + c]
                        if actual == 0:
                            expected = 0
                        elif actual == mine_color:
                            expected = 1
                        else:
                            expected = 2
                        if pred_class[r, c] == expected:
                            correct += 1
                        total += 1
        pct = 100 * correct / total
        print(f"  Layer {layer}, convention '{convention}': {pct:.1f}%")
