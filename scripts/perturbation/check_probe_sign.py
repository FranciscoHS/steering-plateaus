#!/usr/bin/env python3
"""Empirically verify which direction probe_dir pushes predictions."""

import torch, numpy as np
import torch.nn.functional as F
from transformer_lens import HookedTransformer

probe = torch.load("/workspace/mech_int_othelloGPT/mech_int/probes/linear/resid_2_linear.pth",
                   map_location="cpu", weights_only=True)

r, c = 3, 3
# This is what all our scripts use as "probe_dir":
probe_dir = (probe[0, :, r, c, 2] - probe[0, :, r, c, 1]).float()

model = HookedTransformer.from_pretrained("othello-gpt")
model.to("cuda").eval()

# Token mapping
stoi = {}
for sq in range(27):     stoi[sq] = sq + 1
for sq in range(29, 35): stoi[sq] = sq - 1
for sq in range(37, 64): stoi[sq] = sq - 3

FLIP_DIRS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def play_game(rng, n_moves=20):
    board = [0]*64; board[27]=2; board[28]=1; board[35]=1; board[36]=2
    player = 1; moves = []; boards = []
    def wf(sq,p):
        r2,c2=sq//8,sq%8; opp=3-p
        for dr,dc in FLIP_DIRS_8:
            nr,nc=r2+dr,c2+dc; found=False
            while 0<=nr<8 and 0<=nc<8:
                idx=nr*8+nc
                if board[idx]==opp: found=True
                elif board[idx]==p and found: return True
                else: break
                nr+=dr; nc+=dc
        return False
    def dm(sq,p):
        r2,c2=sq//8,sq%8; opp=3-p; board[sq]=p
        for dr,dc in FLIP_DIRS_8:
            tf=[]; nr,nc=r2+dr,c2+dc
            while 0<=nr<8 and 0<=nc<8:
                idx=nr*8+nc
                if board[idx]==opp: tf.append(idx)
                elif board[idx]==p:
                    for f in tf: board[f]=p
                    break
                else: break
                nr+=dr; nc+=dc
    while len(moves) < n_moves:
        legal=[sq for sq in range(64) if sq not in {27,28,35,36} and board[sq]==0 and wf(sq,player)]
        if not legal:
            player=3-player
            legal=[sq for sq in range(64) if sq not in {27,28,35,36} and board[sq]==0 and wf(sq,player)]
            if not legal: break
        sq=legal[rng.integers(len(legal))]; dm(sq,player)
        moves.append(stoi[sq]); boards.append(list(board)); player=3-player
    return moves, boards

rng = np.random.default_rng(42)
moves, boards = play_game(rng)

tokens = torch.tensor(moves[:20], dtype=torch.long).unsqueeze(0).cuda()
with torch.no_grad():
    _, cache = model.run_with_cache(tokens, names_filter=["blocks.2.hook_resid_post"])
act = cache["blocks.2.hook_resid_post"][0, -1].float().cpu()

class_names = ["empty", "mine", "opp"]

# Probe prediction before perturbation
logits = torch.einsum("d, d o -> o", act, probe[0, :, r, c])
pred = logits.argmax().item()
print(f"Board at ({r},{c}): {boards[19][r*8+c]} (1=black, 2=white)")
print(f"Before:      logits=[{logits[0]:.2f}, {logits[1]:.2f}, {logits[2]:.2f}], pred={class_names[pred]}")

for eps in [1.0, 2.0, 5.0]:
    # +probe_dir
    logits_p = torch.einsum("d, d o -> o", act + eps * probe_dir, probe[0, :, r, c])
    # -probe_dir
    logits_m = torch.einsum("d, d o -> o", act - eps * probe_dir, probe[0, :, r, c])
    print(f"+probe_dir x{eps}: logits=[{logits_p[0]:.2f}, {logits_p[1]:.2f}, {logits_p[2]:.2f}], pred={class_names[logits_p.argmax().item()]}")
    print(f"-probe_dir x{eps}: logits=[{logits_m[0]:.2f}, {logits_m[1]:.2f}, {logits_m[2]:.2f}], pred={class_names[logits_m.argmax().item()]}")
