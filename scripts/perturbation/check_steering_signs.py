#!/usr/bin/env python3
"""Check that probe_dir and DoM reliably steer the probe prediction.
Uses angular perturbation on the sphere, same as the sweep experiments."""

import torch, numpy as np
import torch.nn.functional as F
from transformer_lens import HookedTransformer

# Token mapping
def _build_token_maps():
    stoi = {}
    for sq in range(27):        stoi[sq] = sq + 1
    for sq in range(29, 35):    stoi[sq] = sq - 1
    for sq in range(37, 64):    stoi[sq] = sq - 3
    return stoi, {v: k for k, v in stoi.items()}
SQ2TOK, TOK2SQ = _build_token_maps()

FLIP_DIRS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

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

# Setup
model = HookedTransformer.from_pretrained("othello-gpt")
model.to("cuda").eval()
probe = torch.load("/workspace/mech_int_othelloGPT/mech_int/probes/linear/resid_2_linear.pth",
                   map_location="cpu", weights_only=True)

r, c = 3, 3
mine_color = 2  # at position 19 (odd), mine = white

probe_dir = (probe[0, :, r, c, 2] - probe[0, :, r, c, 1]).float()
probe_dir_norm = F.normalize(probe_dir, dim=0)

# Generate games, partition into mine/opp
rng = np.random.default_rng(42)
all_games = []
for _ in range(500):
    m, b = play_random_game(rng)
    if len(m) >= 20:
        all_games.append((m[:20], b[:20]))

mine_idx = [i for i, (m, b) in enumerate(all_games) if b[19][r*8+c] == mine_color]
opp_idx = [i for i, (m, b) in enumerate(all_games) if b[19][r*8+c] == 3 - mine_color]

# Extract activations
hook = "blocks.2.hook_resid_post"
mine_acts = []
for i in mine_idx[:100]:
    tokens = torch.tensor(all_games[i][0], dtype=torch.long).unsqueeze(0).cuda()
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook])
    mine_acts.append(cache[hook][0, -1].float().cpu())

opp_acts = []
for i in opp_idx[:100]:
    tokens = torch.tensor(all_games[i][0], dtype=torch.long).unsqueeze(0).cuda()
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook])
    opp_acts.append(cache[hook][0, -1].float().cpu())

mine_acts_t = torch.stack(mine_acts)
opp_acts_t = torch.stack(opp_acts)

# DoM: mean_mine - mean_opp
dom_raw = mine_acts_t.mean(0) - opp_acts_t.mean(0)
dom_dir = F.normalize(dom_raw, dim=0)

print(f"cos(probe_dir, dom_dir) = {(probe_dir_norm @ dom_dir).item():.4f}")
print()

class_names = ["empty", "mine", "opp"]

def angular_perturb(act, direction, angle_deg):
    """Perturb act on the sphere by angle_deg toward direction."""
    R = act.norm()
    a_unit = act / R
    d = direction.clone()
    d = d - (d @ a_unit) * a_unit
    d = F.normalize(d, dim=0)
    theta = np.deg2rad(angle_deg)
    return R * (np.cos(theta) * a_unit + np.sin(theta) * d)

print("=== Steering 'mine' activations (angular perturbation) ===")
for angle in [5, 10, 20, 30, 45]:
    n_flip_probe = 0
    n_flip_dom = 0
    n_total = len(mine_acts)
    for act in mine_acts:
        # +probe_dir (mine→opp)
        act_p = angular_perturb(act, probe_dir_norm, angle)
        logits_p = torch.einsum("d, d o -> o", act_p, probe[0, :, r, c])
        if logits_p.argmax().item() == 2:  # opp
            n_flip_probe += 1

        # -dom_dir (mine→opp)
        act_d = angular_perturb(act, -dom_dir, angle)
        logits_d = torch.einsum("d, d o -> o", act_d, probe[0, :, r, c])
        if logits_d.argmax().item() == 2:  # opp
            n_flip_dom += 1

    print(f"  {angle:2d} deg:  +probe flips {n_flip_probe}/{n_total}  |  -dom flips {n_flip_dom}/{n_total}")
