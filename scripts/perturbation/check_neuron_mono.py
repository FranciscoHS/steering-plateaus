#!/usr/bin/env python3
"""Test monosemanticity: how much does probe accuracy drop if we replace each
probe direction with its projection onto the best-matching neuron?

For each square/state, compare:
1. Original probe accuracy
2. Accuracy using only the component aligned with the top neuron
"""

import os, pickle
import torch
import numpy as np
import torch.nn.functional as F

from transformer_lens import HookedTransformer

PROBE_DIR = '/workspace/mech_int_othelloGPT/mech_int/probes/linear'
RESULTS_DIR = 'results/othello_plateau'
LAYER = 3
SEQ_LEN = 20
N_GAMES = 2000
SEED = 42
CENTER = {27, 28, 35, 36}

FLIP_DIRS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def _build_token_maps():
    stoi = {}
    for sq in range(27):        stoi[sq] = sq + 1
    for sq in range(29, 35):    stoi[sq] = sq - 1
    for sq in range(37, 64):    stoi[sq] = sq - 3
    return stoi, {v: k for k, v in stoi.items()}
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


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = 'cuda'

    model = HookedTransformer.from_pretrained("othello-gpt")
    model.to(device).eval()
    d_model = model.cfg.d_model

    # Load probe [1, d_model, 8, 8, 3]
    probe = torch.load(f'{PROBE_DIR}/resid_{LAYER}_linear.pth',
                       map_location='cpu', weights_only=True).float()

    # Load W_out for this layer
    W_out = model.blocks[LAYER].mlp.W_out.detach().cpu().float()  # [d_mlp, d_model]
    neuron_dirs = F.normalize(W_out, dim=1)  # [d_mlp, d_model]

    # Generate games
    print("Generating games...")
    all_games = []
    for _ in range(N_GAMES * 2):
        m, b = play_random_game(rng)
        if len(m) >= SEQ_LEN:
            all_games.append((m[:SEQ_LEN], b[:SEQ_LEN]))
        if len(all_games) >= N_GAMES:
            break
    print(f"Got {len(all_games)} games")

    # Extract activations at layer LAYER, position SEQ_LEN-1
    print(f"Extracting activations at layer {LAYER}...")
    hook_name = f"blocks.{LAYER}.hook_resid_post"
    all_acts = []
    all_boards = []
    batch_size = 64
    for i in range(0, len(all_games), batch_size):
        batch = all_games[i:i+batch_size]
        tokens = torch.tensor([g[0] for g in batch], dtype=torch.long).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        acts = cache[hook_name][:, -1, :].float().cpu()  # [batch, d_model]
        all_acts.append(acts)
        for g in batch:
            # Board state at position SEQ_LEN-1 (after move SEQ_LEN)
            # Convention: player 1 = 'moved' = mine, player 2 = opp
            # At position 19, the player who just moved alternates
            # Move 20 is by player (20 % 2 == 0 -> player 1 if 1-indexed, but 0-indexed move 19)
            # Player alternates: move 0 = player 1, move 1 = player 2, ...
            # Move 19 = player 2 (0-indexed), so 'moved' = player 2
            board = g[1][SEQ_LEN - 1]
            # Remap: moved player = mine = 1 in probe convention
            # At move 19 (0-indexed), player who moved = (19 % 2 == 1) -> player 2
            # So player 2 = mine, player 1 = opp
            remapped = []
            for sq_val in board:
                if sq_val == 0: remapped.append(0)    # empty
                elif sq_val == 2: remapped.append(1)   # player 2 = mine (just moved)
                elif sq_val == 1: remapped.append(2)   # player 1 = opp
            all_boards.append(remapped)

    all_acts = torch.cat(all_acts, dim=0)  # [N_GAMES, d_model]
    all_boards = np.array(all_boards)  # [N_GAMES, 64]
    print(f"Activations: {all_acts.shape}")

    # For each playable square, compute original and projected probe accuracy
    state_names = ['empty', 'mine', 'opp']
    results = []

    for r in range(8):
        for c in range(8):
            sq = r * 8 + c
            if sq in CENTER:
                continue

            # Original probe: [d_model, 3]
            W = probe[0, :, r, c, :]  # [d_model, 3]
            logits_orig = all_acts @ W  # [N, 3]
            preds_orig = logits_orig.argmax(dim=1).numpy()
            labels = all_boards[:, sq]
            acc_orig = (preds_orig == labels).mean()

            # For each class direction, find best neuron and project
            W_proj = torch.zeros_like(W)
            alignments = []
            for s in range(3):
                w_s = W[:, s]  # [d_model]
                w_s_norm = F.normalize(w_s.unsqueeze(0), dim=1).squeeze()
                cos_sim = w_s_norm @ neuron_dirs.T  # [d_mlp]
                best_idx = cos_sim.abs().argmax().item()
                best_cos = cos_sim[best_idx].item()
                alignments.append(abs(best_cos))
                # Project: keep only the component along best neuron
                n_dir = neuron_dirs[best_idx]
                W_proj[:, s] = (w_s @ n_dir) * n_dir

            logits_proj = all_acts @ W_proj
            preds_proj = logits_proj.argmax(dim=1).numpy()
            acc_proj = (preds_proj == labels).mean()

            # Per-class accuracy
            per_class_orig = {}
            per_class_proj = {}
            for s, sname in enumerate(state_names):
                mask = labels == s
                if mask.sum() > 0:
                    per_class_orig[sname] = (preds_orig[mask] == s).mean()
                    per_class_proj[sname] = (preds_proj[mask] == s).mean()
                else:
                    per_class_orig[sname] = float('nan')
                    per_class_proj[sname] = float('nan')

            results.append({
                'r': r, 'c': c,
                'acc_orig': acc_orig,
                'acc_proj': acc_proj,
                'alignments': alignments,
                'per_class_orig': per_class_orig,
                'per_class_proj': per_class_proj,
                'class_counts': {sname: int((labels == s).sum()) for s, sname in enumerate(state_names)},
            })

    # Summary
    accs_orig = [r['acc_orig'] for r in results]
    accs_proj = [r['acc_proj'] for r in results]
    print(f"\nLayer {LAYER} probe accuracy:")
    print(f"  Original: mean={np.mean(accs_orig):.4f}, median={np.median(accs_orig):.4f}")
    print(f"  Projected onto top neuron: mean={np.mean(accs_proj):.4f}, median={np.median(accs_proj):.4f}")
    print(f"  Drop: mean={np.mean(accs_orig) - np.mean(accs_proj):.4f}")

    # Per-class summary
    for sname in state_names:
        orig_accs = [r['per_class_orig'][sname] for r in results if not np.isnan(r['per_class_orig'][sname])]
        proj_accs = [r['per_class_proj'][sname] for r in results if not np.isnan(r['per_class_proj'][sname])]
        print(f"\n  {sname}:")
        print(f"    Original: mean={np.mean(orig_accs):.4f}")
        print(f"    Projected: mean={np.mean(proj_accs):.4f}")
        print(f"    Drop: {np.mean(orig_accs) - np.mean(proj_accs):.4f}")

    with open(os.path.join(RESULTS_DIR, f'neuron_mono_L{LAYER}.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {RESULTS_DIR}/neuron_mono_L{LAYER}.pkl")


if __name__ == '__main__':
    main()
