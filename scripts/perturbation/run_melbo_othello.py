#!/usr/bin/env python3
"""Capped MELBO on OthelloGPT.

Finds directions that maximally break the plateau (in cosine distance terms)
at a target angle, then sweeps them to check for the bump-then-dip pattern.

Adapted from run_capped_melbo_chess.py.
"""

import os, pickle, time
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer

# Config
INJECT_LAYER = 2
MEASURE_LAYER = 6
SEQ_LEN = 20
N_GAMES = 500
SEED = 42
RESULTS_DIR = 'results/othello_plateau'
PROBE_PATH = '/workspace/mech_int_othelloGPT/mech_int/probes/linear/resid_2_linear.pth'

# MELBO config
OPT_ANGLE = 10.0       # angle at which to optimize (degrees)
TARGET_COS = 0.1       # target cosine distance (capped reward peaks here)
N_DIRECTIONS = 5
N_STEPS = 300
N_RESTARTS = 3
LR = 0.01

# Sweep config
MAX_ANGLE = 60.0
N_ANGLES = 60
N_RAND = 20

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


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = 'cuda'

    model = HookedTransformer.from_pretrained("othello-gpt")
    model.to(device).eval()
    d_model = model.cfg.d_model

    target_r, target_c = 2, 2
    target_sq = target_r * 8 + target_c
    mine_color = 2

    probe = torch.load(PROBE_PATH, map_location='cpu', weights_only=True)
    probe_mine2empty = F.normalize(
        (probe[0, :, target_r, target_c, 0] - probe[0, :, target_r, target_c, 1]).float(), dim=0)

    # Generate games
    print("Generating games...")
    all_games = []
    for _ in range(N_GAMES):
        m, b = play_random_game(rng)
        if len(m) >= SEQ_LEN:
            all_games.append((m[:SEQ_LEN], b[:SEQ_LEN]))

    mine_idx = [i for i, (m, b) in enumerate(all_games) if b[19][target_sq] == mine_color]
    print(f"Mine: {len(mine_idx)}")

    # Extract activation for MELBO (just use the first one)
    hook = f"blocks.{INJECT_LAYER}.hook_resid_post"
    act_idx = 0
    tokens = torch.tensor(all_games[mine_idx[act_idx]][0], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook])
    resid = cache[hook].float()
    ctx = resid[:, :-1, :]   # [1, seq-1, d_model]
    act = resid[0, -1, :]    # [d_model]
    print(f"Activation norm: {act.norm():.2f}")

    # Forward helpers
    def forward_to_L(ctx_d, a_pert_batch):
        """a_pert_batch: [batch, d_model]"""
        hidden = torch.cat([ctx_d.expand(a_pert_batch.shape[0], -1, -1),
                            a_pert_batch.unsqueeze(1)], dim=1)
        for block in model.blocks[INJECT_LAYER + 1:MEASURE_LAYER + 1]:
            hidden = block(hidden)
        return hidden[:, -1, :].float()

    # Reference at measure layer
    with torch.no_grad():
        ref_L = forward_to_L(ctx, act.unsqueeze(0))  # [1, d_model]

    # --- MELBO optimization ---
    a_norm = act.norm()
    a_unit = act / a_norm
    angle_rad = np.deg2rad(OPT_ANGLE)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    directions = []
    rewards = []
    histories = []

    print(f"\n{'='*60}")
    print(f"CAPPED MELBO (cos dist): target_cos={TARGET_COS}, angle={OPT_ANGLE}°")
    print(f"{'='*60}")

    for d_idx in range(N_DIRECTIONS):
        print(f"\n--- Direction {d_idx+1}/{N_DIRECTIONS} ---")
        best_reward = -float('inf')
        best_direction = None
        best_history = None

        for restart in range(N_RESTARTS):
            direction = torch.randn(d_model, device=device)
            direction = direction - (direction @ a_unit) * a_unit
            for pd in directions:
                direction = direction - (direction @ pd) * pd
            direction = F.normalize(direction, dim=0)
            direction = direction.clone().detach().requires_grad_(True)

            optimizer = torch.optim.Adam([direction], lr=LR)
            history = []

            for step in range(N_STEPS):
                optimizer.zero_grad()

                d = direction / direction.norm()
                d = d - (d @ a_unit) * a_unit
                for pd in directions:
                    d = d - (d @ pd) * pd
                d = d / (d.norm() + 1e-10)

                a_pert = a_norm * (cos_theta * a_unit + sin_theta * d)
                pert_L = forward_to_L(ctx, a_pert.unsqueeze(0))  # [1, d_model]

                cos_dist = 1 - F.cosine_similarity(ref_L, pert_L, dim=-1)  # [1]
                cos_dist = cos_dist.squeeze()
                reward = cos_dist * torch.exp(-cos_dist / TARGET_COS)

                loss = -reward
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    direction.data -= (direction.data @ a_unit) * a_unit
                    for pd in directions:
                        direction.data -= (direction.data @ pd) * pd

                history.append((reward.item(), cos_dist.item()))

                if step % 100 == 0 or step == N_STEPS - 1:
                    print(f"    restart {restart+1}/{N_RESTARTS}, step {step}: "
                          f"reward={reward.item():.6f}, cos_dist={cos_dist.item():.4f}")

            final_reward = history[-1][0]
            if final_reward > best_reward:
                best_reward = final_reward
                with torch.no_grad():
                    d_final = direction / direction.norm()
                    d_final = d_final - (d_final @ a_unit) * a_unit
                    for pd in directions:
                        d_final = d_final - (d_final @ pd) * pd
                    d_final = d_final / d_final.norm()
                    best_direction = d_final.clone()
                best_history = history

        directions.append(best_direction)
        rewards.append(best_reward)
        histories.append(best_history)
        print(f"  Best: reward={best_reward:.6f}, cos_dist={best_history[-1][1]:.4f}")

    # Compare with probe direction
    print(f"\nProbe cosine similarities:")
    for i, d in enumerate(directions):
        cos_probe = (d @ probe_mine2empty.to(device)).item()
        print(f"  MELBO dir {i+1}: cos(probe)={cos_probe:.4f}")

    # --- Sweep: cosine dist + L2 for MELBO dirs, probe, and random ---
    angles_deg = np.linspace(0, MAX_ANGLE, N_ANGLES + 1)[1:]
    angles_rad_sweep = np.deg2rad(angles_deg)
    cos_t = torch.tensor(np.cos(angles_rad_sweep), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(angles_rad_sweep), dtype=torch.float32, device=device)

    with torch.no_grad():
        ref_input = torch.cat([ctx, act[None, None]], dim=1)
        ref_L = forward_to_L(ctx, act.unsqueeze(0))

    def sweep_dir(d_vec):
        d = d_vec.to(device)
        d = d - (d @ a_unit) * a_unit
        d = F.normalize(d, dim=0)
        perturbed = a_norm * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * d[None])
        with torch.no_grad():
            pert_L = forward_to_L(ctx, perturbed)
            cos_dist = (1 - F.cosine_similarity(ref_L, pert_L, dim=-1)).cpu().numpy()
            l2_dist = (pert_L - ref_L).norm(dim=-1).cpu().numpy()
        return cos_dist, l2_dist

    print("\nSweeping MELBO directions...")
    melbo_cos = np.zeros((N_DIRECTIONS, N_ANGLES))
    melbo_l2 = np.zeros((N_DIRECTIONS, N_ANGLES))
    for i, d in enumerate(directions):
        melbo_cos[i], melbo_l2[i] = sweep_dir(d)

    print("Sweeping probe direction...")
    probe_cos, probe_l2 = sweep_dir(probe_mine2empty)

    print("Sweeping MELBO difference directions...")
    from itertools import combinations
    diff_pairs = list(combinations(range(N_DIRECTIONS), 2))
    n_diffs = len(diff_pairs)
    diff_cos = np.zeros((n_diffs, N_ANGLES))
    diff_l2 = np.zeros((n_diffs, N_ANGLES))
    for k, (i, j) in enumerate(diff_pairs):
        diff_dir = F.normalize(directions[i] - directions[j], dim=0)
        diff_cos[k], diff_l2[k] = sweep_dir(diff_dir)

    print("Sweeping random directions...")
    rand_dirs = F.normalize(torch.randn(N_RAND, d_model), dim=-1)
    rand_cos = np.zeros((N_RAND, N_ANGLES))
    rand_l2 = np.zeros((N_RAND, N_ANGLES))
    for k in range(N_RAND):
        rand_cos[k], rand_l2[k] = sweep_dir(rand_dirs[k])

    # Save
    results = {
        'directions': [d.cpu() for d in directions],
        'rewards': rewards,
        'histories': histories,
        'angles_deg': angles_deg,
        'melbo_cos': melbo_cos, 'melbo_l2': melbo_l2,
        'diff_cos': diff_cos, 'diff_l2': diff_l2,
        'diff_pairs': diff_pairs,
        'probe_cos': probe_cos, 'probe_l2': probe_l2,
        'rand_cos': rand_cos, 'rand_l2': rand_l2,
        'config': {'target_r': target_r, 'target_c': target_c,
                   'opt_angle': OPT_ANGLE, 'target_cos': TARGET_COS,
                   'n_directions': N_DIRECTIONS},
    }
    with open(os.path.join(RESULTS_DIR, 'melbo_othello.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Plot: cosine distance and L2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    rand_cos_med = np.median(rand_cos, axis=0)
    rand_cos_std = np.std(rand_cos, axis=0)
    rand_l2_med = np.median(rand_l2, axis=0)
    rand_l2_std = np.std(rand_l2, axis=0)

    for ax, probe_arr, melbo_arr, d_arr, r_med, r_std, ylabel in [
        (ax1, probe_cos, melbo_cos, diff_cos, rand_cos_med, rand_cos_std, 'Cosine distance'),
        (ax2, probe_l2, melbo_l2, diff_l2, rand_l2_med, rand_l2_std, 'L2 distance'),
    ]:
        ax.fill_between(angles_deg, r_med - r_std, r_med + r_std, color='C0', alpha=0.15)
        ax.plot(angles_deg, r_med, color='C0', lw=2, label='Random')
        ax.plot(angles_deg, probe_arr, color='C3', lw=2, label='Probe (mine→empty)')
        for i in range(N_DIRECTIONS):
            ax.plot(angles_deg, melbo_arr[i], color='C2', lw=1.5,
                    alpha=0.6, label='MELBO' if i == 0 else None)
        for k in range(n_diffs):
            ax.plot(angles_deg, d_arr[k], color='C4', lw=1, ls='--',
                    alpha=0.5, label='MELBO diff' if k == 0 else None)
        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'OthelloGPT: MELBO vs MELBO-diffs vs probe vs random, '
                 f'sq ({target_r},{target_c}), opt@{OPT_ANGLE}°', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'melbo_vs_probe.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {plot_path}")


if __name__ == '__main__':
    main()
