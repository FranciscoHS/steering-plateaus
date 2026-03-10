#!/usr/bin/env python3
"""Perturbation sweep along probe directions vs random directions on OthelloGPT.

For several board squares, extract the probe direction (first SVD direction of
the probe weight matrix), sweep perturbations along it, and compare against
random directions.

Usage:
    python scripts/perturbation/sweep_probe_directions.py
"""

import argparse, os, pickle
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer

# Token mapping for OthelloGPT
def _build_token_maps():
    stoi = {}
    for sq in range(27):        stoi[sq] = sq + 1
    for sq in range(29, 35):    stoi[sq] = sq - 1
    for sq in range(37, 64):    stoi[sq] = sq - 3
    return stoi, {v: k for k, v in stoi.items()}
SQ2TOK, TOK2SQ = _build_token_maps()

FLIP_DIRS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
CENTER = {27, 28, 35, 36}

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


def forward_to_L(model, ctx, a_pert_batch, inject_layer, measure_layer):
    """Forward from inject_layer to measure_layer. a_pert_batch: [batch, d_model]"""
    hidden = torch.cat([ctx.expand(a_pert_batch.shape[0], -1, -1),
                        a_pert_batch.unsqueeze(1)], dim=1)
    for block in model.blocks[inject_layer + 1:measure_layer + 1]:
        hidden = block(hidden)
    return hidden[:, -1, :].float()


def sweep_dir(model, ctx, act, ref_L, a_unit, a_norm, d_vec,
              inject_layer, measure_layer, device, angles_deg):
    """Sweep a direction across angles, return cosine distance array."""
    angles_rad = np.deg2rad(angles_deg)
    cos_t = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)

    d = d_vec.to(device).float()
    d = d - (d @ a_unit) * a_unit
    d = F.normalize(d, dim=0)
    perturbed = a_norm * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * d[None])
    with torch.no_grad():
        pert_L = forward_to_L(model, ctx, perturbed, inject_layer, measure_layer)
        cos_dist = (1 - F.cosine_similarity(ref_L, pert_L, dim=-1)).cpu().numpy()
    return cos_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inject_layer', type=int, default=2)
    parser.add_argument('--measure_layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--n_games', type=int, default=500)
    parser.add_argument('--n_squares', type=int, default=8,
                        help='Number of random squares to test')
    parser.add_argument('--n_rand', type=int, default=20)
    parser.add_argument('--max_sweep_angle', type=float, default=60.0)
    parser.add_argument('--n_sweep_angles', type=int, default=120)
    parser.add_argument('--probe_dir', type=str,
                        default='/workspace/mech_int_othelloGPT/mech_int/probes/linear')
    parser.add_argument('--results_dir', type=str, default='results/othello_plateau')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    device = 'cuda'

    model = HookedTransformer.from_pretrained("othello-gpt")
    model.to(device).eval()
    d_model = model.cfg.d_model

    # Load probe
    probe = torch.load(f'{args.probe_dir}/resid_{args.inject_layer}_linear.pth',
                        map_location='cpu', weights_only=True).float()
    # probe shape: [1, d_model, 8, 8, 3]

    # Pick random non-center squares
    playable = [sq for sq in range(64) if sq not in CENTER]
    chosen_sqs = rng.choice(playable, size=args.n_squares, replace=False)
    print(f"Selected squares: {[(sq//8, sq%8) for sq in chosen_sqs]}")

    # Extract probe directions: first SVD direction of W for each square
    probe_directions = {}
    for sq in chosen_sqs:
        r, c = sq // 8, sq % 8
        W = probe[0, :, r, c, :]  # [d_model, 3]
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        d_probe = U[:, 0]  # first left singular vector
        probe_directions[sq] = F.normalize(d_probe, dim=0)
        print(f"  Square ({r},{c}): probe SVD singular values = {S.detach().numpy()}")

    # Generate games and pick an activation
    print("Generating games...")
    all_games = []
    for _ in range(args.n_games):
        m, b = play_random_game(rng)
        if len(m) >= args.seq_len:
            all_games.append((m[:args.seq_len], b[:args.seq_len]))

    # Use first game's activation
    hook_name = f"blocks.{args.inject_layer}.hook_resid_post"
    tokens = torch.tensor(all_games[0][0], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    resid = cache[hook_name].float()
    ctx = resid[:, :-1, :]
    act = resid[0, -1, :]
    a_norm = act.norm()
    a_unit = act / a_norm
    print(f"Activation norm: {a_norm:.2f}")

    board_state = all_games[0][1][args.seq_len - 1]
    print(f"Board state at selected squares:")
    for sq in chosen_sqs:
        state_map = {0: 'empty', 1: 'black', 2: 'white'}
        print(f"  ({sq//8},{sq%8}): {state_map[board_state[sq]]}")

    # Reference
    with torch.no_grad():
        ref_L = forward_to_L(model, ctx, act.unsqueeze(0),
                             args.inject_layer, args.measure_layer)

    angles_deg = np.linspace(0, args.max_sweep_angle, args.n_sweep_angles + 1)[1:]

    # Sweep probe directions
    print("\nSweeping probe directions...")
    probe_curves = {}
    for sq in chosen_sqs:
        r, c = sq // 8, sq % 8
        curve = sweep_dir(model, ctx, act, ref_L, a_unit, a_norm,
                         probe_directions[sq], args.inject_layer,
                         args.measure_layer, device, angles_deg)
        probe_curves[sq] = curve
        print(f"  ({r},{c}): max cd = {curve.max():.4f}")

    # Sweep random directions
    print("Sweeping random directions...")
    rand_curves = np.zeros((args.n_rand, len(angles_deg)))
    for k in range(args.n_rand):
        rand_d = F.normalize(torch.randn(d_model), dim=0)
        rand_curves[k] = sweep_dir(model, ctx, act, ref_L, a_unit, a_norm,
                                   rand_d, args.inject_layer,
                                   args.measure_layer, device, angles_deg)

    # Save
    results = {
        'chosen_sqs': chosen_sqs,
        'probe_directions': {sq: d.cpu() for sq, d in probe_directions.items()},
        'probe_curves': probe_curves,
        'rand_curves': rand_curves,
        'angles_deg': angles_deg,
        'board_state': board_state,
        'args': vars(args),
    }
    pkl_path = os.path.join(args.results_dir, 'probe_sweep.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Random band
    rand_med = np.median(rand_curves, axis=0)
    rand_q25 = np.percentile(rand_curves, 25, axis=0)
    rand_q75 = np.percentile(rand_curves, 75, axis=0)
    ax.fill_between(angles_deg, rand_q25, rand_q75, color='C0', alpha=0.15)
    ax.plot(angles_deg, rand_med, color='C0', lw=2, label='Random (median)')

    # Probe directions
    state_map = {0: 'empty', 1: 'black', 2: 'white'}
    colors = plt.cm.tab10(np.linspace(0, 1, len(chosen_sqs)))
    for i, sq in enumerate(chosen_sqs):
        r, c = sq // 8, sq % 8
        state = state_map[board_state[sq]]
        ax.plot(angles_deg, probe_curves[sq], color=colors[i], lw=1.5,
                label=f'Probe ({r},{c}) [{state}]')

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('Cosine distance')
    ax.set_title(f'OthelloGPT L{args.inject_layer}→L{args.measure_layer}: '
                 f'Probe directions vs random')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'probe_sweep.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {plot_path}")
    print(f"Saved: {pkl_path}")


if __name__ == '__main__':
    main()
