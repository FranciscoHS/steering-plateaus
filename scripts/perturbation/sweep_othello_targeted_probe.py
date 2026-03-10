#!/usr/bin/env python3
"""Targeted probe perturbation on OthelloGPT.

Picks a specific square, filters to activations where that square is "mine",
then perturbs along the mine→opponent probe direction vs random directions.
This tests whether the probe direction breaks the plateau when applied to
activations where it's actually relevant.

Usage:
    python scripts/perturbation/sweep_othello_targeted_probe.py
"""

import os, sys, pickle, time
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
N_GAMES = 500       # generate more to have enough after filtering
N_DIRS = 20         # random directions per activation
MAX_ANGLE = 60.0
N_ANGLES = 240
SEED = 42
RESULTS_DIR = 'results/othello_plateau'
PROBE_PATH = '/workspace/mech_int_othelloGPT/mech_int/probes/linear/resid_2_linear.pth'


# ─── Othello game engine (with board state tracking) ───────────────────

# Token mapping: matches OthelloGPT's actual tokenization (1-indexed, gaps at center)
def _build_token_maps():
    stoi = {}
    for sq in range(27):        stoi[sq] = sq + 1
    for sq in range(29, 35):    stoi[sq] = sq - 1
    for sq in range(37, 64):    stoi[sq] = sq - 3
    itos = {v: k for k, v in stoi.items()}
    return stoi, itos

SQUARE_TO_TOKEN, TOKEN_TO_SQUARE = _build_token_maps()
FLIP_DIRS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def play_random_game(rng):
    """Play a random legal Othello game. Returns (token_list, board_states).

    board_states[k] is the 8x8 board AFTER move k has been played.
    Board values: 0=empty, 1=black, 2=white.
    """
    board = [0] * 64
    board[27] = 2; board[28] = 1; board[35] = 1; board[36] = 2
    player = 1  # 1=black, 2=white; black goes first
    moves = []
    board_states = []

    def would_flip(sq, p):
        r, c = sq // 8, sq % 8
        opp = 3 - p
        for dr, dc in FLIP_DIRS_8:
            nr, nc = r + dr, c + dc
            found = False
            while 0 <= nr < 8 and 0 <= nc < 8:
                idx = nr * 8 + nc
                if board[idx] == opp:
                    found = True
                elif board[idx] == p and found:
                    return True
                else:
                    break
                nr += dr; nc += dc
        return False

    def do_move(sq, p):
        r, c = sq // 8, sq % 8
        opp = 3 - p
        board[sq] = p
        for dr, dc in FLIP_DIRS_8:
            to_flip = []
            nr, nc = r + dr, c + dc
            while 0 <= nr < 8 and 0 <= nc < 8:
                idx = nr * 8 + nc
                if board[idx] == opp:
                    to_flip.append(idx)
                elif board[idx] == p:
                    for f in to_flip:
                        board[f] = p
                    break
                else:
                    break
                nr += dr; nc += dc

    while True:
        legal = [sq for sq in range(64)
                 if sq not in {27, 28, 35, 36} and board[sq] == 0
                 and would_flip(sq, player)]
        if not legal:
            player = 3 - player
            legal = [sq for sq in range(64)
                     if sq not in {27, 28, 35, 36} and board[sq] == 0
                     and would_flip(sq, player)]
            if not legal:
                break
        sq = legal[rng.integers(len(legal))]
        do_move(sq, player)
        moves.append(SQUARE_TO_TOKEN[sq])
        board_states.append(list(board))
        player = 3 - player

    return moves, board_states


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    print("Loading OthelloGPT...")
    model = HookedTransformer.from_pretrained("othello-gpt")
    model.to(device)
    model.eval()
    d_model = model.cfg.d_model
    print(f"d_model={d_model}, n_layers={model.cfg.n_layers}, device={device}")

    # Load probe
    print(f"Loading probe from {PROBE_PATH}")
    probe = torch.load(PROBE_PATH, map_location='cpu', weights_only=True)
    # Shape: (1, 512, 8, 8, 3) = (modes, d_model, rows, cols, classes)
    # Classes: 0=empty, 1=mine, 2=opponent

    # Generate games with board state tracking
    print(f"\nGenerating {N_GAMES} random Othello games...")
    all_games = []
    for _ in range(N_GAMES):
        moves, boards = play_random_game(rng)
        if len(moves) >= SEQ_LEN:
            all_games.append((moves[:SEQ_LEN], boards[:SEQ_LEN]))
    print(f"Got {len(all_games)} games with >= {SEQ_LEN} moves")

    # At position SEQ_LEN-1 (last token), figure out "mine" convention.
    # Move k: black if k even, white if k odd.
    # At position SEQ_LEN-1 = 19 (odd), the player who just moved is white.
    # Convention: at even positions mine=black, at odd positions mine=white.
    last_pos = SEQ_LEN - 1
    last_player = 1 if last_pos % 2 == 0 else 2  # who just moved
    mine_color = last_player  # "mine" = the player who just moved at this position
    print(f"At position {last_pos}: last player={'black' if mine_color==1 else 'white'}, "
          f"mine={mine_color}")

    # Find a good target square: one that's frequently "mine" at the last position
    square_counts = np.zeros((8, 8), dtype=int)
    for moves, boards in all_games:
        board = boards[last_pos]
        for r in range(8):
            for c in range(8):
                if board[r * 8 + c] == mine_color:
                    square_counts[r, c] += 1

    # Pick the square with the most "mine" occurrences
    best_r, best_c = np.unravel_index(square_counts.argmax(), (8, 8))
    n_mine = square_counts[best_r, best_c]
    print(f"\nTarget square: ({best_r},{best_c}), "
          f"mine in {n_mine}/{len(all_games)} games ({100*n_mine/len(all_games):.0f}%)")

    # Filter games where target square is "mine" at last position
    filtered_indices = []
    for idx, (moves, boards) in enumerate(all_games):
        if boards[last_pos][best_r * 8 + best_c] == mine_color:
            filtered_indices.append(idx)
    print(f"Filtered to {len(filtered_indices)} games")

    # Probe direction: mine→opponent for this square
    # probe[0, :, r, c, 1] = "mine" direction, probe[0, :, r, c, 2] = "opponent" direction
    probe_dir_raw = probe[0, :, best_r, best_c, 2] - probe[0, :, best_r, best_c, 1]
    probe_dir = F.normalize(probe_dir_raw.float(), dim=0)
    print(f"Probe direction norm (before normalization): {probe_dir_raw.norm():.3f}")

    # Extract activations for filtered games
    print(f"\nExtracting layer-{INJECT_LAYER} activations...")
    contexts = []
    activations = []
    hook_name = f"blocks.{INJECT_LAYER}.hook_resid_post"

    for idx in filtered_indices:
        moves, _ = all_games[idx]
        tokens = torch.tensor(moves, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        resid = cache[hook_name].float()
        contexts.append(resid[:, :-1, :].cpu())
        activations.append(resid[:, -1, :].squeeze(0).cpu())

    n_sweep = min(len(activations), 50)
    activations_t = torch.stack(activations[:n_sweep])
    print(f"Sweeping {n_sweep} activations, "
          f"norms: mean={activations_t.norm(dim=-1).mean():.2f}")

    # Angle grid
    angles_deg = np.linspace(0, MAX_ANGLE, N_ANGLES + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)
    cos_t = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)

    def sweep_dirs(ctx, act, directions):
        """Sweep multiple directions for one activation. Returns [n_dirs, n_angles]."""
        act_d = act.to(device)
        ctx_d = ctx.to(device)
        R = act_d.norm()
        a_unit = act_d / R

        with torch.no_grad():
            full_ref = torch.cat([ctx_d, act_d[None, None]], dim=1)
            hidden = full_ref
            for block in model.blocks[INJECT_LAYER + 1:MEASURE_LAYER + 1]:
                hidden = block(hidden)
            ref = hidden[:, -1, :].float()

        n_d = directions.shape[0]
        curves = np.zeros((n_d, N_ANGLES))
        for j in range(n_d):
            d = directions[j].to(device)
            d = d - (d @ a_unit) * a_unit
            d = F.normalize(d, dim=0)

            perturbed = R * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * d[None])
            ctx_batch = ctx_d.expand(N_ANGLES, -1, -1)
            full = torch.cat([ctx_batch, perturbed.unsqueeze(1)], dim=1)

            with torch.no_grad():
                hidden = full
                for block in model.blocks[INJECT_LAYER + 1:MEASURE_LAYER + 1]:
                    hidden = block(hidden)
                measured = hidden[:, -1, :].float()
                cos_dist = 1 - F.cosine_similarity(ref, measured, dim=-1)
                curves[j] = cos_dist.cpu().numpy()
        return curves

    # Random directions (fixed across activations for fair comparison)
    random_dirs = F.normalize(torch.randn(N_DIRS, d_model), dim=-1)

    # Sweep: probe direction + random directions for each activation
    print(f"\n=== Sweep: 1 probe dir + {N_DIRS} random dirs x {n_sweep} activations ===")
    probe_curves = np.zeros((n_sweep, N_ANGLES))
    rand_curves = np.zeros((n_sweep, N_DIRS, N_ANGLES))
    t0 = time.time()

    for i in range(n_sweep):
        # Probe direction (single)
        probe_curves[i] = sweep_dirs(
            contexts[i], activations[i], probe_dir.unsqueeze(0))[0]
        # Random directions
        rand_curves[i] = sweep_dirs(contexts[i], activations[i], random_dirs)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_sweep}  ({(i+1)/(time.time()-t0):.1f} act/s)")

    print(f"Done in {time.time()-t0:.1f}s")

    # Save
    results = {
        'config': {
            'inject_layer': INJECT_LAYER,
            'measure_layer': MEASURE_LAYER,
            'seq_len': SEQ_LEN,
            'target_square': (best_r, best_c),
            'mine_color': mine_color,
            'n_sweep': n_sweep,
            'n_dirs': N_DIRS,
            'max_angle': MAX_ANGLE,
            'n_angles': N_ANGLES,
            'seed': SEED,
        },
        'angles_deg': angles_deg,
        'probe_curves': probe_curves,  # [n_sweep, n_angles]
        'rand_curves': rand_curves,    # [n_sweep, n_dirs, n_angles]
    }
    pkl_path = os.path.join(RESULTS_DIR, 'targeted_probe.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # Plot
    rand_per_act = np.median(rand_curves, axis=1)  # [n_sweep, n_angles]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Random directions
    ax.fill_between(angles_deg,
                    np.percentile(rand_per_act, 25, axis=0),
                    np.percentile(rand_per_act, 75, axis=0),
                    color='C0', alpha=0.15)
    ax.plot(angles_deg, np.median(rand_per_act, axis=0),
            color='C0', lw=2.5, label=f'Random directions (n={N_DIRS})')

    # Probe direction (one curve per activation, show median + IQR)
    ax.fill_between(angles_deg,
                    np.percentile(probe_curves, 25, axis=0),
                    np.percentile(probe_curves, 75, axis=0),
                    color='C3', alpha=0.15)
    ax.plot(angles_deg, np.median(probe_curves, axis=0),
            color='C3', lw=2.5,
            label=f'Probe: ({best_r},{best_c}) mine→opp')

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel(f'Cosine distance (at layer {MEASURE_LAYER})')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MAX_ANGLE)
    ax.set_title(f'OthelloGPT: targeted probe vs random, square ({best_r},{best_c}), '
                 f'inject@L{INJECT_LAYER} measure@L{MEASURE_LAYER}')
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'targeted_probe.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Numerical summary
    for name, per_act in [('Random', rand_per_act), ('Probe', probe_curves)]:
        med = np.median(per_act, axis=0)
        for thresh in [0.01, 0.05, 0.1]:
            crossings = np.where(med > thresh)[0]
            if len(crossings):
                print(f"{name}: median crosses {thresh} at {angles_deg[crossings[0]]:.1f}°")
            else:
                print(f"{name}: median never crosses {thresh}")


if __name__ == '__main__':
    main()
