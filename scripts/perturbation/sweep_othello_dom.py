#!/usr/bin/env python3
"""Difference-of-means vs probe direction on OthelloGPT.

For a target square, computes the difference-of-means (DoM) direction
between activations where the square is "mine" vs "opponent", checks
alignment with the probe direction, and runs perturbation sweeps for
both directions (+/- probe, +/- DoM, random).

Usage:
    python scripts/perturbation/sweep_othello_dom.py
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
N_DIRS = 20         # random directions per activation
MAX_ANGLE = 60.0
N_ANGLES = 240
SEED = 42
RESULTS_DIR = 'results/othello_plateau'
PROBE_PATH = '/workspace/mech_int_othelloGPT/mech_int/probes/linear/resid_2_linear.pth'


# ─── Token mapping ───────────────────────────────────────────────────

def _build_token_maps():
    stoi = {}
    for sq in range(27):        stoi[sq] = sq + 1
    for sq in range(29, 35):    stoi[sq] = sq - 1
    for sq in range(37, 64):    stoi[sq] = sq - 3
    itos = {v: k for k, v in stoi.items()}
    return stoi, itos

SQUARE_TO_TOKEN, TOKEN_TO_SQUARE = _build_token_maps()
FLIP_DIRS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ─── Othello game engine ─────────────────────────────────────────────

def play_random_game(rng):
    board = [0] * 64
    board[27] = 2; board[28] = 1; board[35] = 1; board[36] = 2
    player = 1
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
                if board[idx] == opp: found = True
                elif board[idx] == p and found: return True
                else: break
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
                if board[idx] == opp: to_flip.append(idx)
                elif board[idx] == p:
                    for f in to_flip: board[f] = p
                    break
                else: break
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
            if not legal: break
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
    model.to(device).eval()
    d_model = model.cfg.d_model

    # Load probe
    print(f"Loading probe from {PROBE_PATH}")
    probe = torch.load(PROBE_PATH, map_location='cpu', weights_only=True)

    # Generate games
    print(f"\nGenerating {N_GAMES} random Othello games...")
    all_games = []
    for _ in range(N_GAMES):
        m, b = play_random_game(rng)
        if len(m) >= SEQ_LEN:
            all_games.append((m[:SEQ_LEN], b[:SEQ_LEN]))
    print(f"Got {len(all_games)} games with >= {SEQ_LEN} moves")

    # Convention: at position 19 (odd), mine = white = 2
    last_pos = SEQ_LEN - 1
    mine_color = 2  # white at odd position
    opp_color = 1

    # Find target square: most frequent "mine"
    square_counts = np.zeros((8, 8), dtype=int)
    for moves, boards in all_games:
        board = boards[last_pos]
        for r in range(8):
            for c in range(8):
                if board[r * 8 + c] == mine_color:
                    square_counts[r, c] += 1

    best_r, best_c = np.unravel_index(square_counts.argmax(), (8, 8))
    best_sq = best_r * 8 + best_c
    n_mine = square_counts[best_r, best_c]
    print(f"\nTarget square: ({best_r},{best_c}) = sq {best_sq}, "
          f"mine in {n_mine}/{len(all_games)} ({100*n_mine/len(all_games):.0f}%)")

    # Partition games by square state
    mine_indices = []
    opp_indices = []
    for idx, (m, b) in enumerate(all_games):
        state = b[last_pos][best_sq]
        if state == mine_color:
            mine_indices.append(idx)
        elif state == opp_color:
            opp_indices.append(idx)
    print(f"Mine: {len(mine_indices)}, Opponent: {len(opp_indices)}")

    # Extract all activations
    print(f"\nExtracting layer-{INJECT_LAYER} activations...")
    hook_name = f"blocks.{INJECT_LAYER}.hook_resid_post"
    all_contexts = {}
    all_activations = {}

    for idx in mine_indices + opp_indices:
        m, _ = all_games[idx]
        tokens = torch.tensor(m, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        resid = cache[hook_name].float()
        all_contexts[idx] = resid[:, :-1, :].cpu()
        all_activations[idx] = resid[:, -1, :].squeeze(0).cpu()

    # Compute difference-of-means
    mine_acts = torch.stack([all_activations[i] for i in mine_indices])
    opp_acts = torch.stack([all_activations[i] for i in opp_indices])
    dom_raw = mine_acts.mean(dim=0) - opp_acts.mean(dim=0)
    dom_dir = F.normalize(dom_raw, dim=0)
    print(f"DoM norm (before normalization): {dom_raw.norm():.3f}")

    # Probe direction: opponent - mine (same as targeted probe script)
    probe_dir_raw = (probe[0, :, best_r, best_c, 2] - probe[0, :, best_r, best_c, 1]).float()
    probe_dir = F.normalize(probe_dir_raw, dim=0)
    print(f"Probe norm (before normalization): {probe_dir_raw.norm():.3f}")

    # Alignment
    cos_sim = (dom_dir @ probe_dir).item()
    # Note: probe_dir points mine→opp, dom_dir points opp→mine (mean_mine - mean_opp)
    # So we expect negative alignment
    print(f"\nDoM · probe = {cos_sim:.4f}")
    print(f"DoM · (-probe) = {-cos_sim:.4f}")

    # Angle grid
    angles_deg = np.linspace(0, MAX_ANGLE, N_ANGLES + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)
    cos_t = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)

    def sweep_dir(ctx, act, direction):
        """Sweep one direction for one activation. Returns [n_angles]."""
        act_d = act.to(device)
        ctx_d = ctx.to(device)
        R = act_d.norm()
        a_unit = act_d / R

        with torch.no_grad():
            h = torch.cat([ctx_d, act_d[None, None]], dim=1)
            for block in model.blocks[INJECT_LAYER + 1:MEASURE_LAYER + 1]:
                h = block(h)
            ref = h[:, -1, :].float()

        d = direction.to(device)
        d = d - (d @ a_unit) * a_unit
        d = F.normalize(d, dim=0)

        perturbed = R * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * d[None])
        full = torch.cat([ctx_d.expand(N_ANGLES, -1, -1), perturbed.unsqueeze(1)], dim=1)
        with torch.no_grad():
            h2 = full
            for block in model.blocks[INJECT_LAYER + 1:MEASURE_LAYER + 1]:
                h2 = block(h2)
            return (1 - F.cosine_similarity(ref, h2[:, -1, :].float(), dim=-1)).cpu().numpy()

    def sweep_random(ctx, act, n_dirs):
        """Sweep random directions. Returns [n_dirs, n_angles]."""
        curves = np.zeros((n_dirs, N_ANGLES))
        for j in range(n_dirs):
            d = F.normalize(torch.randn(d_model), dim=0)
            curves[j] = sweep_dir(ctx, act, d)
        return curves

    # Sweep on "mine" activations
    n_sweep = min(len(mine_indices), 50)
    sweep_indices = mine_indices[:n_sweep]

    print(f"\n=== Sweep: probe, DoM, random x {n_sweep} activations ===")
    probe_plus_curves = np.zeros((n_sweep, N_ANGLES))
    probe_minus_curves = np.zeros((n_sweep, N_ANGLES))
    dom_plus_curves = np.zeros((n_sweep, N_ANGLES))    # toward mine (reinforce)
    dom_minus_curves = np.zeros((n_sweep, N_ANGLES))   # toward opp (change)
    rand_curves = np.zeros((n_sweep, N_DIRS, N_ANGLES))
    t0 = time.time()

    for i, idx in enumerate(sweep_indices):
        ctx = all_contexts[idx]
        act = all_activations[idx]
        probe_plus_curves[i] = sweep_dir(ctx, act, probe_dir)    # opp - mine direction
        probe_minus_curves[i] = sweep_dir(ctx, act, -probe_dir)  # mine - opp direction
        dom_plus_curves[i] = sweep_dir(ctx, act, dom_dir)        # toward mine
        dom_minus_curves[i] = sweep_dir(ctx, act, -dom_dir)      # toward opp
        rand_curves[i] = sweep_random(ctx, act, N_DIRS)
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
            'n_mine': len(mine_indices),
            'n_opp': len(opp_indices),
            'n_dirs': N_DIRS,
            'seed': SEED,
        },
        'angles_deg': angles_deg,
        'probe_plus_curves': probe_plus_curves,
        'probe_minus_curves': probe_minus_curves,
        'dom_plus_curves': dom_plus_curves,
        'dom_minus_curves': dom_minus_curves,
        'rand_curves': rand_curves,
        'dom_probe_cosine': cos_sim,
        'dom_norm': dom_raw.norm().item(),
        'probe_norm': probe_dir_raw.norm().item(),
    }
    pkl_path = os.path.join(RESULTS_DIR, 'dom_vs_probe.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # Plot
    rand_per_act = np.median(rand_curves, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Random
    ax.fill_between(angles_deg, np.percentile(rand_per_act, 25, axis=0),
                    np.percentile(rand_per_act, 75, axis=0), color='C0', alpha=0.12)
    ax.plot(angles_deg, np.median(rand_per_act, axis=0),
            color='C0', lw=2.5, label='Random')

    # +probe (mine→opp, should break plateau)
    ax.fill_between(angles_deg, np.percentile(probe_plus_curves, 25, axis=0),
                    np.percentile(probe_plus_curves, 75, axis=0), color='C3', alpha=0.12)
    ax.plot(angles_deg, np.median(probe_plus_curves, axis=0),
            color='C3', lw=2.5, label='Probe (mine→opp)')

    # -probe (opp→mine, reinforce)
    ax.fill_between(angles_deg, np.percentile(probe_minus_curves, 25, axis=0),
                    np.percentile(probe_minus_curves, 75, axis=0), color='C3', alpha=0.12, linestyle='--')
    ax.plot(angles_deg, np.median(probe_minus_curves, axis=0),
            color='C3', lw=2.5, ls='--', label='Probe (opp→mine)')

    # +DoM (toward mine, reinforce)
    ax.fill_between(angles_deg, np.percentile(dom_plus_curves, 25, axis=0),
                    np.percentile(dom_plus_curves, 75, axis=0), color='C2', alpha=0.12)
    ax.plot(angles_deg, np.median(dom_plus_curves, axis=0),
            color='C2', lw=2.5, label='DoM (opp→mine)')

    # -DoM (toward opp, should break plateau)
    ax.fill_between(angles_deg, np.percentile(dom_minus_curves, 25, axis=0),
                    np.percentile(dom_minus_curves, 75, axis=0), color='C2', alpha=0.12)
    ax.plot(angles_deg, np.median(dom_minus_curves, axis=0),
            color='C2', lw=2.5, ls='--', label='DoM (mine→opp)')

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel(f'Cosine distance (at layer {MEASURE_LAYER})')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MAX_ANGLE)
    ax.set_title(f'OthelloGPT: DoM vs probe, square ({best_r},{best_c}), '
                 f'cos(DoM,probe)={cos_sim:.3f}')
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'dom_vs_probe.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Summary
    print(f"\nAlignment: cos(DoM, probe) = {cos_sim:.4f}")
    for name, curves in [('Random', rand_per_act),
                         ('Probe mine→opp', probe_plus_curves),
                         ('Probe opp→mine', probe_minus_curves),
                         ('DoM opp→mine', dom_plus_curves),
                         ('DoM mine→opp', dom_minus_curves)]:
        med = np.median(curves, axis=0)
        for thresh in [0.01, 0.05, 0.1]:
            crossings = np.where(med > thresh)[0]
            if len(crossings):
                print(f"{name}: median crosses {thresh} at {angles_deg[crossings[0]]:.1f}°")
            else:
                print(f"{name}: median never crosses {thresh}")


if __name__ == '__main__':
    main()
