#!/usr/bin/env python3
"""Perturbation sweep on OthelloGPT to check for activation plateaus.

Uses TransformerLens to load OthelloGPT. Generates random legal Othello games,
extracts activations at layer 2, perturbs along random directions on the sphere,
and measures cosine distance at layer 6 (2nd to last).

Usage:
    python scripts/perturbation/sweep_othello.py
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
MEASURE_LAYER = 6   # 2nd to last (8 layers: 0-7)
SEQ_LEN = 20        # truncate games to this length
N_GAMES = 200       # games to generate (more than we need, some may be short)
N_SWEEP = 50        # activations to sweep
N_DIRS = 20         # random directions per activation
MAX_ANGLE = 60.0
N_ANGLES = 240
SEED = 42
RESULTS_DIR = 'results/othello_plateau'


# ─── Othello game engine ───────────────────────────────────────────────

# Token mapping: matches OthelloGPT's actual tokenization (1-indexed, gaps at center)
# Token 0 = pass/special, tokens 1-60 = playable squares
def _build_token_maps():
    stoi = {}
    for sq in range(27):        stoi[sq] = sq + 1       # squares 0-26 → tokens 1-27
    for sq in range(29, 35):    stoi[sq] = sq - 1       # squares 29-34 → tokens 28-33
    for sq in range(37, 64):    stoi[sq] = sq - 3       # squares 37-63 → tokens 34-60
    itos = {v: k for k, v in stoi.items()}
    return stoi, itos

SQUARE_TO_TOKEN, TOKEN_TO_SQUARE = _build_token_maps()
FLIP_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def generate_random_game(rng):
    """Play a random legal Othello game, return token sequence (tokens 0-59)."""
    board = [0] * 64  # 0=empty, 1=black, 2=white
    board[27] = 2; board[28] = 1; board[35] = 1; board[36] = 2
    player = 1  # black first
    moves = []

    def would_flip(sq, p):
        r, c = sq // 8, sq % 8
        opp = 3 - p
        for dr, dc in FLIP_DIRS:
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
        for dr, dc in FLIP_DIRS:
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
                break  # game over
        sq = legal[rng.integers(len(legal))]
        do_move(sq, player)
        moves.append(SQUARE_TO_TOKEN[sq])
        player = 3 - player

    return moves


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load OthelloGPT
    print("Loading OthelloGPT...")
    model = HookedTransformer.from_pretrained("othello-gpt")
    model.to(device)
    model.eval()
    d_model = model.cfg.d_model
    n_layers = model.cfg.n_layers
    print(f"d_model={d_model}, n_layers={n_layers}, device={device}")

    # Generate games
    print(f"\nGenerating {N_GAMES} random Othello games...")
    games = []
    for _ in range(N_GAMES):
        moves = generate_random_game(rng)
        if len(moves) >= SEQ_LEN:
            games.append(torch.tensor(moves[:SEQ_LEN], dtype=torch.long))
    print(f"Got {len(games)} games with >= {SEQ_LEN} moves")
    if len(games) < N_SWEEP:
        print(f"WARNING: only {len(games)} games, need {N_SWEEP}")
        N_SWEEP_ACTUAL = len(games)
    else:
        N_SWEEP_ACTUAL = N_SWEEP

    # Extract activations at inject_layer
    print(f"\nExtracting layer-{INJECT_LAYER} activations...")
    contexts = []
    activations = []
    hook_name = f"blocks.{INJECT_LAYER}.hook_resid_post"

    for tokens in games[:N_SWEEP_ACTUAL]:
        tokens_in = tokens.unsqueeze(0).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens_in, names_filter=[hook_name])
        resid = cache[hook_name].float()
        contexts.append(resid[:, :-1, :].cpu())
        activations.append(resid[:, -1, :].squeeze(0).cpu())

    activations_t = torch.stack(activations)
    print(f"Activation norms: mean={activations_t.norm(dim=-1).mean():.2f}, "
          f"std={activations_t.norm(dim=-1).std():.2f}")

    # Angle grid
    angles_deg = np.linspace(0, MAX_ANGLE, N_ANGLES + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)
    cos_t = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)

    # Sweep: for each activation, perturb along random directions
    print(f"\n=== Perturbation sweep ({N_SWEEP_ACTUAL} acts x {N_DIRS} dirs x {N_ANGLES} angles) ===")
    all_curves = np.zeros((N_SWEEP_ACTUAL, N_DIRS, N_ANGLES))
    t0 = time.time()

    for i in range(N_SWEEP_ACTUAL):
        ctx = contexts[i].to(device)       # [1, seq_len-1, d_model]
        act = activations[i].to(device)    # [d_model]
        R = act.norm()
        a_unit = act / R

        # Reference: forward through blocks inject+1 to measure
        with torch.no_grad():
            full_ref = torch.cat([ctx, act[None, None]], dim=1)
            hidden = full_ref
            for block in model.blocks[INJECT_LAYER + 1:MEASURE_LAYER + 1]:
                hidden = block(hidden)
            ref = hidden[:, -1, :].float()  # [1, d_model]

        for j in range(N_DIRS):
            # Random direction, orthogonalized to activation
            d = torch.randn(d_model, device=device)
            d = d - (d @ a_unit) * a_unit
            d = F.normalize(d, dim=0)

            # All angles at once: [N_ANGLES, d_model]
            perturbed = R * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * d[None])

            # Recombine with context
            ctx_batch = ctx.expand(N_ANGLES, -1, -1)
            full = torch.cat([ctx_batch, perturbed.unsqueeze(1)], dim=1)

            with torch.no_grad():
                hidden = full
                for block in model.blocks[INJECT_LAYER + 1:MEASURE_LAYER + 1]:
                    hidden = block(hidden)
                measured = hidden[:, -1, :].float()
                cos_dist = 1 - F.cosine_similarity(ref, measured, dim=-1)
                all_curves[i, j] = cos_dist.cpu().numpy()

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{N_SWEEP_ACTUAL}  ({(i+1)/elapsed:.1f} act/s)")

    total_time = time.time() - t0
    print(f"Done in {total_time:.1f}s")

    # Save results
    results = {
        'config': {
            'inject_layer': INJECT_LAYER,
            'measure_layer': MEASURE_LAYER,
            'seq_len': SEQ_LEN,
            'n_sweep': N_SWEEP_ACTUAL,
            'n_dirs': N_DIRS,
            'max_angle': MAX_ANGLE,
            'n_angles': N_ANGLES,
            'seed': SEED,
        },
        'angles_deg': angles_deg,
        'curves': all_curves,  # [n_sweep, n_dirs, n_angles]
        'norms': activations_t.norm(dim=-1).numpy(),
    }
    pkl_path = os.path.join(RESULTS_DIR, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # Plot: median per-activation (across directions), then median + IQR across activations
    per_act = np.median(all_curves, axis=1)  # [n_sweep, n_angles]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(angles_deg,
                    np.percentile(per_act, 25, axis=0),
                    np.percentile(per_act, 75, axis=0),
                    color='C0', alpha=0.15)
    ax.plot(angles_deg, np.median(per_act, axis=0),
            color='C0', lw=2.5, label=f'OthelloGPT (n={N_SWEEP_ACTUAL})')
    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel(f'Cosine distance (at layer {MEASURE_LAYER})')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MAX_ANGLE)
    ax.set_title(f'OthelloGPT: perturbation sweep, inject@L{INJECT_LAYER} measure@L{MEASURE_LAYER}')
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'plateau_sweep.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Numerical summary
    med = np.median(per_act, axis=0)
    for thresh in [0.01, 0.05, 0.1]:
        crossings = np.where(med > thresh)[0]
        if len(crossings):
            print(f"Median crosses {thresh} at {angles_deg[crossings[0]]:.1f}°")
        else:
            print(f"Median never crosses {thresh}")


if __name__ == '__main__':
    main()
