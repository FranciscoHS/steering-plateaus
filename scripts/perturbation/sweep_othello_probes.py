#!/usr/bin/env python3
"""Perturbation sweep on OthelloGPT: probe directions vs random directions.

Loads OthelloGPT linear probes at layer 2, perturbs activations along probe
directions and random directions, measures cosine distance at layer 6.
Plots both on the same figure.

Usage:
    python scripts/perturbation/sweep_othello_probes.py
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
N_GAMES = 200
N_SWEEP = 50
N_RANDOM_DIRS = 20
MAX_ANGLE = 60.0
N_ANGLES = 240
SEED = 42
RESULTS_DIR = 'results/othello_plateau'
PROBE_PATH = '/workspace/mech_int_othelloGPT/mech_int/probes/linear/resid_2_linear.pth'


# ─── Othello game engine ───────────────────────────────────────────────

# Token mapping: matches OthelloGPT's actual tokenization (1-indexed, gaps at center)
def _build_token_maps():
    stoi = {}
    for sq in range(27):        stoi[sq] = sq + 1
    for sq in range(29, 35):    stoi[sq] = sq - 1
    for sq in range(37, 64):    stoi[sq] = sq - 3
    itos = {v: k for k, v in stoi.items()}
    return stoi, itos

SQUARE_TO_TOKEN, TOKEN_TO_SQUARE = _build_token_maps()
FLIP_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def generate_random_game(rng):
    """Play a random legal Othello game, return token sequence (tokens 0-59)."""
    board = [0] * 64
    board[27] = 2; board[28] = 1; board[35] = 1; board[36] = 2
    player = 1
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
                break
        sq = legal[rng.integers(len(legal))]
        do_move(sq, player)
        moves.append(SQUARE_TO_TOKEN[sq])
        player = 3 - player

    return moves


# ─── Sweep helpers ─────────────────────────────────────────────────────

def sweep_directions(model, ctx, act, directions, inject_layer, measure_layer,
                     cos_t, sin_t, device):
    """Sweep multiple directions for one activation, return [n_dirs, n_angles] cosine distances."""
    n_dirs = directions.shape[0]
    n_angles = len(cos_t)
    R = act.norm()
    a_unit = act / R

    # Reference
    with torch.no_grad():
        full_ref = torch.cat([ctx, act[None, None]], dim=1)
        hidden = full_ref
        for block in model.blocks[inject_layer + 1:measure_layer + 1]:
            hidden = block(hidden)
        ref = hidden[:, -1, :].float()

    results = np.zeros((n_dirs, n_angles))
    for j in range(n_dirs):
        d = directions[j].to(device)
        # Orthogonalize and normalize
        d = d - (d @ a_unit) * a_unit
        d = F.normalize(d, dim=0)

        perturbed = R * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * d[None])
        ctx_batch = ctx.expand(n_angles, -1, -1)
        full = torch.cat([ctx_batch, perturbed.unsqueeze(1)], dim=1)

        with torch.no_grad():
            hidden = full
            for block in model.blocks[inject_layer + 1:measure_layer + 1]:
                hidden = block(hidden)
            measured = hidden[:, -1, :].float()
            cos_dist = 1 - F.cosine_similarity(ref, measured, dim=-1)
            results[j] = cos_dist.cpu().numpy()

    return results


# ─── Main ──────────────────────────────────────────────────────────────

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

    # Load probe directions
    print(f"Loading probe from {PROBE_PATH}")
    probe = torch.load(PROBE_PATH, map_location='cpu', weights_only=True)
    # probe shape: (1, 512, 8, 8, 3) = (modes, d_model, rows, cols, classes)
    # Reshape to (n_probe_dirs, d_model): flatten (rows, cols, classes)
    probe_dirs = probe[0].permute(1, 2, 3, 0).reshape(-1, d_model)  # [192, 512]
    # Normalize to unit vectors
    probe_dirs = F.normalize(probe_dirs.float(), dim=-1)
    n_probe_dirs = probe_dirs.shape[0]
    print(f"Probe directions: {n_probe_dirs} (8x8x3 = 64 squares x 3 classes)")

    # Generate games
    print(f"\nGenerating {N_GAMES} random Othello games...")
    games = []
    for _ in range(N_GAMES):
        moves = generate_random_game(rng)
        if len(moves) >= SEQ_LEN:
            games.append(torch.tensor(moves[:SEQ_LEN], dtype=torch.long))
    print(f"Got {len(games)} games with >= {SEQ_LEN} moves")

    # Extract activations
    print(f"\nExtracting layer-{INJECT_LAYER} activations...")
    contexts = []
    activations = []
    hook_name = f"blocks.{INJECT_LAYER}.hook_resid_post"

    for tokens in games[:N_SWEEP]:
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

    # Random directions
    random_dirs = F.normalize(torch.randn(N_RANDOM_DIRS, d_model), dim=-1)

    # Sweep random directions
    print(f"\n=== Random directions ({N_SWEEP} acts x {N_RANDOM_DIRS} dirs) ===")
    rand_curves = np.zeros((N_SWEEP, N_RANDOM_DIRS, N_ANGLES))
    t0 = time.time()
    for i in range(N_SWEEP):
        ctx = contexts[i].to(device)
        act = activations[i].to(device)
        rand_curves[i] = sweep_directions(
            model, ctx, act, random_dirs, INJECT_LAYER, MEASURE_LAYER,
            cos_t, sin_t, device)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_SWEEP}  ({(i+1)/(time.time()-t0):.1f} act/s)")
    print(f"Random done in {time.time()-t0:.1f}s")

    # Sweep probe directions
    print(f"\n=== Probe directions ({N_SWEEP} acts x {n_probe_dirs} dirs) ===")
    probe_curves = np.zeros((N_SWEEP, n_probe_dirs, N_ANGLES))
    t0 = time.time()
    for i in range(N_SWEEP):
        ctx = contexts[i].to(device)
        act = activations[i].to(device)
        probe_curves[i] = sweep_directions(
            model, ctx, act, probe_dirs, INJECT_LAYER, MEASURE_LAYER,
            cos_t, sin_t, device)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_SWEEP}  ({(i+1)/(time.time()-t0):.1f} act/s)")
    print(f"Probe done in {time.time()-t0:.1f}s")

    # Save
    results = {
        'config': {
            'inject_layer': INJECT_LAYER,
            'measure_layer': MEASURE_LAYER,
            'seq_len': SEQ_LEN,
            'n_sweep': N_SWEEP,
            'n_random_dirs': N_RANDOM_DIRS,
            'n_probe_dirs': n_probe_dirs,
            'max_angle': MAX_ANGLE,
            'n_angles': N_ANGLES,
            'seed': SEED,
        },
        'angles_deg': angles_deg,
        'rand_curves': rand_curves,    # [n_sweep, n_random_dirs, n_angles]
        'probe_curves': probe_curves,  # [n_sweep, n_probe_dirs, n_angles]
        'norms': activations_t.norm(dim=-1).numpy(),
    }
    pkl_path = os.path.join(RESULTS_DIR, 'probe_vs_random.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # Plot: median per-activation (across directions), then median + IQR across activations
    rand_per_act = np.median(rand_curves, axis=1)    # [n_sweep, n_angles]
    probe_per_act = np.median(probe_curves, axis=1)  # [n_sweep, n_angles]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Random
    ax.fill_between(angles_deg,
                    np.percentile(rand_per_act, 25, axis=0),
                    np.percentile(rand_per_act, 75, axis=0),
                    color='C0', alpha=0.15)
    ax.plot(angles_deg, np.median(rand_per_act, axis=0),
            color='C0', lw=2.5, label=f'Random directions (n={N_RANDOM_DIRS})')

    # Probe
    ax.fill_between(angles_deg,
                    np.percentile(probe_per_act, 25, axis=0),
                    np.percentile(probe_per_act, 75, axis=0),
                    color='C3', alpha=0.15)
    ax.plot(angles_deg, np.median(probe_per_act, axis=0),
            color='C3', lw=2.5, label=f'Probe directions (n={n_probe_dirs})')

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel(f'Cosine distance (at layer {MEASURE_LAYER})')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MAX_ANGLE)
    ax.set_title(f'OthelloGPT: probe vs random perturbations, '
                 f'inject@L{INJECT_LAYER} measure@L{MEASURE_LAYER}')
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'probe_vs_random.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Numerical summary
    for name, per_act in [('Random', rand_per_act), ('Probe', probe_per_act)]:
        med = np.median(per_act, axis=0)
        for thresh in [0.01, 0.05, 0.1]:
            crossings = np.where(med > thresh)[0]
            if len(crossings):
                print(f"{name}: median crosses {thresh} at {angles_deg[crossings[0]]:.1f}°")
            else:
                print(f"{name}: median never crosses {thresh}")


if __name__ == '__main__':
    main()
