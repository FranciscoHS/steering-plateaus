#!/usr/bin/env python3
"""Perturbation sweep on OthelloGPT: real activations vs random points.

Samples random points from a Gaussian fitted to real activations (full
covariance), then runs the same perturbation sweep on both. Plots both
on the same figure.

Usage:
    python scripts/perturbation/sweep_othello_real_vs_random.py
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
N_SWEEP = 50        # activations to sweep (both real and random)
N_FIT = 200         # real activations for covariance fitting
N_DIRS = 20
MAX_ANGLE = 60.0
N_ANGLES = 240
SEED = 42
RESULTS_DIR = 'results/othello_plateau'


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


def sample_random_activations(real_activations, n, seed=42, device='cpu'):
    """Sample from Gaussian with mean and full covariance matched to real activations."""
    torch.manual_seed(seed + 999)  # different seed from game generation
    data = real_activations.to(device)
    N, D = data.shape
    mean = data.mean(dim=0)

    centered = data - mean.unsqueeze(0)
    print(f"  Computing {D}x{D} covariance from {N} samples...")
    cov = (centered.T @ centered) / (N - 1)

    try:
        L = torch.linalg.cholesky(cov)
    except torch.linalg.LinAlgError:
        reg = 1e-5 * torch.trace(cov) / D
        print(f"  Cholesky failed, adding regularization eps={reg.item():.2e}")
        L = torch.linalg.cholesky(cov + reg * torch.eye(D, device=device))

    z = torch.randn(n, D, device=device)
    samples = mean.unsqueeze(0) + z @ L.T
    return samples.cpu()


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

    # Generate games
    print(f"\nGenerating {N_GAMES} random Othello games...")
    games = []
    for _ in range(N_GAMES):
        moves = generate_random_game(rng)
        if len(moves) >= SEQ_LEN:
            games.append(torch.tensor(moves[:SEQ_LEN], dtype=torch.long))
    print(f"Got {len(games)} games with >= {SEQ_LEN} moves")

    # Extract activations
    print(f"\nExtracting layer-{INJECT_LAYER} activations from {len(games)} games...")
    contexts = []
    activations = []
    hook_name = f"blocks.{INJECT_LAYER}.hook_resid_post"

    for tokens in games:
        tokens_in = tokens.unsqueeze(0).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens_in, names_filter=[hook_name])
        resid = cache[hook_name].float()
        contexts.append(resid[:, :-1, :].cpu())
        activations.append(resid[:, -1, :].squeeze(0).cpu())

    all_activations = torch.stack(activations)  # [N_GAMES, d_model]
    print(f"Real activation norms: mean={all_activations.norm(dim=-1).mean():.2f}, "
          f"std={all_activations.norm(dim=-1).std():.2f}")

    # Sample random points
    print(f"\nSampling {N_SWEEP} random points (full covariance)...")
    random_acts = sample_random_activations(all_activations, N_SWEEP, seed=SEED, device=device)
    print(f"Random point norms: mean={random_acts.norm(dim=-1).mean():.2f}, "
          f"std={random_acts.norm(dim=-1).std():.2f}")

    # Angle grid
    angles_deg = np.linspace(0, MAX_ANGLE, N_ANGLES + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)
    cos_t = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)

    def sweep_one(ctx, act, n_dirs):
        """Sweep random directions for one activation, return [n_dirs, n_angles]."""
        act_d = act.to(device)
        ctx_d = ctx.to(device)
        R = act_d.norm()
        a_unit = act_d / R

        # Reference
        with torch.no_grad():
            full_ref = torch.cat([ctx_d, act_d[None, None]], dim=1)
            hidden = full_ref
            for block in model.blocks[INJECT_LAYER + 1:MEASURE_LAYER + 1]:
                hidden = block(hidden)
            ref = hidden[:, -1, :].float()

        curves = np.zeros((n_dirs, N_ANGLES))
        for j in range(n_dirs):
            d = torch.randn(d_model, device=device)
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

    # Sweep real activations
    print(f"\n=== Real activations ({N_SWEEP} acts x {N_DIRS} dirs) ===")
    real_curves = np.zeros((N_SWEEP, N_DIRS, N_ANGLES))
    t0 = time.time()
    for i in range(N_SWEEP):
        real_curves[i] = sweep_one(contexts[i], activations[i], N_DIRS)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_SWEEP}  ({(i+1)/(time.time()-t0):.1f} act/s)")
    print(f"Real done in {time.time()-t0:.1f}s")

    # Sweep random points (use real contexts as stand-ins)
    print(f"\n=== Random points ({N_SWEEP} acts x {N_DIRS} dirs) ===")
    rand_curves = np.zeros((N_SWEEP, N_DIRS, N_ANGLES))
    t0 = time.time()
    for i in range(N_SWEEP):
        rand_curves[i] = sweep_one(contexts[i % len(contexts)], random_acts[i], N_DIRS)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_SWEEP}  ({(i+1)/(time.time()-t0):.1f} act/s)")
    print(f"Random done in {time.time()-t0:.1f}s")

    # Save
    results = {
        'config': {
            'inject_layer': INJECT_LAYER,
            'measure_layer': MEASURE_LAYER,
            'seq_len': SEQ_LEN,
            'n_sweep': N_SWEEP,
            'n_dirs': N_DIRS,
            'max_angle': MAX_ANGLE,
            'n_angles': N_ANGLES,
            'seed': SEED,
        },
        'angles_deg': angles_deg,
        'real_curves': real_curves,  # [n_sweep, n_dirs, n_angles]
        'rand_curves': rand_curves,  # [n_sweep, n_dirs, n_angles]
        'real_norms': all_activations[:N_SWEEP].norm(dim=-1).numpy(),
        'rand_norms': random_acts.norm(dim=-1).numpy(),
    }
    pkl_path = os.path.join(RESULTS_DIR, 'real_vs_random_acts.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # Plot
    real_per_act = np.median(real_curves, axis=1)
    rand_per_act = np.median(rand_curves, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(angles_deg,
                    np.percentile(real_per_act, 25, axis=0),
                    np.percentile(real_per_act, 75, axis=0),
                    color='C0', alpha=0.15)
    ax.plot(angles_deg, np.median(real_per_act, axis=0),
            color='C0', lw=2.5, label=f'Real activations (n={N_SWEEP})')

    ax.fill_between(angles_deg,
                    np.percentile(rand_per_act, 25, axis=0),
                    np.percentile(rand_per_act, 75, axis=0),
                    color='C1', alpha=0.15)
    ax.plot(angles_deg, np.median(rand_per_act, axis=0),
            color='C1', lw=2.5, label=f'Random points (n={N_SWEEP})')

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel(f'Cosine distance (at layer {MEASURE_LAYER})')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MAX_ANGLE)
    ax.set_title(f'OthelloGPT: real vs random activations, '
                 f'inject@L{INJECT_LAYER} measure@L{MEASURE_LAYER}')
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'real_vs_random_acts.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Numerical summary
    for name, per_act in [('Real', real_per_act), ('Random', rand_per_act)]:
        med = np.median(per_act, axis=0)
        for thresh in [0.01, 0.05, 0.1]:
            crossings = np.where(med > thresh)[0]
            if len(crossings):
                print(f"{name}: median crosses {thresh} at {angles_deg[crossings[0]]:.1f}°")
            else:
                print(f"{name}: median never crosses {thresh}")


if __name__ == '__main__':
    main()
