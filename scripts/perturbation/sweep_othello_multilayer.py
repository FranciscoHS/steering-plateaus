#!/usr/bin/env python3
"""Multi-layer steering on OthelloGPT.

Perturbs the residual stream at ALL layers simultaneously, using each
layer's probe direction (mine→empty), at the same angular magnitude.
Produces cos dist + P(cf-legal) overlay plot.
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
SEQ_LEN = 20
N_GAMES = 500
MAX_ANGLE = 60.0
N_ANGLES = 60
SEED = 42
RESULTS_DIR = 'results/othello_plateau'
PROBE_DIR = '/workspace/mech_int_othelloGPT/mech_int/probes/linear'
N_LAYERS = 8
MEASURE_LAYER = 7  # penultimate (model has layers 0-7)

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


def get_legal_moves(board, player):
    legal = []
    for sq in range(64):
        if sq in {27, 28, 35, 36} or board[sq] != 0:
            continue
        r, c = sq // 8, sq % 8
        opp = 3 - player
        found_any = False
        for dr, dc in FLIP_DIRS_8:
            nr, nc = r + dr, c + dc
            found = False
            while 0 <= nr < 8 and 0 <= nc < 8:
                idx = nr * 8 + nc
                if board[idx] == opp:
                    found = True
                elif board[idx] == player and found:
                    found_any = True
                    break
                else:
                    break
                nr += dr; nc += dc
            if found_any:
                break
        if found_any:
            legal.append(sq)
    return legal


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = 'cuda'

    model = HookedTransformer.from_pretrained("othello-gpt")
    model.to(device).eval()

    target_r, target_c = 2, 2
    target_sq = target_r * 8 + target_c
    mine_color = 2

    # Load probe directions for all layers
    probe_dirs = {}
    for layer in range(N_LAYERS):
        probe = torch.load(f'{PROBE_DIR}/resid_{layer}_linear.pth',
                           map_location='cpu', weights_only=True)
        # mine→empty direction
        d = (probe[0, :, target_r, target_c, 0] - probe[0, :, target_r, target_c, 1]).float()
        probe_dirs[layer] = F.normalize(d, dim=0).to(device)
    print(f"Loaded probe directions for {N_LAYERS} layers")

    # Generate games
    print("Generating games...")
    all_games = []
    for _ in range(N_GAMES):
        m, b = play_random_game(rng)
        if len(m) >= SEQ_LEN:
            all_games.append((m[:SEQ_LEN], b[:SEQ_LEN]))

    mine_idx = [i for i, (m, b) in enumerate(all_games) if b[19][target_sq] == mine_color]
    print(f"Mine: {len(mine_idx)}")

    # Compute counterfactual legal moves
    next_player = 1
    legal_actual_tokens = []
    legal_cf_empty_tokens = []
    tokens_list = []

    for i in mine_idx[:100]:
        tokens_list.append(torch.tensor(all_games[i][0], dtype=torch.long))
        board_actual = list(all_games[i][1][19])
        legal_actual_tokens.append(set(SQ2TOK[sq] for sq in get_legal_moves(board_actual, next_player)))
        board_cf = list(board_actual)
        board_cf[target_sq] = 0
        legal_cf_empty_tokens.append(set(SQ2TOK[sq] for sq in get_legal_moves(board_cf, next_player)))

    n_sweep = min(len(tokens_list), 50)
    print(f"Sweeping {n_sweep} activations")

    angles_deg = np.linspace(0, MAX_ANGLE, N_ANGLES + 1)[1:]
    angles_rad = torch.tensor(np.deg2rad(angles_deg), dtype=torch.float32, device=device)
    cos_t = torch.cos(angles_rad)
    sin_t = torch.sin(angles_rad)

    def make_steering_hook(layer, batch_size):
        """Hook that perturbs the last-position activation on the sphere
        toward probe_dirs[layer] by angle[batch_idx]."""
        direction = probe_dirs[layer]

        def hook_fn(activation, hook):
            # activation: [batch, seq, d_model]
            # First element in batch is reference (no perturbation)
            # Elements 1..N_ANGLES get angular perturbation
            act = activation[:, -1, :].float()  # [batch, d_model]

            for b in range(1, batch_size):
                a = act[b]
                R = a.norm()
                a_unit = a / R
                d = direction - (direction @ a_unit) * a_unit
                d = F.normalize(d, dim=0)
                theta_idx = b - 1
                activation[b, -1, :] = R * (cos_t[theta_idx] * a_unit + sin_t[theta_idx] * d)
            return activation
        return hook_fn

    def sweep_one_activation(i):
        """Run one activation through multi-layer steering. Returns cos_dist, kl, p_cf, p_actual."""
        tokens = tokens_list[i].unsqueeze(0).to(device)
        batch_size = 1 + N_ANGLES  # ref + N_ANGLES perturbations
        tokens_batch = tokens.expand(batch_size, -1)

        # Build hooks for all layers
        fwd_hooks = []
        for layer in range(N_LAYERS):
            hook_name = f"blocks.{layer}.hook_resid_post"
            fwd_hooks.append((hook_name, make_steering_hook(layer, batch_size)))

        # Capture penultimate layer
        captured_L = {}
        def capture_hook(activation, hook):
            captured_L['act'] = activation[:, -1, :].float().clone()
            return activation
        fwd_hooks.append((f"blocks.{MEASURE_LAYER}.hook_resid_post", capture_hook))

        with torch.no_grad():
            logits = model.run_with_hooks(tokens_batch, fwd_hooks=fwd_hooks)

        last_logits = logits[:, -1, :]  # [batch, vocab]
        probs = F.softmax(last_logits, dim=-1)
        ref_probs = probs[0:1]
        pert_probs = probs[1:]

        # KL
        kl = (ref_probs * (ref_probs.log() - pert_probs.log())).sum(dim=-1)

        # Cosine distance at measure layer
        ref_L = captured_L['act'][0:1]
        pert_L = captured_L['act'][1:]
        cos_dist = 1 - F.cosine_similarity(ref_L, pert_L, dim=-1)

        # P(cf-legal) and P(actual-legal)
        cf_toks = list(legal_cf_empty_tokens[i])
        actual_toks = list(legal_actual_tokens[i])
        p_cf = pert_probs[:, cf_toks].sum(dim=-1) if cf_toks else torch.zeros(N_ANGLES, device=device)
        p_actual = pert_probs[:, actual_toks].sum(dim=-1) if actual_toks else torch.zeros(N_ANGLES, device=device)

        return (cos_dist.cpu().numpy(), kl.cpu().numpy(),
                p_cf.cpu().numpy(), p_actual.cpu().numpy())

    # Sweep: multi-layer probe
    print("Multi-layer probe sweep...")
    ml_cos = np.zeros((n_sweep, N_ANGLES))
    ml_kl = np.zeros((n_sweep, N_ANGLES))
    ml_pcf = np.zeros((n_sweep, N_ANGLES))
    ml_pa = np.zeros((n_sweep, N_ANGLES))
    t0 = time.time()
    for i in range(n_sweep):
        ml_cos[i], ml_kl[i], ml_pcf[i], ml_pa[i] = sweep_one_activation(i)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_sweep} ({(i+1)/(time.time()-t0):.1f} act/s)")

    # Sweep: random directions at all layers (for comparison)
    N_RAND = 20
    d_model = model.cfg.d_model
    print(f"Random multi-layer sweep ({n_sweep} acts x {N_RAND} dirs)...")

    rand_cos = np.zeros((n_sweep, N_RAND, N_ANGLES))
    rand_pcf = np.zeros((n_sweep, N_RAND, N_ANGLES))

    for k in range(N_RAND):
        # Random direction per layer
        rand_layer_dirs = {layer: F.normalize(torch.randn(d_model), dim=0).to(device)
                           for layer in range(N_LAYERS)}
        # Temporarily swap probe_dirs
        orig_dirs = dict(probe_dirs)
        for layer in range(N_LAYERS):
            probe_dirs[layer] = rand_layer_dirs[layer]

        for i in range(n_sweep):
            c, _, p, _ = sweep_one_activation(i)
            rand_cos[i, k] = c
            rand_pcf[i, k] = p

        # Restore
        for layer in range(N_LAYERS):
            probe_dirs[layer] = orig_dirs[layer]

        if (k + 1) % 5 == 0:
            print(f"  Random dir {k+1}/{N_RAND}")

    # Save
    save_data = {
        'angles_deg': angles_deg,
        'ml_cos': ml_cos, 'ml_kl': ml_kl, 'ml_pcf': ml_pcf, 'ml_pa': ml_pa,
        'rand_cos': rand_cos, 'rand_pcf': rand_pcf,
        'config': {'r': target_r, 'c': target_c, 'n_sweep': n_sweep, 'n_rand': N_RAND},
    }
    with open(os.path.join(RESULTS_DIR, 'multilayer_steering_2_2.pkl'), 'wb') as f:
        pickle.dump(save_data, f)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    rand_cos_med = np.median(np.median(rand_cos, axis=1), axis=0)
    ax1.plot(angles_deg, rand_cos_med, color='C0', lw=2, label='Random (cos dist)')
    ax1.plot(angles_deg, np.median(ml_cos, axis=0), color='C3', lw=2, label='Probe all layers (cos dist)')

    rand_pcf_med = np.median(np.median(rand_pcf, axis=1), axis=0)
    ax2.plot(angles_deg, rand_pcf_med, color='C0', lw=2, ls='--', label='Random (P(cf legal))')
    ax2.plot(angles_deg, np.median(ml_pcf, axis=0), color='C3', lw=2, ls='--', label='Probe all layers (P(cf legal))')

    ax1.set_xlabel('Perturbation angle (degrees)')
    ax1.set_ylabel(f'Cosine distance (layer {MEASURE_LAYER})')
    ax2.set_ylabel('P(counterfactual-legal moves)')
    ax1.set_xlim(0, MAX_ANGLE)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'OthelloGPT: multi-layer steering (mine→empty), square ({target_r},{target_c})')
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'multilayer_cos_2_2.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {plot_path}")
    med_pcf = np.median(ml_pcf, axis=0)
    med_pa = np.median(ml_pa, axis=0)
    med_cos = np.median(ml_cos, axis=0)
    print(f"Probe all layers:")
    print(f"  P(cf-legal) at 0°: {med_pcf[0]:.3f}, at 30°: {med_pcf[N_ANGLES//2]:.3f}, at 60°: {med_pcf[-1]:.3f}")
    print(f"  P(actual-legal) at 0°: {med_pa[0]:.3f}, at 30°: {med_pa[N_ANGLES//2]:.3f}, at 60°: {med_pa[-1]:.3f}")
    print(f"  Cos dist at 0°: {med_cos[0]:.4f}, at 30°: {med_cos[N_ANGLES//2]:.4f}, at 60°: {med_cos[-1]:.4f}")


if __name__ == '__main__':
    main()
