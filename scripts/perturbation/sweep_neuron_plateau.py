#!/usr/bin/env python3
"""Perturbation sweep along neuron W_out directions that align with 'empty' probes.

Inject at layer 3, measure cosine distance at layer 6.
Compare: top-aligned neuron dirs, probe dirs, random dirs.
"""

import os, pickle, time
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer

INJECT_LAYER = 3
MEASURE_LAYER = 6
SEQ_LEN = 20
N_GAMES = 500
MAX_ANGLE = 60.0
N_ANGLES = 120
SEED = 42
RESULTS_DIR = 'results/othello_plateau'
PROBE_DIR = '/workspace/mech_int_othelloGPT/mech_int/probes/linear'

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

    # Load probe at inject layer
    probe = torch.load(f'{PROBE_DIR}/resid_{INJECT_LAYER}_linear.pth',
                        map_location='cpu', weights_only=True).float()

    # Get neuron W_out directions at inject layer
    W_out = model.blocks[INJECT_LAYER].mlp.W_out.detach().cpu().float()  # [d_mlp, d_model]
    neuron_dirs = F.normalize(W_out, dim=1)  # [d_mlp, d_model]
    d_mlp = neuron_dirs.shape[0]

    # Find neurons most aligned with 'empty' probe directions
    empty_probe_dirs = []
    empty_probe_labels = []
    for r in range(8):
        for c in range(8):
            sq = r * 8 + c
            if sq in CENTER:
                continue
            d = F.normalize(probe[0, :, r, c, 0].float(), dim=0)  # empty class
            empty_probe_dirs.append(d)
            empty_probe_labels.append(f"({r},{c})")
    empty_probe_dirs = torch.stack(empty_probe_dirs)  # [60, d_model]

    # For each empty probe, find best-matching neuron
    cos_sim = empty_probe_dirs @ neuron_dirs.T  # [60, d_mlp]
    max_cos, best_neuron_idx = cos_sim.abs().max(dim=1)

    # Pick top 5 most aligned neuron-probe pairs
    top5_vals, top5_probe_idx = max_cos.topk(5)
    print("Top 5 neuron-empty probe alignments:")
    neuron_directions = {}
    for i in range(5):
        pi = top5_probe_idx[i].item()
        ni = best_neuron_idx[pi].item()
        cos_val = top5_vals[i].item()
        label = f"Neuron {ni} ({empty_probe_labels[pi]} empty, cos={cos_val:.3f})"
        print(f"  {label}")
        neuron_directions[label] = neuron_dirs[ni]

    # Also get corresponding empty probe directions for comparison
    probe_directions = {}
    for i in range(5):
        pi = top5_probe_idx[i].item()
        label = f"Probe {empty_probe_labels[pi]} empty"
        probe_directions[label] = empty_probe_dirs[pi]

    # Generate games and extract activations
    print("\nGenerating games...")
    all_games = []
    for _ in range(N_GAMES * 2):
        m, b = play_random_game(rng)
        if len(m) >= SEQ_LEN:
            all_games.append((m[:SEQ_LEN], b[:SEQ_LEN]))
        if len(all_games) >= N_GAMES:
            break

    hook = f"blocks.{INJECT_LAYER}.hook_resid_post"
    all_contexts = []
    all_acts = []
    for i in range(min(100, len(all_games))):
        tokens = torch.tensor(all_games[i][0], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook])
        resid = cache[hook].float()
        all_contexts.append(resid[:, :-1, :].cpu())
        all_acts.append(resid[:, -1, :].squeeze(0).cpu())

    n_sweep = min(len(all_acts), 50)
    print(f"Sweeping {n_sweep} activations")

    # Angle grid
    angles_deg = np.linspace(0, MAX_ANGLE, N_ANGLES + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)
    cos_t = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)

    def forward_to_L(resid):
        h = resid
        for block in model.blocks[INJECT_LAYER + 1:MEASURE_LAYER + 1]:
            h = block(h)
        return h

    def sweep_one_dir(i, direction):
        """Returns cos_dist [N_ANGLES]."""
        act_d = all_acts[i].to(device)
        ctx_d = all_contexts[i].to(device)
        R = act_d.norm()
        a_unit = act_d / R

        d = direction.to(device)
        d = d - (d @ a_unit) * a_unit
        d = F.normalize(d, dim=0)

        with torch.no_grad():
            ref_input = torch.cat([ctx_d, act_d[None, None]], dim=1)
            ref_L = forward_to_L(ref_input)[:, -1, :].float()

            perturbed = R * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * d[None])
            ctx_batch = ctx_d.expand(N_ANGLES, -1, -1)
            full_input = torch.cat([ctx_batch, perturbed.unsqueeze(1)], dim=1)

            pert_L = forward_to_L(full_input)[:, -1, :].float()
            cos_dist = 1 - F.cosine_similarity(ref_L, pert_L, dim=-1)

        return cos_dist.cpu().numpy()

    # Sweep all direction types
    all_directions = {}
    all_directions.update(neuron_directions)
    all_directions.update(probe_directions)

    results = {}
    for name, d_vec in all_directions.items():
        cos_all = np.zeros((n_sweep, N_ANGLES))
        t0 = time.time()
        for i in range(n_sweep):
            cos_all[i] = sweep_one_dir(i, d_vec)
            if (i + 1) % 10 == 0:
                print(f"  {name}: {i+1}/{n_sweep} ({(i+1)/(time.time()-t0):.1f} act/s)")
        results[name] = cos_all

    # Random directions
    N_RAND = 20
    rand_dirs = F.normalize(torch.randn(N_RAND, d_model), dim=-1)
    rand_cos = np.zeros((n_sweep, N_RAND, N_ANGLES))
    t0 = time.time()
    print(f"Random directions ({n_sweep} acts x {N_RAND} dirs)...")
    for i in range(n_sweep):
        for k in range(N_RAND):
            rand_cos[i, k] = sweep_one_dir(i, rand_dirs[k])
        if (i + 1) % 10 == 0:
            print(f"  Random: {i+1}/{n_sweep} ({(i+1)/(time.time()-t0):.1f} act/s)")

    # Save
    save_data = {
        'angles_deg': angles_deg,
        'results': results,
        'rand_cos': rand_cos,
        'config': {'inject_layer': INJECT_LAYER, 'measure_layer': MEASURE_LAYER,
                   'n_sweep': n_sweep, 'n_rand': N_RAND},
    }
    save_path = os.path.join(RESULTS_DIR, 'neuron_plateau_L3_L6.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved: {save_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Random band
    rand_med = np.median(np.median(rand_cos, axis=1), axis=0)
    rand_lo = np.percentile(np.median(rand_cos, axis=1), 25, axis=0)
    rand_hi = np.percentile(np.median(rand_cos, axis=1), 75, axis=0)
    ax.fill_between(angles_deg, rand_lo, rand_hi, color='C0', alpha=0.15)
    ax.plot(angles_deg, rand_med, 'C0-', lw=2, label='Random')

    # Neuron directions
    for name in neuron_directions:
        med = np.median(results[name], axis=0)
        ax.plot(angles_deg, med, lw=2, ls='-', label=name)

    # Probe directions (dashed)
    for name in probe_directions:
        med = np.median(results[name], axis=0)
        ax.plot(angles_deg, med, lw=2, ls='--', label=name)

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel(f'Cosine distance (layer {MEASURE_LAYER})')
    ax.set_title(f'OthelloGPT: neuron vs probe plateau-breaking (inject L{INJECT_LAYER}, measure L{MEASURE_LAYER})')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MAX_ANGLE)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'neuron_plateau_L3_L6.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


if __name__ == '__main__':
    main()
