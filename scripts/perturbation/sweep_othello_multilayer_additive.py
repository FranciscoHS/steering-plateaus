#!/usr/bin/env python3
"""Multi-layer ADDITIVE steering on OthelloGPT.

Instead of angular perturbation, adds alpha * probe_dir at every layer.
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
N_MAGS = 60
SEED = 42
RESULTS_DIR = 'results/othello_plateau'
PROBE_DIR = '/workspace/mech_int_othelloGPT/mech_int/probes/linear'
N_LAYERS = 8
MEASURE_LAYER = 7

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
    d_model = model.cfg.d_model

    target_r, target_c = 2, 2
    target_sq = target_r * 8 + target_c
    mine_color = 2

    # Load probe directions for all layers
    probe_dirs = {}
    for layer in range(N_LAYERS):
        probe = torch.load(f'{PROBE_DIR}/resid_{layer}_linear.pth',
                           map_location='cpu', weights_only=True)
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

    # Get activation norms to calibrate magnitude range
    hook = "blocks.4.hook_resid_post"
    sample_tokens = torch.tensor(all_games[mine_idx[0]][0], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        _, cache = model.run_with_cache(sample_tokens, names_filter=[hook])
    act_norm = cache[hook][0, -1].float().norm().item()
    print(f"Sample activation norm (layer 4): {act_norm:.1f}")

    # Sweep magnitudes: 0 to 2x activation norm
    MAX_MAG = act_norm * 2
    magnitudes = np.linspace(0, MAX_MAG, N_MAGS + 1)[1:]
    mags_t = torch.tensor(magnitudes, dtype=torch.float32, device=device)
    print(f"Magnitude range: 0 to {MAX_MAG:.1f} (act norm = {act_norm:.1f})")

    # Legal moves
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

    def make_additive_hook(layer, batch_size):
        direction = probe_dirs[layer]
        def hook_fn(activation, hook):
            for b in range(1, batch_size):
                activation[b, -1, :] = activation[b, -1, :] + mags_t[b - 1] * direction
            return activation
        return hook_fn

    def sweep_one(i):
        tokens = tokens_list[i].unsqueeze(0).to(device)
        batch_size = 1 + N_MAGS
        tokens_batch = tokens.expand(batch_size, -1)

        fwd_hooks = []
        for layer in range(N_LAYERS):
            fwd_hooks.append((f"blocks.{layer}.hook_resid_post",
                              make_additive_hook(layer, batch_size)))

        captured = {}
        def capture_hook(activation, hook):
            captured['act'] = activation[:, -1, :].float().clone()
            return activation
        fwd_hooks.append((f"blocks.{MEASURE_LAYER}.hook_resid_post", capture_hook))

        with torch.no_grad():
            logits = model.run_with_hooks(tokens_batch, fwd_hooks=fwd_hooks)

        probs = F.softmax(logits[:, -1, :], dim=-1)
        ref_probs = probs[0:1]
        pert_probs = probs[1:]

        kl = (ref_probs * (ref_probs.log() - pert_probs.log())).sum(dim=-1)
        cos_dist = 1 - F.cosine_similarity(captured['act'][0:1], captured['act'][1:], dim=-1)

        cf_toks = list(legal_cf_empty_tokens[i])
        actual_toks = list(legal_actual_tokens[i])
        p_cf = pert_probs[:, cf_toks].sum(dim=-1) if cf_toks else torch.zeros(N_MAGS, device=device)
        p_actual = pert_probs[:, actual_toks].sum(dim=-1) if actual_toks else torch.zeros(N_MAGS, device=device)

        return (cos_dist.cpu().numpy(), kl.cpu().numpy(),
                p_cf.cpu().numpy(), p_actual.cpu().numpy())

    # Probe sweep
    print(f"Multi-layer additive probe sweep ({n_sweep} acts)...")
    ml_cos = np.zeros((n_sweep, N_MAGS))
    ml_kl = np.zeros((n_sweep, N_MAGS))
    ml_pcf = np.zeros((n_sweep, N_MAGS))
    ml_pa = np.zeros((n_sweep, N_MAGS))
    t0 = time.time()
    for i in range(n_sweep):
        ml_cos[i], ml_kl[i], ml_pcf[i], ml_pa[i] = sweep_one(i)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_sweep} ({(i+1)/(time.time()-t0):.1f} act/s)")

    # Random sweep
    N_RAND = 20
    print(f"Random multi-layer additive sweep ({n_sweep} acts x {N_RAND} dirs)...")
    rand_cos = np.zeros((n_sweep, N_RAND, N_MAGS))
    rand_pcf = np.zeros((n_sweep, N_RAND, N_MAGS))
    orig_dirs = dict(probe_dirs)
    for k in range(N_RAND):
        for layer in range(N_LAYERS):
            probe_dirs[layer] = F.normalize(torch.randn(d_model), dim=0).to(device)
        for i in range(n_sweep):
            c, _, p, _ = sweep_one(i)
            rand_cos[i, k] = c
            rand_pcf[i, k] = p
        if (k + 1) % 5 == 0:
            print(f"  Random dir {k+1}/{N_RAND}")
    for layer in range(N_LAYERS):
        probe_dirs[layer] = orig_dirs[layer]

    # Save
    save_data = {
        'magnitudes': magnitudes, 'act_norm': act_norm,
        'ml_cos': ml_cos, 'ml_kl': ml_kl, 'ml_pcf': ml_pcf, 'ml_pa': ml_pa,
        'rand_cos': rand_cos, 'rand_pcf': rand_pcf,
        'config': {'r': target_r, 'c': target_c, 'n_sweep': n_sweep, 'n_rand': N_RAND},
    }
    with open(os.path.join(RESULTS_DIR, 'multilayer_additive_2_2.pkl'), 'wb') as f:
        pickle.dump(save_data, f)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    rand_cos_med = np.median(np.median(rand_cos, axis=1), axis=0)
    ax1.plot(magnitudes, rand_cos_med, color='C0', lw=2, label='Random (cos dist)')
    ax1.plot(magnitudes, np.median(ml_cos, axis=0), color='C3', lw=2, label='Probe all layers (cos dist)')

    rand_pcf_med = np.median(np.median(rand_pcf, axis=1), axis=0)
    ax2.plot(magnitudes, rand_pcf_med, color='C0', lw=2, ls='--', label='Random (P(cf legal))')
    ax2.plot(magnitudes, np.median(ml_pcf, axis=0), color='C3', lw=2, ls='--', label='Probe all layers (P(cf legal))')

    ax1.set_xlabel('Steering magnitude (per layer)')
    ax1.set_ylabel(f'Cosine distance (layer {MEASURE_LAYER})')
    ax2.set_ylabel('P(counterfactual-legal moves)')
    ax1.set_xlim(0, MAX_MAG)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='center left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'OthelloGPT: multi-layer additive steering (mine→empty), square ({target_r},{target_c})')
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'multilayer_additive_cos_2_2.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {plot_path}")
    med_pcf = np.median(ml_pcf, axis=0)
    med_pa = np.median(ml_pa, axis=0)
    med_cos = np.median(ml_cos, axis=0)
    print(f"Probe all layers (additive):")
    print(f"  P(cf-legal) at mag {magnitudes[0]:.1f}: {med_pcf[0]:.3f}, "
          f"at {magnitudes[N_MAGS//2]:.1f}: {med_pcf[N_MAGS//2]:.3f}, "
          f"at {magnitudes[-1]:.1f}: {med_pcf[-1]:.3f}")
    print(f"  P(actual-legal) at mag {magnitudes[0]:.1f}: {med_pa[0]:.3f}, "
          f"at {magnitudes[N_MAGS//2]:.1f}: {med_pa[N_MAGS//2]:.3f}, "
          f"at {magnitudes[-1]:.1f}: {med_pa[-1]:.3f}")
    print(f"  Cos dist at mag {magnitudes[0]:.1f}: {med_cos[0]:.4f}, "
          f"at {magnitudes[N_MAGS//2]:.1f}: {med_cos[N_MAGS//2]:.4f}, "
          f"at {magnitudes[-1]:.1f}: {med_cos[-1]:.4f}")


if __name__ == '__main__':
    main()
