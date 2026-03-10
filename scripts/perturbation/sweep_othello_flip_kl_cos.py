#!/usr/bin/env python3
"""Overlay behavioral steering metric, cosine distance, and KL-div.

For each angle along probe/DoM/random directions, measures:
1. Cosine distance at layer 6
2. KL divergence on logits
3. Behavioral: probability mass on moves legal under the counterfactual board
   (where target square is flipped from mine to opponent)

All forward passes batched over angles.
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
MAX_ANGLE = 180.0
N_ANGLES = 180
SEED = 42
RESULTS_DIR = 'results/othello_plateau'
PROBE_PATH = '/workspace/mech_int_othelloGPT/mech_int/probes/linear/resid_2_linear.pth'

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
    """Get legal moves for player on given board."""
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

    probe = torch.load(PROBE_PATH, map_location='cpu', weights_only=True)
    target_r, target_c = 2, 2
    target_sq = target_r * 8 + target_c
    mine_color = 2  # white at position 19 (odd)
    opp_color = 1

    # +probe_mine2opp = mine→opp
    probe_mine2opp = F.normalize(
        (probe[0, :, target_r, target_c, 2] - probe[0, :, target_r, target_c, 1]).float(), dim=0)
    # +probe_mine2empty = mine→empty
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
    opp_idx = [i for i, (m, b) in enumerate(all_games) if b[19][target_sq] == opp_color]
    print(f"Mine: {len(mine_idx)}, Opp: {len(opp_idx)}")

    # Extract activations and compute legal moves for actual + counterfactual boards
    hook = f"blocks.{INJECT_LAYER}.hook_resid_post"
    mine_contexts, mine_acts = [], []
    legal_actual_tokens = []    # legal moves under actual board
    legal_cf_opp_tokens = []    # counterfactual: target square → opp
    legal_cf_empty_tokens = []  # counterfactual: target square → empty

    # Next player at position 20 is black (player 1), since white just moved at pos 19
    next_player = 1

    for i in mine_idx[:100]:
        tokens = torch.tensor(all_games[i][0], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook])
        resid = cache[hook].float()
        mine_contexts.append(resid[:, :-1, :].cpu())
        mine_acts.append(resid[:, -1, :].squeeze(0).cpu())

        board_actual = list(all_games[i][1][19])
        legal_a = get_legal_moves(board_actual, next_player)
        legal_actual_tokens.append(set(SQ2TOK[sq] for sq in legal_a))

        # Counterfactual: mine → opp
        board_cf_opp = list(board_actual)
        board_cf_opp[target_sq] = opp_color
        legal_cf_opp_tokens.append(set(SQ2TOK[sq] for sq in get_legal_moves(board_cf_opp, next_player)))

        # Counterfactual: mine → empty
        board_cf_empty = list(board_actual)
        board_cf_empty[target_sq] = 0
        legal_cf_empty_tokens.append(set(SQ2TOK[sq] for sq in get_legal_moves(board_cf_empty, next_player)))

    n_sweep = min(len(mine_acts), 50)

    # Check how often legal sets differ for each counterfactual
    for cf_name, cf_tokens in [('mine->opp', legal_cf_opp_tokens), ('mine->empty', legal_cf_empty_tokens)]:
        n_differ = sum(1 for i in range(n_sweep) if legal_actual_tokens[i] != cf_tokens[i])
        print(f"Legal sets differ ({cf_name}): {n_differ}/{n_sweep}")
        for i in range(min(3, n_sweep)):
            cf_only = cf_tokens[i] - legal_actual_tokens[i]
            actual_only = legal_actual_tokens[i] - cf_tokens[i]
            print(f"  Game {i}: actual_only={len(actual_only)}, cf_only={len(cf_only)}, "
                  f"shared={len(legal_actual_tokens[i] & cf_tokens[i])}")

    print(f"\nSweeping {n_sweep} activations")

    # Angle grid
    angles_deg = np.linspace(0, MAX_ANGLE, N_ANGLES + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)
    cos_t = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)

    def forward_to_logits(resid):
        h = resid
        for block in model.blocks[INJECT_LAYER + 1:]:
            h = block(h)
        h = model.ln_final(h)
        return model.unembed(h)

    def forward_to_L6(resid):
        h = resid
        for block in model.blocks[INJECT_LAYER + 1:MEASURE_LAYER + 1]:
            h = block(h)
        return h

    def sweep_one_dir(i, direction, cf_tokens_list):
        """Sweep one direction for activation i, batched over angles.
        cf_tokens_list: the per-activation list of counterfactual-legal token sets.
        Returns (cos_dists, kl_divs, p_cf, p_actual, entropy) each [N_ANGLES]."""
        act_d = mine_acts[i].to(device)
        ctx_d = mine_contexts[i].to(device)
        R = act_d.norm()
        a_unit = act_d / R

        d = direction.to(device)
        d = d - (d @ a_unit) * a_unit
        d = F.normalize(d, dim=0)

        with torch.no_grad():
            ref_input = torch.cat([ctx_d, act_d[None, None]], dim=1)
            ref_logits = forward_to_logits(ref_input)[:, -1, :]
            ref_probs = F.softmax(ref_logits, dim=-1)
            ref_L6 = forward_to_L6(ref_input)[:, -1, :].float()

            perturbed = R * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * d[None])
            ctx_batch = ctx_d.expand(N_ANGLES, -1, -1)
            full_input = torch.cat([ctx_batch, perturbed.unsqueeze(1)], dim=1)

            pert_logits = forward_to_logits(full_input)[:, -1, :]
            pert_probs = F.softmax(pert_logits, dim=-1)

            # KL(ref || pert)
            kl = (ref_probs * (ref_probs.log() - pert_probs.log())).sum(dim=-1)

            # Cosine distance at L6
            pert_L6 = forward_to_L6(full_input)[:, -1, :].float()
            cos_dist = 1 - F.cosine_similarity(ref_L6, pert_L6, dim=-1)

            # Probability mass on counterfactual-legal and actual-legal tokens
            cf_toks = list(cf_tokens_list[i])
            actual_toks = list(legal_actual_tokens[i])

            p_cf = pert_probs[:, cf_toks].sum(dim=-1) if cf_toks else torch.zeros(N_ANGLES, device=device)
            p_actual = pert_probs[:, actual_toks].sum(dim=-1) if actual_toks else torch.zeros(N_ANGLES, device=device)

            # Entropy of output distribution
            entropy = -(pert_probs * pert_probs.log().clamp(min=-100)).sum(dim=-1)

        return (cos_dist.cpu().numpy(), kl.cpu().numpy(),
                p_cf.cpu().numpy(), p_actual.cpu().numpy(), entropy.cpu().numpy())

    # Sweep probe directions (each with its own counterfactual legal set)
    directions = {
        'Probe (mine->empty)': (probe_mine2empty, legal_cf_empty_tokens),
    }

    results = {}
    for name, (d_vec, cf_toks) in directions.items():
        cos_all = np.zeros((n_sweep, N_ANGLES))
        kl_all = np.zeros((n_sweep, N_ANGLES))
        p_cf_all = np.zeros((n_sweep, N_ANGLES))
        p_actual_all = np.zeros((n_sweep, N_ANGLES))
        ent_all = np.zeros((n_sweep, N_ANGLES))
        t0 = time.time()
        for i in range(n_sweep):
            cos_all[i], kl_all[i], p_cf_all[i], p_actual_all[i], ent_all[i] = sweep_one_dir(i, d_vec, cf_toks)
            if (i + 1) % 10 == 0:
                print(f"  {name}: {i+1}/{n_sweep} ({(i+1)/(time.time()-t0):.1f} act/s)")
        results[name] = {
            'cos_dist': cos_all,
            'kl_div': kl_all,
            'p_cf': p_cf_all,
            'p_actual': p_actual_all,
            'entropy': ent_all,
        }

    # Random directions (use mine->empty counterfactual for P(cf-legal) since it's the more informative one)
    N_RAND = 20
    rand_dirs = F.normalize(torch.randn(N_RAND, d_model), dim=-1)
    rand_cos = np.zeros((n_sweep, N_RAND, N_ANGLES))
    rand_kl = np.zeros((n_sweep, N_RAND, N_ANGLES))
    rand_p_cf = np.zeros((n_sweep, N_RAND, N_ANGLES))
    rand_p_actual = np.zeros((n_sweep, N_RAND, N_ANGLES))
    rand_ent = np.zeros((n_sweep, N_RAND, N_ANGLES))
    t0 = time.time()
    print(f"Random directions ({n_sweep} acts x {N_RAND} dirs)...")
    for i in range(n_sweep):
        for k in range(N_RAND):
            rand_cos[i, k], rand_kl[i, k], rand_p_cf[i, k], rand_p_actual[i, k], rand_ent[i, k] = \
                sweep_one_dir(i, rand_dirs[k], legal_cf_empty_tokens)
        if (i + 1) % 10 == 0:
            print(f"  Random: {i+1}/{n_sweep} ({(i+1)/(time.time()-t0):.1f} act/s)")

    # Save
    save_data = {
        'angles_deg': angles_deg,
        'results': results,
        'rand_cos': rand_cos, 'rand_kl': rand_kl,
        'rand_p_cf': rand_p_cf, 'rand_p_actual': rand_p_actual,
        'rand_ent': rand_ent,
        'config': {'r': target_r, 'c': target_c, 'n_sweep': n_sweep, 'n_rand': N_RAND},
    }
    with open(os.path.join(RESULTS_DIR, 'flip_kl_cos_L2_2_2.pkl'), 'wb') as f:
        pickle.dump(save_data, f)

    # --- Plot 1: P(cf legal) + cosine distance ---
    dir_colors = {'Probe (mine->empty)': 'C3'}

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    rand_cos_med = np.median(np.median(rand_cos, axis=1), axis=0)
    ax1.plot(angles_deg, rand_cos_med, color='C0', lw=2, label='Random (cos dist)')
    for name, color in dir_colors.items():
        ax1.plot(angles_deg, np.median(results[name]['cos_dist'], axis=0),
                 color=color, lw=2, label=f'{name} (cos dist)')

    rand_pcf_med = np.median(np.median(rand_p_cf, axis=1), axis=0)
    ax2.plot(angles_deg, rand_pcf_med, color='C0', lw=2, ls='--', label='Random (P(cf legal))')
    for name, color in dir_colors.items():
        ax2.plot(angles_deg, np.median(results[name]['p_cf'], axis=0),
                 color=color, lw=2, ls='--', label=f'{name} (P(cf legal))')

    ax1.set_xlabel('Perturbation angle (degrees)')
    ax1.set_ylabel(f'Cosine distance (layer {MEASURE_LAYER})')
    ax2.set_ylabel('P(counterfactual-legal moves)')
    ax1.set_xlim(0, MAX_ANGLE)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'OthelloGPT: inject L{INJECT_LAYER}, P(cf-legal) + cos dist, square ({target_r},{target_c})')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'flip_cos_L2_2_2.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Plot 2: Entropy ---
    fig, ax = plt.subplots(figsize=(10, 6))
    rand_ent_med = np.median(np.median(rand_ent, axis=1), axis=0)
    rand_ent_std = np.std(np.median(rand_ent, axis=1), axis=0)
    ax.fill_between(angles_deg, rand_ent_med - rand_ent_std, rand_ent_med + rand_ent_std,
                     color='C0', alpha=0.15)
    ax.plot(angles_deg, rand_ent_med, color='C0', lw=2, label='Random')
    for name, color in dir_colors.items():
        ent_med = np.median(results[name]['entropy'], axis=0)
        ent_std = np.std(results[name]['entropy'], axis=0)
        ax.fill_between(angles_deg, ent_med - ent_std, ent_med + ent_std,
                         color=color, alpha=0.15)
        ax.plot(angles_deg, ent_med, color=color, lw=2, label=name)
    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('Entropy (nats)')
    ax.set_title(f'OthelloGPT: output entropy, inject L{INJECT_LAYER}, square ({target_r},{target_c})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'entropy_L2_2_2.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Print summary
    print(f"\nSaved: {RESULTS_DIR}/flip_cos_L2_2_2.png")
    print(f"Saved: {RESULTS_DIR}/entropy_L2_2_2.png")
    for name in dir_colors:
        med_pcf = np.median(results[name]['p_cf'], axis=0)
        med_pa = np.median(results[name]['p_actual'], axis=0)
        med_kl = np.median(results[name]['kl_div'], axis=0)
        med_cos = np.median(results[name]['cos_dist'], axis=0)
        print(f"\n{name}:")
        print(f"  P(cf-legal) at 0°: {med_pcf[0]:.3f}, at 30°: {med_pcf[N_ANGLES//2]:.3f}, at 60°: {med_pcf[-1]:.3f}")
        print(f"  P(actual-legal) at 0°: {med_pa[0]:.3f}, at 30°: {med_pa[N_ANGLES//2]:.3f}, at 60°: {med_pa[-1]:.3f}")
        print(f"  KL div at 0°: {med_kl[0]:.4f}, at 30°: {med_kl[N_ANGLES//2]:.4f}, at 60°: {med_kl[-1]:.4f}")
        print(f"  Cos dist at 0°: {med_cos[0]:.4f}, at 30°: {med_cos[N_ANGLES//2]:.4f}, at 60°: {med_cos[-1]:.4f}")


if __name__ == '__main__':
    main()
