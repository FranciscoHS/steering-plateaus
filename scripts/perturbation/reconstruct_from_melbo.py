#!/usr/bin/env python3
"""Reconstruction quality vs number of MELBO directions.

Given a set of MELBO directions found at a given layer, reconstruct activations
using least-squares projection onto K directions, and measure cosine similarity.

Baselines: random directions, PCA top-K.

Assumes MELBO directions have already been found and saved.
"""

import os, sys, pickle, argparse
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer

SEED = 42
SEQ_LEN = 20
N_GAMES = 2000
RESULTS_DIR = 'results/othello_plateau'
PROBE_DIR = '/workspace/mech_int_othelloGPT/mech_int/probes/linear'

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


def reconstruct_least_squares(activations, directions_matrix):
    """Reconstruct activations using least-squares onto directions.

    activations: [N, d_model]
    directions_matrix: [K, d_model]

    Returns: [N, d_model] reconstructed activations
    """
    # c = (D^T D)^{-1} D^T x^T  -> [K, N]
    # x_hat = D^T c -> [d_model, N]
    D = directions_matrix.T  # [d_model, K]
    DtD = D.T @ D  # [K, K]
    DtX = D.T @ activations.T  # [K, N]
    # Solve DtD @ c = DtX
    c = torch.linalg.solve(DtD, DtX)  # [K, N]
    x_hat = D @ c  # [d_model, N]
    return x_hat.T  # [N, d_model]


def cosine_sim_batch(a, b):
    """Mean cosine similarity between corresponding rows of a and b."""
    return F.cosine_similarity(a, b, dim=1).mean().item()


CENTER = {27, 28, 35, 36}

def probe_accuracy(activations, probe, board_labels):
    """Compute mean probe accuracy across all playable squares.

    activations: [N, d_model]
    probe: [1, d_model, 8, 8, 3]
    board_labels: [N, 64] with values 0/1/2 (empty/mine/opp)

    Returns: mean accuracy across all playable squares.
    """
    correct = 0
    total = 0
    for r in range(8):
        for c in range(8):
            sq = r * 8 + c
            if sq in CENTER:
                continue
            W = probe[0, :, r, c, :]  # [d_model, 3]
            logits = activations @ W   # [N, 3]
            preds = logits.argmax(dim=1).numpy()
            labels = board_labels[:, sq]
            correct += (preds == labels).sum()
            total += len(labels)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--melbo_pkl', type=str, default='results/othello_plateau/melbo_othello.pkl')
    parser.add_argument('--n_rand', type=int, default=10, help='Number of random baseline repeats')
    parser.add_argument('--sae_pt', type=str, default=None, help='Path to SAE ae.pt weights')
    args = parser.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = 'cuda'

    model = HookedTransformer.from_pretrained("othello-gpt")
    model.to(device).eval()
    d_model = model.cfg.d_model

    # Load MELBO directions
    print(f"Loading MELBO directions from {args.melbo_pkl}...")
    with open(args.melbo_pkl, 'rb') as f:
        melbo_data = pickle.load(f)
    melbo_dirs = torch.stack(melbo_data['directions']).float()  # [K_melbo, d_model]
    K_melbo = melbo_dirs.shape[0]
    print(f"  {K_melbo} MELBO directions")

    # Generate games and extract activations
    print("Generating games...")
    all_games = []
    for _ in range(N_GAMES * 2):
        m, b = play_random_game(rng)
        if len(m) >= SEQ_LEN:
            all_games.append((m[:SEQ_LEN], b[:SEQ_LEN]))
        if len(all_games) >= N_GAMES:
            break
    print(f"Got {len(all_games)} games")

    # Load probe
    probe = torch.load(f'{PROBE_DIR}/resid_{args.layer}_linear.pth',
                        map_location='cpu', weights_only=True).float()
    print(f"Probe shape: {probe.shape}")

    hook_name = f"blocks.{args.layer}.hook_resid_post"
    all_acts = []
    all_boards = []
    batch_size = 64
    for i in range(0, len(all_games), batch_size):
        batch = all_games[i:i+batch_size]
        tokens = torch.tensor([g[0] for g in batch], dtype=torch.long).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        all_acts.append(cache[hook_name][:, -1, :].float().cpu())
        for g in batch:
            board = g[1][SEQ_LEN - 1]
            # Remap: at move SEQ_LEN-1 (0-indexed), player who moved = player 2
            # player 2 = mine (1), player 1 = opp (2)
            remapped = []
            for sq_val in board:
                if sq_val == 0: remapped.append(0)
                elif sq_val == 2: remapped.append(1)
                elif sq_val == 1: remapped.append(2)
            all_boards.append(remapped)
    all_acts = torch.cat(all_acts, dim=0)  # [N, d_model]
    all_boards = np.array(all_boards)  # [N, 64]
    print(f"Activations: {all_acts.shape}")

    # Baseline probe accuracy on original activations
    orig_probe_acc = probe_accuracy(all_acts, probe, all_boards)
    print(f"Original probe accuracy: {orig_probe_acc:.4f}")

    # Center activations (subtract mean) for PCA
    act_mean = all_acts.mean(dim=0)
    acts_centered = all_acts - act_mean

    # PCA
    print("Computing PCA...")
    U, S, Vt = torch.linalg.svd(acts_centered, full_matrices=False)
    pca_dirs = Vt  # [d_model, d_model] — rows are PCs

    # Load SAE if provided
    sae_dirs = None
    if args.sae_pt:
        print(f"Loading SAE from {args.sae_pt}...")
        sae_sd = torch.load(args.sae_pt, map_location='cpu', weights_only=True)
        sae_enc_w = sae_sd['encoder.weight']  # [dict_size, d_model]
        sae_enc_b = sae_sd['encoder.bias']    # [dict_size]
        sae_dec_w = sae_sd['decoder.weight']  # [d_model, dict_size]
        sae_bias = sae_sd['bias']             # [d_model]
        dict_size = sae_enc_w.shape[0]
        print(f"  SAE dict_size={dict_size}")

        # Compute mean activation per feature to rank them
        sae_acts = F.relu(all_acts @ sae_enc_w.T + sae_enc_b)  # [N, dict_size]
        mean_act = sae_acts.mean(dim=0)  # [dict_size]
        sorted_idx = mean_act.argsort(descending=True)
        # SAE directions = decoder columns, ordered by mean activation
        sae_dirs = F.normalize(sae_dec_w.T[sorted_idx], dim=1)  # [dict_size, d_model]
        print(f"  Top feature mean activations: {mean_act[sorted_idx[:5]].tolist()}")

    # K values to test
    K_values = list(range(1, min(K_melbo + 1, d_model + 1)))

    # Reconstruction quality for each K
    print("Computing reconstruction quality...")

    melbo_cos = []; melbo_probe = []
    pca_cos = []; pca_probe = []
    sae_cos = []; sae_probe = []
    rand_cos_mean = []; rand_cos_std = []
    rand_probe_mean = []; rand_probe_std = []

    for K in K_values:
        # MELBO
        if K <= K_melbo:
            x_hat = reconstruct_least_squares(all_acts, melbo_dirs[:K])
            melbo_cos.append(cosine_sim_batch(all_acts, x_hat))
            melbo_probe.append(probe_accuracy(x_hat, probe, all_boards))
        else:
            melbo_cos.append(float('nan'))
            melbo_probe.append(float('nan'))

        # PCA (on centered data, then add mean back)
        x_hat_pca = reconstruct_least_squares(acts_centered, pca_dirs[:K]) + act_mean
        pca_cos.append(cosine_sim_batch(all_acts, x_hat_pca))
        pca_probe.append(probe_accuracy(x_hat_pca, probe, all_boards))

        # SAE (top-K most active features, least-squares)
        if sae_dirs is not None and K <= sae_dirs.shape[0]:
            x_hat_sae = reconstruct_least_squares(all_acts, sae_dirs[:K])
            sae_cos.append(cosine_sim_batch(all_acts, x_hat_sae))
            sae_probe.append(probe_accuracy(x_hat_sae, probe, all_boards))
        else:
            sae_cos.append(float('nan'))
            sae_probe.append(float('nan'))

        # Random (multiple repeats)
        rand_cos_this = []
        rand_probe_this = []
        for _ in range(args.n_rand):
            rand_d = F.normalize(torch.randn(K, d_model), dim=1)
            x_hat_rand = reconstruct_least_squares(all_acts, rand_d)
            rand_cos_this.append(cosine_sim_batch(all_acts, x_hat_rand))
            rand_probe_this.append(probe_accuracy(x_hat_rand, probe, all_boards))
        rand_cos_mean.append(np.mean(rand_cos_this))
        rand_cos_std.append(np.std(rand_cos_this))
        rand_probe_mean.append(np.mean(rand_probe_this))
        rand_probe_std.append(np.std(rand_probe_this))

        sae_str = f", SAE={sae_cos[-1]:.4f}/{sae_probe[-1]:.4f}" if sae_dirs is not None else ""
        if K % 10 == 0 or K == K_melbo:
            print(f"  K={K}: MELBO={melbo_cos[-1]:.4f}/{melbo_probe[-1]:.4f}, "
                  f"PCA={pca_cos[-1]:.4f}/{pca_probe[-1]:.4f}{sae_str}, "
                  f"Rand={rand_cos_mean[-1]:.4f}/{rand_probe_mean[-1]:.4f}")

    # Save
    results = {
        'K_values': K_values,
        'melbo_cos': melbo_cos, 'melbo_probe': melbo_probe,
        'pca_cos': pca_cos, 'pca_probe': pca_probe,
        'sae_cos': sae_cos, 'sae_probe': sae_probe,
        'rand_cos_mean': rand_cos_mean, 'rand_cos_std': rand_cos_std,
        'rand_probe_mean': rand_probe_mean, 'rand_probe_std': rand_probe_std,
        'orig_probe_acc': orig_probe_acc,
        'K_melbo': K_melbo,
        'layer': args.layer,
    }
    pkl_path = os.path.join(RESULTS_DIR, 'reconstruction.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {pkl_path}")

    # Plot: two panels — cosine similarity and probe accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    K_arr = np.array(K_values)
    melbo_arr = np.array(melbo_cos)
    valid = ~np.isnan(melbo_arr)

    # Left: cosine similarity
    ax1.plot(K_arr[valid], melbo_arr[valid], 'C2-o', lw=2, markersize=3, label='MELBO')
    ax1.plot(K_arr, pca_cos, 'C1-', lw=2, label='PCA')
    if sae_dirs is not None:
        sae_arr = np.array(sae_cos)
        sae_valid = ~np.isnan(sae_arr)
        ax1.plot(K_arr[sae_valid], sae_arr[sae_valid], 'C3-s', lw=2, markersize=3, label='SAE')
    rand_m = np.array(rand_cos_mean)
    rand_s = np.array(rand_cos_std)
    ax1.fill_between(K_arr, rand_m - rand_s, rand_m + rand_s, color='C0', alpha=0.15)
    ax1.plot(K_arr, rand_m, 'C0-', lw=2, label='Random')
    ax1.set_xlabel('Number of directions (K)')
    ax1.set_ylabel('Cosine similarity')
    ax1.set_title('Reconstruction: cosine similarity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: probe accuracy
    melbo_p = np.array(melbo_probe)
    ax2.plot(K_arr[valid], melbo_p[valid], 'C2-o', lw=2, markersize=3, label='MELBO')
    ax2.plot(K_arr, pca_probe, 'C1-', lw=2, label='PCA')
    if sae_dirs is not None:
        sae_p = np.array(sae_probe)
        ax2.plot(K_arr[sae_valid], sae_p[sae_valid], 'C3-s', lw=2, markersize=3, label='SAE')
    rand_pm = np.array(rand_probe_mean)
    rand_ps = np.array(rand_probe_std)
    ax2.fill_between(K_arr, rand_pm - rand_ps, rand_pm + rand_ps, color='C0', alpha=0.15)
    ax2.plot(K_arr, rand_pm, 'C0-', lw=2, label='Random')
    ax2.axhline(orig_probe_acc, color='k', ls='--', lw=1, label=f'Original ({orig_probe_acc:.3f})')
    ax2.set_xlabel('Number of directions (K)')
    ax2.set_ylabel('Probe accuracy')
    ax2.set_title('Reconstruction: probe accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'OthelloGPT Layer {args.layer}: Reconstruction quality', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'reconstruction.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


if __name__ == '__main__':
    main()
