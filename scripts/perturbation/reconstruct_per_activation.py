#!/usr/bin/env python3
"""Per-activation reconstruction quality vs K (small K, e.g. 1-10).

For each activation individually, pick its top-K directions (by projection
magnitude for MELBO/PCA/random, by encoder activation for SAE), reconstruct
via least-squares on those K directions, measure cosine sim and probe accuracy.

This respects sparsity: different activations use different directions.
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
CENTER = {27, 28, 35, 36}

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


def reconstruct_single(act, directions_subset):
    """Reconstruct a single activation [d] using directions [K, d] via least-squares."""
    D = directions_subset.T  # [d, K]
    DtD = D.T @ D  # [K, K]
    Dtx = D.T @ act  # [K]
    c = torch.linalg.solve(DtD, Dtx)  # [K]
    return D @ c  # [d]


def omp_reconstruct_single(act, all_dirs, K):
    """Reconstruct a single activation using orthogonal matching pursuit.

    Greedily selects K directions: at each step, pick the direction most
    correlated with the residual, then do least-squares on all selected
    directions so far.

    act: [d]
    all_dirs: [M, d]
    K: number of directions to select

    Returns: [d] reconstructed activation
    """
    residual = act.clone()
    selected = []
    for _ in range(K):
        # Find direction most correlated with residual
        correlations = (all_dirs @ residual).abs()  # [M]
        # Exclude already selected
        for idx in selected:
            correlations[idx] = -1.0
        best = correlations.argmax().item()
        selected.append(best)
        # Least-squares on all selected directions so far
        D = all_dirs[selected].T  # [d, k]
        c = torch.linalg.solve(D.T @ D, D.T @ act)  # [k]
        x_hat = D @ c
        residual = act - x_hat
    return x_hat


def per_act_reconstruct(all_acts, all_dirs, K, ranking_scores=None, method='topk'):
    """For each activation, select K directions and reconstruct.

    method: 'topk' — pick top-K by ranking_scores (default for SAE/PCA)
            'omp'  — orthogonal matching pursuit (greedy residual-based selection)

    Returns: [N, d] reconstructed activations
    """
    N, d = all_acts.shape

    if method == 'omp':
        x_hat = torch.zeros_like(all_acts)
        for i in range(N):
            x_hat[i] = omp_reconstruct_single(all_acts[i], all_dirs, K)
        return x_hat

    # Top-K selection
    if ranking_scores is None:
        ranking_scores = (all_acts @ all_dirs.T).abs()  # [N, M]

    x_hat = torch.zeros_like(all_acts)
    for i in range(N):
        top_idx = ranking_scores[i].topk(K).indices
        x_hat[i] = reconstruct_single(all_acts[i], all_dirs[top_idx])
    return x_hat


def probe_accuracy(activations, probe, board_labels, occupied_only=False):
    """Mean probe accuracy across playable squares.

    If occupied_only=True, only count squares where the ground truth is not empty.
    """
    correct = 0
    total = 0
    for r in range(8):
        for c in range(8):
            sq = r * 8 + c
            if sq in CENTER:
                continue
            W = probe[0, :, r, c, :]
            logits = activations @ W
            preds = logits.argmax(dim=1).numpy()
            labels = board_labels[:, sq]
            if occupied_only:
                occ = labels != 0
                correct += (preds[occ] == labels[occ]).sum()
                total += occ.sum()
            else:
                correct += (preds == labels).sum()
                total += len(labels)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--melbo_pkl', type=str,
                        default='results/iterative_melbo_othello/L2_meas6_angle15.0_frac0.1_drift5.0/results.pkl')
    parser.add_argument('--sae_pt', type=str, default=None)
    parser.add_argument('--max_k', type=int, default=15)
    parser.add_argument('--n_rand', type=int, default=5)
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
    melbo_dirs = torch.stack(melbo_data['directions']).float()
    K_melbo = melbo_dirs.shape[0]
    print(f"  {K_melbo} MELBO directions")

    # Generate games
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

    # Extract activations and board states
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
            remapped = []
            for sq_val in board:
                if sq_val == 0: remapped.append(0)
                elif sq_val == 2: remapped.append(1)
                elif sq_val == 1: remapped.append(2)
            all_boards.append(remapped)
    all_acts = torch.cat(all_acts, dim=0)
    all_boards = np.array(all_boards)
    N = all_acts.shape[0]
    print(f"Activations: {all_acts.shape}")

    orig_probe_acc = probe_accuracy(all_acts, probe, all_boards)
    orig_probe_acc_occ = probe_accuracy(all_acts, probe, all_boards, occupied_only=True)
    print(f"Original probe accuracy: {orig_probe_acc:.4f} (all), {orig_probe_acc_occ:.4f} (occupied)")

    # Pool size: cap all methods to same number of directions for fair comparison
    pool_size = K_melbo
    print(f"Pool size (capped to MELBO count): {pool_size}")

    # PCA: top pool_size PCs by variance explained
    act_mean = all_acts.mean(dim=0)
    acts_centered = all_acts - act_mean
    U, S, Vt = torch.linalg.svd(acts_centered, full_matrices=False)
    pca_dirs = Vt[:pool_size]  # [pool_size, d]

    # SAE
    sae_dec_dirs = None
    sae_ranking = None
    if args.sae_pt:
        print(f"Loading SAE from {args.sae_pt}...")
        sae_sd = torch.load(args.sae_pt, map_location='cpu', weights_only=True)
        sae_enc_w = sae_sd['encoder.weight']
        sae_enc_b = sae_sd['encoder.bias']
        sae_dec_w = sae_sd['decoder.weight']  # [d_model, dict_size]
        dict_size = sae_enc_w.shape[0]
        print(f"  SAE dict_size={dict_size}")

        # Per-activation SAE feature activations (for ranking)
        sae_feature_acts = F.relu(all_acts @ sae_enc_w.T + sae_enc_b)  # [N, dict_size]
        sae_ranking = sae_feature_acts  # rank by encoder activation magnitude
        # Decoder columns as directions
        sae_dec_dirs_all = F.normalize(sae_dec_w.T, dim=1)  # [dict_size, d_model]

        # Cap to pool_size: keep top by mean activation
        mean_act = sae_feature_acts.mean(dim=0)
        top_sae_idx = mean_act.argsort(descending=True)[:pool_size]
        sae_dec_dirs = sae_dec_dirs_all[top_sae_idx]  # [pool_size, d_model]
        sae_ranking = sae_feature_acts[:, top_sae_idx]  # [N, pool_size]
        print(f"  SAE capped to {pool_size} directions (top by mean activation)")

        # Stats
        n_active = (sae_feature_acts > 0).float().sum(dim=1)
        print(f"  Mean active features per activation (full): {n_active.mean():.1f} "
              f"(median: {n_active.median():.0f}, min: {n_active.min():.0f}, max: {n_active.max():.0f})")

    # K values: 1-10 by 1, then by 10 up to max_k
    K_values = list(range(1, min(11, args.max_k + 1)))
    if args.max_k > 10:
        K_values += list(range(20, args.max_k + 1, 10))

    print(f"\nPer-activation reconstruction (K=1..{args.max_k})...")

    # Store full per-activation cosine sim distributions
    melbo_cos_all = []; melbo_probe = []; melbo_probe_occ = []
    pca_cos_all = []; pca_probe = []; pca_probe_occ = []
    sae_cos_all = []; sae_probe = []; sae_probe_occ = []
    sae_native_cos_all = []; sae_native_probe = []; sae_native_probe_occ = []
    rand_cos_all = []  # list of [n_rand, N] arrays per K
    rand_probe_mean = []; rand_probe_std = []
    rand_probe_occ_mean = []; rand_probe_occ_std = []

    # Save per-activation reconstructions for each K
    saved_reconstructions = {}

    for K in K_values:
        recons_this_K = {}

        # MELBO: OMP selection (greedy residual-based)
        x_hat = per_act_reconstruct(all_acts, melbo_dirs, K, method='omp')
        melbo_cos_all.append(F.cosine_similarity(all_acts, x_hat, dim=1).numpy())
        melbo_probe.append(probe_accuracy(x_hat, probe, all_boards))
        melbo_probe_occ.append(probe_accuracy(x_hat, probe, all_boards, occupied_only=True))
        recons_this_K['melbo'] = x_hat

        # PCA: per-activation ranking
        x_hat_pca = per_act_reconstruct(acts_centered, pca_dirs, K)
        x_hat_pca = x_hat_pca + act_mean
        pca_cos_all.append(F.cosine_similarity(all_acts, x_hat_pca, dim=1).numpy())
        pca_probe.append(probe_accuracy(x_hat_pca, probe, all_boards))
        pca_probe_occ.append(probe_accuracy(x_hat_pca, probe, all_boards, occupied_only=True))
        recons_this_K['pca'] = x_hat_pca

        # SAE (least-squares): rank by encoder activation
        if sae_dec_dirs is not None:
            x_hat_sae = per_act_reconstruct(all_acts, sae_dec_dirs, K, ranking_scores=sae_ranking)
            sae_cos_all.append(F.cosine_similarity(all_acts, x_hat_sae, dim=1).numpy())
            sae_probe.append(probe_accuracy(x_hat_sae, probe, all_boards))
            sae_probe_occ.append(probe_accuracy(x_hat_sae, probe, all_boards, occupied_only=True))
            recons_this_K['sae'] = x_hat_sae
        else:
            sae_cos_all.append(np.full(N, np.nan))
            sae_probe.append(float('nan'))
            sae_probe_occ.append(float('nan'))

        # SAE (native): encoder → restrict to pool → top-K → decoder
        if args.sae_pt:
            acts_centered_sae = all_acts - sae_sd['bias']
            feat_acts = F.relu(acts_centered_sae @ sae_enc_w.T + sae_enc_b)  # [N, dict_size]
            feat_acts_pool = feat_acts.clone()
            mask = torch.ones(dict_size, dtype=torch.bool)
            mask[top_sae_idx] = False
            feat_acts_pool[:, mask] = 0.0
            topk_vals, topk_idx = feat_acts_pool.topk(K, dim=1)
            feat_sparse = torch.zeros_like(feat_acts_pool)
            feat_sparse.scatter_(1, topk_idx, topk_vals)
            x_hat_native = feat_sparse @ sae_dec_w.T + sae_sd['bias']  # [N, d_model]
            sae_native_cos_all.append(F.cosine_similarity(all_acts, x_hat_native, dim=1).numpy())
            sae_native_probe.append(probe_accuracy(x_hat_native, probe, all_boards))
            sae_native_probe_occ.append(probe_accuracy(x_hat_native, probe, all_boards, occupied_only=True))
            recons_this_K['sae_native'] = x_hat_native
        else:
            sae_native_cos_all.append(np.full(N, np.nan))
            sae_native_probe.append(float('nan'))
            sae_native_probe_occ.append(float('nan'))

        # Random: same pool size as others
        rand_this_cos = []
        rand_probe_this = []
        rand_probe_occ_this = []
        for _ in range(args.n_rand):
            rand_d = F.normalize(torch.randn(pool_size, d_model), dim=1)
            x_hat_rand = per_act_reconstruct(all_acts, rand_d, K)
            rand_this_cos.append(F.cosine_similarity(all_acts, x_hat_rand, dim=1).numpy())
            rand_probe_this.append(probe_accuracy(x_hat_rand, probe, all_boards))
            rand_probe_occ_this.append(probe_accuracy(x_hat_rand, probe, all_boards, occupied_only=True))
        rand_cos_all.append(np.array(rand_this_cos))  # [n_rand, N]
        rand_probe_mean.append(np.mean(rand_probe_this))
        rand_probe_std.append(np.std(rand_probe_this))
        rand_probe_occ_mean.append(np.mean(rand_probe_occ_this))
        rand_probe_occ_std.append(np.std(rand_probe_occ_this))

        saved_reconstructions[K] = recons_this_K

        sae_str = f", SAE={sae_cos_all[-1].mean():.4f}/{sae_probe_occ[-1]:.4f}" if sae_dec_dirs is not None else ""
        native_str = f", Native={sae_native_cos_all[-1].mean():.4f}/{sae_native_probe_occ[-1]:.4f}" if args.sae_pt else ""
        print(f"  K={K}: MELBO={melbo_cos_all[-1].mean():.4f}/{melbo_probe_occ[-1]:.4f}, "
              f"PCA={pca_cos_all[-1].mean():.4f}/{pca_probe_occ[-1]:.4f}{sae_str}{native_str}, "
              f"Rand={rand_cos_all[-1].mean():.4f}/{rand_probe_occ_mean[-1]:.4f}")

    # Compute summary stats from full distributions
    def summarize(cos_list):
        return {
            'mean': [c.mean() for c in cos_list],
            'max': [c.max() for c in cos_list],
            'min': [c.min() for c in cos_list],
            'p5': [np.percentile(c, 5) for c in cos_list],
            'p95': [np.percentile(c, 95) for c in cos_list],
        }

    melbo_stats = summarize(melbo_cos_all)
    pca_stats = summarize(pca_cos_all)
    sae_stats = summarize(sae_cos_all)
    sae_native_stats = summarize(sae_native_cos_all)
    # For random, average across repeats first then summarize
    rand_mean_per_act = [arr.mean(axis=0) for arr in rand_cos_all]  # list of [N] arrays
    rand_stats = summarize(rand_mean_per_act)

    # Save
    results = {
        'K_values': K_values,
        'melbo_stats': melbo_stats, 'melbo_probe': melbo_probe, 'melbo_probe_occ': melbo_probe_occ,
        'pca_stats': pca_stats, 'pca_probe': pca_probe, 'pca_probe_occ': pca_probe_occ,
        'sae_stats': sae_stats, 'sae_probe': sae_probe, 'sae_probe_occ': sae_probe_occ,
        'sae_native_stats': sae_native_stats, 'sae_native_probe': sae_native_probe,
        'sae_native_probe_occ': sae_native_probe_occ,
        'rand_stats': rand_stats,
        'rand_probe_mean': rand_probe_mean, 'rand_probe_std': rand_probe_std,
        'rand_probe_occ_mean': rand_probe_occ_mean, 'rand_probe_occ_std': rand_probe_occ_std,
        'orig_probe_acc': orig_probe_acc,
        'orig_probe_acc_occ': orig_probe_acc_occ,
        'K_melbo': K_melbo,
        'pool_size': pool_size,
        'layer': args.layer,
        'all_acts': all_acts,
        'all_boards': all_boards,
        'reconstructions': saved_reconstructions,
    }
    pkl_path = os.path.join(RESULTS_DIR, 'reconstruction_per_act.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # Plot: 3 panels — mean cos sim, max cos sim, probe accuracy
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    K_arr = np.array(K_values)

    # Helper to plot one cos sim panel
    def plot_cos_panel(ax, stat_key, title):
        ax.plot(K_arr, melbo_stats[stat_key], 'C2-o', lw=2, ms=5, label='MELBO')
        ax.plot(K_arr, pca_stats[stat_key], 'C1-^', lw=2, ms=5, label='PCA')
        if sae_dec_dirs is not None:
            ax.plot(K_arr, sae_stats[stat_key], 'C3-s', lw=2, ms=5, label='SAE (LS)')
            ax.plot(K_arr, sae_native_stats[stat_key], 'C4-d', lw=2, ms=5, label='SAE (native)')
        ax.plot(K_arr, rand_stats[stat_key], 'C0-', lw=2, label='Random')
        ax.set_xlabel('K (directions per activation)')
        ax.set_ylabel('Cosine similarity')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plot_cos_panel(ax1, 'mean', 'Mean cosine similarity')
    plot_cos_panel(ax2, 'max', 'Max cosine similarity')

    # Probe accuracy (occupied only)
    ax3.plot(K_arr, melbo_probe_occ, 'C2-o', lw=2, ms=5, label='MELBO')
    ax3.plot(K_arr, pca_probe_occ, 'C1-^', lw=2, ms=5, label='PCA')
    if sae_dec_dirs is not None:
        ax3.plot(K_arr, sae_probe_occ, 'C3-s', lw=2, ms=5, label='SAE (LS)')
        ax3.plot(K_arr, sae_native_probe_occ, 'C4-d', lw=2, ms=5, label='SAE (native)')
    rm = np.array(rand_probe_occ_mean); rs = np.array(rand_probe_occ_std)
    ax3.fill_between(K_arr, rm - rs, rm + rs, color='C0', alpha=0.15)
    ax3.plot(K_arr, rm, 'C0-', lw=2, label='Random')
    ax3.axhline(orig_probe_acc_occ, color='k', ls='--', lw=1, label=f'Original ({orig_probe_acc_occ:.3f})')
    ax3.set_xlabel('K (directions per activation)')
    ax3.set_ylabel('Probe accuracy (occupied)')
    ax3.set_title('Probe accuracy (occupied squares)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f'OthelloGPT Layer {args.layer}: Per-activation reconstruction (pool={pool_size})',
                 fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'reconstruction_per_act.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


if __name__ == '__main__':
    main()
