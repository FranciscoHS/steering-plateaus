#!/usr/bin/env python3
"""Iterative capped MELBO with local optimization between rounds on OthelloGPT.

Each round at a fixed opt_angle:
  1. Find directions via capped MELBO until reward drops below
     reward_fraction * best_reward_this_round
  2. Refine each new direction via cone-constrained local opt (max_drift degrees)

Stops when a round adds zero new directions (refinement didn't open up space
for the next round to find anything new).

Usage:
    python scripts/perturbation/run_iterative_melbo_othello.py
    python scripts/perturbation/run_iterative_melbo_othello.py --opt_angle 5 --reward_fraction 0.1
"""

import argparse, os, pickle, time
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer

# Token mapping for OthelloGPT
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


def forward_to_L(model, ctx, a_pert_batch, inject_layer, measure_layer):
    """Forward from inject_layer to measure_layer. a_pert_batch: [batch, d_model]"""
    hidden = torch.cat([ctx.expand(a_pert_batch.shape[0], -1, -1),
                        a_pert_batch.unsqueeze(1)], dim=1)
    for block in model.blocks[inject_layer + 1:measure_layer + 1]:
        hidden = block(hidden)
    return hidden[:, -1, :].float()


def capped_melbo_one_direction(model, ctx, act, ref_L, a_unit, a_norm,
                                angle_rad, target_cos, prev_directions,
                                inject_layer, measure_layer, device, d_model,
                                n_steps=300, lr=0.01, n_restarts=3,
                                init_directions=None):
    """Find one direction via capped MELBO, orthogonalized to prev_directions.

    If init_directions is provided (list of vectors), uses them as initialization
    for the first len(init_directions) restarts instead of random.
    """
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    best_reward = -float('inf')
    best_direction = None
    best_history = None

    for restart in range(n_restarts):
        if init_directions is not None and restart < len(init_directions):
            direction = init_directions[restart].to(device).float().clone()
        else:
            direction = torch.randn(d_model, device=device)
        direction = direction - (direction @ a_unit) * a_unit
        for pd in prev_directions:
            direction = direction - (direction @ pd) * pd
        direction = F.normalize(direction, dim=0)
        direction = direction.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([direction], lr=lr)
        history = []

        for step in range(n_steps):
            optimizer.zero_grad()

            d = direction / direction.norm()
            d = d - (d @ a_unit) * a_unit
            for pd in prev_directions:
                d = d - (d @ pd) * pd
            d = d / (d.norm() + 1e-10)

            a_pert = a_norm * (cos_theta * a_unit + sin_theta * d)
            pert_L = forward_to_L(model, ctx, a_pert.unsqueeze(0),
                                  inject_layer, measure_layer)

            cos_dist = 1 - F.cosine_similarity(ref_L, pert_L, dim=-1)
            cos_dist = cos_dist.squeeze()
            reward = cos_dist * torch.exp(-cos_dist / target_cos)

            loss = -reward
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                direction.data -= (direction.data @ a_unit) * a_unit
                for pd in prev_directions:
                    direction.data -= (direction.data @ pd) * pd

            history.append((reward.item(), cos_dist.item()))

            if step % 100 == 0 or step == n_steps - 1:
                print(f"      restart {restart+1}/{n_restarts}, step {step}: "
                      f"reward={reward.item():.6f}, cos_dist={cos_dist.item():.4f}")

        final_reward = history[-1][0]
        if final_reward > best_reward:
            best_reward = final_reward
            with torch.no_grad():
                d_final = direction / direction.norm()
                d_final = d_final - (d_final @ a_unit) * a_unit
                for pd in prev_directions:
                    d_final = d_final - (d_final @ pd) * pd
                d_final = d_final / d_final.norm()
                best_direction = d_final.clone()
            best_history = history

    return best_direction, best_reward, best_history


def refine_in_cone(model, ctx, act, ref_L, a_unit, a_norm,
                   d_start, max_drift_deg, angle_rad,
                   inject_layer, measure_layer, device, d_model,
                   n_steps=200, lr=0.01, n_restarts=3):
    """Refine a direction within a cone of max_drift_deg around d_start.

    Maximizes cosine distance (uncapped) subject to angular constraint.
    Returns: refined_direction, final_cd, actual_drift_deg
    """
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    cos_drift_min = np.cos(np.deg2rad(max_drift_deg))

    best_cd = -1.0
    best_dir = None

    for restart in range(n_restarts):
        if max_drift_deg < 1.0:
            direction = d_start.clone()
        else:
            noise = torch.randn(d_model, device=device)
            noise = noise - (noise @ d_start) * d_start
            noise = F.normalize(noise, dim=0)
            rand_angle = np.deg2rad(max_drift_deg * np.random.uniform(0, 0.5))
            direction = np.cos(rand_angle) * d_start + np.sin(rand_angle) * noise

        direction = F.normalize(direction, dim=0)
        direction = direction.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([direction], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()

            d = direction / direction.norm()
            d = d - (d @ a_unit) * a_unit
            d = d / (d.norm() + 1e-10)

            a_pert = a_norm * (cos_theta * a_unit + sin_theta * d)
            pert_L = forward_to_L(model, ctx, a_pert.unsqueeze(0),
                                  inject_layer, measure_layer)

            cos_dist = 1 - F.cosine_similarity(ref_L, pert_L, dim=-1)
            cos_dist = cos_dist.squeeze()

            loss = -cos_dist
            loss.backward()
            optimizer.step()

            # Project back into cone
            with torch.no_grad():
                direction.data = F.normalize(direction.data, dim=0)
                cos_sim = (direction.data @ d_start).item()
                if cos_sim < cos_drift_min:
                    d_parallel = (direction.data @ d_start) * d_start
                    d_perp = direction.data - d_parallel
                    if d_perp.norm() > 1e-8:
                        d_perp = F.normalize(d_perp, dim=0)
                    sin_max = np.sin(np.deg2rad(max_drift_deg))
                    direction.data = cos_drift_min * d_start + sin_max * d_perp
                    direction.data = F.normalize(direction.data, dim=0)

        final_cd = cos_dist.item()
        if final_cd > best_cd:
            best_cd = final_cd
            best_dir = F.normalize(direction.data.clone().detach(), dim=0)

    drift = np.degrees(np.arccos(np.clip((best_dir @ d_start).item(), -1, 1)))
    return best_dir, best_cd, drift


def sweep_dir(model, ctx, act, ref_L, a_unit, a_norm, d_vec,
              inject_layer, measure_layer, device, angles_deg):
    """Sweep a direction across angles, return cosine distance array."""
    angles_rad = np.deg2rad(angles_deg)
    cos_t = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)

    d = d_vec.to(device)
    d = d - (d @ a_unit) * a_unit
    d = F.normalize(d, dim=0)
    perturbed = a_norm * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * d[None])
    with torch.no_grad():
        pert_L = forward_to_L(model, ctx, perturbed, inject_layer, measure_layer)
        cos_dist = (1 - F.cosine_similarity(ref_L, pert_L, dim=-1)).cpu().numpy()
    return cos_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inject_layer', type=int, default=2)
    parser.add_argument('--measure_layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--n_games', type=int, default=500)
    parser.add_argument('--act_idx', type=int, default=0,
                        help='Index into mine-color games')
    parser.add_argument('--target_sq', type=str, default='2,2',
                        help='Target square as row,col')

    # MELBO config
    parser.add_argument('--opt_angle', type=float, default=5.0)
    parser.add_argument('--target_cos', type=float, default=0.1)
    parser.add_argument('--reward_fraction', type=float, default=0.1,
                        help='Stop adding directions when reward < fraction * best_reward_this_round')

    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)

    # Local opt config
    parser.add_argument('--refine_steps', type=int, default=200)
    parser.add_argument('--refine_restarts', type=int, default=3)
    parser.add_argument('--max_drift', type=float, default=5.0,
                        help='Max angular drift during refinement (degrees)')
    parser.add_argument('--refine_lr', type=float, default=0.01)

    # Sweep config
    parser.add_argument('--max_sweep_angle', type=float, default=60.0)
    parser.add_argument('--n_sweep_angles', type=int, default=60)
    parser.add_argument('--n_rand', type=int, default=20)

    parser.add_argument('--seed_directions', type=str, default=None,
                        help='Path to results.pkl from a previous run; loads directions as frozen seeds')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint .pkl to resume from (loads directions + round state)')
    parser.add_argument('--frame_seed', action='store_true',
                        help='Seed MELBO restarts with deficit eigenvectors of the frame operator')
    parser.add_argument('--n_frame_seeds', type=int, default=None,
                        help='Number of deficit eigenvectors to use as seeds (default: n_restarts)')
    parser.add_argument('--ckpt_every', type=int, default=10,
                        help='Checkpoint every N directions within a round')
    parser.add_argument('--probe_path', type=str,
                        default='/workspace/mech_int_othelloGPT/mech_int/probes/linear/resid_2_linear.pth')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    target_r, target_c = [int(x) for x in args.target_sq.split(',')]
    target_sq = target_r * 8 + target_c
    mine_color = 2

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = (f'results/iterative_melbo_othello/'
                           f'L{args.inject_layer}_meas{args.measure_layer}'
                           f'_angle{args.opt_angle}_frac{args.reward_fraction}'
                           f'_drift{args.max_drift}')
    os.makedirs(args.results_dir, exist_ok=True)
    device = 'cuda'

    # Load model
    model = HookedTransformer.from_pretrained("othello-gpt")
    model.to(device).eval()
    d_model = model.cfg.d_model

    # Load probe if available
    probe_dir = None
    if os.path.exists(args.probe_path):
        probe = torch.load(args.probe_path, map_location='cpu', weights_only=True)
        probe_dir = F.normalize(
            (probe[0, :, target_r, target_c, 0] - probe[0, :, target_r, target_c, 1]).float(), dim=0)
        print(f"Loaded probe for square ({target_r},{target_c})")
    else:
        print(f"Probe not found at {args.probe_path}, skipping probe comparison")

    # Generate games
    print("Generating games...")
    all_games = []
    for _ in range(args.n_games):
        m, b = play_random_game(rng)
        if len(m) >= args.seq_len:
            all_games.append((m[:args.seq_len], b[:args.seq_len]))

    mine_idx = [i for i, (m, b) in enumerate(all_games) if b[args.seq_len-1][target_sq] == mine_color]
    print(f"Games with mine on ({target_r},{target_c}): {len(mine_idx)}")

    # Extract activation
    hook_name = f"blocks.{args.inject_layer}.hook_resid_post"
    tokens = torch.tensor(all_games[mine_idx[args.act_idx]][0],
                         dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    resid = cache[hook_name].float()
    ctx = resid[:, :-1, :]   # [1, seq-1, d_model]
    act = resid[0, -1, :]    # [d_model]
    a_norm = act.norm()
    a_unit = act / a_norm
    print(f"Activation norm: {a_norm:.2f}")

    # Reference at measure layer
    with torch.no_grad():
        ref_L = forward_to_L(model, ctx, act.unsqueeze(0),
                             args.inject_layer, args.measure_layer)

    angle_rad = np.deg2rad(args.opt_angle)

    # === Iterative MELBO with local opt ===
    all_directions = []  # accumulated across rounds
    round_logs = []
    rnd = 0
    best_reward_ever = None

    # Load seed directions if provided (frozen — not refined, only used for orthogonality)
    n_seed = 0
    if args.seed_directions:
        seed_data = pickle.load(open(args.seed_directions, 'rb'))
        seed_dirs = seed_data['directions']
        for d in seed_dirs:
            all_directions.append(d.to(device).float())
        n_seed = len(all_directions)
        print(f"Loaded {n_seed} seed directions from {args.seed_directions}")

    # Resume from checkpoint if provided
    if args.resume_checkpoint:
        ckpt = pickle.load(open(args.resume_checkpoint, 'rb'))
        all_directions = [d.to(device).float() for d in ckpt['directions']]
        round_logs = ckpt.get('round_logs', [])
        rnd = ckpt.get('round', 0)
        best_reward_ever = ckpt.get('best_reward_ever', None)
        if best_reward_ever is None and round_logs:
            # Reconstruct from round logs
            for rl in round_logs:
                for d in rl.get('pre_refine', []):
                    if best_reward_ever is None or d['reward'] > best_reward_ever:
                        best_reward_ever = d['reward']
        # Count seeds: if seed_directions was used in original run, those are still seeds
        if args.seed_directions:
            n_seed = len(pickle.load(open(args.seed_directions, 'rb'))['directions'])
        print(f"Resumed from {args.resume_checkpoint}: {len(all_directions)} directions, "
              f"round {rnd}, best_reward_ever={best_reward_ever}")

    # Compute frame operator deficit eigenvectors for seeding
    def compute_deficit_seeds(directions, n_seeds):
        """Return bottom eigenvectors of the frame operator S = D^T D."""
        if len(directions) == 0:
            return None
        D = torch.stack(directions).float()  # [N, d]
        S = D.T @ D  # [d, d]
        eigvals, eigvecs = torch.linalg.eigh(S)
        # Bottom eigenvectors (smallest eigenvalues) = deficit directions
        deficit_dirs = [eigvecs[:, i] for i in range(min(n_seeds, d_model))]
        print(f"  Frame operator: eigenval range [{eigvals[0].item():.2f}, "
              f"{eigvals[-1].item():.2f}], using bottom {len(deficit_dirs)} as seeds")
        return deficit_dirs

    n_frame_seeds = args.n_frame_seeds if args.n_frame_seeds else args.n_restarts

    while True:
        rnd += 1
        print(f"\n{'='*60}")
        print(f"ROUND {rnd} — {len(all_directions)} directions so far")
        print(f"{'='*60}")

        # Compute deficit seeds at the start of each round
        deficit_seeds = None
        if args.frame_seed and len(all_directions) > 0:
            deficit_seeds = compute_deficit_seeds(all_directions, n_frame_seeds)

        round_info = {
            'round': rnd,
            'pre_refine': [],
            'post_refine': [],
        }

        # Phase 1: Find directions until reward drops below threshold
        new_directions = []
        d_idx = 0

        while True:
            d_idx += 1
            print(f"\n  --- Direction {d_idx} "
                  f"(global #{len(all_directions)+len(new_directions)+1}) ---")
            prev = all_directions + new_directions
            direction, reward, history = capped_melbo_one_direction(
                model, ctx, act, ref_L, a_unit, a_norm,
                angle_rad, args.target_cos, prev,
                args.inject_layer, args.measure_layer, device, d_model,
                n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts,
                init_directions=deficit_seeds)

            if best_reward_ever is None or reward > best_reward_ever:
                best_reward_ever = reward

            print(f"    reward={reward:.6f}, cos_dist={history[-1][1]:.4f}, "
                  f"ratio={reward/best_reward_ever:.3f}")

            if reward < args.reward_fraction * best_reward_ever:
                print(f"    Reward dropped below {args.reward_fraction:.0%} of "
                      f"best ever ({best_reward_ever:.6f}), stopping round")
                break

            new_directions.append(direction)
            round_info['pre_refine'].append({
                'reward': reward, 'cos_dist': history[-1][1],
                'direction': direction.cpu(),
            })

            # Mid-round checkpoint every N directions
            if len(new_directions) % args.ckpt_every == 0:
                mid_ckpt = {
                    'round': rnd,
                    'directions': [d.cpu() for d in all_directions + new_directions],
                    'round_logs': round_logs + [round_info],
                    'best_reward_ever': best_reward_ever,
                    'args': vars(args),
                    'mid_round': True,
                    'n_new_unrefined': len(new_directions),
                }
                mid_path = os.path.join(args.results_dir, f'checkpoint_round{rnd}_mid.pkl')
                with open(mid_path, 'wb') as f:
                    pickle.dump(mid_ckpt, f)
                print(f"    Mid-round checkpoint: {len(all_directions)+len(new_directions)} dirs")

        n_new = len(new_directions)
        print(f"\n  Round {rnd}: found {n_new} directions")

        if n_new == 0:
            print("  No new directions found, stopping")
            round_logs.append(round_info)
            break

        # Phase 2: Refine each new direction
        print(f"\n  --- Local optimization (drift cap={args.max_drift}°) ---")
        refined_directions = []
        for i, d_start in enumerate(new_directions):
            refined, cd, drift = refine_in_cone(
                model, ctx, act, ref_L, a_unit, a_norm,
                d_start, args.max_drift, angle_rad,
                args.inject_layer, args.measure_layer, device, d_model,
                n_steps=args.refine_steps, lr=args.refine_lr,
                n_restarts=args.refine_restarts)
            refined_directions.append(refined)
            print(f"    Dir {i+1}: cd {round_info['pre_refine'][i]['cos_dist']:.4f}"
                  f" → {cd:.4f}, drift={drift:.1f}°")
            round_info['post_refine'].append({
                'cd': cd, 'drift': drift,
                'direction': refined.cpu(),
            })

        all_directions.extend(refined_directions)
        round_logs.append(round_info)

        # Checkpoint after each round
        ckpt = {
            'round': rnd,
            'directions': [d.cpu() for d in all_directions],
            'round_logs': round_logs,
            'best_reward_ever': best_reward_ever,
            'args': vars(args),
        }
        ckpt_path = os.path.join(args.results_dir, f'checkpoint_round{rnd}.pkl')
        with open(ckpt_path, 'wb') as f:
            pickle.dump(ckpt, f)
        print(f"  Checkpointed: {ckpt_path}")

    n_rounds = rnd
    n_total = len(all_directions)
    n_new = n_total - n_seed
    print(f"\n{'='*60}")
    print(f"DONE: {n_new} new directions across {n_rounds} rounds"
          f"{f' (+{n_seed} seeds)' if n_seed else ''}")
    print(f"{'='*60}")

    # Pairwise cosines
    if n_total > 0:
        D = torch.stack(all_directions)
        G = (D @ D.T).abs()
        G.fill_diagonal_(0)
        print(f"Max |cos_sim|: {G.max().item():.4f}")
        print(f"Mean |cos_sim|: {G.sum().item() / max(n_total * (n_total - 1), 1):.6f}")
    else:
        G = torch.zeros(0, 0)

    # Probe comparison
    if probe_dir is not None and n_total > 0:
        print(f"\nProbe cosine similarities:")
        for i, d in enumerate(all_directions):
            cos_probe = (d @ probe_dir.to(device)).item()
            print(f"  Dir {i+1}: cos(probe)={cos_probe:.4f}")

    # === Sweep ===
    if n_total == 0:
        print("No directions found, skipping sweep")
        return

    angles_deg = np.linspace(0, args.max_sweep_angle, args.n_sweep_angles + 1)[1:]

    print("\nSweeping MELBO directions...")
    melbo_cos = np.zeros((n_total, len(angles_deg)))
    for i, d in enumerate(all_directions):
        melbo_cos[i] = sweep_dir(model, ctx, act, ref_L, a_unit, a_norm, d,
                                 args.inject_layer, args.measure_layer,
                                 device, angles_deg)

    if probe_dir is not None:
        print("Sweeping probe direction...")
        probe_cos = sweep_dir(model, ctx, act, ref_L, a_unit, a_norm,
                              probe_dir, args.inject_layer, args.measure_layer,
                              device, angles_deg)
    else:
        probe_cos = None

    print("Sweeping random directions...")
    rand_dirs = F.normalize(torch.randn(args.n_rand, d_model), dim=-1)
    rand_cos = np.zeros((args.n_rand, len(angles_deg)))
    for k in range(args.n_rand):
        rand_cos[k] = sweep_dir(model, ctx, act, ref_L, a_unit, a_norm,
                                rand_dirs[k], args.inject_layer,
                                args.measure_layer, device, angles_deg)

    # Build round_boundaries for plotting
    round_boundaries = []
    cumulative = 0
    for rl in round_logs:
        n_in_round = len(rl['post_refine'])
        cumulative += n_in_round
        round_boundaries.append(cumulative)

    # Save
    results = {
        'directions': [d.cpu() for d in all_directions],
        'round_logs': round_logs,
        'round_boundaries': round_boundaries,
        'angles_deg': angles_deg,
        'melbo_cos': melbo_cos,
        'probe_cos': probe_cos,
        'rand_cos': rand_cos,
        'gram_matrix': G.cpu().numpy(),
        'args': vars(args),
    }
    save_path = os.path.join(args.results_dir, 'results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    # Plot
    n_rounds_with_dirs = len([rl for rl in round_logs if len(rl['post_refine']) > 0])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: sweep curves
    ax = axes[0]
    rand_med = np.median(rand_cos, axis=0)
    rand_std = np.std(rand_cos, axis=0)
    ax.fill_between(angles_deg, rand_med - rand_std, rand_med + rand_std,
                    color='C0', alpha=0.15)
    ax.plot(angles_deg, rand_med, color='C0', lw=2, label='Random')

    if probe_cos is not None:
        ax.plot(angles_deg, probe_cos, color='C3', lw=2, label='Probe')

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, max(n_rounds_with_dirs, 1)))
    dir_idx = 0
    color_idx = 0
    for rl in round_logs:
        n_in_round = len(rl['post_refine'])
        if n_in_round == 0:
            continue
        for j in range(n_in_round):
            ax.plot(angles_deg, melbo_cos[dir_idx], color=colors[color_idx],
                    lw=1.5, alpha=0.6,
                    label=f'Round {rl["round"]}' if j == 0 else None)
            dir_idx += 1
        color_idx += 1

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('Cosine distance')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'OthelloGPT L{args.inject_layer}→L{args.measure_layer}, '
                 f'opt@{args.opt_angle}°')

    # Right: Gram matrix
    ax = axes[1]
    im = ax.imshow(G.cpu().numpy(), cmap='RdBu_r', vmin=0, vmax=0.3)
    plt.colorbar(im, ax=ax)
    for b in round_boundaries[:-1]:
        ax.axhline(b - 0.5, color='white', lw=1)
        ax.axvline(b - 0.5, color='white', lw=1)
    ax.set_title(f'|cos_sim| ({n_total} dirs)')

    plt.suptitle(f'Iterative capped MELBO: {n_total} dirs in {n_rounds} rounds, '
                 f'drift={args.max_drift}°, frac={args.reward_fraction}',
                 fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'iterative_melbo.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {plot_path}")
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    main()
