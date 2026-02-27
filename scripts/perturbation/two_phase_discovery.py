#!/usr/bin/env python3
"""Two-phase feature discovery: QR-constrained discovery + unconstrained refinement.

Phase 1 (Discovery): k directions optimized jointly with QR re-orthogonalization
  after each gradient step. Finds k distinct neighborhoods in feature space.

Phase 2 (Refinement): each direction refined independently (no orthogonality),
  letting it snap to the nearest true feature. Then deduplicated.

Usage:
    python scripts/perturbation/two_phase_discovery.py \
        --model chess_16L --layer 2 --measure_layer 14 \
        --k 128 --alpha 8.0 --phase1_steps 500 --phase2_steps 200
"""
import argparse, os, sys, time, pickle
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (load_model, forward_from_layer_to_layer,
                       _get_blocks, _run_block, _is_rope_model,
                       _get_position_embeddings, _is_cat_pos_emb_model,
                       _get_cat_pos_emb)
from src.data import generate_chess_games


def forward_checkpointed(model, ctx, points, start_layer, end_layer):
    """Forward with gradient checkpointing, from start_layer to end_layer (inclusive)."""
    context_batch = ctx.expand(points.shape[0], -1, -1)
    hidden = torch.cat([context_batch, points], dim=1)

    blocks = _get_blocks(model)
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, hidden)
    elif _is_cat_pos_emb_model(model):
        pos_emb = _get_cat_pos_emb(model, hidden.shape[1])

    for block in blocks[start_layer + 1:end_layer + 1]:
        hidden = torch.utils.checkpoint.checkpoint(
            _run_block, block, hidden, model, pos_emb, use_reentrant=False)

    return hidden[:, -1, :]


def compute_melbo(model, ctx, act, ref_out, directions, angle_deg, layer,
                  measure_layer, sub_batch, cap=None):
    """Compute MELBO objective (cosine distance) for a batch of directions.

    Uses spherical perturbation: p = R * (cos(θ) * a_hat + sin(θ) * d_orth).

    Args:
        directions: (k, d) Parameter tensor
        cap: if not None, clamp cosine distance to this value
    Returns:
        total_loss: scalar (negative mean cosine distance, for minimization)
        cos_dists: (k,) detached tensor of per-direction cosine distances
    """
    model_dtype = next(model.parameters()).dtype
    k = directions.shape[0]
    R = act.float().norm()
    a_unit = act.float() / R
    theta = np.deg2rad(angle_deg)

    n_sub = (k + sub_batch - 1) // sub_batch
    cos_dists_parts = []

    for sb_start in range(0, k, sub_batch):
        sb_end = min(sb_start + sub_batch, k)
        sb_dirs = F.normalize(directions[sb_start:sb_end], dim=1)

        # Orthogonalize w.r.t. activation direction
        projs = (sb_dirs @ a_unit).unsqueeze(1)
        d_orth = sb_dirs - projs * a_unit.unsqueeze(0)
        d_orth = F.normalize(d_orth, dim=1)

        # Spherical perturbation
        perturbed = R * (np.cos(theta) * a_unit.unsqueeze(0)
                         + np.sin(theta) * d_orth)
        perturbed = perturbed.to(model_dtype)

        # Forward to measure layer
        p_out = forward_checkpointed(
            model, ctx, perturbed.unsqueeze(1), layer, measure_layer)

        # Cosine distance
        ref_exp = ref_out.expand(p_out.shape[0], -1)
        cd = 1.0 - F.cosine_similarity(ref_exp, p_out.float(), dim=-1)

        if cap is not None:
            cd_loss = torch.clamp(cd, max=cap)
        else:
            cd_loss = cd

        sub_loss = -cd_loss.mean() / n_sub
        sub_loss.backward()
        cos_dists_parts.append(cd.detach())

    cos_dists = torch.cat(cos_dists_parts)
    return cos_dists


def phase1_discovery(model, ctx, act, ref_out, k, angle_deg, n_steps, lr,
                     sub_batch, layer, measure_layer, device, d_model, cap=None):
    """Phase 1: joint optimization with QR re-orthogonalization.

    Finds k distinct neighborhoods in feature space by keeping directions
    strictly orthogonal throughout optimization.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 1: Discovery — k={k}, angle={angle_deg}°, "
          f"{n_steps} steps, cap={cap}")
    print(f"{'='*60}")

    # Initialize k random orthogonal directions via QR
    raw = torch.randn(d_model, k, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(raw, mode='reduced')  # (d_model, k)
    directions = torch.nn.Parameter(Q.T.clone())  # (k, d_model)

    optimizer = torch.optim.Adam([directions], lr=lr)
    history = []
    t0 = time.time()

    for step in range(n_steps):
        optimizer.zero_grad()
        cos_dists = compute_melbo(
            model, ctx, act, ref_out, directions, angle_deg,
            layer, measure_layer, sub_batch, cap=cap)

        optimizer.step()

        # QR re-orthogonalization
        with torch.no_grad():
            Q, _ = torch.linalg.qr(directions.data.T, mode='reduced')  # (d, k)
            directions.data = Q.T  # (k, d)

        # Logging
        if step % 25 == 0 or step == n_steps - 1:
            with torch.no_grad():
                G = directions.data @ directions.data.T
                off_diag = G.abs()
                off_diag.fill_diagonal_(0)
                max_cos = off_diag.max().item()

                n_above_01 = (cos_dists > 0.1).sum().item()
                n_above_001 = (cos_dists > 0.01).sum().item()

            elapsed = time.time() - t0
            print(f"  step {step:4d} ({elapsed:.0f}s): "
                  f"mean_cd={cos_dists.mean().item():.4f}, "
                  f"max|cos|={max_cos:.6f}, "
                  f"n>0.1={n_above_01}/{k}, n>0.01={n_above_001}/{k}")

            history.append({
                'step': step, 'phase': 1,
                'mean_cd': cos_dists.mean().item(),
                'max_cos': max_cos,
                'n_above_01': n_above_01,
                'n_above_001': n_above_001,
                'elapsed': elapsed,
            })

    final_dirs = directions.data.detach().clone()
    final_scores = cos_dists.detach().clone()
    print(f"  Phase 1 done: {(final_scores > 0.1).sum()}/{k} with cd>0.1, "
          f"{(final_scores > 0.01).sum()}/{k} with cd>0.01")

    return final_dirs, final_scores, history


def phase2_refinement(model, ctx, act, ref_out, directions, scores,
                      angle_deg, n_steps, lr, layer, measure_layer,
                      device, sub_batch, max_drift_deg=30.0):
    """Phase 2: independent refinement with drift cap.

    Each direction is optimized individually to snap to the nearest true feature.
    After each step, if the direction has drifted more than max_drift_deg from
    its phase-1 starting point, it is projected back to the drift boundary.

    The drift cap prevents directions from jumping to different features.
    Expected worst-case distortion from QR on the i-th of k directions in R^n
    is arccos(sqrt(1 - (i-1)/n)); for i=128, n=512 this is ~30°.
    """
    k, d_model = directions.shape
    cos_drift_min = np.cos(np.deg2rad(max_drift_deg))

    print(f"\n{'='*60}")
    print(f"PHASE 2: Refinement — {k} directions, {n_steps} steps each, "
          f"drift cap={max_drift_deg}°")
    print(f"{'='*60}")

    p1_dirs = F.normalize(directions.clone(), dim=1)  # phase 1 anchors
    refined_dirs = directions.clone()
    refined_scores = scores.clone()
    n_capped = 0
    history = []
    t0 = time.time()

    for i in range(k):
        d_i = torch.nn.Parameter(refined_dirs[i:i+1].clone())  # (1, d_model)
        anchor = p1_dirs[i]  # unit vector
        opt_i = torch.optim.Adam([d_i], lr=lr)

        for step in range(n_steps):
            opt_i.zero_grad()
            cd = compute_melbo(
                model, ctx, act, ref_out, d_i, angle_deg,
                layer, measure_layer, sub_batch=1, cap=None)
            opt_i.step()
            with torch.no_grad():
                d_i.data = F.normalize(d_i.data, dim=1)
                # Enforce drift cap: if cos_sim with anchor < threshold,
                # project back toward anchor
                cos_sim = (d_i.data[0] @ anchor).item()
                if cos_sim < cos_drift_min:
                    # Move back: blend toward anchor until exactly at boundary
                    # d_new = alpha * anchor + beta * d_orth, where d_orth is
                    # the component of d orthogonal to anchor
                    d_orth = d_i.data[0] - (d_i.data[0] @ anchor) * anchor
                    d_orth = F.normalize(d_orth, dim=0)
                    sin_max = np.sin(np.deg2rad(max_drift_deg))
                    d_i.data[0] = cos_drift_min * anchor + sin_max * d_orth
                    d_i.data = F.normalize(d_i.data, dim=1)
                    n_capped += 1

        refined_dirs[i] = d_i.data[0].detach()
        refined_scores[i] = cd[0].detach()

        # Compute final drift from phase 1
        drift_cos = (refined_dirs[i] @ p1_dirs[i]).item()
        drift_deg = np.rad2deg(np.arccos(np.clip(drift_cos, -1, 1)))

        if (i + 1) % 10 == 0 or i == k - 1:
            elapsed = time.time() - t0
            print(f"  {i+1}/{k} refined ({elapsed:.0f}s): "
                  f"cd={refined_scores[i].item():.4f}, "
                  f"drift={drift_deg:.1f}°")

            history.append({
                'step': i, 'phase': 2,
                'cd': refined_scores[i].item(),
                'drift_deg': drift_deg,
                'elapsed': elapsed,
            })

    # Summary of drifts
    all_drifts = []
    for i in range(k):
        dc = (refined_dirs[i] @ p1_dirs[i]).item()
        all_drifts.append(np.rad2deg(np.arccos(np.clip(dc, -1, 1))))
    all_drifts = np.array(all_drifts)
    print(f"  Phase 2 done: {(refined_scores > 0.1).sum()}/{k} with cd>0.1")
    print(f"  Drift stats: mean={all_drifts.mean():.1f}°, "
          f"median={np.median(all_drifts):.1f}°, "
          f"max={all_drifts.max():.1f}°, "
          f"n_capped={n_capped} projections")

    return refined_dirs, refined_scores, history


def deduplicate(directions, scores, tau):
    """Remove near-duplicate directions, keeping the higher-scoring one.

    For each pair with |cos_sim| > tau, discard the one with lower score.
    """
    k = directions.shape[0]
    G = (directions @ directions.T).abs()
    G.fill_diagonal_(0)

    keep = torch.ones(k, dtype=torch.bool)

    # Greedily remove: sort by score descending, mark duplicates of higher-scored dirs
    order = scores.argsort(descending=True)
    for idx in range(k):
        i = order[idx].item()
        if not keep[i]:
            continue
        for jdx in range(idx + 1, k):
            j = order[jdx].item()
            if keep[j] and G[i, j] > tau:
                keep[j] = False

    kept_dirs = directions[keep]
    kept_scores = scores[keep]
    kept_indices = torch.where(keep)[0]

    n_removed = k - keep.sum().item()
    print(f"  Deduplication (tau={tau}): {k} → {keep.sum().item()} "
          f"({n_removed} removed)")

    return kept_dirs, kept_scores, kept_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--measure_layer', type=int, default=14)
    parser.add_argument('--act_idx', type=int, default=137)
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--k', type=int, default=128,
                        help='Number of directions to discover')
    parser.add_argument('--alpha', type=float, default=8.0,
                        help='Perturbation angle in degrees')
    parser.add_argument('--cap', type=float, default=None,
                        help='Cap cosine distance in phase 1 (None = no cap)')
    parser.add_argument('--phase1_steps', type=int, default=500)
    parser.add_argument('--phase2_steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--sub_batch', type=int, default=64)
    parser.add_argument('--max_drift', type=float, default=30.0,
                        help='Max drift in degrees from phase 1 direction during refinement')
    parser.add_argument('--dedup_tau', type=float, default=0.1,
                        help='Deduplication threshold on |cos_sim|')
    parser.add_argument('--resume_p1', type=str, default=None,
                        help='Path to phase 1 checkpoint to skip phase 1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_dir = (f"results/two_phase/{args.model}_L{args.layer}"
                   f"_meas{args.measure_layer}_k{args.k}")
    os.makedirs(results_dir, exist_ok=True)

    # Load model
    model, tokenizer, device = load_model(args.model)
    model_dtype = next(model.parameters()).dtype
    d_model = model._plateau_info['d_model']
    print(f"Model: {args.model}, d_model={d_model}, device={device}")
    print(f"k={args.k}, alpha={args.alpha}°, "
          f"phase1={args.phase1_steps} steps, phase2={args.phase2_steps} steps")

    # Extract activation
    n_games = max(args.act_idx + 1, 50)
    token_ids_list = generate_chess_games(n_games, args.seq_len, seed=args.seed)

    ctx = act = None
    batch_size = 32
    count = 0
    for i in range(0, n_games, batch_size):
        bt = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            out = model(bt, output_hidden_states=True)
            h = out.hidden_states[args.layer + 1]
        local_idx = args.act_idx - count
        if 0 <= local_idx < bt.shape[0]:
            ctx = h[local_idx:local_idx+1, :-1, :].to(device, dtype=model_dtype)
            act = h[local_idx, -1, :].to(device, dtype=model_dtype)
        count += bt.shape[0]
        del out, h, bt

    print(f"Activation {args.act_idx}, norm={act.float().norm():.2f}")

    # Reference output
    with torch.no_grad():
        ref_out = forward_from_layer_to_layer(
            model, ctx, act.unsqueeze(0).unsqueeze(0),
            args.layer, args.measure_layer).float()

    # Phase 1: Discovery (or resume from checkpoint)
    if args.resume_p1:
        print(f"Resuming from phase 1 checkpoint: {args.resume_p1}")
        ckpt = pickle.load(open(args.resume_p1, 'rb'))
        dirs_p1 = ckpt['phase1_dirs'].to(device)
        scores_p1 = ckpt['phase1_scores'].to(device)
        hist_p1 = ckpt['phase1_history']
        print(f"  Loaded {dirs_p1.shape[0]} directions, "
              f"{(scores_p1 > 0.1).sum()}/{dirs_p1.shape[0]} with cd>0.1")
    else:
        dirs_p1, scores_p1, hist_p1 = phase1_discovery(
            model, ctx, act, ref_out, args.k, args.alpha, args.phase1_steps,
            args.lr, args.sub_batch, args.layer, args.measure_layer,
            device, d_model, cap=args.cap)

        # Checkpoint after phase 1
        ckpt = {
            'phase1_dirs': dirs_p1.cpu(), 'phase1_scores': scores_p1.cpu(),
            'phase1_history': hist_p1, 'args': vars(args),
        }
        pickle.dump(ckpt, open(os.path.join(results_dir, 'checkpoint_p1.pkl'), 'wb'))
        print("Checkpointed phase 1")

    # Phase 2: Refinement
    dirs_p2, scores_p2, hist_p2 = phase2_refinement(
        model, ctx, act, ref_out, dirs_p1, scores_p1,
        args.alpha, args.phase2_steps, args.lr,
        args.layer, args.measure_layer, device, args.sub_batch,
        max_drift_deg=args.max_drift)

    # Deduplication
    dirs_final, scores_final, kept_idx = deduplicate(
        dirs_p2, scores_p2, args.dedup_tau)

    # Final stats
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    n_final = dirs_final.shape[0]
    G = (dirs_final @ dirs_final.T).abs()
    G.fill_diagonal_(0)
    print(f"Directions: {args.k} → {n_final} after dedup (tau={args.dedup_tau})")
    print(f"cd > 0.01: {(scores_final > 0.01).sum()}/{n_final}")
    print(f"cd > 0.1: {(scores_final > 0.1).sum()}/{n_final}")
    print(f"Max |cos_sim|: {G.max().item():.4f}")
    print(f"Mean |cos_sim|: {G.sum().item() / max(n_final * (n_final - 1), 1):.4f}")

    # Save
    save_data = {
        'directions': dirs_final.cpu(),
        'scores': scores_final.cpu(),
        'kept_indices': kept_idx.cpu(),
        'phase1_dirs': dirs_p1.cpu(),
        'phase1_scores': scores_p1.cpu(),
        'phase2_dirs': dirs_p2.cpu(),
        'phase2_scores': scores_p2.cpu(),
        'gram_matrix': G.cpu().numpy(),
        'phase1_history': hist_p1,
        'phase2_history': hist_p2,
        'args': vars(args),
    }
    pickle.dump(save_data, open(os.path.join(results_dir, 'features.pkl'), 'wb'))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Phase 1 training curve
    steps = [h['step'] for h in hist_p1]
    axes[0].plot(steps, [h['mean_cd'] for h in hist_p1], 'b-')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Mean cosine distance')
    axes[0].set_title('Phase 1: Discovery')
    axes[0].grid(True, alpha=0.3)

    # Score distributions: phase 1 vs phase 2 vs final
    axes[1].hist(scores_p1.cpu().numpy(), bins=30, alpha=0.5, label='Phase 1', color='blue')
    axes[1].hist(scores_p2.cpu().numpy(), bins=30, alpha=0.5, label='Phase 2', color='orange')
    axes[1].hist(scores_final.cpu().numpy(), bins=30, alpha=0.5, label='Final', color='green')
    axes[1].set_xlabel('Cosine distance')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Score distributions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Gram matrix of final directions
    im = axes[2].imshow(G.cpu().numpy(), cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    axes[2].set_title(f'|cos_sim| (final, n={n_final})')
    plt.colorbar(im, ax=axes[2])

    fig.suptitle(f'Two-phase discovery: {args.model} L{args.layer}→L{args.measure_layer}, '
                 f'k={args.k}, alpha={args.alpha}°', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'summary.png'),
                dpi=150, bbox_inches='tight')

    print(f"\nSaved to {results_dir}/")


if __name__ == '__main__':
    main()
