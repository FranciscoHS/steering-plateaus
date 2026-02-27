#!/usr/bin/env python3
"""Progressive feature discovery with capped plateau-breaking and hard coherence projection.

Discovers features in phases with progressively relaxed breaking angle:
- Phase 1: strict angle (8°), finds "loud" features
- Phase 2+: relaxed angles, finds quieter features
- All phases use capped cosine distance (no reward beyond cap)
- Hard Gram-Schmidt projection enforces coherence: after each step,
  any pair exceeding the threshold gets corrected by projecting the
  worse-performing direction away from the better one.

Usage:
    python scripts/perturbation/progressive_feature_discovery.py \
        --model chess_16L --layer 2 --measure_layer 14 \
        --phases "8:100:500,12:100:500,16:100:500,20:100:500"
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


def welch_bound(k, d):
    """Welch bound on max coherence for k unit vectors in R^d."""
    if k <= d:
        return 0.0
    return np.sqrt((k - d) / (d * (k - 1)))


def enforce_coherence(directions, cos_dists, frozen_dirs, threshold):
    """Hard Gram-Schmidt projection to enforce max coherence below threshold.

    For each pair of directions (including frozen) exceeding the threshold,
    projects the worse-performing direction (lower cos_dist) away from the
    better one. Frozen directions are never modified.

    Args:
        directions: (new_k, d_model) Parameter tensor — modified in-place
        cos_dists: (new_k,) detached tensor of cosine distances (plateau-breaking scores)
        frozen_dirs: (n_frozen, d_model) tensor or None
        threshold: max allowed |cos sim| between any pair
    Returns:
        n_corrections: number of corrections made
    """
    with torch.no_grad():
        new_k = directions.shape[0]
        d = F.normalize(directions.data, dim=1)
        n_frozen = frozen_dirs.shape[0] if frozen_dirs is not None else 0

        n_corrections = 0

        # Check new-vs-frozen: always correct the new direction
        if frozen_dirs is not None and n_frozen > 0:
            G_cross = d @ frozen_dirs.T  # (new_k, n_frozen)
            violations = G_cross.abs() > threshold
            for i in range(new_k):
                for j in range(n_frozen):
                    if violations[i, j]:
                        # Project new direction i away from frozen direction j
                        d[i] = d[i] - (d[i] @ frozen_dirs[j]) * frozen_dirs[j]
                        d[i] = d[i] / d[i].norm()
                        n_corrections += 1

        # Check new-vs-new: correct the worse performer
        G_new = d @ d.T  # (new_k, new_k)
        for i in range(new_k):
            for j in range(i + 1, new_k):
                if G_new[i, j].abs() > threshold:
                    # Project worse performer away from better
                    if cos_dists[i] < cos_dists[j]:
                        d[i] = d[i] - (d[i] @ d[j]) * d[j]
                        d[i] = d[i] / d[i].norm()
                    else:
                        d[j] = d[j] - (d[j] @ d[i]) * d[i]
                        d[j] = d[j] / d[j].norm()
                    n_corrections += 1

        directions.data = d
        return n_corrections


def run_phase(model, ctx, act, ref_out, frozen_dirs, new_k, angle_deg, cap,
              threshold, n_steps, lr, sub_batch, layer, measure_layer,
              device, d_model, phase_idx):
    """Run one phase of progressive feature discovery.

    Args:
        frozen_dirs: (n_frozen, d_model) tensor of already-found directions, or None
        new_k: number of new directions to optimize
        angle_deg: perturbation angle for this phase
        cap: cosine distance cap (no reward beyond this)
        threshold: max coherence threshold for hard projection
    Returns:
        new_dirs: (new_k, d_model) tensor of discovered directions
        cos_dists: (new_k,) tensor of final cosine distances
        history: list of dicts with training metrics
    """
    model_dtype = next(model.parameters()).dtype
    R = act.float().norm()
    a_unit = act.float() / R
    theta = np.deg2rad(angle_deg)
    n_frozen = frozen_dirs.shape[0] if frozen_dirs is not None else 0
    total = n_frozen + new_k

    # Initialize new directions
    directions = torch.randn(new_k, d_model, device=device, dtype=torch.float32)
    directions = F.normalize(directions, dim=1)
    directions = torch.nn.Parameter(directions)

    optimizer = torch.optim.Adam([directions], lr=lr)
    history = []
    t0 = time.time()

    for step in range(n_steps):
        optimizer.zero_grad()

        # --- Feature effect: sub-batched with capped cosine distance ---
        n_sub_batches = (new_k + sub_batch - 1) // sub_batch
        cos_dists_detached = []

        for sb_start in range(0, new_k, sub_batch):
            sb_end = min(sb_start + sub_batch, new_k)
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

            # Cosine distance, CAPPED
            ref_exp = ref_out.expand(p_out.shape[0], -1)
            cd = 1.0 - F.cosine_similarity(ref_exp, p_out.float(), dim=-1)
            cd_capped = torch.clamp(cd, max=cap)

            sub_feat_loss = -cd_capped.mean() / n_sub_batches
            sub_feat_loss.backward()
            cos_dists_detached.append(cd.detach())

        cos_dists = torch.cat(cos_dists_detached)
        feature_loss = cos_dists.mean()

        optimizer.step()

        # Normalize to unit sphere
        with torch.no_grad():
            directions.data = F.normalize(directions.data, dim=1)

        # Hard coherence projection
        n_corrections = enforce_coherence(
            directions, cos_dists, frozen_dirs, threshold)

        # --- Logging ---
        if step % 25 == 0 or step == n_steps - 1:
            with torch.no_grad():
                d_eval = F.normalize(directions, dim=1)
                if frozen_dirs is not None and n_frozen > 0:
                    all_eval = torch.cat([frozen_dirs, d_eval], dim=0)
                else:
                    all_eval = d_eval
                G_eval = all_eval @ all_eval.T
                off_diag = G_eval.abs()
                off_diag.fill_diagonal_(0)
                max_cos = off_diag.max().item()
                mean_cos = off_diag.sum().item() / (total * (total - 1))

                n_above_cap = (cos_dists > cap).sum().item()
                n_above_01 = (cos_dists > 0.1).sum().item()

            elapsed = time.time() - t0
            wb = welch_bound(total, d_model)
            print(f"  step {step:4d} ({elapsed:.0f}s): "
                  f"feat={feature_loss.item():.4f}, "
                  f"max|cos|={max_cos:.4f} (thresh={threshold:.4f}, Welch={wb:.4f}), "
                  f"mean|cos|={mean_cos:.4f}, "
                  f"n>cap={n_above_cap}/{new_k}, n>0.1={n_above_01}/{new_k}, "
                  f"corrections={n_corrections}")

            history.append({
                'step': step, 'feature_loss': feature_loss.item(),
                'max_cos': max_cos, 'mean_cos': mean_cos,
                'n_above_cap': n_above_cap, 'n_above_01': n_above_01,
                'n_corrections': n_corrections,
                'elapsed': elapsed, 'phase': phase_idx, 'angle': angle_deg,
            })

    final_dirs = F.normalize(directions, dim=1).detach()
    final_cos = cos_dists.detach()

    n_good = (final_cos > 0.01).sum().item()
    n_great = (final_cos > cap).sum().item()
    print(f"  Phase {phase_idx} done: {n_great}/{new_k} above cap={cap}, "
          f"{n_good}/{new_k} above 0.01, max|cos|={max_cos:.4f}")

    return final_dirs, final_cos, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--measure_layer', type=int, default=14)
    parser.add_argument('--act_idx', type=int, default=137)
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--phases', type=str, default='8:100:500,12:100:500,16:100:500,20:100:500',
                        help='Comma-separated phases as angle:n_dirs:n_steps')
    parser.add_argument('--cap', type=float, default=0.1,
                        help='Cosine distance cap (no reward beyond this)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Max coherence threshold. Default: Welch bound + 0.01 margin')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--sub_batch', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Parse phases
    phases = []
    for p in args.phases.split(','):
        parts = p.strip().split(':')
        phases.append({
            'angle': float(parts[0]),
            'n_dirs': int(parts[1]),
            'n_steps': int(parts[2]),
        })
    total_dirs = sum(p['n_dirs'] for p in phases)
    print(f"Schedule: {len(phases)} phases, {total_dirs} total directions")
    for i, p in enumerate(phases):
        print(f"  Phase {i}: {p['angle']}°, {p['n_dirs']} dirs, {p['n_steps']} steps")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Compute threshold from Welch bound if not specified
    # We use the final total number of directions for the bound
    if args.threshold is None:
        d_model_est = 512  # will be updated after model load
    else:
        threshold = args.threshold

    results_dir = (f"results/progressive_features/{args.model}_L{args.layer}"
                   f"_meas{args.measure_layer}_cap{args.cap}")
    os.makedirs(results_dir, exist_ok=True)

    # Load model
    model, tokenizer, device = load_model(args.model)
    model_dtype = next(model.parameters()).dtype
    d_model = model._plateau_info['d_model']
    print(f"Model: {args.model}, d_model={d_model}, device={device}")

    if args.threshold is None:
        wb = welch_bound(total_dirs, d_model)
        threshold = wb + 0.01
        print(f"Threshold: Welch({total_dirs}, {d_model}) + 0.01 = {wb:.4f} + 0.01 = {threshold:.4f}")
    else:
        threshold = args.threshold
        print(f"Threshold: {threshold:.4f} (user-specified)")

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

    R = act.float().norm()
    print(f"Activation {args.act_idx}, norm={R:.2f}")

    # Reference output at measure layer (unperturbed)
    with torch.no_grad():
        ref_out = forward_from_layer_to_layer(
            model, ctx, act.unsqueeze(0).unsqueeze(0),
            args.layer, args.measure_layer).float()

    # Run phases
    all_dirs = None  # frozen directions accumulate here
    all_cos_dists = []
    all_history = []
    phase_boundaries = [0]

    for phase_idx, phase in enumerate(phases):
        print(f"\n{'='*60}")
        print(f"PHASE {phase_idx}: angle={phase['angle']}°, "
              f"n_dirs={phase['n_dirs']}, n_steps={phase['n_steps']}, "
              f"threshold={threshold:.4f}")
        if all_dirs is not None:
            print(f"  Frozen directions: {all_dirs.shape[0]}")
        print(f"{'='*60}")

        new_dirs, new_cos, history = run_phase(
            model, ctx, act, ref_out,
            frozen_dirs=all_dirs,
            new_k=phase['n_dirs'],
            angle_deg=phase['angle'],
            cap=args.cap,
            threshold=threshold,
            n_steps=phase['n_steps'],
            lr=args.lr,
            sub_batch=args.sub_batch,
            layer=args.layer,
            measure_layer=args.measure_layer,
            device=device,
            d_model=d_model,
            phase_idx=phase_idx,
        )

        # Accumulate
        if all_dirs is None:
            all_dirs = new_dirs
        else:
            all_dirs = torch.cat([all_dirs, new_dirs], dim=0)
        all_cos_dists.append(new_cos.cpu().numpy())
        all_history.extend(history)
        phase_boundaries.append(all_dirs.shape[0])

        # Checkpoint after each phase
        ckpt = {
            'directions': all_dirs.cpu(),
            'cos_dists': np.concatenate(all_cos_dists),
            'history': all_history,
            'phase_boundaries': phase_boundaries,
            'phases': phases,
            'threshold': threshold,
            'args': vars(args),
        }
        ckpt_path = os.path.join(results_dir, 'checkpoint.pkl')
        pickle.dump(ckpt, open(ckpt_path, 'wb'))
        print(f"  Checkpointed: {all_dirs.shape[0]} total directions")

    # Final stats
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    total = all_dirs.shape[0]
    G = all_dirs @ all_dirs.T
    off_diag = G.abs()
    off_diag.fill_diagonal_(0)
    all_cos_np = np.concatenate(all_cos_dists)

    wb = welch_bound(total, d_model)
    print(f"Total directions: {total}")
    print(f"Cos dist > 0.01: {(all_cos_np > 0.01).sum()}/{total}")
    print(f"Cos dist > 0.1: {(all_cos_np > 0.1).sum()}/{total}")
    print(f"Max |cos sim|: {off_diag.max().item():.4f} (Welch={wb:.4f}, threshold={threshold:.4f})")
    print(f"Mean |cos sim|: {off_diag.sum().item()/(total*(total-1)):.4f}")

    # Save final
    save_data = {
        'directions': all_dirs.cpu(),
        'cos_dists': all_cos_np,
        'gram_matrix': G.cpu().numpy(),
        'history': all_history,
        'phase_boundaries': phase_boundaries,
        'phases': phases,
        'threshold': threshold,
        'args': vars(args),
    }
    pickle.dump(save_data, open(os.path.join(results_dir, 'features.pkl'), 'wb'))

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    phase_colors = plt.cm.tab10(np.linspace(0, 1, len(phases)))

    for h_idx, h in enumerate(all_history):
        pi = h['phase']
        axes[0, 0].scatter(h_idx, h['feature_loss'], color=phase_colors[pi], s=3)
        axes[0, 1].scatter(h_idx, h['max_cos'], color=phase_colors[pi], s=3)
        axes[1, 0].scatter(h_idx, h['n_corrections'], color=phase_colors[pi], s=3)
        axes[1, 1].scatter(h_idx, h['n_above_01'], color=phase_colors[pi], s=3)

    axes[0, 0].set_ylabel('Mean cosine distance')
    axes[0, 0].set_title('Feature effect (higher = better)')
    axes[0, 1].set_ylabel('Max |cos sim|')
    axes[0, 1].set_title('Coherence')
    axes[0, 1].axhline(threshold, ls='--', color='red', alpha=0.5, label=f'threshold={threshold:.4f}')
    axes[0, 1].axhline(wb, ls='--', color='gray', alpha=0.5, label=f'Welch={wb:.4f}')
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Coherence corrections per step')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Directions with cos_dist > 0.1')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Training step (global)')

    fig.suptitle(f'Progressive feature discovery: {args.model} L{args.layer}→L{args.measure_layer}, '
                 f'cap={args.cap}, threshold={threshold:.4f}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'),
                dpi=150, bbox_inches='tight')

    print(f"\nSaved to {results_dir}/")


if __name__ == '__main__':
    main()
