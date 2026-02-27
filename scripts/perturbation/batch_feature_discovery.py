#!/usr/bin/env python3
"""Batch feature discovery using frame potential regularization.

Simultaneously optimizes k directions (k > d_model to force superposition) to:
1. Maximize downstream response when perturbing along each direction
2. Minimize frame potential (encourage angular spread)

Usage:
    python scripts/perturbation/batch_feature_discovery.py \
        --model chess_16L --layer 2 --k 1024 --n_steps 500
"""
import argparse, os, sys, time, pickle
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_final, _get_blocks, _run_block, _is_rope_model, _get_position_embeddings, _is_cat_pos_emb_model, _get_cat_pos_emb
from src.data import generate_chess_games


def forward_checkpointed(model, ctx, points, start_layer):
    """Like forward_from_layer_to_final but with gradient checkpointing per block."""
    context_batch = ctx.expand(points.shape[0], -1, -1)
    hidden = torch.cat([context_batch, points], dim=1)

    blocks = _get_blocks(model)
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, hidden)
    elif _is_cat_pos_emb_model(model):
        pos_emb = _get_cat_pos_emb(model, hidden.shape[1])

    for block in blocks[start_layer + 1:]:
        hidden = torch.utils.checkpoint.checkpoint(
            _run_block, block, hidden, model, pos_emb, use_reentrant=False)

    return hidden[:, -1, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='chess_16L')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--act_idx', type=int, default=137)
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--k', type=int, default=1024,
                        help='Number of directions to optimize')
    parser.add_argument('--angle', type=float, default=10.0,
                        help='Perturbation angle in degrees')
    parser.add_argument('--fp_lambda', type=float, default=0.1,
                        help='Frame potential weight')
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--sub_batch', type=int, default=64,
                        help='Sub-batch size for forward passes (memory vs speed)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_every', type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_dir = (f"results/batch_features/{args.model}_layer{args.layer}"
                   f"_k{args.k}_angle{args.angle}_fp{args.fp_lambda}")
    os.makedirs(results_dir, exist_ok=True)

    # Load model
    model, tokenizer, device = load_model(args.model)
    model_dtype = next(model.parameters()).dtype
    d_model = model._plateau_info['d_model']
    print(f"Model: {args.model}, d_model={d_model}, device={device}")

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
    a_unit = act.float() / R
    print(f"Activation {args.act_idx}, norm={R:.2f}, d_model={d_model}")

    # Reference output (unperturbed)
    with torch.no_grad():
        ref_out = forward_from_layer_to_final(
            model, ctx, act.unsqueeze(0).unsqueeze(0), args.layer)
        ref_out = ref_out.float()  # (1, d_model)

    # Initialize directions on unit sphere
    directions = torch.randn(args.k, d_model, device=device, dtype=torch.float32)
    directions = F.normalize(directions, dim=1)
    directions = torch.nn.Parameter(directions)

    optimizer = torch.optim.Adam([directions], lr=args.lr)
    theta = np.deg2rad(args.angle)

    history = []
    t0 = time.time()

    for step in range(args.n_steps):
        optimizer.zero_grad()

        # --- Feature effect: accumulate gradients sub-batch by sub-batch ---
        # This avoids storing the full computation graph for all k directions.
        n_sub_batches = (args.k + args.sub_batch - 1) // args.sub_batch
        cos_dists_detached = []
        for sb_start in range(0, args.k, args.sub_batch):
            sb_end = min(sb_start + args.sub_batch, args.k)

            # Re-normalize inside each sub-batch (creates fresh graph)
            sb_dirs = F.normalize(directions[sb_start:sb_end], dim=1)
            sb_size = sb_dirs.shape[0]

            # Orthogonalize w.r.t. activation direction
            projs = (sb_dirs @ a_unit).unsqueeze(1)  # (sb, 1)
            d_orth = sb_dirs - projs * a_unit.unsqueeze(0)  # (sb, d)
            d_orth = F.normalize(d_orth, dim=1)

            # Spherical perturbation
            perturbed = R * (np.cos(theta) * a_unit.unsqueeze(0)
                             + np.sin(theta) * d_orth)  # (sb, d)
            perturbed = perturbed.to(model_dtype)

            # Forward pass (differentiable w.r.t. perturbed)
            p_out = forward_checkpointed(
                model, ctx, perturbed.unsqueeze(1), args.layer)  # (sb, d)

            # Cosine distance
            ref_exp = ref_out.expand(sb_size, -1)
            cd = 1.0 - F.cosine_similarity(ref_exp, p_out.float(), dim=-1)  # (sb,)

            # Backward for this sub-batch (accumulates into directions.grad)
            sub_feat_loss = -cd.mean() / n_sub_batches
            sub_feat_loss.backward()

            cos_dists_detached.append(cd.detach())

        cos_dists = torch.cat(cos_dists_detached)  # (k,) detached
        feature_loss = cos_dists.mean()

        # --- Frame potential (cheap, can do full batch) ---
        d_clean = F.normalize(directions, dim=1)  # (k, d_model)
        G = d_clean @ d_clean.T  # (k, k)
        fp = (G ** 2).sum() - (G.diagonal() ** 2).sum()
        fp_per_pair = fp / (args.k * (args.k - 1))

        fp_loss = args.fp_lambda * fp_per_pair
        fp_loss.backward()  # accumulates onto existing gradients

        optimizer.step()

        # Project back to unit sphere
        with torch.no_grad():
            directions.data = F.normalize(directions.data, dim=1)

        # --- Logging ---
        if step % 25 == 0 or step == args.n_steps - 1:
            with torch.no_grad():
                G_eval = F.normalize(directions, dim=1) @ F.normalize(directions, dim=1).T
                off_diag = G_eval.abs()
                off_diag.fill_diagonal_(0)
                max_cos = off_diag.max().item()
                mean_cos = off_diag.sum().item() / (args.k * (args.k - 1))
                n_good = (cos_dists.detach() > 0.01).sum().item()
                n_great = (cos_dists.detach() > 0.1).sum().item()
                welch = 1.0 / np.sqrt(d_model)

            elapsed = time.time() - t0
            print(f"Step {step:4d} ({elapsed:.0f}s): "
                  f"feat={feature_loss.item():.4f}, fp={fp_per_pair.item():.6f}, "
                  f"max|cos|={max_cos:.3f} (Welch={welch:.3f}), "
                  f"mean|cos|={mean_cos:.3f}, "
                  f"n>0.01={n_good}, n>0.1={n_great}")

            history.append({
                'step': step, 'feature_loss': feature_loss.item(),
                'fp': fp_per_pair.item(),
                'loss': -feature_loss.item() + fp_loss.item(),
                'max_cos': max_cos, 'mean_cos': mean_cos,
                'n_good': n_good, 'n_great': n_great,
                'elapsed': elapsed,
            })

        # --- Checkpointing ---
        if (step + 1) % args.checkpoint_every == 0 or step == args.n_steps - 1:
            with torch.no_grad():
                final_dirs = F.normalize(directions, dim=1)
            ckpt = {
                'directions': final_dirs.cpu(),
                'cos_dists': cos_dists.detach().cpu().numpy(),
                'history': history,
                'args': vars(args),
                'step': step,
            }
            ckpt_path = os.path.join(results_dir, f'checkpoint_step{step}.pkl')
            pickle.dump(ckpt, open(ckpt_path, 'wb'))

    # Final evaluation and save
    with torch.no_grad():
        final_dirs = F.normalize(directions, dim=1)
        all_cos = torch.zeros(args.k, device=device)
        for sb_start in range(0, args.k, args.sub_batch):
            sb_end = min(sb_start + args.sub_batch, args.k)
            sb_dirs = final_dirs[sb_start:sb_end]
            sb_size = sb_dirs.shape[0]
            projs = (sb_dirs @ a_unit).unsqueeze(1)
            d_orth = sb_dirs - projs * a_unit.unsqueeze(0)
            d_orth = F.normalize(d_orth, dim=1)
            perturbed = R * (np.cos(theta) * a_unit.unsqueeze(0)
                             + np.sin(theta) * d_orth)
            perturbed = perturbed.to(model_dtype)
            p_out = forward_from_layer_to_final(
                model, ctx, perturbed.unsqueeze(1), args.layer)
            ref_exp = ref_out.expand(sb_size, -1)
            cd = 1.0 - F.cosine_similarity(ref_exp, p_out.float(), dim=-1)
            all_cos[sb_start:sb_end] = cd

        G_final = final_dirs @ final_dirs.T
        off_diag_final = G_final.abs()
        off_diag_final.fill_diagonal_(0)

    save_data = {
        'directions': final_dirs.cpu(),
        'cos_dists': all_cos.cpu().numpy(),
        'gram_matrix': G_final.cpu().numpy(),
        'history': history,
        'args': vars(args),
    }
    pkl_path = os.path.join(results_dir, 'batch_features.pkl')
    pickle.dump(save_data, open(pkl_path, 'wb'))

    # Summary
    cos_np = all_cos.cpu().numpy()
    print(f"\n=== Results ===")
    print(f"Directions: {args.k} in d={d_model} (ratio={args.k/d_model:.1f}x)")
    print(f"Features with cos_dist > 0.01: {(cos_np > 0.01).sum()}/{args.k}")
    print(f"Features with cos_dist > 0.1: {(cos_np > 0.1).sum()}/{args.k}")
    print(f"Cos_dist: mean={cos_np.mean():.4f}, median={np.median(cos_np):.4f}, "
          f"max={cos_np.max():.4f}")
    print(f"Coherence: max={off_diag_final.max().item():.3f}, "
          f"mean={off_diag_final.sum().item()/(args.k*(args.k-1)):.3f}, "
          f"Welch bound={1/np.sqrt(d_model):.3f}")

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = [h['step'] for h in history]
    axes[0, 0].plot(steps, [h['feature_loss'] for h in history])
    axes[0, 0].set_ylabel('Mean cosine distance')
    axes[0, 0].set_title('Feature effect (higher = better)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, [h['fp'] for h in history])
    axes[0, 1].set_ylabel('Frame potential per pair')
    axes[0, 1].set_title('Frame potential (lower = better)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, [h['max_cos'] for h in history], label='Max |cos|')
    axes[1, 0].plot(steps, [h['mean_cos'] for h in history], label='Mean |cos|')
    axes[1, 0].axhline(1/np.sqrt(d_model), ls='--', color='gray', label='Welch bound')
    axes[1, 0].set_ylabel('Coherence')
    axes[1, 0].set_title('Direction coherence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, [h['n_good'] for h in history], label='>0.01')
    axes[1, 1].plot(steps, [h['n_great'] for h in history], label='>0.1')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_title('Directions above threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f'Batch feature discovery: k={args.k}, layer={args.layer}, '
                 f'angle={args.angle}°, λ_fp={args.fp_lambda}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nSaved to {results_dir}/")


if __name__ == '__main__':
    main()
