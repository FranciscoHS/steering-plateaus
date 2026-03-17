#!/usr/bin/env python3
"""
Delta optimization v8: multi-restart Jacobian minimization with cosine distance floor.

Key differences from v6:
- Multiple random restarts (e.g. 10x30 steps instead of 1x300)
- Cached reference Jacobians: pass --ref_checkpoint to skip recomputation
- Keeps the best direction (lowest combined loss) across restarts

Usage:
    python scripts/jacobian/run_delta_optimization_v8.py --angle 10 --cos_target 0.05
    python scripts/jacobian/run_delta_optimization_v8.py --angle 10 --cos_target 0.05 \
        --ref_checkpoint results/delta_opt_v6_10deg_cos05/checkpoint.pkl
    python scripts/jacobian/run_delta_optimization_v8.py --n_restarts 20 --n_steps 30
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F

from src.model import (
    load_model, forward_from_layer_to_final,
    _get_final_norm, _get_lm_head, _get_blocks,
)
from src.data import load_fineweb_fixed_length
from src.analysis import compute_jacobian_norm_vjp


def save_checkpoint(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {path}")


def extract_split_activations(model, tokenizer, token_ids_list, layer, device,
                              n_act, n_ctx, batch_size=64):
    """Extract activations and split into activation set + context set."""
    total = n_act + n_ctx
    all_last_token = []
    all_prefix = []

    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        batch_tokens = torch.stack(token_ids_list[i:batch_end]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        all_last_token.append(hidden[:, -1, :].cpu())
        all_prefix.append(hidden[:, :-1, :].cpu())

    all_last_token = torch.cat(all_last_token, dim=0)
    all_prefix_list = []
    for batch_prefix in all_prefix:
        for j in range(batch_prefix.shape[0]):
            all_prefix_list.append(batch_prefix[j:j+1])

    activations = all_last_token[:n_act]
    contexts = all_prefix_list[n_act:n_act + n_ctx]
    return activations, contexts


def spherical_perturb(a, direction, angle_rad):
    """Spherical perturbation, differentiable w.r.t. direction."""
    R = a.norm()
    a_hat = a / R
    proj = (direction * a_hat).sum()
    v_perp = direction - proj * a_hat
    v_perp = v_perp / (v_perp.norm() + 1e-10)
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    return cos_t * a + sin_t * R * v_perp


def compute_cosine_distance(model, ctx, a_orig, a_pert, layer):
    """Cosine distance between final-layer representations. Differentiable w.r.t. a_pert."""
    d = a_orig.shape[-1]
    with torch.no_grad():
        ref_final = forward_from_layer_to_final(model, ctx, a_orig.reshape(1, 1, d), layer)
    pert_final = forward_from_layer_to_final(model, ctx, a_pert.reshape(1, 1, d), layer)
    cos_sim = F.cosine_similarity(ref_final.float(), pert_final.float(), dim=-1)
    return (1.0 - cos_sim).squeeze()


def compute_kl(model, ctx, a_orig, a_pert, layer):
    """KL(p_orig || p_pert) from logits. For evaluation only."""
    d = a_orig.shape[-1]
    with torch.no_grad():
        ref_final = forward_from_layer_to_final(model, ctx, a_orig.reshape(1, 1, d), layer)
        ref_logits = _get_lm_head(model)(_get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1))
        ref_lp = F.log_softmax(ref_logits.float(), dim=-1)
        ref_p = ref_lp.exp()
        pert_final = forward_from_layer_to_final(model, ctx, a_pert.reshape(1, 1, d), layer)
        pert_logits = _get_lm_head(model)(_get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1))
        pert_lp = F.log_softmax(pert_logits.float(), dim=-1)
        kl = (ref_p * (ref_lp - pert_lp)).sum(dim=-1)
    return kl.squeeze().item()


def optimization_step(model, direction, activations, contexts, layer,
                      batch_indices, n_frob_probes, device, angle_rad,
                      cos_target, cos_lambda):
    """One step: minimize ||J||_F^2 + cosine distance floor penalty."""
    d_model = direction.shape[0]
    B = len(batch_indices)
    total_frob_val = 0.0
    total_cos_loss = 0.0
    total_cos_dist = 0.0

    for i in batch_indices:
        a_i = activations[i].to(device).detach()
        ctx_i = contexts[i].to(device).detach()

        # --- Cosine distance ---
        d_unit_cos = direction / (direction.norm() + 1e-10)
        a_pert_cos = spherical_perturb(a_i, d_unit_cos, angle_rad)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            cos_dist = compute_cosine_distance(model, ctx_i, a_i, a_pert_cos, layer)
        total_cos_dist += cos_dist.item() / B
        cos_violation = torch.clamp(cos_target - cos_dist, min=0.0)
        cos_loss = cos_lambda * cos_violation ** 2 / B
        cos_loss.backward()
        total_cos_loss += cos_loss.item()

        # --- Frobenius norm ---
        d_unit_frob = direction / (direction.norm() + 1e-10)
        a_pert_frob = spherical_perturb(a_i, d_unit_frob, angle_rad)
        point = a_pert_frob.reshape(1, 1, d_model)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            final_act = forward_from_layer_to_final(model, ctx_i, point, layer)
        frob_sq = torch.tensor(0.0, device=device)
        for k in range(n_frob_probes):
            u = torch.randn(1, d_model, device=device)
            s = (final_act * u).sum()
            g = torch.autograd.grad(s, direction, create_graph=True, retain_graph=True)[0]
            frob_sq = frob_sq + (g ** 2).sum()
        frob_sq = frob_sq / n_frob_probes
        R = a_i.norm().item()
        inflation = (R * np.sin(angle_rad)) ** 2
        frob_loss = frob_sq / (inflation * B)
        frob_loss.backward()
        total_frob_val += frob_sq.item() / (inflation * B)

    return total_frob_val, total_cos_loss, total_cos_dist


def evaluate_direction(model, direction, activations, contexts, layer,
                       angle_rad, device, n_eval=None):
    """Evaluate a direction on all activations. Returns mean cos_dist and mean frob."""
    n = n_eval or len(activations)
    cos_dists = []
    d_model = direction.shape[0]
    for i in range(n):
        a_i = activations[i].to(device)
        ctx_i = contexts[i].to(device)
        a_pert = spherical_perturb(a_i, direction, angle_rad).detach()
        with torch.no_grad():
            cos_dist = compute_cosine_distance(model, ctx_i, a_i, a_pert, layer).item()
        cos_dists.append(cos_dist)
    return np.mean(cos_dists), np.array(cos_dists)


def main():
    parser = argparse.ArgumentParser(description='Delta optimization v8 (multi-restart)')
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=100)
    parser.add_argument('--n_ctx', type=int, default=100)
    parser.add_argument('--angle', type=float, default=10)
    parser.add_argument('--cos_target', type=float, default=0.05)
    parser.add_argument('--cos_lambda', type=float, default=2000000)
    parser.add_argument('--n_frob_probes', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--n_restarts', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=30)
    parser.add_argument('--ref_checkpoint', type=str, default=None,
                        help='Path to checkpoint with cached activations/ref Jacobians')
    parser.add_argument('--results_dir', type=str, default='results/delta_opt_v8')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.results_dir, 'checkpoint.pkl')
    angle_rad = np.deg2rad(args.angle)

    d_model = {'gpt2': 768, 'gpt2-medium': 1024, 'gpt2-large': 1280, 'gpt2-xl': 1600}[args.model]

    print(f"Configuration:")
    print(f"  Model: {args.model} (d_model={d_model})")
    print(f"  Layer: {args.layer}")
    print(f"  Angle: {args.angle} deg")
    print(f"  Cosine distance floor: {args.cos_target}, lambda: {args.cos_lambda}")
    print(f"  Frob probes: {args.n_frob_probes}")
    print(f"  Restarts: {args.n_restarts}, steps/restart: {args.n_steps}")
    print(f"  Ref checkpoint: {args.ref_checkpoint or 'none (will compute)'}")
    print("-" * 60)

    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    # =========================================================
    # Stage 0: Load or compute activations + references
    # =========================================================
    if args.ref_checkpoint and os.path.exists(args.ref_checkpoint):
        print(f"\nLoading cached data from {args.ref_checkpoint}")
        with open(args.ref_checkpoint, 'rb') as f:
            ref_cp = pickle.load(f)
        activations = ref_cp['activations']
        contexts = ref_cp['contexts']
        ref_frob_norms = np.array(ref_cp['ref_frob_norms'])
        # Use random baselines if available and angle matches
        ref_args = ref_cp.get('args', {})
        if ref_args.get('angle') == args.angle and 'random_cos_dists' in ref_cp:
            random_frob_norms = np.array(ref_cp['random_frob_norms'])
            random_cos_dists = np.array(ref_cp['random_cos_dists'])
            random_kls = np.array(ref_cp['random_kls'])
            print(f"Reused random baselines from ref checkpoint")
        else:
            random_frob_norms = random_cos_dists = random_kls = None
        print(f"Activations: {activations.shape}, ref ||J||_F: {ref_frob_norms.mean():.1f}")
    else:
        print(f"\n=== Stage 0: Extracting activations ===")
        n_total = args.n_act + args.n_ctx
        token_ids_list = load_fineweb_fixed_length(
            n_total, tokenizer, seq_len=args.seq_len, seed=args.seed
        )
        activations, contexts = extract_split_activations(
            model, tokenizer, token_ids_list, args.layer, device,
            args.n_act, args.n_ctx
        )
        print(f"Activations shape: {activations.shape}, mean norm: {activations.norm(dim=-1).mean():.1f}")

        print(f"\nComputing reference ||J||_F...")
        ref_frob_norms = []
        t0 = time.time()
        for i in range(args.n_act):
            frob_norm, _ = compute_jacobian_norm_vjp(
                model, contexts[i].to(device), activations[i].to(device),
                args.layer, n_probes=50
            )
            ref_frob_norms.append(frob_norm)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{args.n_act}  ({(i+1)/(time.time()-t0):.1f}/s)")
        ref_frob_norms = np.array(ref_frob_norms)
        print(f"Reference ||J||_F: mean={ref_frob_norms.mean():.1f}, std={ref_frob_norms.std():.1f}")
        random_frob_norms = random_cos_dists = random_kls = None

    # Compute random baselines if not cached
    if random_cos_dists is None:
        print(f"\nBaseline: random directions at {args.angle} deg...")
        random_frob_norms = []
        random_cos_dists = []
        random_kls = []
        n_random = 5
        for i in range(min(20, args.n_act)):
            a_i = activations[i].to(device)
            ctx_i = contexts[i].to(device)
            for _ in range(n_random):
                rand_dir = torch.randn(d_model, device=device)
                rand_dir = rand_dir / rand_dir.norm()
                a_hat = a_i / a_i.norm()
                v = rand_dir - (rand_dir @ a_hat) * a_hat
                v = v / v.norm()
                a_pert = np.cos(angle_rad) * a_i + np.sin(angle_rad) * a_i.norm() * v

                frob, _ = compute_jacobian_norm_vjp(model, ctx_i, a_pert.detach(), args.layer, n_probes=50)
                random_frob_norms.append(frob)

                with torch.no_grad():
                    cos_dist = compute_cosine_distance(model, ctx_i, a_i, a_pert.detach(), args.layer).item()
                random_cos_dists.append(cos_dist)

                kl = compute_kl(model, ctx_i, a_i, a_pert.detach(), args.layer)
                random_kls.append(kl)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/20")
        random_frob_norms = np.array(random_frob_norms)
        random_cos_dists = np.array(random_cos_dists)
        random_kls = np.array(random_kls)
        cos_deg = np.degrees(np.arccos(np.clip(1.0 - random_cos_dists, -1, 1)))
        print(f"Random @ {args.angle} deg: cos_dist={random_cos_dists.mean():.4f} ({cos_deg.mean():.1f} deg), KL={random_kls.mean():.3f}")

    # Save base checkpoint with activations + refs
    checkpoint = {
        'args': vars(args),
        'activations': activations,
        'contexts': contexts,
        'ref_frob_norms': ref_frob_norms,
        'random_frob_norms': random_frob_norms,
        'random_cos_dists': random_cos_dists,
        'random_kls': random_kls,
    }

    # =========================================================
    # Stage 1: Multi-restart optimization
    # =========================================================
    print(f"\n=== Stage 1: Multi-restart optimization ({args.n_restarts} x {args.n_steps} steps) ===")

    best_direction = None
    best_cos_dist = -1.0
    best_frob = float('inf')
    best_restart = -1
    all_restart_results = []

    t0_all = time.time()
    for restart in range(args.n_restarts):
        # Fresh random direction
        direction = torch.randn(d_model, device=device)
        direction = (direction / direction.norm()).requires_grad_(True)
        optimizer = torch.optim.Adam([direction], lr=args.lr)

        t0 = time.time()
        last_cos_dist = 0.0
        last_frob = 0.0
        for step in range(args.n_steps):
            batch_indices = np.random.choice(args.n_act, args.batch_size, replace=False)
            optimizer.zero_grad()
            frob_val, cos_loss, cos_dist_mean = optimization_step(
                model, direction, activations, contexts, args.layer,
                batch_indices, args.n_frob_probes, device, angle_rad,
                args.cos_target, args.cos_lambda
            )
            optimizer.step()
            with torch.no_grad():
                direction.data.div_(direction.data.norm())
            last_cos_dist = cos_dist_mean
            last_frob = np.sqrt(frob_val) if frob_val > 0 else 0

        # Evaluate on more samples
        d_eval = direction.detach() / direction.detach().norm()
        mean_cos_dist, _ = evaluate_direction(
            model, d_eval, activations, contexts, args.layer,
            angle_rad, device, n_eval=min(30, args.n_act)
        )

        elapsed = time.time() - t0
        cd_angle = np.degrees(np.arccos(np.clip(1.0 - mean_cos_dist, -1, 1)))
        print(f"  Restart {restart+1}/{args.n_restarts}: cos_dist={mean_cos_dist:.4f} ({cd_angle:.1f}deg)  "
              f"||J||_F~{last_frob:.0f}  ({elapsed:.0f}s)")

        all_restart_results.append({
            'direction': d_eval.cpu(),
            'mean_cos_dist': mean_cos_dist,
            'last_frob': last_frob,
        })

        # Keep best by cos_dist (primary), then frob (secondary)
        if mean_cos_dist > best_cos_dist:
            best_cos_dist = mean_cos_dist
            best_frob = last_frob
            best_direction = d_eval.cpu().clone()
            best_restart = restart

    cd_best = np.degrees(np.arccos(np.clip(1.0 - best_cos_dist, -1, 1)))
    print(f"\nBest: restart {best_restart+1}, cos_dist={best_cos_dist:.4f} ({cd_best:.1f}deg), "
          f"||J||_F~{best_frob:.0f}  (total {time.time()-t0_all:.0f}s)")

    checkpoint['opt_state'] = {
        'direction': best_direction,
        'step': args.n_steps - 1,
        'history': {'frob_loss': [], 'cos_loss': [], 'cos_dist_mean': [],
                    'grad_norm': [], 'cosine_sim': []},
    }
    checkpoint['all_restarts'] = all_restart_results
    checkpoint['stage'] = 'optimization_done'
    save_checkpoint(checkpoint, checkpoint_path)

    # =========================================================
    # Stage 2: Evaluation
    # =========================================================
    print(f"\n=== Stage 2: Evaluation ===")

    direction_opt = best_direction.to(device)
    direction_opt = direction_opt / direction_opt.norm()

    shifted_frob_norms = []
    shifted_cos_dists = []
    shifted_kls = []
    t0 = time.time()
    for i in range(args.n_act):
        a_i = activations[i].to(device)
        ctx_i = contexts[i].to(device)
        a_pert = spherical_perturb(a_i, direction_opt, angle_rad).detach()

        frob, _ = compute_jacobian_norm_vjp(model, ctx_i, a_pert, args.layer, n_probes=50)
        shifted_frob_norms.append(frob)

        with torch.no_grad():
            cos_dist = compute_cosine_distance(model, ctx_i, a_i, a_pert, args.layer).item()
        shifted_cos_dists.append(cos_dist)

        kl = compute_kl(model, ctx_i, a_i, a_pert, args.layer)
        shifted_kls.append(kl)

        if (i + 1) % 20 == 0:
            cd_angle = np.degrees(np.arccos(np.clip(1.0 - cos_dist, -1, 1)))
            print(f"  {i+1}/{args.n_act}  ||J||_F={frob:.1f}  cos_dist={cos_dist:.4f} ({cd_angle:.1f}deg)  "
                  f"KL={kl:.3f}  ({(i+1)/(time.time()-t0):.1f}/s)")

    shifted_frob_norms = np.array(shifted_frob_norms)
    shifted_cos_dists = np.array(shifted_cos_dists)
    shifted_kls = np.array(shifted_kls)

    rand_angles = np.degrees(np.arccos(np.clip(1.0 - random_cos_dists, -1, 1)))
    opt_angles = np.degrees(np.arccos(np.clip(1.0 - shifted_cos_dists, -1, 1)))
    target_angle = np.degrees(np.arccos(np.clip(1.0 - args.cos_target, -1, 1)))

    print(f"\nResults:")
    print(f"  ||J||_F original:       mean={ref_frob_norms.mean():.1f}, std={ref_frob_norms.std():.1f}")
    print(f"  ||J||_F random@{args.angle}deg:   mean={random_frob_norms.mean():.1f}, std={random_frob_norms.std():.1f}")
    print(f"  ||J||_F optimized@{args.angle}deg: mean={shifted_frob_norms.mean():.1f}, std={shifted_frob_norms.std():.1f}")
    print(f"  Downstream cosine angle:")
    print(f"    Random:    mean={rand_angles.mean():.1f} deg, median={np.median(rand_angles):.1f} deg")
    print(f"    Optimized: mean={opt_angles.mean():.1f} deg, median={np.median(opt_angles):.1f} deg")
    print(f"    Target:    {target_angle:.1f} deg (cos_dist={args.cos_target})")
    cos_above = (shifted_cos_dists >= args.cos_target).sum()
    print(f"  cos_dist >= {args.cos_target}: {cos_above}/{len(shifted_cos_dists)}")
    print(f"  KL random:    mean={random_kls.mean():.3f}")
    print(f"  KL optimized: mean={shifted_kls.mean():.3f}")

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ||J||_F histogram
    ax = axes[0]
    all_vals = np.concatenate([ref_frob_norms, random_frob_norms, shifted_frob_norms])
    bins = np.linspace(0, min(all_vals.max() * 1.1, np.percentile(all_vals, 99) * 1.5), 40)
    ax.hist(ref_frob_norms, bins=bins, alpha=0.5, label=f'Original ({ref_frob_norms.mean():.0f})', color='C0')
    ax.hist(shifted_frob_norms, bins=bins, alpha=0.5, label=f'Optimized ({shifted_frob_norms.mean():.0f})', color='C1')
    ax.hist(random_frob_norms, bins=bins, alpha=0.3, label=f'Random ({random_frob_norms.mean():.0f})', color='gray')
    ax.set_xlabel('||J||_F')
    ax.set_title('Jacobian Frobenius norm')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Cosine distance histogram
    ax = axes[1]
    bins = np.linspace(0, max(shifted_cos_dists.max(), random_cos_dists.max()) * 1.1, 30)
    ax.hist(random_cos_dists, bins=bins, alpha=0.5, label=f'Random ({np.median(rand_angles):.1f}deg)', color='gray')
    ax.hist(shifted_cos_dists, bins=bins, alpha=0.5, label=f'Optimized ({np.median(opt_angles):.1f}deg)', color='C1')
    ax.axvline(args.cos_target, color='red', linestyle='--', label=f'Target ({target_angle:.0f}deg)')
    ax.set_xlabel('Cosine distance')
    ax.set_title(f'Downstream cos dist at {args.angle} deg')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Restart comparison
    ax = axes[2]
    restart_cos = [r['mean_cos_dist'] for r in all_restart_results]
    restart_angles = [np.degrees(np.arccos(np.clip(1.0 - c, -1, 1))) for c in restart_cos]
    ax.bar(range(1, len(restart_angles)+1), restart_angles, color='C1', alpha=0.7)
    ax.axhline(target_angle, color='red', linestyle='--', label=f'Target ({target_angle:.0f}deg)')
    rand_mean_angle = rand_angles.mean()
    ax.axhline(rand_mean_angle, color='gray', linestyle='--', label=f'Random ({rand_mean_angle:.1f}deg)')
    ax.set_xlabel('Restart')
    ax.set_ylabel('Mean downstream angle (deg)')
    ax.set_title('Per-restart performance')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'v8: Multi-restart min-J + cos>{args.cos_target} at {args.angle}deg ({args.n_restarts}x{args.n_steps} steps)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, f'eval_{args.angle}deg.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    results = {
        'args': vars(args),
        'angle': args.angle,
        'direction': direction_opt.cpu(),
        'ref_frob_norms': ref_frob_norms,
        'random_frob_norms': random_frob_norms,
        'random_cos_dists': random_cos_dists,
        'random_kls': random_kls,
        'shifted_frob_norms': shifted_frob_norms,
        'shifted_cos_dists': shifted_cos_dists,
        'shifted_kls': shifted_kls,
        'all_restarts': all_restart_results,
    }
    results_path = os.path.join(args.results_dir, f'results_{args.angle}deg.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved: {results_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
