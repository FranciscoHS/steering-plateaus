#!/usr/bin/env python3
"""
Delta optimization v9: minimize Jacobian norm with KL divergence floor.

Key idea: find directions that keep low ||J||_F (stay on the manifold) while
causing meaningful output change (high KL divergence). This avoids the v8
problem where cosine distance constraints just find long-plateau random-ish
directions, since KL directly measures whether the model's output distribution
actually changes.

Loss = ||J||_F^2 + kl_lambda * max(0, kl_target - KL(p_orig || p_pert))^2

Based on v8 (multi-restart), but replaces cosine distance floor with KL floor.

Usage:
    python scripts/delta_optimization/run_delta_optimization_v9.py --angle 10 --kl_target 0.5
    python scripts/delta_optimization/run_delta_optimization_v9.py --angle 10 --kl_target 1.0 \
        --ref_checkpoint results/delta_opt_v8/checkpoint.pkl
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


def extract_activations_with_context(model, token_ids_list, layer, device,
                                     n_act, batch_size=64):
    """Extract last-token activations paired with their own prefix context."""
    all_last_token = []
    all_prefix = []

    for i in range(0, n_act, batch_size):
        batch_end = min(i + batch_size, n_act)
        batch_tokens = torch.stack(token_ids_list[i:batch_end]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        all_last_token.append(hidden[:, -1, :].cpu())
        all_prefix.append(hidden[:, :-1, :].cpu())

    all_last_token = torch.cat(all_last_token, dim=0)
    contexts = []
    for batch_prefix in all_prefix:
        for j in range(batch_prefix.shape[0]):
            contexts.append(batch_prefix[j:j+1])

    return all_last_token, contexts


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
    """Cosine distance between final-layer representations (for evaluation)."""
    d = a_orig.shape[-1]
    with torch.no_grad():
        ref_final = forward_from_layer_to_final(model, ctx, a_orig.reshape(1, 1, d), layer)
        pert_final = forward_from_layer_to_final(model, ctx, a_pert.reshape(1, 1, d), layer)
        cos_sim = F.cosine_similarity(ref_final.float(), pert_final.float(), dim=-1)
    return (1.0 - cos_sim).squeeze().item()


def compute_kl_eval(model, ctx, a_orig, a_pert, layer):
    """KL(p_orig || p_pert) from logits. For evaluation only (no grad)."""
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


def compute_kl_differentiable(model, ctx, a_orig, a_pert, layer):
    """KL(p_orig || p_pert), differentiable w.r.t. a_pert.

    The ref distribution is detached (fixed target). Gradients flow through
    the perturbed path only, which depends on the direction parameter.
    """
    d = a_orig.shape[-1]
    # Reference: no grad needed
    with torch.no_grad():
        ref_final = forward_from_layer_to_final(model, ctx, a_orig.reshape(1, 1, d), layer)
        ref_logits = _get_lm_head(model)(_get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1))
        ref_lp = F.log_softmax(ref_logits.float(), dim=-1)
        ref_p = ref_lp.exp()

    # Perturbed: needs grad for direction optimization
    pert_final = forward_from_layer_to_final(model, ctx, a_pert.reshape(1, 1, d), layer)
    pert_logits = _get_lm_head(model)(_get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1))
    pert_lp = F.log_softmax(pert_logits.float(), dim=-1)

    kl = (ref_p * (ref_lp - pert_lp)).sum(dim=-1)
    return kl.squeeze()


def optimization_step(model, direction, activations, contexts, layer,
                      batch_indices, n_frob_probes, device, angle_rad,
                      kl_target, kl_lambda, frob_weight=1.0):
    """One step: minimize frob_weight * ||J||_F^2 + KL floor penalty."""
    d_model = direction.shape[0]
    B = len(batch_indices)
    total_frob_val = 0.0
    total_kl_loss = 0.0
    total_kl_val = 0.0

    for i in batch_indices:
        a_i = activations[i].to(device).detach()
        ctx_i = contexts[i].to(device).detach()

        # --- KL divergence floor ---
        d_unit_kl = direction / (direction.norm() + 1e-10)
        a_pert_kl = spherical_perturb(a_i, d_unit_kl, angle_rad)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            kl_val = compute_kl_differentiable(model, ctx_i, a_i, a_pert_kl, layer)
        total_kl_val += kl_val.item() / B
        # Penalize when KL is below target
        kl_violation = torch.clamp(kl_target - kl_val, min=0.0)
        kl_loss = kl_lambda * kl_violation ** 2 / B
        kl_loss.backward()
        total_kl_loss += kl_loss.item()

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
        frob_loss = frob_weight * frob_sq / (inflation * B)
        frob_loss.backward()
        total_frob_val += frob_sq.item() / (inflation * B)

    return total_frob_val, total_kl_loss, total_kl_val


def evaluate_direction(model, direction, activations, contexts, layer,
                       angle_rad, device, n_eval=None):
    """Evaluate a direction: returns mean KL and per-sample KLs."""
    n = n_eval or len(activations)
    kls = []
    for i in range(n):
        a_i = activations[i].to(device)
        ctx_i = contexts[i].to(device)
        a_pert = spherical_perturb(a_i, direction, angle_rad).detach()
        kl = compute_kl_eval(model, ctx_i, a_i, a_pert, layer)
        kls.append(kl)
    return np.mean(kls), np.array(kls)


def main():
    parser = argparse.ArgumentParser(description='Delta optimization v9 (min J + KL floor)')
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=100)
    parser.add_argument('--angle', type=float, default=10)
    parser.add_argument('--kl_target', type=float, default=1.0,
                        help='Minimum KL divergence floor')
    parser.add_argument('--kl_lambda', type=float, default=100.0,
                        help='Penalty weight for KL floor violation')
    parser.add_argument('--frob_weight', type=float, default=1.0,
                        help='Weight on Frobenius norm term (reduce to prioritize KL)')
    parser.add_argument('--n_frob_probes', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--n_restarts', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=30)
    parser.add_argument('--ref_checkpoint', type=str, default=None,
                        help='Path to checkpoint with cached activations/ref Jacobians')
    parser.add_argument('--results_dir', type=str, default='results/delta_opt_v9')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.results_dir, 'checkpoint.pkl')
    angle_rad = np.deg2rad(args.angle)

    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Layer: {args.layer}")
    print(f"  Angle: {args.angle} deg")
    print(f"  KL floor: {args.kl_target}, lambda: {args.kl_lambda}, frob_weight: {args.frob_weight}")
    print(f"  Frob probes: {args.n_frob_probes}")
    print(f"  Restarts: {args.n_restarts}, steps/restart: {args.n_steps}")
    print(f"  Ref checkpoint: {args.ref_checkpoint or 'none (will compute)'}")
    print("-" * 60)

    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model.config.hidden_size
    print(f"  d_model={d_model}")

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
        # Reuse random baselines if angle matches
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
        token_ids_list = load_fineweb_fixed_length(
            args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed
        )
        activations, contexts = extract_activations_with_context(
            model, token_ids_list, args.layer, device, args.n_act
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

                cos_dist = compute_cosine_distance(model, ctx_i, a_i, a_pert.detach(), args.layer)
                random_cos_dists.append(cos_dist)

                kl = compute_kl_eval(model, ctx_i, a_i, a_pert.detach(), args.layer)
                random_kls.append(kl)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/20")
        random_frob_norms = np.array(random_frob_norms)
        random_cos_dists = np.array(random_cos_dists)
        random_kls = np.array(random_kls)
        print(f"Random @ {args.angle} deg: KL={random_kls.mean():.3f} (std={random_kls.std():.3f}), "
              f"cos_dist={random_cos_dists.mean():.4f}")

    # Save base checkpoint
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
    best_kl = -1.0
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
        last_kl = 0.0
        last_frob = 0.0
        for step in range(args.n_steps):
            batch_indices = np.random.choice(args.n_act, args.batch_size, replace=False)
            optimizer.zero_grad()
            frob_val, kl_loss, kl_mean = optimization_step(
                model, direction, activations, contexts, args.layer,
                batch_indices, args.n_frob_probes, device, angle_rad,
                args.kl_target, args.kl_lambda, args.frob_weight
            )
            optimizer.step()
            with torch.no_grad():
                direction.data.div_(direction.data.norm())
            last_kl = kl_mean
            last_frob = np.sqrt(frob_val) if frob_val > 0 else 0

        # Evaluate on more samples
        d_eval = direction.detach() / direction.detach().norm()
        mean_kl, _ = evaluate_direction(
            model, d_eval, activations, contexts, args.layer,
            angle_rad, device, n_eval=min(30, args.n_act)
        )

        elapsed = time.time() - t0
        print(f"  Restart {restart+1}/{args.n_restarts}: KL={mean_kl:.3f}  "
              f"||J||_F~{last_frob:.0f}  ({elapsed:.0f}s)")

        all_restart_results.append({
            'direction': d_eval.cpu(),
            'mean_kl': mean_kl,
            'last_frob': last_frob,
        })

        # Keep best: highest KL that still has reasonable Jacobian
        # Primary: KL above target; secondary: lowest Jacobian
        if mean_kl > best_kl:
            best_kl = mean_kl
            best_frob = last_frob
            best_direction = d_eval.cpu().clone()
            best_restart = restart

    print(f"\nBest: restart {best_restart+1}, KL={best_kl:.3f}, "
          f"||J||_F~{best_frob:.0f}  (total {time.time()-t0_all:.0f}s)")

    checkpoint['opt_state'] = {
        'direction': best_direction,
        'step': args.n_steps - 1,
    }
    checkpoint['all_restarts'] = all_restart_results
    checkpoint['stage'] = 'optimization_done'
    save_checkpoint(checkpoint, checkpoint_path)

    # =========================================================
    # Stage 2: Full evaluation
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

        cos_dist = compute_cosine_distance(model, ctx_i, a_i, a_pert, args.layer)
        shifted_cos_dists.append(cos_dist)

        kl = compute_kl_eval(model, ctx_i, a_i, a_pert, args.layer)
        shifted_kls.append(kl)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{args.n_act}  ||J||_F={frob:.1f}  KL={kl:.3f}  "
                  f"cos_dist={cos_dist:.4f}  ({(i+1)/(time.time()-t0):.1f}/s)")

    shifted_frob_norms = np.array(shifted_frob_norms)
    shifted_cos_dists = np.array(shifted_cos_dists)
    shifted_kls = np.array(shifted_kls)

    print(f"\nResults:")
    print(f"  ||J||_F original:       mean={ref_frob_norms.mean():.1f}, std={ref_frob_norms.std():.1f}")
    print(f"  ||J||_F random@{args.angle}deg:   mean={random_frob_norms.mean():.1f}, std={random_frob_norms.std():.1f}")
    print(f"  ||J||_F optimized@{args.angle}deg: mean={shifted_frob_norms.mean():.1f}, std={shifted_frob_norms.std():.1f}")
    print(f"  KL random:    mean={random_kls.mean():.3f}, std={random_kls.std():.3f}")
    print(f"  KL optimized: mean={shifted_kls.mean():.3f}, std={shifted_kls.std():.3f}")
    print(f"  KL target:    {args.kl_target}")
    kl_above = (shifted_kls >= args.kl_target).sum()
    print(f"  KL >= {args.kl_target}: {kl_above}/{len(shifted_kls)}")
    print(f"  cos_dist random:    mean={random_cos_dists.mean():.4f}")
    print(f"  cos_dist optimized: mean={shifted_cos_dists.mean():.4f}")

    # =========================================================
    # Plot
    # =========================================================
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

    # KL histogram
    ax = axes[1]
    all_kls = np.concatenate([random_kls, shifted_kls])
    bins = np.linspace(0, max(np.percentile(all_kls, 99) * 1.2, args.kl_target * 2), 30)
    ax.hist(random_kls, bins=bins, alpha=0.5, label=f'Random ({random_kls.mean():.3f})', color='gray')
    ax.hist(shifted_kls, bins=bins, alpha=0.5, label=f'Optimized ({shifted_kls.mean():.3f})', color='C1')
    ax.axvline(args.kl_target, color='red', linestyle='--', label=f'Target ({args.kl_target})')
    ax.set_xlabel('KL(p_orig || p_pert)')
    ax.set_title(f'KL divergence at {args.angle} deg')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ||J||_F vs KL scatter
    ax = axes[2]
    ax.scatter(random_kls, random_frob_norms, alpha=0.3, s=15, color='gray', label='Random')
    ax.scatter(shifted_kls, shifted_frob_norms, alpha=0.5, s=20, color='C1', label='Optimized')
    ax.axhline(ref_frob_norms.mean(), color='C0', linestyle='--', alpha=0.5, label=f'Orig ||J||_F ({ref_frob_norms.mean():.0f})')
    ax.axvline(args.kl_target, color='red', linestyle='--', alpha=0.5, label=f'KL target')
    ax.set_xlabel('KL(p_orig || p_pert)')
    ax.set_ylabel('||J||_F')
    ax.set_title('||J||_F vs KL')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'v9: min-J + KL>{args.kl_target} at {args.angle}deg ({args.n_restarts}x{args.n_steps} steps)', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, f'eval_{args.angle}deg_kl{args.kl_target}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {plot_path}")

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
    results_path = os.path.join(args.results_dir, f'results_{args.angle}deg_kl{args.kl_target}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved: {results_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
