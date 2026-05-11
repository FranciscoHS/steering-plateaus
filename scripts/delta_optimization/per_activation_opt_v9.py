#!/usr/bin/env python3
"""
Per-activation delta optimization v9, then average.

For each of the N activations, independently optimize a direction that has
high KL-div while keeping Jacobian norm low (same loss as v9). Then average
all N directions and evaluate the average on all activations.

Compare to the joint optimization direction from the standard v9 run.

Usage:
    python scripts/delta_optimization/per_activation_opt_v9.py \
        --ref_checkpoint results/delta_opt_v9/checkpoint.pkl \
        --joint_results results/delta_opt_v9/results_10deg_kl1.0.pkl
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
    _get_final_norm, _get_lm_head,
)
from src.analysis import compute_jacobian_norm_vjp


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


def compute_kl_differentiable(model, ctx, a_orig, a_pert, layer):
    """KL(p_orig || p_pert), differentiable w.r.t. a_pert."""
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
    return kl.squeeze()


def compute_kl_eval(model, ctx, a_orig, a_pert, layer):
    """KL(p_orig || p_pert), no grad."""
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


def optimize_single_activation(model, a_i, ctx_i, layer, device, angle_rad,
                                kl_target, kl_lambda, frob_weight,
                                n_frob_probes, n_restarts, n_steps, lr):
    """Run v9 optimization for a single activation. Returns best direction (cpu)."""
    d_model = a_i.shape[0]
    a_i = a_i.to(device).detach()
    ctx_i = ctx_i.to(device).detach()

    best_direction = None
    best_kl = -1.0

    for restart in range(n_restarts):
        direction = torch.randn(d_model, device=device)
        direction = (direction / direction.norm()).requires_grad_(True)
        optimizer = torch.optim.Adam([direction], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()

            # KL floor
            d_unit = direction / (direction.norm() + 1e-10)
            a_pert = spherical_perturb(a_i, d_unit, angle_rad)
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                kl_val = compute_kl_differentiable(model, ctx_i, a_i, a_pert, layer)
            kl_violation = torch.clamp(kl_target - kl_val, min=0.0)
            kl_loss = kl_lambda * kl_violation ** 2
            kl_loss.backward()

            # Frobenius norm
            d_unit2 = direction / (direction.norm() + 1e-10)
            a_pert2 = spherical_perturb(a_i, d_unit2, angle_rad)
            point = a_pert2.reshape(1, 1, d_model)
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
            frob_loss = frob_weight * frob_sq / inflation
            frob_loss.backward()

            optimizer.step()
            with torch.no_grad():
                direction.data.div_(direction.data.norm())

        # Evaluate this restart
        d_eval = (direction.detach() / direction.detach().norm())
        a_pert_eval = spherical_perturb(a_i, d_eval, angle_rad).detach()
        kl_eval = compute_kl_eval(model, ctx_i, a_i, a_pert_eval, layer)

        if kl_eval > best_kl:
            best_kl = kl_eval
            best_direction = d_eval.cpu().clone()

    return best_direction, best_kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_checkpoint', type=str,
                        default='results/delta_opt_v9/checkpoint.pkl',
                        help='v9 checkpoint with cached activations')
    parser.add_argument('--joint_results', type=str,
                        default='results/delta_opt_v9/results_10deg_kl1.0.pkl',
                        help='v9 joint optimization results for comparison')
    parser.add_argument('--angle', type=float, default=10)
    parser.add_argument('--kl_target', type=float, default=1.0)
    parser.add_argument('--kl_lambda', type=float, default=100.0)
    parser.add_argument('--frob_weight', type=float, default=1.0)
    parser.add_argument('--n_frob_probes', type=int, default=5)
    parser.add_argument('--n_restarts', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--results_dir', type=str,
                        default='results/delta_opt_v9_per_act')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    angle_rad = np.deg2rad(args.angle)

    # Load cached activations from v9 checkpoint
    print(f"Loading checkpoint: {args.ref_checkpoint}")
    with open(args.ref_checkpoint, 'rb') as f:
        cp = pickle.load(f)
    activations = cp['activations']
    contexts = cp['contexts']
    ref_frob_norms = np.array(cp['ref_frob_norms'])
    n_act = activations.shape[0]
    print(f"  {n_act} activations, d_model={activations.shape[1]}")

    # Load joint direction for comparison
    print(f"Loading joint results: {args.joint_results}")
    with open(args.joint_results, 'rb') as f:
        joint = pickle.load(f)
    joint_direction = joint['direction']  # already unit norm
    joint_kls = joint['shifted_kls']
    random_kls = joint['random_kls']
    print(f"  Joint KL: mean={joint_kls.mean():.4f}")
    print(f"  Random KL: mean={random_kls.mean():.4f}")

    # Load model
    model_name = cp['args']['model']
    layer = cp['args']['layer']
    print(f"\nLoading model: {model_name}")
    model, tokenizer, device = load_model(model_name)

    print(f"\nConfiguration:")
    print(f"  angle={args.angle}, kl_target={args.kl_target}, kl_lambda={args.kl_lambda}")
    print(f"  frob_weight={args.frob_weight}, n_frob_probes={args.n_frob_probes}")
    print(f"  n_restarts={args.n_restarts}, n_steps={args.n_steps}, lr={args.lr}")
    print("-" * 60)

    # =========================================================
    # Stage 1: Per-activation optimization
    # =========================================================
    print(f"\n=== Per-activation optimization ({n_act} activations) ===")

    per_act_directions = []
    per_act_kls = []  # KL on the activation it was optimized for
    checkpoint_path = os.path.join(args.results_dir, 'checkpoint.pkl')

    # Check for existing checkpoint to resume
    start_idx = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        if 'per_act_directions' in ckpt and 'per_act_kls' in ckpt:
            per_act_directions = ckpt['per_act_directions']
            per_act_kls = ckpt['per_act_kls']
            start_idx = len(per_act_directions)
            print(f"  Resuming from activation {start_idx}")

    t0 = time.time()
    for i in range(start_idx, n_act):
        direction_i, kl_i = optimize_single_activation(
            model, activations[i], contexts[i], layer, device, angle_rad,
            args.kl_target, args.kl_lambda, args.frob_weight,
            args.n_frob_probes, args.n_restarts, args.n_steps, args.lr
        )
        per_act_directions.append(direction_i)
        per_act_kls.append(kl_i)

        elapsed = time.time() - t0
        eta = elapsed / (i - start_idx + 1) * (n_act - i - 1) if i > start_idx else 0
        print(f"  [{i+1}/{n_act}] KL={kl_i:.4f}  "
              f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        # Checkpoint every 10 activations
        if (i + 1) % 10 == 0:
            ckpt_data = {
                'args': vars(args),
                'per_act_directions': per_act_directions,
                'per_act_kls': per_act_kls,
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(ckpt_data, f)
            print(f"  Checkpoint saved ({i+1}/{n_act})")

    per_act_kls = np.array(per_act_kls)
    print(f"\nPer-activation KLs (on own activation):")
    print(f"  mean={per_act_kls.mean():.4f}, std={per_act_kls.std():.4f}")
    print(f"  min={per_act_kls.min():.4f}, max={per_act_kls.max():.4f}")
    print(f"  fraction >= kl_target: {(per_act_kls >= args.kl_target).mean():.1%}")

    # =========================================================
    # Stage 2: Average directions and evaluate
    # =========================================================
    print(f"\n=== Averaging and evaluating ===")

    # Stack and average
    dir_matrix = torch.stack(per_act_directions)  # [n_act, d_model]
    avg_direction = dir_matrix.mean(dim=0)
    avg_direction = avg_direction / avg_direction.norm()
    avg_direction_dev = avg_direction.to(device)

    # Also check: how aligned are the per-activation directions?
    cosines = F.cosine_similarity(dir_matrix[None, :, :], dir_matrix[:, None, :], dim=-1)
    # Upper triangle only (exclude diagonal)
    triu_idx = torch.triu_indices(n_act, n_act, offset=1)
    pairwise_cos = cosines[triu_idx[0], triu_idx[1]]
    print(f"\nPairwise cosine similarity between per-activation directions:")
    print(f"  mean={pairwise_cos.mean():.4f}, std={pairwise_cos.std():.4f}")
    print(f"  min={pairwise_cos.min():.4f}, max={pairwise_cos.max():.4f}")

    # Cosine of avg with joint
    joint_dev = joint_direction.to(device)
    joint_dev = joint_dev / joint_dev.norm()
    cos_avg_joint = F.cosine_similarity(avg_direction_dev.unsqueeze(0),
                                         joint_dev.unsqueeze(0)).item()
    print(f"\nCosine(avg_direction, joint_direction) = {cos_avg_joint:.4f}")

    # Evaluate: averaged direction on all activations
    print(f"\nEvaluating averaged direction on all {n_act} activations...")
    avg_kls = []
    for i in range(n_act):
        a_i = activations[i].to(device)
        ctx_i = contexts[i].to(device)
        a_pert = spherical_perturb(a_i, avg_direction_dev, angle_rad).detach()
        kl = compute_kl_eval(model, ctx_i, a_i, a_pert, layer)
        avg_kls.append(kl)
    avg_kls = np.array(avg_kls)

    # Evaluate: joint direction on all activations (re-measure for consistency)
    print(f"Evaluating joint direction on all {n_act} activations...")
    joint_kls_remeasured = []
    for i in range(n_act):
        a_i = activations[i].to(device)
        ctx_i = contexts[i].to(device)
        a_pert = spherical_perturb(a_i, joint_dev, angle_rad).detach()
        kl = compute_kl_eval(model, ctx_i, a_i, a_pert, layer)
        joint_kls_remeasured.append(kl)
    joint_kls_remeasured = np.array(joint_kls_remeasured)

    # Evaluate: each per-activation direction on its own activation (already have this)
    # Also evaluate each per-act direction on ALL activations (cross-eval)
    print(f"Evaluating per-activation directions cross-activation...")
    cross_kl_matrix = np.zeros((n_act, n_act))
    for i in range(n_act):
        d_i = per_act_directions[i].to(device)
        d_i = d_i / d_i.norm()
        for j in range(n_act):
            a_j = activations[j].to(device)
            ctx_j = contexts[j].to(device)
            a_pert = spherical_perturb(a_j, d_i, angle_rad).detach()
            kl = compute_kl_eval(model, ctx_j, a_j, a_pert, layer)
            cross_kl_matrix[i, j] = kl
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_act}")

    # =========================================================
    # Results summary
    # =========================================================
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")

    print(f"\nKL divergence at {args.angle} deg (all {n_act} activations):")
    print(f"  Random baseline:        mean={random_kls.mean():.4f}")
    print(f"  Joint (100-act) opt:    mean={joint_kls_remeasured.mean():.4f}")
    print(f"  Avg of per-act opts:    mean={avg_kls.mean():.4f}")
    print(f"  Per-act on own (diag):  mean={per_act_kls.mean():.4f}")
    print(f"  Per-act cross (off-diag mean): mean={cross_kl_matrix[~np.eye(n_act, dtype=bool)].mean():.4f}")

    # =========================================================
    # Plot
    # =========================================================
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1: KL comparison histogram
    ax = axes[0]
    all_kls_concat = np.concatenate([random_kls, joint_kls_remeasured, avg_kls])
    bins = np.linspace(0, max(np.percentile(all_kls_concat, 99) * 1.5, 0.1), 35)
    ax.hist(random_kls, bins=bins, alpha=0.4, label=f'Random ({random_kls.mean():.4f})', color='gray')
    ax.hist(joint_kls_remeasured, bins=bins, alpha=0.5, label=f'Joint-100 ({joint_kls_remeasured.mean():.4f})', color='C0')
    ax.hist(avg_kls, bins=bins, alpha=0.5, label=f'Avg-of-per-act ({avg_kls.mean():.4f})', color='C1')
    ax.set_xlabel('KL(p_orig || p_pert)')
    ax.set_title('KL divergence comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Per-activation KLs (own activation vs cross-activation)
    ax = axes[1]
    diag_kls = np.diag(cross_kl_matrix)
    off_diag_means = []
    for i in range(n_act):
        mask = np.ones(n_act, dtype=bool)
        mask[i] = False
        off_diag_means.append(cross_kl_matrix[i, mask].mean())
    off_diag_means = np.array(off_diag_means)
    ax.scatter(diag_kls, off_diag_means, alpha=0.5, s=15)
    lim = max(diag_kls.max(), off_diag_means.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('KL on own activation')
    ax.set_ylabel('Mean KL on other activations')
    ax.set_title('Per-act direction generalization')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Pairwise cosine similarity histogram
    ax = axes[2]
    ax.hist(pairwise_cos.numpy(), bins=40, alpha=0.7, color='C2')
    ax.axvline(cos_avg_joint, color='red', linestyle='--',
               label=f'cos(avg, joint)={cos_avg_joint:.3f}')
    ax.set_xlabel('Cosine similarity')
    ax.set_title('Pairwise cos sim of per-act directions')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Per-activation v9 opt → average vs joint ({n_act} acts, {args.angle}°)', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {plot_path}")

    # Save results
    results = {
        'args': vars(args),
        'per_act_directions': per_act_directions,
        'per_act_kls': per_act_kls,
        'avg_direction': avg_direction.cpu(),
        'avg_kls': avg_kls,
        'joint_direction': joint_direction,
        'joint_kls_remeasured': joint_kls_remeasured,
        'random_kls': random_kls,
        'cross_kl_matrix': cross_kl_matrix,
        'pairwise_cosines': pairwise_cos.numpy(),
        'cos_avg_joint': cos_avg_joint,
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved: {results_path}")


if __name__ == '__main__':
    main()
