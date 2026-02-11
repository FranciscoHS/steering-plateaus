#!/usr/bin/env python3
"""
Constrained MELBO: find the direction within the language subspace (span of D)
that maximizes KL divergence when used as a spherical perturbation.

D = matrix of (English - Chinese) activation differences at a given layer.
The optimization finds which component of the language steering vector does
the most "work" (causes the most behavioral change).

Usage:
    python scripts/run_melbo_constrained.py --model qwen3-0.6b
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_allpos,
    _get_blocks, _is_rope_model, _get_position_embeddings, _run_block,
    _get_final_norm, _get_lm_head,
)
from src.data import load_fineweb_fixed_length
from data.bilingual_prompts import ENGLISH_PROMPTS, CHINESE_PROMPTS


def extract_bilingual_diffs(model, tokenizer, device, layer):
    """Extract difference matrix D = en_acts - zh_acts at given layer."""
    def get_acts(prompts):
        acts = []
        for p in prompts:
            tokens = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            acts.append(outputs.hidden_states[layer + 1][:, -1, :].squeeze(0).cpu())
        return torch.stack(acts)

    print(f"Extracting bilingual activations at layer {layer}...")
    en_acts = get_acts(ENGLISH_PROMPTS)
    zh_acts = get_acts(CHINESE_PROMPTS)
    D = en_acts - zh_acts

    mean_vec = D.mean(dim=0)
    mean_dir = mean_vec / mean_vec.norm()
    print(f"  D shape: {D.shape}, mean norm: {mean_vec.norm():.2f}")

    return D, mean_dir


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device,
                                   batch_size=64):
    """Extract layer-L context and final-token activation for fixed-length sequences."""
    all_contexts = []
    all_activations = []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].cpu())
        all_activations.append(hidden[:, -1, :].cpu())
    return torch.cat(all_contexts, 0), torch.cat(all_activations, 0)


def forward_to_hidden_and_logits(model, hidden_states, start_layer):
    """Forward pass returning both final-block hidden state and logits (last token)."""
    hidden = hidden_states
    blocks = _get_blocks(model)
    rope = _is_rope_model(model)
    if rope:
        pos_emb = _get_position_embeddings(model, hidden)
    for block in blocks[start_layer + 1:]:
        hidden = _run_block(block, hidden, rope, pos_emb if rope else None)
    final_hidden = hidden[:, -1, :]
    normed = _get_final_norm(model)(final_hidden)
    logits = _get_lm_head(model)(normed)
    return final_hidden, logits


def build_perturbed_hidden(contexts, activations, direction, angle_rad):
    """Construct hidden states after spherical perturbation along direction.

    Args:
        contexts: [N, seq_len-1, d] on device
        activations: [N, d] on device
        direction: [d] unit vector (may carry gradients)
        angle_rad: float

    Returns:
        hidden: [N, seq_len, d]
    """
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms

    # Project direction orthogonal to each activation
    proj = (direction.unsqueeze(0) * a_unit).sum(dim=-1, keepdim=True)
    u_perp = direction.unsqueeze(0) - proj * a_unit
    u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)

    a_pert = a_norms * (cos_t * a_unit + sin_t * u_perp)
    return torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)


def optimize_direction(model, contexts, activations, ref_log_probs,
                       V, angle_rad, layer, device,
                       n_steps=300, lr=0.01, n_restarts=5,
                       prev_alphas=None):
    """Find direction in span(V) maximizing mean KL divergence.

    Args:
        V: [d_model, r] right singular vectors of D (orthonormal columns), on device
        prev_alphas: list of [r] unit vectors for orthogonality

    Returns:
        best_alpha, best_direction, best_kl, best_history
    """
    r = V.shape[1]
    best_kl = -float('inf')
    best_alpha = None
    best_history = None

    for restart in range(n_restarts):
        # Random init, projected to orthogonal complement of previous directions
        alpha = torch.randn(r, device=device)
        if prev_alphas:
            for pa in prev_alphas:
                alpha = alpha - (alpha @ pa) * pa
        alpha = alpha / alpha.norm()
        alpha = alpha.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([alpha], lr=lr)
        history = []

        for step in range(n_steps):
            optimizer.zero_grad()

            # Unit direction with orthogonality
            a_normed = alpha / alpha.norm()
            if prev_alphas:
                for pa in prev_alphas:
                    a_normed = a_normed - (a_normed @ pa) * pa
                a_normed = a_normed / (a_normed.norm() + 1e-10)
            direction = V @ a_normed

            # Perturbed forward pass
            hidden = build_perturbed_hidden(contexts, activations, direction, angle_rad)
            logits = forward_from_layer_allpos(model, hidden, layer)
            pert_log_probs = F.log_softmax(logits, dim=-1)

            # KL(ref || pert)
            kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1).mean()

            loss = -kl
            loss.backward()
            optimizer.step()

            # Project alpha to orthogonal subspace after step
            if prev_alphas:
                with torch.no_grad():
                    for pa in prev_alphas:
                        alpha.data -= (alpha.data @ pa) * pa

            kl_val = kl.item()
            history.append(kl_val)

            if step % 50 == 0 or step == n_steps - 1:
                print(f"    restart {restart+1}/{n_restarts}, step {step}: "
                      f"KL = {kl_val:.6f}")

        final_kl = history[-1]
        if final_kl > best_kl:
            best_kl = final_kl
            with torch.no_grad():
                a_final = alpha / alpha.norm()
                if prev_alphas:
                    for pa in prev_alphas:
                        a_final = a_final - (a_final @ pa) * pa
                    a_final = a_final / a_final.norm()
                best_alpha = a_final.clone()
            best_history = history

    best_direction = (V @ best_alpha).detach()
    best_direction = best_direction / best_direction.norm()
    return best_alpha, best_direction, best_kl, best_history


def sweep_single_direction(model, contexts, activations, ref_final, ref_log_probs,
                           direction, angles_rad, layer, device):
    """Perturbation sweep for one fixed direction across all activations.

    Returns: kl_divs [N, n_angles], l2_dists [N, n_angles]
    """
    N = activations.shape[0]
    n_angles = len(angles_rad)
    kl_divs = np.zeros((N, n_angles))
    l2_dists = np.zeros((N, n_angles))

    direction_dev = direction.to(device)

    for ang_idx, theta in enumerate(angles_rad):
        with torch.no_grad():
            hidden = build_perturbed_hidden(contexts, activations, direction_dev, theta)
            pert_final, pert_logits = forward_to_hidden_and_logits(model, hidden, layer)
            pert_log_probs = F.log_softmax(pert_logits, dim=-1)

            l2 = (pert_final - ref_final).norm(dim=-1)
            kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)

            l2_dists[:, ang_idx] = l2.cpu().numpy()
            kl_divs[:, ang_idx] = kl.cpu().numpy()

    return kl_divs, l2_dists


def sweep_random_directions(model, contexts, activations, ref_final, ref_log_probs,
                            angles_rad, layer, device, n_dirs=10):
    """Perturbation sweep with random directions (different per activation).

    Returns: kl_divs [N, n_angles, n_dirs], l2_dists [N, n_angles, n_dirs]
    """
    N, d = activations.shape
    n_angles = len(angles_rad)
    kl_divs = np.zeros((N, n_angles, n_dirs))
    l2_dists = np.zeros((N, n_angles, n_dirs))

    # Pre-compute random orthogonal directions
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms
    v_perps = torch.zeros(N, n_dirs, d, device=device)
    for i in range(N):
        v = torch.randn(n_dirs, d, device=device)
        v = v - (v @ a_unit[i]).unsqueeze(1) * a_unit[i].unsqueeze(0)
        v_perps[i] = F.normalize(v, dim=-1)

    for ang_idx, theta in enumerate(angles_rad):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        for dir_idx in range(n_dirs):
            v = v_perps[:, dir_idx, :]  # [N, d], already orthogonal
            a_pert = a_norms * (cos_t * a_unit + sin_t * v)

            with torch.no_grad():
                hidden = torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)
                pert_final, pert_logits = forward_to_hidden_and_logits(
                    model, hidden, layer)
                pert_log_probs = F.log_softmax(pert_logits, dim=-1)

                l2 = (pert_final - ref_final).norm(dim=-1)
                kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)

                l2_dists[:, ang_idx, dir_idx] = l2.cpu().numpy()
                kl_divs[:, ang_idx, dir_idx] = kl.cpu().numpy()

    return kl_divs, l2_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--opt_angle', type=float, default=15.0,
                        help='Angle in degrees for optimization')
    parser.add_argument('--n_act', type=int, default=50)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_directions', type=int, default=3)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_angles', type=int, default=60,
                        help='Number of angles for evaluation sweep')
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/melbo_constrained/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Phase 1: Setup =====
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    # Bilingual difference matrix
    D, mean_dir = extract_bilingual_diffs(model, tokenizer, device, args.layer)

    # SVD of D
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    V = Vh.T  # [d_model, r]
    r = V.shape[1]
    pc1 = Vh[0]
    print(f"SVD: rank r={r}, top SV={S[0]:.2f}, "
          f"PC1 var explained={S[0]**2 / (S**2).sum():.3f}")
    print(f"cos(PC1, mean): {(pc1 @ mean_dir).abs().item():.3f}")

    # ===== Phase 2: FineWeb activations =====
    print(f"\nLoading {args.n_act} FineWeb sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.layer} activations...")
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Activation shape: {activations.shape}, "
          f"mean norm: {activations.norm(dim=-1).mean():.2f}")

    # Move to device
    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)
    V_dev = V.to(device)

    # Pre-compute reference (unperturbed)
    with torch.no_grad():
        ref_hidden = torch.cat(
            [contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_hidden_and_logits(
            model, ref_hidden, args.layer)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # ===== Phase 3: Optimization =====
    angle_rad = np.deg2rad(args.opt_angle)
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION: top-{args.n_directions} directions in span(D)")
    print(f"Angle: {args.opt_angle} deg, steps: {args.n_steps}, "
          f"restarts: {args.n_restarts}")
    print(f"{'='*60}")

    melbo_alphas = []
    melbo_directions = []
    melbo_kls = []
    melbo_histories = []

    for d_idx in range(args.n_directions):
        print(f"\n--- Direction {d_idx+1}/{args.n_directions} ---")
        t0 = time.time()
        alpha, direction, best_kl, history = optimize_direction(
            model, contexts_dev, activations_dev, ref_log_probs,
            V_dev, angle_rad, args.layer, device,
            n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts,
            prev_alphas=melbo_alphas if melbo_alphas else None
        )
        elapsed = time.time() - t0

        melbo_alphas.append(alpha)
        melbo_directions.append(direction.cpu())
        melbo_kls.append(best_kl)
        melbo_histories.append(history)

        cos_mean = (direction @ mean_dir.to(device)).abs().item()
        cos_pc1 = (direction @ pc1.to(device)).abs().item()
        print(f"  Best KL: {best_kl:.6f}")
        print(f"  cos(MELBO_{d_idx+1}, mean): {cos_mean:.3f}")
        print(f"  cos(MELBO_{d_idx+1}, PC1): {cos_pc1:.3f}")
        print(f"  Time: {elapsed:.1f}s")

        if d_idx > 0:
            for prev_idx in range(d_idx):
                cos_prev = (direction @ melbo_directions[prev_idx].to(device)).abs().item()
                print(f"  cos(MELBO_{d_idx+1}, MELBO_{prev_idx+1}): {cos_prev:.4f}")

    # KL for mean and PC1 at the optimization angle (for comparison)
    with torch.no_grad():
        for name, d in [("mean", mean_dir), ("PC1", pc1)]:
            hidden = build_perturbed_hidden(
                contexts_dev, activations_dev, d.to(device), angle_rad)
            logits = forward_from_layer_allpos(model, hidden, args.layer)
            lp = F.log_softmax(logits, dim=-1)
            kl = (ref_log_probs.exp() * (ref_log_probs - lp)).sum(dim=-1).mean()
            print(f"\n{name} direction KL at {args.opt_angle} deg: {kl.item():.6f}")

    # Save optimization checkpoint
    opt_results = {
        'args': vars(args),
        'melbo_alphas': [a.cpu() for a in melbo_alphas],
        'melbo_directions': melbo_directions,
        'melbo_kls': melbo_kls,
        'melbo_histories': melbo_histories,
        'mean_dir': mean_dir,
        'pc1': pc1,
        'singular_values': S,
    }
    with open(os.path.join(args.results_dir, 'optimization_results.pkl'), 'wb') as f:
        pickle.dump(opt_results, f)
    print(f"\nSaved optimization checkpoint")

    # ===== Phase 4: Evaluation sweep =====
    angles_deg = np.linspace(0, 180, args.n_angles + 1)[1:]
    angles_rad_sweep = np.deg2rad(angles_deg)

    print(f"\n{'='*60}")
    print(f"EVALUATION SWEEP: {len(angles_deg)} angles, 0-180 deg")
    print(f"{'='*60}")

    sweep_results = {}

    # Named directions
    named_dirs = [('mean', mean_dir), ('PC1', pc1)]
    for i, d in enumerate(melbo_directions):
        named_dirs.append((f'MELBO_{i+1}', d))

    for name, direction in named_dirs:
        print(f"Sweeping: {name}...", end=' ')
        t0 = time.time()
        kl, l2 = sweep_single_direction(
            model, contexts_dev, activations_dev, ref_final, ref_log_probs,
            direction, angles_rad_sweep, args.layer, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}
        print(f"{time.time()-t0:.1f}s")

    # Random directions
    print(f"Sweeping: random ({args.n_random_dirs} dirs/act)...", end=' ')
    t0 = time.time()
    rand_kl, rand_l2 = sweep_random_directions(
        model, contexts_dev, activations_dev, ref_final, ref_log_probs,
        angles_rad_sweep, args.layer, device, n_dirs=args.n_random_dirs)
    sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}
    print(f"{time.time()-t0:.1f}s")

    # ===== Phase 5: Save and plot =====
    all_results = {
        **opt_results,
        'angles_deg': angles_deg,
        'sweep_results': sweep_results,
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved results to {args.results_dir}/results.pkl")

    # --- Plots ---
    colors = {
        'random': 'gray', 'mean': 'green', 'PC1': 'blue',
        'MELBO_1': 'red', 'MELBO_2': 'orange', 'MELBO_3': 'purple',
    }

    # 1. Sweep comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for metric_idx, (key, ylabel) in enumerate(
            [('kl', 'KL divergence'), ('l2', 'L2 distance (final hidden)')]):
        ax = axes[metric_idx]

        # Random baseline
        rand_data = sweep_results['random'][key]
        rand_per_act = np.median(rand_data, axis=2)
        rand_median = np.median(rand_per_act, axis=0)
        rand_q25 = np.percentile(rand_per_act, 25, axis=0)
        rand_q75 = np.percentile(rand_per_act, 75, axis=0)
        ax.fill_between(angles_deg, rand_q25, rand_q75, color='gray', alpha=0.2)
        ax.plot(angles_deg, rand_median, color='gray', lw=2,
                label='Random', linestyle='--')

        # Named directions
        for name in ['mean', 'PC1', 'MELBO_1', 'MELBO_2', 'MELBO_3']:
            if name not in sweep_results:
                continue
            data = sweep_results[name][key]
            median = np.median(data, axis=0)
            q25 = np.percentile(data, 25, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            color = colors.get(name, 'black')
            ax.plot(angles_deg, median, color=color, lw=2, label=name)
            ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.1)

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)

    fig.suptitle(f'Constrained MELBO: {args.model}, layer {args.layer}\n'
                 f'Optimized at {args.opt_angle} deg', fontsize=12)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'sweep_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # 2. Optimization history
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, history in enumerate(melbo_histories):
        ax.plot(history, label=f'MELBO_{i+1} (final={history[-1]:.4f})',
                color=colors.get(f'MELBO_{i+1}', 'black'))
    ax.set_xlabel('Optimization step')
    ax.set_ylabel('KL divergence')
    ax.set_title(f'Optimization convergence ({args.opt_angle} deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(args.results_dir, 'optimization_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for i, kl in enumerate(melbo_kls):
        d = melbo_directions[i]
        cos_m = (d @ mean_dir).abs().item()
        cos_p = (d @ pc1).abs().item()
        print(f"MELBO_{i+1}: KL={kl:.6f}, cos(mean)={cos_m:.3f}, cos(PC1)={cos_p:.3f}")

    # Random KL at optimization angle for comparison
    opt_ang_idx = np.argmin(np.abs(angles_deg - args.opt_angle))
    rand_kl_at_opt = np.median(sweep_results['random']['kl'][:, opt_ang_idx, :])
    mean_kl_at_opt = np.median(sweep_results['mean']['kl'][:, opt_ang_idx])
    print(f"\nAt {args.opt_angle} deg:")
    print(f"  Random median KL: {rand_kl_at_opt:.6f}")
    print(f"  Mean dir median KL: {mean_kl_at_opt:.6f}")
    for i, kl in enumerate(melbo_kls):
        print(f"  MELBO_{i+1} mean KL: {kl:.6f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
