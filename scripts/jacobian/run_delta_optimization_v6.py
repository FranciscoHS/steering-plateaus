#!/usr/bin/env python3
"""
Delta optimization v6: minimize Jacobian with enforced large downstream cosine distance.

Find a direction d such that at a fixed angle theta:
  a_perturbed = cos(theta)*a + sin(theta)*||a||*v_perp(d, a)

The perturbed activations have:
  1. cosine_distance(final_orig, final_pert) >= target  (large downstream change)
  2. ||J||_F at perturbed point is minimized

Loss = ||J||_F^2 + lambda * max(0, cos_target - cos_dist)^2

where cos_dist = 1 - cos_sim(final_orig, final_pert), so cos_dist=1 means 90 deg.

The idea: if we enforce e.g. 90 deg cosine distance downstream, the direction
*cannot* stay in the plateau, so either the optimization fails (no such direction
exists) or it finds something feature-like.

Usage:
    python scripts/jacobian/run_delta_optimization_v6.py
    python scripts/jacobian/run_delta_optimization_v6.py --angle 30 --cos_target 1.0
    python scripts/jacobian/run_delta_optimization_v6.py --angle 50 --cos_target 0.9
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
    """
    Cosine distance between final-layer representations.
    Returns 1 - cos_sim, so 1.0 = orthogonal, 2.0 = antiparallel.
    Differentiable w.r.t. a_pert.
    """
    d = a_orig.shape[-1]

    with torch.no_grad():
        ref_final = forward_from_layer_to_final(model, ctx, a_orig.reshape(1, 1, d), layer)

    pert_final = forward_from_layer_to_final(model, ctx, a_pert.reshape(1, 1, d), layer)

    cos_sim = F.cosine_similarity(ref_final.float(), pert_final.float(), dim=-1)
    cos_dist = 1.0 - cos_sim
    return cos_dist.squeeze()


def compute_kl(model, ctx, a_orig, a_pert, layer):
    """KL(p_orig || p_pert) from logits. For evaluation only (no grad needed)."""
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

        # --- Cosine distance computation (separate forward pass) ---
        d_unit_cos = direction / (direction.norm() + 1e-10)
        a_pert_cos = spherical_perturb(a_i, d_unit_cos, angle_rad)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            cos_dist = compute_cosine_distance(model, ctx_i, a_i, a_pert_cos, layer)

        cos_dist_val = cos_dist.item()
        total_cos_dist += cos_dist_val / B

        cos_violation = torch.clamp(cos_target - cos_dist, min=0.0)
        cos_loss = cos_lambda * cos_violation ** 2 / B
        cos_loss.backward()
        total_cos_loss += cos_loss.item()

        # --- Frobenius norm computation (separate forward pass) ---
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

        # Normalize by chain-rule inflation
        R = a_i.norm().item()
        inflation = (R * np.sin(angle_rad)) ** 2
        frob_loss = frob_sq / (inflation * B)
        frob_loss.backward()
        total_frob_val += frob_sq.item() / (inflation * B)

    return total_frob_val, total_cos_loss, total_cos_dist


def main():
    parser = argparse.ArgumentParser(description='Delta optimization v6 (minimize Jacobian + cosine distance floor)')
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=100)
    parser.add_argument('--n_ctx', type=int, default=100)
    parser.add_argument('--angle', type=float, default=30,
                        help='Fixed perturbation angle in degrees')
    parser.add_argument('--cos_target', type=float, default=1.0,
                        help='Min cosine distance downstream (1.0 = orthogonal, 2.0 = antiparallel)')
    parser.add_argument('--cos_lambda', type=float, default=10000,
                        help='Penalty weight for cosine distance floor violation')
    parser.add_argument('--n_frob_probes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--results_dir', type=str, default='results/delta_opt_v6')
    parser.add_argument('--resume', action='store_true')
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
    print(f"  Cosine distance floor: {args.cos_target} (1.0=orthogonal), lambda: {args.cos_lambda}")
    print(f"  Loss: ||J||_F^2 + lambda*max(0, cos_target - cos_dist)^2")
    print(f"  Frob probes: {args.n_frob_probes}")
    print(f"  Optimization: B={args.batch_size}, lr={args.lr}, steps={args.n_steps}")
    print("-" * 60)

    # Resume
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        stage = checkpoint.get('stage', 'start')
        print(f"Resuming from stage: {stage}")
    else:
        checkpoint = {'stage': 'start', 'args': vars(args)}
        stage = 'start'

    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    # =========================================================
    # Stage 0: Extract activations + compute references
    # =========================================================
    if stage == 'start':
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

        # Reference ||J||_F at original activations
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

        # Baseline: random direction at this angle
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
        print(f"Random @ {args.angle} deg:")
        print(f"  ||J||_F mean={random_frob_norms.mean():.1f}")
        print(f"  cos_dist mean={random_cos_dists.mean():.4f} (angle={cos_deg.mean():.1f} deg)")
        print(f"  KL mean={random_kls.mean():.3f}")

        checkpoint.update({
            'activations': activations,
            'contexts': contexts,
            'ref_frob_norms': ref_frob_norms,
            'random_frob_norms': random_frob_norms,
            'random_cos_dists': random_cos_dists,
            'random_kls': random_kls,
            'stage': 'activations_ready',
        })
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'activations_ready'

    # =========================================================
    # Stage 1: Optimization
    # =========================================================
    if stage in ['activations_ready', 'optimization_in_progress']:
        print(f"\n=== Stage 1: Direction optimization ===")

        activations = checkpoint['activations']
        contexts = checkpoint['contexts']
        ref_frob_norms = np.array(checkpoint['ref_frob_norms'])
        print(f"  Ref ||J||_F: {ref_frob_norms.mean():.1f}")
        print(f"  Cosine dist floor: {args.cos_target}, lambda: {args.cos_lambda}")

        opt_state = checkpoint.get('opt_state', None)
        if opt_state is not None:
            direction = opt_state['direction'].to(device).requires_grad_(True)
            start_step = opt_state['step'] + 1
            history = opt_state['history']
            print(f"Resuming from step {start_step}")
        else:
            direction = torch.randn(d_model, device=device)
            direction = (direction / direction.norm()).requires_grad_(True)
            start_step = 0
            history = {'frob_loss': [], 'cos_loss': [], 'cos_dist_mean': [],
                       'grad_norm': [], 'cosine_sim': []}

        optimizer = torch.optim.Adam([direction], lr=args.lr)
        if opt_state is not None and 'optimizer_state' in opt_state:
            optimizer.load_state_dict(opt_state['optimizer_state'])

        prev_dir = direction.detach().clone() / direction.detach().norm()

        t0 = time.time()
        for step in range(start_step, args.n_steps):
            batch_indices = np.random.choice(args.n_act, args.batch_size, replace=False)

            optimizer.zero_grad()
            frob_val, cos_loss, cos_dist_mean = optimization_step(
                model, direction, activations, contexts, args.layer,
                batch_indices, args.n_frob_probes, device, angle_rad,
                args.cos_target, args.cos_lambda
            )

            grad_norm = direction.grad.norm().item() if direction.grad is not None else 0.0
            optimizer.step()

            with torch.no_grad():
                direction.data.div_(direction.data.norm())

            curr_dir = direction.detach() / direction.detach().norm()
            cos_sim = (curr_dir * prev_dir).sum().item()
            prev_dir = curr_dir.clone()

            history['frob_loss'].append(frob_val)
            history['cos_loss'].append(cos_loss)
            history['cos_dist_mean'].append(cos_dist_mean)
            history['grad_norm'].append(grad_norm)
            history['cosine_sim'].append(cos_sim)

            if (step + 1) % 10 == 0:
                elapsed = time.time() - t0
                steps_done = step - start_step + 1
                rate = steps_done / elapsed
                eta = (args.n_steps - step - 1) / rate if rate > 0 else 0
                frob_est = np.sqrt(frob_val) if frob_val > 0 else 0
                # Convert cos_dist to angle
                cd_angle = np.degrees(np.arccos(np.clip(1.0 - cos_dist_mean, -1, 1)))
                print(f"  Step {step+1}/{args.n_steps}  ||J||_F~{frob_est:.0f}  "
                      f"cos_loss={cos_loss:.1f}  cos_dist={cos_dist_mean:.4f} ({cd_angle:.1f}deg)  "
                      f"grad={grad_norm:.4f}  cos={cos_sim:.4f}  "
                      f"({rate:.1f} st/s, ETA {eta:.0f}s)")

            if (step + 1) % 50 == 0:
                checkpoint['opt_state'] = {
                    'direction': direction.detach().cpu(),
                    'step': step,
                    'history': history,
                    'optimizer_state': optimizer.state_dict(),
                }
                checkpoint['stage'] = 'optimization_in_progress'
                save_checkpoint(checkpoint, checkpoint_path)

        checkpoint['opt_state'] = {
            'direction': direction.detach().cpu(),
            'step': args.n_steps - 1,
            'history': history,
            'optimizer_state': optimizer.state_dict(),
        }
        checkpoint['stage'] = 'optimization_done'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'optimization_done'
        print(f"\nOptimization done in {time.time()-t0:.1f}s")

    # =========================================================
    # Stage 2: Evaluation
    # =========================================================
    if stage == 'optimization_done':
        print(f"\n=== Stage 2: Evaluation ===")

        activations = checkpoint['activations']
        contexts = checkpoint['contexts']
        direction_opt = checkpoint['opt_state']['direction'].to(device)
        direction_opt = direction_opt / direction_opt.norm()
        history = checkpoint['opt_state']['history']
        ref_frob_norms = np.array(checkpoint['ref_frob_norms'])
        random_frob_norms = np.array(checkpoint['random_frob_norms'])
        random_cos_dists = np.array(checkpoint['random_cos_dists'])
        random_kls = np.array(checkpoint['random_kls'])

        # Evaluate on all activations
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

        # Convert to angles for reporting
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

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Top left: ||J||_F histogram
        ax = axes[0, 0]
        all_vals = np.concatenate([ref_frob_norms, random_frob_norms, shifted_frob_norms])
        bins = np.linspace(0, min(all_vals.max() * 1.1, np.percentile(all_vals, 99) * 1.5), 40)
        ax.hist(ref_frob_norms, bins=bins, alpha=0.5, label=f'Original ({ref_frob_norms.mean():.0f})', color='C0')
        ax.hist(shifted_frob_norms, bins=bins, alpha=0.5, label=f'Optimized ({shifted_frob_norms.mean():.0f})', color='C1')
        ax.hist(random_frob_norms, bins=bins, alpha=0.3, label=f'Random ({random_frob_norms.mean():.0f})', color='gray')
        ax.set_xlabel('||J||_F')
        ax.set_title('Jacobian Frobenius norm')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Top middle: cosine distance histogram
        ax = axes[0, 1]
        bins = np.linspace(0, max(shifted_cos_dists.max(), random_cos_dists.max()) * 1.1, 30)
        ax.hist(random_cos_dists, bins=bins, alpha=0.5, label=f'Random ({np.median(rand_angles):.1f}deg)', color='gray')
        ax.hist(shifted_cos_dists, bins=bins, alpha=0.5, label=f'Optimized ({np.median(opt_angles):.1f}deg)', color='C1')
        ax.axvline(args.cos_target, color='red', linestyle='--', label=f'Target ({target_angle:.0f}deg)')
        ax.set_xlabel('Cosine distance (1=orthogonal)')
        ax.set_title(f'Downstream cosine dist at {args.angle} deg')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Top right: direction stability
        ax = axes[0, 2]
        ax.plot(history['cosine_sim'])
        ax.set_xlabel('Step')
        ax.set_ylabel('Cosine similarity')
        ax.set_title('Direction stability')
        ax.grid(True, alpha=0.3)

        # Bottom left: frob loss
        ax = axes[1, 0]
        ax.plot(history['frob_loss'], alpha=0.5)
        window = 20
        if len(history['frob_loss']) > window:
            sm = np.convolve(history['frob_loss'], np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(history['frob_loss'])), sm, color='red', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('||J||_F^2 (batch mean)')
        ax.set_title('Frob minimization')
        ax.grid(True, alpha=0.3)

        # Bottom middle: cosine loss
        ax = axes[1, 1]
        ax.plot(history['cos_loss'], alpha=0.5)
        if len(history['cos_loss']) > window:
            sm = np.convolve(history['cos_loss'], np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(history['cos_loss'])), sm, color='red', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Cosine penalty')
        ax.set_title('Cosine dist floor loss')
        ax.grid(True, alpha=0.3)

        # Bottom right: mean cosine dist over training
        ax = axes[1, 2]
        cd_angles = [np.degrees(np.arccos(np.clip(1.0 - cd, -1, 1))) for cd in history['cos_dist_mean']]
        ax.plot(cd_angles, alpha=0.5)
        if len(cd_angles) > window:
            sm = np.convolve(cd_angles, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(cd_angles)), sm, color='red', linewidth=2)
        ax.axhline(target_angle, color='red', linestyle='--', alpha=0.7, label=f'Target ({target_angle:.0f}deg)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Downstream angle (deg)')
        ax.set_title('Cosine angle during training')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f'v6: Minimize ||J||_F + cos_dist>{args.cos_target} at {args.angle}deg (layer {args.layer})', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(args.results_dir, f'eval_{args.angle}deg.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Save
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
            'history': history,
        }
        results_path = os.path.join(args.results_dir, f'results_{args.angle}deg.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved: {results_path}")

        checkpoint['stage'] = 'done'
        save_checkpoint(checkpoint, checkpoint_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
