#!/usr/bin/env python3
"""
Delta optimization v4: spherical perturbation at fixed angle.

Parameterize a unit direction d in R^d_model. For each activation a_i:
  v_perp = d - (d . a_hat) * a_hat, normalized
  a_perturbed = cos(theta) * a + sin(theta) * ||a|| * v_perp

Minimize mean ||J||_F at the perturbed points.

This ensures we're perturbing by a fixed angle (e.g. 50 deg), so we're
definitely off the plateau, and asks: can the optimizer find a direction
that minimizes the Jacobian norm out there?

Usage:
    python scripts/delta_optimization/run_delta_optimization_v4.py
    python scripts/delta_optimization/run_delta_optimization_v4.py --angle 50 --n_steps 300
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model, forward_from_layer_to_final, _get_blocks
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

        if (i // batch_size) % 10 == 0:
            print(f"  Extracted {batch_end}/{total}", end='\r')

    print()

    all_last_token = torch.cat(all_last_token, dim=0)
    all_prefix_list = []
    for batch_prefix in all_prefix:
        for j in range(batch_prefix.shape[0]):
            all_prefix_list.append(batch_prefix[j:j+1])

    activations = all_last_token[:n_act]
    contexts = all_prefix_list[n_act:n_act + n_ctx]

    return activations, contexts


def spherical_perturb(a, direction, angle_rad):
    """Spherical perturbation: a_perturbed = cos(theta)*a + sin(theta)*||a||*v_perp.

    direction is projected orthogonal to a and normalized to get v_perp.
    Returns a_perturbed (differentiable w.r.t. direction).
    """
    R = a.norm()
    a_hat = a / R

    # Project direction orthogonal to a
    proj = (direction * a_hat).sum()
    v_perp = direction - proj * a_hat
    v_perp = v_perp / (v_perp.norm() + 1e-10)

    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    return cos_t * a + sin_t * R * v_perp


def optimization_step(model, direction, activations, contexts, layer,
                      batch_indices, n_frob_probes, device, angle_rad):
    """One optimization step: minimize ||J||_F at spherically-perturbed points.

    We minimize ||J||_F^2 (equivalent minimizer to ||J||_F, avoids sqrt in gradient).
    Each activation gets one forward pass; Hutchinson probes reuse the graph.
    """
    d_model = direction.shape[0]
    B = len(batch_indices)
    total_loss = 0.0

    for i in batch_indices:
        a_i = activations[i].to(device).detach()
        ctx_i = contexts[i].to(device).detach()

        # Recompute d_unit per activation so the graph stays alive
        d_unit = direction / (direction.norm() + 1e-10)
        a_pert = spherical_perturb(a_i, d_unit, angle_rad)
        point = a_pert.reshape(1, 1, d_model)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            final_act = forward_from_layer_to_final(model, ctx_i, point, layer)

        # ||J||_F^2 via Hutchinson (all probes share one forward pass)
        frob_sq = torch.tensor(0.0, device=device)
        for k in range(n_frob_probes):
            u = torch.randn(1, d_model, device=device)
            s = (final_act * u).sum()
            g = torch.autograd.grad(s, direction, create_graph=True, retain_graph=True)[0]
            frob_sq = frob_sq + (g ** 2).sum()
        frob_sq = frob_sq / n_frob_probes

        loss = frob_sq / B
        loss.backward()
        total_loss += frob_sq.item() / B

    return np.sqrt(total_loss)  # return ||J||_F estimate for logging


def main():
    parser = argparse.ArgumentParser(description='Delta optimization v4 (spherical, minimize ||J||_F)')
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=100)
    parser.add_argument('--n_ctx', type=int, default=100)
    parser.add_argument('--angle', type=float, default=50,
                        help='Fixed perturbation angle in degrees')
    parser.add_argument('--n_frob_probes', type=int, default=10,
                        help='Hutchinson probes for Frobenius norm')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--results_dir', type=str, default='results/delta_opt_v4')
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
    print(f"  Angle: {args.angle} deg ({angle_rad:.4f} rad)")
    print(f"  Loss: minimize ||J||_F at spherically-perturbed points")
    print(f"  Frob probes: {args.n_frob_probes}")
    print(f"  Optimization: B={args.batch_size}, lr={args.lr}, steps={args.n_steps}")
    print("-" * 60)

    # Resume logic
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        stage = checkpoint.get('stage', 'start')
        print(f"Resuming from stage: {stage}")
    else:
        checkpoint = {'stage': 'start', 'args': vars(args)}
        stage = 'start'

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    print(f"Device: {device}")

    # =========================================================
    # Stage 0: Extract activations
    # =========================================================
    if stage == 'start':
        print(f"\n=== Stage 0: Extracting activations ===")

        n_total = args.n_act + args.n_ctx
        token_ids_list = load_fineweb_fixed_length(
            n_total, tokenizer, seq_len=args.seq_len, seed=args.seed
        )

        print(f"Extracting activations at layer {args.layer}...")
        activations, contexts = extract_split_activations(
            model, tokenizer, token_ids_list, args.layer, device,
            args.n_act, args.n_ctx
        )
        print(f"Activations shape: {activations.shape}")
        print(f"Mean activation norm: {activations.norm(dim=-1).mean():.1f}")

        # Reference: ||J||_F at the original (unperturbed) activations
        print(f"\nComputing reference ||J||_F at original activations...")
        ref_frob_norms = []
        t0 = time.time()
        for i in range(args.n_act):
            a_i = activations[i].to(device)
            ctx_i = contexts[i].to(device)
            frob_norm, _ = compute_jacobian_norm_vjp(
                model, ctx_i, a_i, args.layer, n_probes=50
            )
            ref_frob_norms.append(frob_norm)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{args.n_act}  ({(i+1)/(time.time()-t0):.1f}/s)")
        ref_frob_norms = np.array(ref_frob_norms)
        print(f"Reference ||J||_F: mean={ref_frob_norms.mean():.1f}, std={ref_frob_norms.std():.1f}")

        # Also measure ||J||_F at random perturbation directions (baseline)
        print(f"\nComputing ||J||_F at {args.angle} deg along random directions (baseline)...")
        random_frob_norms = []
        n_random_dirs = 5
        t0 = time.time()
        for i in range(min(20, args.n_act)):
            a_i = activations[i].to(device)
            ctx_i = contexts[i].to(device)
            for _ in range(n_random_dirs):
                rand_dir = torch.randn(d_model, device=device)
                rand_dir = rand_dir / rand_dir.norm()
                a_pert = spherical_perturb(a_i, rand_dir, angle_rad)
                frob_norm, _ = compute_jacobian_norm_vjp(
                    model, ctx_i, a_pert.detach(), args.layer, n_probes=50
                )
                random_frob_norms.append(frob_norm)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/20")
        random_frob_norms = np.array(random_frob_norms)
        print(f"Random-direction ||J||_F at {args.angle} deg: mean={random_frob_norms.mean():.1f}, "
              f"std={random_frob_norms.std():.1f}")

        checkpoint['activations'] = activations
        checkpoint['contexts'] = contexts
        checkpoint['ref_frob_norms'] = ref_frob_norms
        checkpoint['random_frob_norms'] = random_frob_norms
        checkpoint['stage'] = 'activations_ready'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'activations_ready'
        del token_ids_list

    # =========================================================
    # Stage 1: Optimization
    # =========================================================
    if stage in ['activations_ready', 'optimization_in_progress']:
        print(f"\n=== Stage 1: Direction optimization (minimize ||J||_F at {args.angle} deg) ===")

        activations = checkpoint['activations']
        contexts = checkpoint['contexts']
        ref_frob = np.array(checkpoint['ref_frob_norms'])
        random_frob = np.array(checkpoint['random_frob_norms'])
        print(f"  Reference ||J||_F at original: mean={ref_frob.mean():.1f}")
        print(f"  Reference ||J||_F at {args.angle} deg random: mean={random_frob.mean():.1f}")

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
            history = {'loss': [], 'grad_norm': [], 'cosine_sim': []}

        optimizer = torch.optim.Adam([direction], lr=args.lr)
        if opt_state is not None and 'optimizer_state' in opt_state:
            optimizer.load_state_dict(opt_state['optimizer_state'])

        prev_dir = direction.detach().clone() / direction.detach().norm()

        t0 = time.time()
        for step in range(start_step, args.n_steps):
            batch_indices = np.random.choice(args.n_act, args.batch_size, replace=False)

            optimizer.zero_grad()
            loss_val = optimization_step(
                model, direction, activations, contexts, args.layer,
                batch_indices, args.n_frob_probes, device, angle_rad
            )

            grad_norm = direction.grad.norm().item() if direction.grad is not None else 0.0
            optimizer.step()

            # Re-normalize direction to unit sphere
            with torch.no_grad():
                direction.data.div_(direction.data.norm())

            curr_dir = direction.detach() / direction.detach().norm()
            cos_sim = (curr_dir * prev_dir).sum().item()
            prev_dir = curr_dir.clone()

            history['loss'].append(loss_val)
            history['grad_norm'].append(grad_norm)
            history['cosine_sim'].append(cos_sim)

            if (step + 1) % 10 == 0:
                elapsed = time.time() - t0
                steps_done = step - start_step + 1
                rate = steps_done / elapsed
                eta = (args.n_steps - step - 1) / rate if rate > 0 else 0
                print(f"  Step {step+1}/{args.n_steps}  ||J||_F={loss_val:.1f}  "
                      f"grad={grad_norm:.4f}  cos={cos_sim:.4f}  "
                      f"({rate:.1f} steps/s, ETA {eta:.0f}s)")

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

        total_time = time.time() - t0
        print(f"\nOptimization done in {total_time:.1f}s ({total_time/60:.1f} min)")

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

        # Evaluate ||J||_F at optimized direction for all activations
        shifted_frob_norms = []
        t0 = time.time()
        for i in range(args.n_act):
            a_i = activations[i].to(device)
            ctx_i = contexts[i].to(device)
            a_pert = spherical_perturb(a_i, direction_opt, angle_rad).detach()
            frob_norm, _ = compute_jacobian_norm_vjp(
                model, ctx_i, a_pert, args.layer, n_probes=50
            )
            shifted_frob_norms.append(frob_norm)
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{args.n_act}  ||J||_F={frob_norm:.1f}  ({(i+1)/elapsed:.1f}/s)")

        shifted_frob_norms = np.array(shifted_frob_norms)

        print(f"\nResults:")
        print(f"  ||J||_F at original activations:     mean={ref_frob_norms.mean():.1f}, std={ref_frob_norms.std():.1f}")
        print(f"  ||J||_F at {args.angle} deg random dir:     mean={random_frob_norms.mean():.1f}, std={random_frob_norms.std():.1f}")
        print(f"  ||J||_F at {args.angle} deg optimized dir:  mean={shifted_frob_norms.mean():.1f}, std={shifted_frob_norms.std():.1f}")
        print(f"  Reduction vs random: {(1 - shifted_frob_norms.mean() / random_frob_norms.mean()) * 100:.1f}%")

        # Plot
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Left: ||J||_F histogram
        ax = axes[0]
        all_vals = np.concatenate([ref_frob_norms, random_frob_norms, shifted_frob_norms])
        bins = np.linspace(all_vals.min() * 0.8, all_vals.max() * 1.1, 30)
        ax.hist(ref_frob_norms, bins=bins, alpha=0.5, label=f'Original (mean={ref_frob_norms.mean():.1f})',
                color='C0', edgecolor='black', linewidth=0.5)
        ax.hist(random_frob_norms, bins=bins, alpha=0.5, label=f'Random @ {args.angle}° (mean={random_frob_norms.mean():.1f})',
                color='gray', edgecolor='black', linewidth=0.5)
        ax.hist(shifted_frob_norms, bins=bins, alpha=0.5, label=f'Optimized @ {args.angle}° (mean={shifted_frob_norms.mean():.1f})',
                color='C1', edgecolor='black', linewidth=0.5)
        ax.set_xlabel('||J||_F')
        ax.set_title('Jacobian Frobenius norm')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Middle: loss curve
        ax = axes[1]
        ax.plot(history['loss'], alpha=0.5)
        # Smoothed
        window = 20
        if len(history['loss']) > window:
            smoothed = np.convolve(history['loss'], np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(history['loss'])), smoothed, color='red', linewidth=2)
        ax.axhline(ref_frob_norms.mean(), color='C0', linestyle='--', label=f'Original mean ({ref_frob_norms.mean():.1f})')
        ax.axhline(random_frob_norms.mean(), color='gray', linestyle='--', label=f'Random mean ({random_frob_norms.mean():.1f})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean ||J||_F (batch)')
        ax.set_title('Loss curve')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: direction stability
        ax = axes[2]
        ax.plot(history['cosine_sim'])
        ax.set_xlabel('Step')
        ax.set_ylabel('Cosine similarity (step to step)')
        ax.set_title('Direction stability')
        ax.grid(True, alpha=0.3)

        fig.suptitle(f'v4: Minimize ||J||_F at {args.angle}° spherical perturbation (layer {args.layer})', fontsize=12)
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
            'shifted_frob_norms': shifted_frob_norms,
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
