#!/usr/bin/env python3
"""
Delta optimization v5: spherical perturbation with KL floor + Jacobian matching.

Find a direction d such that at a fixed angle theta:
  a_perturbed = cos(theta)*a + sin(theta)*||a||*v_perp(d, a)

The perturbed activations have:
  1. KL(original || perturbed) >= kl_min  (we've left the plateau)
  2. ||J||_F at perturbed point ≈ target  (endpoint is "real-like")

Loss = ((||J||_F - target) / target)^2 + lambda * max(0, kl_min - KL)^2

Usage:
    python scripts/jacobian/run_delta_optimization_v5.py
    python scripts/jacobian/run_delta_optimization_v5.py --angle 50 --kl_min 0.5
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


def compute_kl(model, ctx, a_orig, a_pert, layer):
    """KL(p_orig || p_pert) from logits. Differentiable w.r.t. a_pert."""
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


def optimization_step(model, direction, activations, contexts, layer,
                      batch_indices, n_frob_probes, device, angle_rad,
                      kl_min, kl_lambda):
    """One step: minimize ||J||_F^2 + KL floor penalty."""
    d_model = direction.shape[0]
    B = len(batch_indices)
    total_frob_val = 0.0
    total_kl_loss = 0.0
    total_kl_val = 0.0

    for i in batch_indices:
        a_i = activations[i].to(device).detach()
        ctx_i = contexts[i].to(device).detach()

        # --- KL computation (separate forward pass) ---
        d_unit_kl = direction / (direction.norm() + 1e-10)
        a_pert_kl = spherical_perturb(a_i, d_unit_kl, angle_rad)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            kl = compute_kl(model, ctx_i, a_i, a_pert_kl, layer)

        kl_val = kl.item()
        total_kl_val += kl_val / B

        kl_violation = torch.clamp(kl_min - kl, min=0.0)
        kl_loss = kl_lambda * kl_violation ** 2 / B
        kl_loss.backward()
        total_kl_loss += kl_loss.item()

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

        # Minimize ||J||_F^2, normalized by chain-rule inflation
        # d(a_pert)/d(direction) has scale ~||a||*sin(theta), so ||J_chain||^2 ~ ||J_model||^2 * R^2*sin^2
        # Divide out to get ~model-scale values
        R = a_i.norm().item()
        inflation = (R * np.sin(angle_rad)) ** 2
        frob_loss = frob_sq / (inflation * B)
        frob_loss.backward()
        total_frob_val += frob_sq.item() / (inflation * B)

    return total_frob_val, total_kl_loss, total_kl_val


def main():
    parser = argparse.ArgumentParser(description='Delta optimization v5 (spherical, KL floor + Frob match)')
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=100)
    parser.add_argument('--n_ctx', type=int, default=100)
    parser.add_argument('--angle', type=float, default=50,
                        help='Fixed perturbation angle in degrees')
    parser.add_argument('--kl_min', type=float, default=0.5,
                        help='Minimum KL divergence (floor constraint)')
    parser.add_argument('--kl_lambda', type=float, default=10000,
                        help='Penalty weight for KL floor violation')
    parser.add_argument('--n_frob_probes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--results_dir', type=str, default='results/delta_opt_v5')
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
    print(f"  KL floor: {args.kl_min}, lambda: {args.kl_lambda}")
    print(f"  Loss: ||J||_F^2 + lambda*max(0, kl_min - KL)^2")
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
        target_frob = float(ref_frob_norms.mean())
        print(f"Reference ||J||_F: mean={ref_frob_norms.mean():.1f}, std={ref_frob_norms.std():.1f}")
        print(f"Target ||J||_F: {target_frob:.1f}")

        # Baseline: random direction at this angle
        print(f"\nBaseline: random directions at {args.angle} deg...")
        random_frob_norms = []
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
                    kl = compute_kl(model, ctx_i, a_i, a_pert.detach(), args.layer).item()
                random_kls.append(kl)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/20")

        random_frob_norms = np.array(random_frob_norms)
        random_kls = np.array(random_kls)
        print(f"Random @ {args.angle} deg: ||J||_F mean={random_frob_norms.mean():.1f}, "
              f"KL mean={random_kls.mean():.3f}, median={np.median(random_kls):.3f}")

        checkpoint.update({
            'activations': activations,
            'contexts': contexts,
            'ref_frob_norms': ref_frob_norms,
            'target_frob': target_frob,
            'random_frob_norms': random_frob_norms,
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
        target_frob = checkpoint['target_frob']
        target_frob = checkpoint['target_frob']
        print(f"  Ref ||J||_F: {target_frob:.1f}, KL floor: {args.kl_min}, lambda: {args.kl_lambda}")

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
            history = {'frob_loss': [], 'kl_loss': [], 'kl_mean': [],
                       'grad_norm': [], 'cosine_sim': []}

        optimizer = torch.optim.Adam([direction], lr=args.lr)
        if opt_state is not None and 'optimizer_state' in opt_state:
            optimizer.load_state_dict(opt_state['optimizer_state'])

        prev_dir = direction.detach().clone() / direction.detach().norm()

        t0 = time.time()
        for step in range(start_step, args.n_steps):
            batch_indices = np.random.choice(args.n_act, args.batch_size, replace=False)

            optimizer.zero_grad()
            frob_val, kl_loss, kl_mean = optimization_step(
                model, direction, activations, contexts, args.layer,
                batch_indices, args.n_frob_probes, device, angle_rad,
                args.kl_min, args.kl_lambda
            )

            grad_norm = direction.grad.norm().item() if direction.grad is not None else 0.0
            optimizer.step()

            with torch.no_grad():
                direction.data.div_(direction.data.norm())

            curr_dir = direction.detach() / direction.detach().norm()
            cos_sim = (curr_dir * prev_dir).sum().item()
            prev_dir = curr_dir.clone()

            history['frob_loss'].append(frob_val)
            history['kl_loss'].append(kl_loss)
            history['kl_mean'].append(kl_mean)
            history['grad_norm'].append(grad_norm)
            history['cosine_sim'].append(cos_sim)

            if (step + 1) % 10 == 0:
                elapsed = time.time() - t0
                steps_done = step - start_step + 1
                rate = steps_done / elapsed
                eta = (args.n_steps - step - 1) / rate if rate > 0 else 0
                frob_est = np.sqrt(frob_val) if frob_val > 0 else 0
                print(f"  Step {step+1}/{args.n_steps}  ||J||_F~{frob_est:.0f}  "
                      f"kl_loss={kl_loss:.1f}  KL={kl_mean:.3f}  "
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
        target_frob = checkpoint['target_frob']
        random_frob_norms = np.array(checkpoint['random_frob_norms'])
        random_kls = np.array(checkpoint['random_kls'])

        # Evaluate on all activations
        shifted_frob_norms = []
        shifted_kls = []
        t0 = time.time()
        for i in range(args.n_act):
            a_i = activations[i].to(device)
            ctx_i = contexts[i].to(device)
            a_pert = spherical_perturb(a_i, direction_opt, angle_rad).detach()

            frob, _ = compute_jacobian_norm_vjp(model, ctx_i, a_pert, args.layer, n_probes=50)
            shifted_frob_norms.append(frob)

            with torch.no_grad():
                kl = compute_kl(model, ctx_i, a_i, a_pert, args.layer).item()
            shifted_kls.append(kl)

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{args.n_act}  ||J||_F={frob:.1f}  KL={kl:.3f}  "
                      f"({(i+1)/(time.time()-t0):.1f}/s)")

        shifted_frob_norms = np.array(shifted_frob_norms)
        shifted_kls = np.array(shifted_kls)

        print(f"\nResults:")
        print(f"  ||J||_F original:       mean={ref_frob_norms.mean():.1f}, std={ref_frob_norms.std():.1f}")
        print(f"  ||J||_F random@{args.angle}deg:   mean={random_frob_norms.mean():.1f}, std={random_frob_norms.std():.1f}")
        print(f"  ||J||_F optimized@{args.angle}deg: mean={shifted_frob_norms.mean():.1f}, std={shifted_frob_norms.std():.1f}")
        print(f"  KL random@{args.angle}deg:    mean={random_kls.mean():.3f}, median={np.median(random_kls):.3f}")
        print(f"  KL optimized@{args.angle}deg:  mean={shifted_kls.mean():.3f}, median={np.median(shifted_kls):.3f}")
        kl_above = (shifted_kls >= args.kl_min).sum()
        print(f"  KL >= {args.kl_min}: {kl_above}/{len(shifted_kls)}")

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
        ax.axvline(target_frob, color='C0', linestyle='--', alpha=0.7)
        ax.set_xlabel('||J||_F')
        ax.set_title('Jacobian Frobenius norm')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Top middle: KL histogram
        ax = axes[0, 1]
        bins = np.linspace(0, max(shifted_kls.max(), random_kls.max()) * 1.1, 30)
        ax.hist(random_kls, bins=bins, alpha=0.5, label=f'Random ({np.median(random_kls):.2f})', color='gray')
        ax.hist(shifted_kls, bins=bins, alpha=0.5, label=f'Optimized ({np.median(shifted_kls):.2f})', color='C1')
        ax.axvline(args.kl_min, color='red', linestyle='--', label=f'KL floor ({args.kl_min})')
        ax.set_xlabel('KL divergence')
        ax.set_title(f'KL at {args.angle} deg')
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

        # Bottom middle: KL loss
        ax = axes[1, 1]
        ax.plot(history['kl_loss'], alpha=0.5)
        if len(history['kl_loss']) > window:
            sm = np.convolve(history['kl_loss'], np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(history['kl_loss'])), sm, color='red', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('KL penalty')
        ax.set_title('KL floor loss')
        ax.grid(True, alpha=0.3)

        # Bottom right: mean KL over training
        ax = axes[1, 2]
        ax.plot(history['kl_mean'], alpha=0.5)
        if len(history['kl_mean']) > window:
            sm = np.convolve(history['kl_mean'], np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(history['kl_mean'])), sm, color='red', linewidth=2)
        ax.axhline(args.kl_min, color='red', linestyle='--', alpha=0.7, label=f'KL floor ({args.kl_min})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean KL')
        ax.set_title('KL during training')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f'v5: Minimize ||J||_F + KL>{args.kl_min} at {args.angle}° (layer {args.layer})', fontsize=12)
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
            'target_frob': target_frob,
            'random_frob_norms': random_frob_norms,
            'random_kls': random_kls,
            'shifted_frob_norms': shifted_frob_norms,
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
