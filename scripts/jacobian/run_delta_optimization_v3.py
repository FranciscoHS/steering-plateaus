#!/usr/bin/env python3
"""
Delta optimization v3: match BOTH top-1 singular value AND Frobenius norm.

Loss: (sigma_1(J) - target_sigma1)^2 + (||J||_F - target_frob)^2

sigma_1 via power iteration + Hutchinson probe.
||J||_F via standard Hutchinson estimator (same as v2).

Usage:
    python scripts/jacobian/run_delta_optimization_v3.py
    python scripts/jacobian/run_delta_optimization_v3.py --n_steps 200 --power_iters 10
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
from src.analysis import compute_jacobian_norm_vjp, compute_top_singular_values


def save_checkpoint(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Checkpoint loaded: {path}")
    return data


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


def optimization_step(model, delta, activations, contexts, layer, batch_indices,
                      power_iters, n_frob_probes, device, target_sigma1, target_frob):
    """
    One optimization step matching BOTH top-1 SV and Frobenius norm.

    Loss = (sigma_1_est - target_sigma1)^2 + (||J||_F_est - target_frob)^2
    """
    d_model = delta.shape[0]
    B = len(batch_indices)
    total_loss = 0.0
    total_sv_loss = 0.0
    total_frob_loss = 0.0

    for i in batch_indices:
        a_i = activations[i].to(device).detach()
        ctx_i = contexts[i].to(device).detach()

        point = (a_i + delta).reshape(1, 1, d_model)

        # Forward pass (through delta's graph)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            final_act = forward_from_layer_to_final(model, ctx_i, point, layer)

        # === Part 1: Frobenius norm via Hutchinson ===
        frob_sq = torch.tensor(0.0, device=device)
        for k in range(n_frob_probes):
            u = torch.randn(1, d_model, device=device)
            s = (final_act * u).sum()
            g = torch.autograd.grad(s, delta, create_graph=True, retain_graph=True)[0]
            frob_sq = frob_sq + (g ** 2).sum()
        frob_sq = frob_sq / n_frob_probes
        frob_est = torch.sqrt(frob_sq + 1e-8)
        frob_loss = (frob_est - target_frob) ** 2 / B

        # === Part 2: Top-1 SV via power iteration ===
        # Power iteration to find top right singular vector (detached from graph)
        v = torch.randn(d_model, device=device)
        v = v / v.norm()

        for pi in range(power_iters):
            v_det = v.detach()

            # Jv via finite differences
            eps = 1e-4
            with torch.no_grad():
                point_pert = (a_i + delta.detach() + eps * v_det).reshape(1, 1, d_model)
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    final_pert = forward_from_layer_to_final(model, ctx_i, point_pert, layer)
                Jv = (final_pert - final_act.detach()).squeeze() / eps

            # J^T(Jv) via backprop
            point_g = (a_i + delta.detach()).reshape(1, 1, d_model).requires_grad_(True)
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                final_g = forward_from_layer_to_final(model, ctx_i, point_g, layer)
            (final_g.squeeze() * Jv.detach()).sum().backward()
            JtJv = point_g.grad.reshape(-1)

            v = JtJv / (JtJv.norm() + 1e-10)

        # Differentiable sigma_1 estimate via eps-trick + Hutchinson
        v_final = v.detach()
        eps_param = torch.zeros(1, device=device, requires_grad=True)
        point_param = (a_i + delta + eps_param * v_final).reshape(1, 1, d_model)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            final_param = forward_from_layer_to_final(model, ctx_i, point_param, layer)

        # ||Jv||^2 via Hutchinson: E_u[(Jv . u)^2]
        n_sv_probes = 5
        sigma1_sq = torch.tensor(0.0, device=device)
        for _ in range(n_sv_probes):
            u = torch.randn(1, d_model, device=device)
            s = (final_param * u).sum()
            ds_deps = torch.autograd.grad(s, eps_param, create_graph=True, retain_graph=True)[0]
            sigma1_sq = sigma1_sq + ds_deps ** 2
        sigma1_sq = sigma1_sq / n_sv_probes

        sigma1_est = torch.sqrt(sigma1_sq + 1e-8)
        sv_loss = (sigma1_est - target_sigma1) ** 2 / B

        # Combined loss
        act_loss = frob_loss + sv_loss
        act_loss.backward()
        total_loss += act_loss.item()
        total_sv_loss += sv_loss.item()
        total_frob_loss += frob_loss.item()

    return total_loss, total_sv_loss, total_frob_loss


def main():
    parser = argparse.ArgumentParser(description='Delta optimization v3 (match top-1 SV + Frobenius)')
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=100)
    parser.add_argument('--n_ctx', type=int, default=100)
    parser.add_argument('--power_iters', type=int, default=5,
                        help='Power iterations per step for top-1 SV')
    parser.add_argument('--n_frob_probes', type=int, default=5,
                        help='Hutchinson probes for Frobenius norm')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--r', type=float, default=0,
                        help='Delta norm (0 = auto from mean pairwise distance)')
    parser.add_argument('--results_dir', type=str, default='results/delta_opt_v3')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.results_dir, 'checkpoint.pkl')

    d_model = {'gpt2': 768, 'gpt2-medium': 1024, 'gpt2-large': 1280, 'gpt2-xl': 1600}[args.model]

    print(f"Configuration:")
    print(f"  Model: {args.model} (d_model={d_model})")
    print(f"  Layer: {args.layer}")
    print(f"  Loss: (sigma_1 - target)^2 + (||J||_F - target)^2  [JOINT MATCHING]")
    print(f"  Power iterations: {args.power_iters}, Frob probes: {args.n_frob_probes}")
    print(f"  Optimization: B={args.batch_size}, lr={args.lr}, steps={args.n_steps}")
    print(f"  Delta norm r: {'auto' if args.r == 0 else args.r}")
    print("-" * 60)

    # Resume logic
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path)
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
        t0 = time.time()

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

        dists = torch.cdist(activations.unsqueeze(0), activations.unsqueeze(0)).squeeze(0)
        mask = torch.triu(torch.ones(args.n_act, args.n_act, dtype=torch.bool), diagonal=1)
        pairwise_dists = dists[mask].numpy()
        print(f"Pairwise L2: mean={pairwise_dists.mean():.1f}, median={np.median(pairwise_dists):.1f}")

        r = args.r if args.r > 0 else pairwise_dists.mean()
        print(f"Using r = {r:.1f}")

        checkpoint['activations'] = activations
        checkpoint['contexts'] = contexts
        checkpoint['pairwise_dists'] = pairwise_dists
        checkpoint['r'] = r
        checkpoint['stage'] = 'activations_ready'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'activations_ready'
        del token_ids_list

    # =========================================================
    # Stage 1: Reference sigma_1 and ||J||_F (cross-context)
    # =========================================================
    if stage in ['activations_ready', 'ref_sv_in_progress']:
        print(f"\n=== Stage 1: Reference sigma_1 + ||J||_F ===")

        activations = checkpoint['activations']
        contexts = checkpoint['contexts']

        ref_sigma1s = checkpoint.get('ref_sigma1s', [])
        ref_frob_norms = checkpoint.get('ref_frob_norms', [])
        done = len(ref_sigma1s)

        t0 = time.time()
        for i in range(done, args.n_act):
            a_i = activations[i].to(device)
            ctx_i = contexts[i].to(device)

            point = a_i.reshape(1, 1, -1)
            svs, _ = compute_top_singular_values(model, ctx_i, point, layer=args.layer, k=5)
            ref_sigma1s.append(float(svs[0]))

            frob_norm, _ = compute_jacobian_norm_vjp(
                model, ctx_i, a_i, args.layer, n_probes=50
            )
            ref_frob_norms.append(frob_norm)

            elapsed = time.time() - t0
            rate = (i - done + 1) / elapsed if elapsed > 0 else 0
            print(f"  {i+1}/{args.n_act}  sigma1={svs[0]:.1f}  ||J||_F={frob_norm:.1f}  "
                  f"({rate:.2f}/s)")

            if (i + 1) % 10 == 0:
                checkpoint['ref_sigma1s'] = ref_sigma1s
                checkpoint['ref_frob_norms'] = ref_frob_norms
                checkpoint['stage'] = 'ref_sv_in_progress'
                save_checkpoint(checkpoint, checkpoint_path)

        sigma1_arr = np.array(ref_sigma1s)
        frob_arr = np.array(ref_frob_norms)
        target_sigma1 = float(sigma1_arr.mean())
        target_frob = float(frob_arr.mean())
        print(f"\nReference sigma_1: mean={sigma1_arr.mean():.1f}, std={sigma1_arr.std():.1f}")
        print(f"Reference ||J||_F: mean={frob_arr.mean():.1f}, std={frob_arr.std():.1f}")
        print(f"Targets: sigma_1={target_sigma1:.1f}, ||J||_F={target_frob:.1f}")

        checkpoint['ref_sigma1s'] = ref_sigma1s
        checkpoint['ref_frob_norms'] = ref_frob_norms
        checkpoint['target_sigma1'] = target_sigma1
        checkpoint['target_frob'] = target_frob
        checkpoint['stage'] = 'ref_sv_done'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'ref_sv_done'

    # =========================================================
    # Stage 2: Optimization (match both)
    # =========================================================
    if stage in ['ref_sv_done', 'optimization_in_progress']:
        print(f"\n=== Stage 2: Delta optimization (joint matching) ===")

        activations = checkpoint['activations']
        contexts = checkpoint['contexts']
        r = checkpoint['r']
        target_sigma1 = checkpoint['target_sigma1']
        target_frob = checkpoint['target_frob']
        print(f"  r = {r:.1f}, target sigma_1 = {target_sigma1:.1f}, target ||J||_F = {target_frob:.1f}")

        opt_state = checkpoint.get('opt_state', None)
        if opt_state is not None:
            delta = opt_state['delta'].to(device).requires_grad_(True)
            start_step = opt_state['step'] + 1
            history = opt_state['history']
            print(f"Resuming from step {start_step}")
        else:
            delta_init = torch.randn(d_model, device=device)
            delta_init = delta_init / delta_init.norm() * r
            delta = delta_init.requires_grad_(True)
            start_step = 0
            history = {'loss': [], 'sv_loss': [], 'frob_loss': [],
                       'grad_norm': [], 'cosine_sim': []}

        optimizer = torch.optim.Adam([delta], lr=args.lr)
        if opt_state is not None and 'optimizer_state' in opt_state:
            optimizer.load_state_dict(opt_state['optimizer_state'])

        prev_direction = delta.detach().clone() / delta.detach().norm()

        t0 = time.time()
        for step in range(start_step, args.n_steps):
            batch_indices = np.random.choice(args.n_act, args.batch_size, replace=False)

            optimizer.zero_grad()
            loss_val, sv_loss, frob_loss = optimization_step(
                model, delta, activations, contexts, args.layer,
                batch_indices, args.power_iters, args.n_frob_probes,
                device, target_sigma1, target_frob
            )

            grad_norm = delta.grad.norm().item() if delta.grad is not None else 0.0
            optimizer.step()

            with torch.no_grad():
                delta.data.mul_(r / delta.data.norm())

            curr_direction = delta.detach() / delta.detach().norm()
            cos_sim = (curr_direction * prev_direction).sum().item()
            prev_direction = curr_direction.clone()

            history['loss'].append(loss_val)
            history['sv_loss'].append(sv_loss)
            history['frob_loss'].append(frob_loss)
            history['grad_norm'].append(grad_norm)
            history['cosine_sim'].append(cos_sim)

            if (step + 1) % 10 == 0:
                elapsed = time.time() - t0
                steps_done = step - start_step + 1
                rate = steps_done / elapsed
                eta = (args.n_steps - step - 1) / rate if rate > 0 else 0
                print(f"  Step {step+1}/{args.n_steps}  loss={loss_val:.1f} "
                      f"(sv={sv_loss:.1f} frob={frob_loss:.1f})  "
                      f"grad={grad_norm:.4f}  cos={cos_sim:.4f}  "
                      f"({rate:.1f} steps/s, ETA {eta:.0f}s)")

            if (step + 1) % 50 == 0:
                checkpoint['opt_state'] = {
                    'delta': delta.detach().cpu(),
                    'step': step,
                    'history': history,
                    'optimizer_state': optimizer.state_dict(),
                }
                checkpoint['stage'] = 'optimization_in_progress'
                save_checkpoint(checkpoint, checkpoint_path)

        checkpoint['opt_state'] = {
            'delta': delta.detach().cpu(),
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
    # Stage 3: Evaluation
    # =========================================================
    if stage == 'optimization_done':
        print(f"\n=== Stage 3: Evaluation ===")

        activations = checkpoint['activations']
        contexts = checkpoint['contexts']
        delta_opt = checkpoint['opt_state']['delta'].to(device)
        history = checkpoint['opt_state']['history']
        ref_sigma1s = np.array(checkpoint['ref_sigma1s'])
        ref_frob_norms = np.array(checkpoint['ref_frob_norms'])
        r = checkpoint['r']
        target_sigma1 = checkpoint['target_sigma1']
        target_frob = checkpoint['target_frob']

        print(f"  Delta norm: {delta_opt.norm().item():.1f} (target r={r:.1f})")
        print(f"  Targets: sigma_1={target_sigma1:.1f}, ||J||_F={target_frob:.1f}")

        shifted_sigma1s = []
        shifted_frob_norms = []
        t0 = time.time()
        for i in range(args.n_act):
            shifted = (activations[i].to(device) + delta_opt).detach()
            ctx_i = contexts[i].to(device)

            point = shifted.reshape(1, 1, -1)
            svs, _ = compute_top_singular_values(model, ctx_i, point, layer=args.layer, k=5)
            shifted_sigma1s.append(float(svs[0]))

            frob_norm, _ = compute_jacobian_norm_vjp(
                model, ctx_i, shifted, args.layer, n_probes=50
            )
            shifted_frob_norms.append(frob_norm)

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  {i+1}/{args.n_act}  sigma1={svs[0]:.1f}  ||J||_F={frob_norm:.1f}  "
                  f"({rate:.2f}/s)")

        shifted_sigma1s = np.array(shifted_sigma1s)
        shifted_frob_norms = np.array(shifted_frob_norms)

        print(f"\nResults:")
        print(f"  sigma_1 — Real: mean={ref_sigma1s.mean():.1f}, std={ref_sigma1s.std():.1f}")
        print(f"  sigma_1 — Shifted: mean={shifted_sigma1s.mean():.1f}, std={shifted_sigma1s.std():.1f}")
        print(f"  ||J||_F — Real: mean={ref_frob_norms.mean():.1f}, std={ref_frob_norms.std():.1f}")
        print(f"  ||J||_F — Shifted: mean={shifted_frob_norms.mean():.1f}, std={shifted_frob_norms.std():.1f}")

        # Plot
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Top left: sigma_1 histogram
        ax = axes[0, 0]
        bins = np.linspace(0, max(ref_sigma1s.max(), shifted_sigma1s.max()) * 1.1, 30)
        ax.hist(ref_sigma1s, bins=bins, alpha=0.6, label='Real', color='C0', edgecolor='black', linewidth=0.5)
        ax.hist(shifted_sigma1s, bins=bins, alpha=0.6, label='Shifted', color='C1', edgecolor='black', linewidth=0.5)
        ax.axvline(ref_sigma1s.mean(), color='C0', linestyle='--')
        ax.axvline(shifted_sigma1s.mean(), color='C1', linestyle='--')
        ax.set_xlabel('sigma_1')
        ax.set_title('Top-1 singular value (optimized)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Top middle: ||J||_F histogram
        ax = axes[0, 1]
        bins = np.linspace(0, max(ref_frob_norms.max(), shifted_frob_norms.max()) * 1.1, 30)
        ax.hist(ref_frob_norms, bins=bins, alpha=0.6, label='Real', color='C0', edgecolor='black', linewidth=0.5)
        ax.hist(shifted_frob_norms, bins=bins, alpha=0.6, label='Shifted', color='C1', edgecolor='black', linewidth=0.5)
        ax.axvline(ref_frob_norms.mean(), color='C0', linestyle='--')
        ax.axvline(shifted_frob_norms.mean(), color='C1', linestyle='--')
        ax.set_xlabel('||J||_F')
        ax.set_title('Frobenius norm (optimized)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Top right: direction stability
        ax = axes[0, 2]
        ax.plot(history['cosine_sim'])
        ax.set_xlabel('Step')
        ax.set_ylabel('Cosine similarity')
        ax.set_title('Direction stability')
        ax.grid(True, alpha=0.3)

        # Bottom left: total loss
        ax = axes[1, 0]
        ax.plot(history['loss'], alpha=0.5, label='Total')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Total loss')
        ax.grid(True, alpha=0.3)

        # Bottom middle: sv loss
        ax = axes[1, 1]
        ax.plot(history['sv_loss'], color='C1', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('(sigma_1 - target)^2')
        ax.set_title('SV loss')
        ax.grid(True, alpha=0.3)

        # Bottom right: frob loss
        ax = axes[1, 2]
        ax.plot(history['frob_loss'], color='C2', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('(||J||_F - target)^2')
        ax.set_title('Frobenius loss')
        ax.grid(True, alpha=0.3)

        fig.suptitle(f'v3: Match sigma_1 + ||J||_F (layer {args.layer}, r={r:.0f})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(args.results_dir, f'eval_r{r:.0f}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Save results
        results = {
            'args': vars(args),
            'r': r,
            'target_sigma1': target_sigma1,
            'target_frob': target_frob,
            'delta': delta_opt.cpu(),
            'ref_sigma1s': ref_sigma1s,
            'shifted_sigma1s': shifted_sigma1s,
            'ref_frob_norms': ref_frob_norms,
            'shifted_frob_norms': shifted_frob_norms,
            'history': history,
        }
        results_path = os.path.join(args.results_dir, f'results_r{r:.0f}.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved: {results_path}")

        checkpoint['stage'] = 'done'
        save_checkpoint(checkpoint, checkpoint_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
