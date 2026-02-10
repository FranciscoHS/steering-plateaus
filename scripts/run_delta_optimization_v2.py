#!/usr/bin/env python3
"""
Delta optimization v2: find a vector delta such that a_i + delta has MATCHING
Jacobian norm to real activations (not just minimized).

Loss: mean_i (||J(a_i + delta)||_F - target_norm)^2
where target_norm = mean ||J(a_i)||_F from reference measurements.

Otherwise identical to v1 — projected gradient descent on sphere of radius r,
cross-context evaluation.

Usage:
    python scripts/run_delta_optimization_v2.py
    python scripts/run_delta_optimization_v2.py --n_steps 200
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model, forward_from_layer_to_final
from src.data import load_fineweb_fixed_length
from src.analysis import compute_jacobian_norm_vjp
from src.visualization import plot_three_way_comparison


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
    """
    Extract activations and split into activation set + context set.
    Uses separate FineWeb sequences for each set (cross-context evaluation).
    """
    total = n_act + n_ctx
    all_last_token = []
    all_prefix = []

    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        batch_tokens = torch.stack(token_ids_list[i:batch_end]).to(device)

        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)

        hidden = outputs.hidden_states[layer + 1]  # [batch, seq_len, d_model]

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

    # Split: first n_act for activations, next n_ctx for contexts
    activations = all_last_token[:n_act]
    contexts = all_prefix_list[n_act:n_act + n_ctx]

    return activations, contexts


def optimization_step(model, delta, activations, contexts, layer, batch_indices,
                      n_probes, device, target_norm):
    """
    One optimization step with matching loss: (||J(a_i+delta)||_F - target)^2.

    Accumulates K probes into a differentiable ||J||_F^2 estimate, takes sqrt,
    then computes matching loss before backward.
    """
    d_model = delta.shape[0]
    B = len(batch_indices)
    total_loss = 0.0

    for i in batch_indices:
        a_i = activations[i].to(device).detach()
        ctx_i = contexts[i].to(device).detach()

        point = (a_i + delta).reshape(1, 1, d_model)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            final_act = forward_from_layer_to_final(model, ctx_i, point, layer)

        # Hutchinson estimate of ||J||_F^2 (differentiable)
        frob_sq = torch.tensor(0.0, device=device)
        for k in range(n_probes):
            u = torch.randn(1, d_model, device=device)
            s = (final_act * u).sum()
            g = torch.autograd.grad(s, delta, create_graph=True, retain_graph=True)[0]
            frob_sq = frob_sq + (g ** 2).sum()
        frob_sq = frob_sq / n_probes

        # Matching loss: (||J||_F - target)^2
        frob_norm = torch.sqrt(frob_sq + 1e-8)  # small eps for numerical stability
        act_loss = (frob_norm - target_norm) ** 2 / B
        act_loss.backward()
        total_loss += act_loss.item()

    return total_loss


def main():
    parser = argparse.ArgumentParser(description='Delta optimization v2 (matching loss)')
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=100,
                        help='Number of activations')
    parser.add_argument('--n_ctx', type=int, default=100,
                        help='Number of cross-contexts')
    parser.add_argument('--n_probes_opt', type=int, default=5,
                        help='Hutchinson probes during optimization')
    parser.add_argument('--n_probes_eval', type=int, default=100,
                        help='Hutchinson probes during evaluation')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Activations per optimization step')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--r', type=float, default=0,
                        help='Delta norm (0 = auto from mean pairwise distance)')
    parser.add_argument('--results_dir', type=str, default='results/delta_opt_v2')
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
    print(f"  Seq length: {args.seq_len}")
    print(f"  Activations: {args.n_act}, Contexts: {args.n_ctx}")
    print(f"  Optimization: B={args.batch_size}, K={args.n_probes_opt}, "
          f"lr={args.lr}, steps={args.n_steps}")
    print(f"  Loss: (||J(a_i+delta)||_F - target)^2  [MATCHING]")
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
    # Stage 0: Extract activations, compute distances
    # =========================================================
    if stage == 'start':
        print(f"\n=== Stage 0: Extracting activations & computing distances ===")
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
        print(f"Contexts: {len(contexts)} x {contexts[0].shape}")

        # Pairwise L2 distances among activations
        print(f"Computing pairwise distances...")
        dists = torch.cdist(activations.unsqueeze(0), activations.unsqueeze(0)).squeeze(0)
        mask = torch.triu(torch.ones(args.n_act, args.n_act, dtype=torch.bool), diagonal=1)
        pairwise_dists = dists[mask].numpy()

        print(f"Pairwise L2 distances ({len(pairwise_dists)} pairs):")
        print(f"  Mean: {pairwise_dists.mean():.1f}")
        print(f"  Median: {np.median(pairwise_dists):.1f}")
        print(f"  Std: {pairwise_dists.std():.1f}")
        print(f"  Min: {pairwise_dists.min():.1f}, Max: {pairwise_dists.max():.1f}")

        act_norms = activations.norm(dim=1).numpy()
        print(f"Activation norms: mean={act_norms.mean():.1f}, std={act_norms.std():.1f}")

        r = args.r if args.r > 0 else pairwise_dists.mean()
        print(f"Using r = {r:.1f}")

        print(f"Stage 0 took {time.time()-t0:.1f}s")

        checkpoint['activations'] = activations
        checkpoint['contexts'] = contexts
        checkpoint['pairwise_dists'] = pairwise_dists
        checkpoint['r'] = r
        checkpoint['stage'] = 'activations_ready'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'activations_ready'

        del token_ids_list

    # =========================================================
    # Stage 1: Reference Jacobian norms (cross-context)
    # =========================================================
    if stage in ['activations_ready', 'ref_norms_in_progress']:
        print(f"\n=== Stage 1: Reference Jacobian norms ({args.n_probes_eval} probes) ===")
        print(f"J: layer {args.layer} last-token -> final layer last-token (pre-ln_f)")

        activations = checkpoint['activations']
        contexts = checkpoint['contexts']

        ref_norms = checkpoint.get('ref_norms', [])
        ref_stds = checkpoint.get('ref_stds', [])
        done = len(ref_norms)

        t0 = time.time()
        for i in range(done, args.n_act):
            a_i = activations[i].to(device)
            ctx_i = contexts[i].to(device)

            frob_norm, frob_std = compute_jacobian_norm_vjp(
                model, ctx_i, a_i, args.layer, n_probes=args.n_probes_eval
            )
            ref_norms.append(frob_norm)
            ref_stds.append(frob_std)

            elapsed = time.time() - t0
            rate = (i - done + 1) / elapsed if elapsed > 0 else 0
            eta = (args.n_act - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{args.n_act}  ||J||_F={frob_norm:.1f}  "
                  f"({rate:.2f}/s, ETA {eta:.0f}s)", end='\r')

            if (i + 1) % 10 == 0:
                checkpoint['ref_norms'] = ref_norms
                checkpoint['ref_stds'] = ref_stds
                checkpoint['stage'] = 'ref_norms_in_progress'
                save_checkpoint(checkpoint, checkpoint_path)

        print()

        ref_norms_arr = np.array(ref_norms)
        target_norm = float(ref_norms_arr.mean())
        print(f"Reference ||J||_F (cross-context):")
        print(f"  Mean: {ref_norms_arr.mean():.1f}")
        print(f"  Std: {ref_norms_arr.std():.1f}")
        print(f"  Min: {ref_norms_arr.min():.1f}, Max: {ref_norms_arr.max():.1f}")
        print(f"  Target norm for matching loss: {target_norm:.1f}")

        checkpoint['ref_norms'] = ref_norms
        checkpoint['ref_stds'] = ref_stds
        checkpoint['target_norm'] = target_norm
        checkpoint['stage'] = 'ref_norms_done'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'ref_norms_done'

    # =========================================================
    # Stage 2: Optimization (matching loss)
    # =========================================================
    if stage in ['ref_norms_done', 'optimization_in_progress']:
        print(f"\n=== Stage 2: Delta optimization (matching loss) ===")

        activations = checkpoint['activations']
        contexts = checkpoint['contexts']
        r = checkpoint['r']
        target_norm = checkpoint['target_norm']
        print(f"  r = {r:.1f}, target ||J||_F = {target_norm:.1f}")
        print(f"  lr = {args.lr}, B = {args.batch_size}, K = {args.n_probes_opt}")

        # Initialize or resume delta
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
            history = {'loss': [], 'grad_norm': [], 'cosine_sim': [], 'frob_est': []}

        optimizer = torch.optim.Adam([delta], lr=args.lr)
        if opt_state is not None and 'optimizer_state' in opt_state:
            optimizer.load_state_dict(opt_state['optimizer_state'])

        prev_direction = delta.detach().clone() / delta.detach().norm()

        t0 = time.time()
        for step in range(start_step, args.n_steps):
            batch_indices = np.random.choice(args.n_act, args.batch_size, replace=False)

            optimizer.zero_grad()
            loss_val = optimization_step(
                model, delta, activations, contexts, args.layer,
                batch_indices, args.n_probes_opt, device, target_norm
            )

            grad_norm = delta.grad.norm().item() if delta.grad is not None else 0.0

            optimizer.step()

            # Project back to sphere of radius r
            with torch.no_grad():
                delta.data.mul_(r / delta.data.norm())

            # Track direction change
            curr_direction = delta.detach() / delta.detach().norm()
            cos_sim = (curr_direction * prev_direction).sum().item()
            prev_direction = curr_direction.clone()

            history['loss'].append(loss_val)
            history['grad_norm'].append(grad_norm)
            history['cosine_sim'].append(cos_sim)

            if (step + 1) % 10 == 0:
                elapsed = time.time() - t0
                steps_done = step - start_step + 1
                rate = steps_done / elapsed
                eta = (args.n_steps - step - 1) / rate if rate > 0 else 0
                # loss_val is mean (||J||_F - target)^2; sqrt gives RMS deviation
                rms_dev = np.sqrt(max(loss_val, 0))
                print(f"  Step {step+1}/{args.n_steps}  loss={loss_val:.1f}  "
                      f"RMS_dev={rms_dev:.1f}  grad={grad_norm:.4f}  "
                      f"cos={cos_sim:.4f}  ({rate:.1f} steps/s, ETA {eta:.0f}s)")

            if (step + 1) % 50 == 0:
                checkpoint['opt_state'] = {
                    'delta': delta.detach().cpu(),
                    'step': step,
                    'history': history,
                    'optimizer_state': optimizer.state_dict(),
                }
                checkpoint['stage'] = 'optimization_in_progress'
                save_checkpoint(checkpoint, checkpoint_path)

        # Save final state
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
        print(f"\n=== Stage 3: Evaluation ({args.n_probes_eval} probes) ===")

        activations = checkpoint['activations']
        contexts = checkpoint['contexts']
        delta_opt = checkpoint['opt_state']['delta'].to(device)
        ref_norms = np.array(checkpoint['ref_norms'])
        r = checkpoint['r']
        target_norm = checkpoint['target_norm']

        print(f"  Optimized delta norm: {delta_opt.norm().item():.1f} (target r={r:.1f})")
        print(f"  Target ||J||_F: {target_norm:.1f}")

        # Compute ||J(a_i + delta)||_F for all activations
        shifted_norms = []
        t0 = time.time()
        for i in range(args.n_act):
            shifted_point = (activations[i].to(device) + delta_opt).detach()
            ctx_i = contexts[i].to(device)

            frob_norm, _ = compute_jacobian_norm_vjp(
                model, ctx_i, shifted_point, args.layer,
                n_probes=args.n_probes_eval
            )
            shifted_norms.append(frob_norm)

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (args.n_act - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{args.n_act}  ||J||_F={frob_norm:.1f}  "
                  f"({rate:.2f}/s, ETA {eta:.0f}s)", end='\r')

        print()

        shifted_norms = np.array(shifted_norms)

        print(f"\nResults (||J||_F, layer {args.layer} last-token -> final layer):")
        print(f"  Real (cross-ctx):    mean={ref_norms.mean():.1f}, std={ref_norms.std():.1f}")
        print(f"  Shifted (a_i+delta): mean={shifted_norms.mean():.1f}, std={shifted_norms.std():.1f}")
        print(f"  Target norm:         {target_norm:.1f}")
        print(f"  Delta norm:          {r:.1f}")

        # Plot
        plot_three_way_comparison(
            {
                'Real (cross-context)': ref_norms,
                f'Shifted (r={r:.0f})': shifted_norms,
            },
            metric_name="Jacobian Frobenius Norm ||J||_F",
            title=f"Delta Opt v2 (matching): Layer {args.layer}, r={r:.0f}",
            save_path=os.path.join(args.results_dir, f'eval_r{r:.0f}.png')
        )

        # Plot optimization history
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(history['loss'])
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss: (||J||_F - target)^2')
        axes[0].set_title('Matching Loss')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history['grad_norm'])
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Gradient Norm')
        axes[1].set_title('Gradient Norm')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(history['cosine_sim'])
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Cosine Similarity')
        axes[2].set_title('Direction Stability')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(args.results_dir, f'training_r{r:.0f}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Save final results
        results = {
            'args': vars(args),
            'r': r,
            'target_norm': target_norm,
            'delta': delta_opt.cpu(),
            'ref_norms': ref_norms,
            'shifted_norms': shifted_norms,
            'history': history,
            'pairwise_dists': checkpoint.get('pairwise_dists'),
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
