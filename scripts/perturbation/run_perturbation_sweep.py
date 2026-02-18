#!/usr/bin/env python3
"""
Perturbation sweep: directly measure activation plateau phenomenon.

Perturbs real vs random activations by varying magnitudes in random directions,
measures downstream effect via L2 distance on final-layer activations and KL
divergence on output logits.

Three conditions:
  1. real_same_ctx: real activation with its own context
  2. real_cross_ctx: real activation with a different context
  3. random: norm-matched random activation with a borrowed context

Usage:
    python scripts/run_perturbation_sweep.py
    python scripts/run_perturbation_sweep.py --model qwen3-0.6b --layer 6
    python scripts/run_perturbation_sweep.py --resume
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
    load_model, forward_from_layer_to_final, forward_from_layer,
    _get_final_norm, _get_lm_head,
)
from src.data import load_fineweb_fixed_length


def save_checkpoint(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Checkpoint loaded: {path}")
    return data


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device, batch_size=64):
    """
    Extract layer-L context (prefix positions) and final-token activation for fixed-length sequences.

    Returns:
        contexts: [n, seq_len-1, d_model] tensor (CPU)
        activations: [n, d_model] tensor (CPU)
    """
    all_contexts = []
    all_activations = []

    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)

        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)

        # hidden_states[layer+1] = residual stream after block `layer`
        hidden = outputs.hidden_states[layer + 1]  # [batch, seq_len, d]

        all_contexts.append(hidden[:, :-1, :].cpu())
        all_activations.append(hidden[:, -1, :].cpu())

        if (i // batch_size) % 10 == 0:
            print(f"  Extracted {min(i + batch_size, len(token_ids_list))}/{len(token_ids_list)}", end='\r')

    print()
    contexts = torch.cat(all_contexts, dim=0)
    activations = torch.cat(all_activations, dim=0)
    return contexts, activations


def run_perturbation_sweep(model, context, activation, directions, magnitudes,
                           layer, device, n_dirs):
    """
    Run perturbation sweep for a single activation.

    Args:
        model: the model
        context: [1, seq_len-1, d_model] - prefix context on CPU
        activation: [d_model] - final-token activation on CPU
        directions: [n_dirs, d_model] - unit direction vectors on CPU
        magnitudes: [n_mags] - absolute perturbation magnitudes
        layer: start layer
        device: torch device
        n_dirs: number of directions

    Returns:
        l2_dists: [n_mags, n_dirs] array
        kl_divs: [n_mags, n_dirs] array
    """
    n_mags = len(magnitudes)
    l2_dists = np.zeros((n_mags, n_dirs))
    kl_divs = np.zeros((n_mags, n_dirs))

    ctx = context.unsqueeze(0).to(device)  # [1, seq_len-1, d_model]
    act = activation.to(device)  # [d_model]
    dirs = directions.to(device)  # [n_dirs, d_model]

    # Reference: unperturbed forward pass
    with torch.no_grad():
        ref_point = act.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
        ref_final = forward_from_layer_to_final(model, ctx, ref_point, layer)  # [1, d_model]
        ref_ln = _get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1)
        ref_logits = _get_lm_head(model)(ref_ln)  # [1, vocab_size]
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)  # [1, vocab_size]

    for mag_idx, eps in enumerate(magnitudes):
        # Batch all directions: [n_dirs, 1, d_model]
        perturbed = (act.unsqueeze(0) + eps * dirs).unsqueeze(1)  # [n_dirs, 1, d_model]

        with torch.no_grad():
            pert_final = forward_from_layer_to_final(model, ctx, perturbed, layer)  # [n_dirs, d_model]

            # L2 distance in final-layer activation space
            l2 = (pert_final - ref_final).norm(dim=-1)  # [n_dirs]
            l2_dists[mag_idx] = l2.cpu().numpy()

            # KL divergence on logits
            pert_ln = _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)
            pert_logits = _get_lm_head(model)(pert_ln)  # [n_dirs, vocab_size]
            pert_log_probs = F.log_softmax(pert_logits, dim=-1)
            ref_probs = ref_log_probs.exp()  # [1, vocab_size]
            # KL(ref || pert) = sum ref * (log_ref - log_pert)
            kl = (ref_probs * (ref_log_probs - pert_log_probs)).sum(dim=-1)  # [n_dirs]
            kl_divs[mag_idx] = kl.cpu().numpy()

    return l2_dists, kl_divs


def plot_perturbation_sweep(magnitudes_relative, results, model_name, layer, save_dir):
    """Plot L2 distance and KL divergence vs perturbation magnitude."""
    conditions = ['real_same_ctx', 'real_cross_ctx', 'random']
    labels = ['Real (same ctx)', 'Real (cross ctx)', 'Random']
    colors = ['blue', 'green', 'red']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for metric_idx, (metric_key, metric_name) in enumerate([('l2', 'L2 distance'), ('kl', 'KL divergence')]):
        ax = axes[metric_idx]

        for cond, label, color in zip(conditions, labels, colors):
            data = results[cond][metric_key]  # [n_act, n_mags, n_dirs]
            # Aggregate: median over directions, then median/IQR over activations
            per_act_median = np.median(data, axis=2)  # [n_act, n_mags]
            median = np.median(per_act_median, axis=0)  # [n_mags]
            q25 = np.percentile(per_act_median, 25, axis=0)
            q75 = np.percentile(per_act_median, 75, axis=0)

            ax.plot(magnitudes_relative, median, color=color, label=label, linewidth=2)
            ax.fill_between(magnitudes_relative, q25, q75, color=color, alpha=0.15)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Perturbation magnitude (relative to mean activation norm)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs perturbation magnitude')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{model_name}, layer {layer}', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'perturbation_sweep.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Perturbation sweep experiment')
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=100,
                        help='Number of activations for sweep')
    parser.add_argument('--n_ctx', type=int, default=100,
                        help='Number of cross-contexts (separate FineWeb sequences)')
    parser.add_argument('--n_dirs', type=int, default=10,
                        help='Random directions per activation per magnitude')
    parser.add_argument('--n_mags', type=int, default=30,
                        help='Number of magnitude points (log-spaced)')
    parser.add_argument('--mag_min', type=float, default=1e-3,
                        help='Min perturbation magnitude (relative to mean act norm)')
    parser.add_argument('--mag_max', type=float, default=10.0,
                        help='Max perturbation magnitude (relative to mean act norm)')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Results directory (auto: results/perturbation_sweep/{model})')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/perturbation_sweep/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.results_dir, 'checkpoint.pkl')

    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Layer: {args.layer}")
    print(f"  Seq length: {args.seq_len}")
    print(f"  Activations: {args.n_act}, Cross-contexts: {args.n_ctx}")
    print(f"  Directions: {args.n_dirs}, Magnitudes: {args.n_mags}")
    print(f"  Magnitude range: {args.mag_min} - {args.mag_max} (relative)")
    print(f"  Results dir: {args.results_dir}")
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
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    # =========================================================
    # Stage 0: Extract activations and prepare data
    # =========================================================
    if stage == 'start':
        print(f"\n=== Stage 0: Extracting activations ===")
        t0 = time.time()

        n_total = args.n_act + args.n_ctx
        print(f"Loading {n_total} FineWeb sequences (seq_len={args.seq_len})...")
        token_ids_list = load_fineweb_fixed_length(
            n_total, tokenizer, seq_len=args.seq_len, seed=args.seed
        )

        print(f"Extracting layer-{args.layer} activations...")
        contexts, activations = extract_context_and_activation(
            model, tokenizer, token_ids_list, args.layer, device
        )
        print(f"Contexts shape: {contexts.shape}, Activations shape: {activations.shape}")

        # Split: first n_act → real activations + same-contexts; last n_ctx → cross-contexts
        same_contexts = contexts[:args.n_act]       # [n_act, seq_len-1, d_model]
        real_activations = activations[:args.n_act]  # [n_act, d_model]
        cross_contexts = contexts[args.n_act:]       # [n_ctx, seq_len-1, d_model]

        # Compute mean activation norm for magnitude normalization
        act_norms = real_activations.norm(dim=-1)  # [n_act]
        act_norm_mean = act_norms.mean().item()
        print(f"Mean activation norm: {act_norm_mean:.2f}")

        # Absolute magnitudes
        magnitudes_relative = np.logspace(np.log10(args.mag_min), np.log10(args.mag_max), args.n_mags)
        magnitudes_absolute = magnitudes_relative * act_norm_mean
        print(f"Magnitude range (absolute): {magnitudes_absolute[0]:.4f} - {magnitudes_absolute[-1]:.2f}")

        # Norm-matched random activations
        random_activations = torch.randn(args.n_act, d_model)
        random_activations = F.normalize(random_activations, dim=-1) * act_norm_mean
        print(f"Random activations shape: {random_activations.shape}, norms ≈ {act_norm_mean:.2f}")

        # Pre-sample random directions (shared across conditions for paired comparison)
        directions = torch.randn(args.n_act, args.n_dirs, d_model)
        directions = F.normalize(directions, dim=-1)
        print(f"Directions shape: {directions.shape}")

        checkpoint['same_contexts'] = same_contexts
        checkpoint['real_activations'] = real_activations
        checkpoint['cross_contexts'] = cross_contexts
        checkpoint['random_activations'] = random_activations
        checkpoint['directions'] = directions
        checkpoint['magnitudes_relative'] = magnitudes_relative
        checkpoint['magnitudes_absolute'] = magnitudes_absolute
        checkpoint['act_norm_mean'] = act_norm_mean
        checkpoint['stage'] = 'data_ready'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'data_ready'

        print(f"Stage 0 took {time.time()-t0:.1f}s")
        del token_ids_list, contexts, activations

    # Load shared data
    same_contexts = checkpoint['same_contexts']
    real_activations = checkpoint['real_activations']
    cross_contexts = checkpoint['cross_contexts']
    random_activations = checkpoint['random_activations']
    directions = checkpoint['directions']
    magnitudes_relative = checkpoint['magnitudes_relative']
    magnitudes_absolute = checkpoint['magnitudes_absolute']

    # =========================================================
    # Stage 1: Perturbation sweep — real, same-context
    # =========================================================
    if stage in ['data_ready', 'real_same_ctx_in_progress']:
        print(f"\n=== Stage 1: Perturbation sweep — real, same-context ===")
        t0 = time.time()

        l2_all = checkpoint.get('real_same_ctx_l2', np.zeros((args.n_act, args.n_mags, args.n_dirs)))
        kl_all = checkpoint.get('real_same_ctx_kl', np.zeros((args.n_act, args.n_mags, args.n_dirs)))
        start_i = checkpoint.get('real_same_ctx_done', 0)

        for i in range(start_i, args.n_act):
            l2, kl = run_perturbation_sweep(
                model, same_contexts[i], real_activations[i], directions[i],
                magnitudes_absolute, args.layer, device, args.n_dirs
            )
            l2_all[i] = l2
            kl_all[i] = kl

            elapsed = time.time() - t0
            rate = (i - start_i + 1) / elapsed if elapsed > 0 else 0
            eta = (args.n_act - i - 1) / rate if rate > 0 else 0
            print(f"  real_same_ctx {i+1}/{args.n_act}  ({rate:.1f} act/s, ETA {eta:.0f}s)", end='\r')

            if (i + 1) % 10 == 0:
                checkpoint['real_same_ctx_l2'] = l2_all
                checkpoint['real_same_ctx_kl'] = kl_all
                checkpoint['real_same_ctx_done'] = i + 1
                checkpoint['stage'] = 'real_same_ctx_in_progress'
                save_checkpoint(checkpoint, checkpoint_path)

        print()
        checkpoint['real_same_ctx_l2'] = l2_all
        checkpoint['real_same_ctx_kl'] = kl_all
        checkpoint['real_same_ctx_done'] = args.n_act
        checkpoint['stage'] = 'real_same_ctx_done'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'real_same_ctx_done'
        print(f"Stage 1 took {time.time()-t0:.1f}s")

    # =========================================================
    # Stage 2: Perturbation sweep — real, cross-context
    # =========================================================
    if stage in ['real_same_ctx_done', 'real_cross_ctx_in_progress']:
        print(f"\n=== Stage 2: Perturbation sweep — real, cross-context ===")
        t0 = time.time()

        l2_all = checkpoint.get('real_cross_ctx_l2', np.zeros((args.n_act, args.n_mags, args.n_dirs)))
        kl_all = checkpoint.get('real_cross_ctx_kl', np.zeros((args.n_act, args.n_mags, args.n_dirs)))
        start_i = checkpoint.get('real_cross_ctx_done', 0)

        n_ctx = cross_contexts.shape[0]
        for i in range(start_i, args.n_act):
            ctx_idx = i % n_ctx
            l2, kl = run_perturbation_sweep(
                model, cross_contexts[ctx_idx], real_activations[i], directions[i],
                magnitudes_absolute, args.layer, device, args.n_dirs
            )
            l2_all[i] = l2
            kl_all[i] = kl

            elapsed = time.time() - t0
            rate = (i - start_i + 1) / elapsed if elapsed > 0 else 0
            eta = (args.n_act - i - 1) / rate if rate > 0 else 0
            print(f"  real_cross_ctx {i+1}/{args.n_act}  ({rate:.1f} act/s, ETA {eta:.0f}s)", end='\r')

            if (i + 1) % 10 == 0:
                checkpoint['real_cross_ctx_l2'] = l2_all
                checkpoint['real_cross_ctx_kl'] = kl_all
                checkpoint['real_cross_ctx_done'] = i + 1
                checkpoint['stage'] = 'real_cross_ctx_in_progress'
                save_checkpoint(checkpoint, checkpoint_path)

        print()
        checkpoint['real_cross_ctx_l2'] = l2_all
        checkpoint['real_cross_ctx_kl'] = kl_all
        checkpoint['real_cross_ctx_done'] = args.n_act
        checkpoint['stage'] = 'real_cross_ctx_done'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'real_cross_ctx_done'
        print(f"Stage 2 took {time.time()-t0:.1f}s")

    # =========================================================
    # Stage 3: Perturbation sweep — random
    # =========================================================
    if stage in ['real_cross_ctx_done', 'random_in_progress']:
        print(f"\n=== Stage 3: Perturbation sweep — random ===")
        t0 = time.time()

        l2_all = checkpoint.get('random_l2', np.zeros((args.n_act, args.n_mags, args.n_dirs)))
        kl_all = checkpoint.get('random_kl', np.zeros((args.n_act, args.n_mags, args.n_dirs)))
        start_i = checkpoint.get('random_done', 0)

        n_ctx = cross_contexts.shape[0]
        for i in range(start_i, args.n_act):
            ctx_idx = i % n_ctx
            l2, kl = run_perturbation_sweep(
                model, cross_contexts[ctx_idx], random_activations[i], directions[i],
                magnitudes_absolute, args.layer, device, args.n_dirs
            )
            l2_all[i] = l2
            kl_all[i] = kl

            elapsed = time.time() - t0
            rate = (i - start_i + 1) / elapsed if elapsed > 0 else 0
            eta = (args.n_act - i - 1) / rate if rate > 0 else 0
            print(f"  random {i+1}/{args.n_act}  ({rate:.1f} act/s, ETA {eta:.0f}s)", end='\r')

            if (i + 1) % 10 == 0:
                checkpoint['random_l2'] = l2_all
                checkpoint['random_kl'] = kl_all
                checkpoint['random_done'] = i + 1
                checkpoint['stage'] = 'random_in_progress'
                save_checkpoint(checkpoint, checkpoint_path)

        print()
        checkpoint['random_l2'] = l2_all
        checkpoint['random_kl'] = kl_all
        checkpoint['random_done'] = args.n_act
        checkpoint['stage'] = 'random_done'
        save_checkpoint(checkpoint, checkpoint_path)
        stage = 'random_done'
        print(f"Stage 3 took {time.time()-t0:.1f}s")

    # =========================================================
    # Stage 4: Plot and save results
    # =========================================================
    if stage == 'random_done':
        print(f"\n=== Stage 4: Plotting and saving results ===")

        results = {
            'real_same_ctx': {
                'l2': checkpoint['real_same_ctx_l2'],
                'kl': checkpoint['real_same_ctx_kl'],
            },
            'real_cross_ctx': {
                'l2': checkpoint['real_cross_ctx_l2'],
                'kl': checkpoint['real_cross_ctx_kl'],
            },
            'random': {
                'l2': checkpoint['random_l2'],
                'kl': checkpoint['random_kl'],
            },
        }

        plot_perturbation_sweep(magnitudes_relative, results, args.model, args.layer, args.results_dir)

        # Print summary
        print(f"\nSummary (median over activations and directions):")
        for cond_name, cond_label in [('real_same_ctx', 'Real (same ctx)'),
                                       ('real_cross_ctx', 'Real (cross ctx)'),
                                       ('random', 'Random')]:
            l2_data = results[cond_name]['l2']  # [n_act, n_mags, n_dirs]
            kl_data = results[cond_name]['kl']
            # Median over dirs, then median over activations
            l2_med = np.median(np.median(l2_data, axis=2), axis=0)
            kl_med = np.median(np.median(kl_data, axis=2), axis=0)
            print(f"  {cond_label}:")
            print(f"    L2 at smallest pert: {l2_med[0]:.4f}, at largest: {l2_med[-1]:.2f}")
            print(f"    KL at smallest pert: {kl_med[0]:.6f}, at largest: {kl_med[-1]:.4f}")

        # Save results pickle
        save_data = {
            'args': vars(args),
            'magnitudes_relative': magnitudes_relative,
            'magnitudes_absolute': checkpoint['magnitudes_absolute'],
            'act_norm_mean': checkpoint['act_norm_mean'],
            'results': results,
        }
        results_path = os.path.join(args.results_dir, 'results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"\nResults saved: {results_path}")

        checkpoint['stage'] = 'done'
        save_checkpoint(checkpoint, checkpoint_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
