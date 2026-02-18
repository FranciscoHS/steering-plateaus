#!/usr/bin/env python3
"""
Perturbation sweep with linearly-spaced absolute magnitudes.

Simplified: only real (same-ctx) vs random. Linear magnitudes 0-100.
Produces linear-scale plots for comparison with collaborator results.

Usage:
    python scripts/run_perturbation_sweep_linear.py --model gpt2-large
    python scripts/run_perturbation_sweep_linear.py --model qwen3-0.6b
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
    load_model, forward_from_layer_to_final,
    _get_final_norm, _get_lm_head,
)
from src.data import load_fineweb_fixed_length


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device, batch_size=64):
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

    contexts = torch.cat(all_contexts, dim=0)
    activations = torch.cat(all_activations, dim=0)
    return contexts, activations


def run_sweep(model, context, activation, directions, magnitudes, layer, device, n_dirs):
    """Run perturbation sweep for a single activation. Returns l2[n_mags, n_dirs], kl[n_mags, n_dirs]."""
    n_mags = len(magnitudes)
    l2_dists = np.zeros((n_mags, n_dirs))
    kl_divs = np.zeros((n_mags, n_dirs))

    ctx = context.unsqueeze(0).to(device)
    act = activation.to(device)
    dirs = directions.to(device)

    with torch.no_grad():
        ref_point = act.unsqueeze(0).unsqueeze(0)
        ref_final = forward_from_layer_to_final(model, ctx, ref_point, layer)
        ref_ln = _get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1)
        ref_logits = _get_lm_head(model)(ref_ln)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    for mag_idx, eps in enumerate(magnitudes):
        perturbed = (act.unsqueeze(0) + eps * dirs).unsqueeze(1)
        with torch.no_grad():
            pert_final = forward_from_layer_to_final(model, ctx, perturbed, layer)
            l2_dists[mag_idx] = (pert_final - ref_final).norm(dim=-1).cpu().numpy()

            pert_ln = _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)
            pert_logits = _get_lm_head(model)(pert_ln)
            pert_log_probs = F.log_softmax(pert_logits, dim=-1)
            ref_probs = ref_log_probs.exp()
            kl = (ref_probs * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            kl_divs[mag_idx] = kl.cpu().numpy()

    return l2_dists, kl_divs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=100)
    parser.add_argument('--n_dirs', type=int, default=10)
    parser.add_argument('--mag_max', type=float, default=100.0, help='Max absolute perturbation magnitude')
    parser.add_argument('--n_mags', type=int, default=50, help='Number of linearly-spaced magnitude points')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/perturbation_sweep_linear/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # Linear magnitudes from 0 (exclusive) to mag_max
    magnitudes = np.linspace(0, args.mag_max, args.n_mags + 1)[1:]  # skip 0

    print(f"Config: model={args.model}, layer={args.layer}, n_act={args.n_act}, "
          f"n_dirs={args.n_dirs}, n_mags={len(magnitudes)}, mag_range=[{magnitudes[0]:.1f}, {magnitudes[-1]:.1f}]")

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    # Extract activations
    print(f"Loading {args.n_act} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)

    print(f"Extracting layer-{args.layer} activations...")
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device
    )

    act_norm_mean = activations.norm(dim=-1).mean().item()
    print(f"Mean activation norm: {act_norm_mean:.2f}")

    # Norm-matched random activations
    random_activations = torch.randn(args.n_act, d_model)
    random_activations = F.normalize(random_activations, dim=-1) * act_norm_mean

    # Shared random directions
    directions = torch.randn(args.n_act, args.n_dirs, d_model)
    directions = F.normalize(directions, dim=-1)

    # Sweep: real
    print(f"\n=== Real activations ===")
    real_l2 = np.zeros((args.n_act, len(magnitudes), args.n_dirs))
    real_kl = np.zeros((args.n_act, len(magnitudes), args.n_dirs))
    t0 = time.time()
    for i in range(args.n_act):
        real_l2[i], real_kl[i] = run_sweep(
            model, contexts[i], activations[i], directions[i], magnitudes, args.layer, device, args.n_dirs
        )
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
    print(f"Real done in {time.time()-t0:.1f}s")

    # Sweep: random
    print(f"\n=== Random activations ===")
    rand_l2 = np.zeros((args.n_act, len(magnitudes), args.n_dirs))
    rand_kl = np.zeros((args.n_act, len(magnitudes), args.n_dirs))
    t0 = time.time()
    for i in range(args.n_act):
        rand_l2[i], rand_kl[i] = run_sweep(
            model, contexts[i], random_activations[i], directions[i], magnitudes, args.layer, device, args.n_dirs
        )
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
    print(f"Random done in {time.time()-t0:.1f}s")

    # Save results
    results = {
        'args': vars(args),
        'magnitudes': magnitudes,
        'act_norm_mean': act_norm_mean,
        'real': {'l2': real_l2, 'kl': real_kl},
        'random': {'l2': rand_l2, 'kl': rand_kl},
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Plots
    conditions = [('real', 'Real'), ('random', 'Random')]
    colors = ['blue', 'red']
    np.random.seed(42)
    n_individual = 10

    for plot_type in ['median', 'individual']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for metric_idx, (key, name) in enumerate([('l2', 'L2 distance'), ('kl', 'KL divergence')]):
            ax = axes[metric_idx]
            for (cond_key, label), color in zip(conditions, colors):
                d = results[cond_key][key]
                per_act = np.median(d, axis=2)  # [n_act, n_mags]

                if plot_type == 'median':
                    median = np.median(per_act, axis=0)
                    q25 = np.percentile(per_act, 25, axis=0)
                    q75 = np.percentile(per_act, 75, axis=0)
                    ax.plot(magnitudes, median, color=color, label=label, linewidth=2)
                    ax.fill_between(magnitudes, q25, q75, color=color, alpha=0.15)
                else:
                    chosen = np.random.choice(per_act.shape[0], n_individual, replace=False)
                    for j, idx in enumerate(chosen):
                        ax.plot(magnitudes, per_act[idx], color=color, alpha=0.4, linewidth=1.5,
                                label=label if j == 0 else None)

            ax.set_xlabel('Perturbation magnitude (absolute)')
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{args.model}, layer {args.layer} (linear scale, {plot_type})', fontsize=14)
        plt.tight_layout()
        path = os.path.join(args.results_dir, f'sweep_{plot_type}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
