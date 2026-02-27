#!/usr/bin/env python3
"""
Compute logit entropy for original vs shifted activations across all 100
optimization prompts. One-shot patch at layer 6, measure next-token entropy.

Produces histogram comparing base vs v1-shifted vs v2-shifted.

Usage:
    python scripts/jacobian/entropy_histogram_shifted.py
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length


def get_entropy(model, input_ids, delta, layer, device):
    """Single forward pass with one-shot patch, return entropy."""
    input_ids = input_ids.to(device)

    def patch_hook(module, input, output):
        hidden = output[0]
        hidden[0, -1, :] = hidden[0, -1, :] + delta
        return (hidden,) + output[1:]

    layers = _get_blocks(model)
    handle = layers[layer].register_forward_hook(patch_hook)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)

    handle.remove()

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * probs.log()).sum().item()
    return entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_prompts', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--v1_results', type=str, default='results/delta_opt/results_r66.pkl')
    parser.add_argument('--v2_results', type=str, default='results/delta_opt_v2/results_r66.pkl')
    parser.add_argument('--v3b_results', type=str, default='results/delta_opt_v3b/results_r66.pkl')
    parser.add_argument('--results_dir', type=str, default='results/delta_steer_spherical')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load deltas
    deltas = {}
    for label, path in [('v1 (minimize)', args.v1_results), ('v2 (match)', args.v2_results),
                         ('v3b (sv+frob)', args.v3b_results)]:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                results = pickle.load(f)
            deltas[label] = results['delta']
            print(f"Loaded {label}: norm={results['delta'].norm().item():.1f}")
        else:
            print(f"Skipping {label}: {path} not found")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    for key in deltas:
        deltas[key] = deltas[key].to(device)

    d_model = deltas[list(deltas.keys())[0]].shape[0]
    zero_delta = torch.zeros(d_model, device=device)

    # Load sequences
    print(f"\nLoading FineWeb sequences (seed={args.seed}, seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(200, tokenizer, seq_len=args.seq_len, seed=args.seed)
    opt_sequences = token_ids_list[:100]

    # Compute entropies
    entropies = {'base': []}
    for label in deltas:
        entropies[label] = []

    print(f"\nComputing entropies for {args.n_prompts} prompts...")
    for i in range(args.n_prompts):
        input_ids = opt_sequences[i].unsqueeze(0)

        h_base = get_entropy(model, input_ids, zero_delta, args.layer, device)
        entropies['base'].append(h_base)

        for label, delta in deltas.items():
            h_shift = get_entropy(model, input_ids, delta, args.layer, device)
            entropies[label].append(h_shift)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{args.n_prompts}")

    # Print stats
    for label, vals in entropies.items():
        arr = np.array(vals)
        print(f"{label:20s}: mean={arr.mean():.2f}, std={arr.std():.2f}, "
              f"median={np.median(arr):.2f}, min={arr.min():.2f}, max={arr.max():.2f}")

    # Plot: two separate histograms
    bins = np.linspace(0, 10, 30)
    base_arr = np.array(entropies['base'])

    for label, color in [('v1 (minimize)', 'C1'), ('v2 (match)', 'C2'), ('v3b (sv+frob)', 'C3')]:
        if label not in entropies:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        shifted_arr = np.array(entropies[label])

        ax.hist(base_arr, bins=bins, alpha=0.6, label='Base', color='C0', edgecolor='black', linewidth=0.5)
        ax.hist(shifted_arr, bins=bins, alpha=0.6, label=label, color=color, edgecolor='black', linewidth=0.5)

        # Add median/std annotations
        ax.axvline(np.median(base_arr), color='C0', linestyle='--', linewidth=1.5)
        ax.axvline(np.median(shifted_arr), color=color, linestyle='--', linewidth=1.5)

        base_text = f'Base: med={np.median(base_arr):.2f}, std={base_arr.std():.2f}'
        shift_text = f'{label}: med={np.median(shifted_arr):.2f}, std={shifted_arr.std():.2f}'
        ax.text(0.98, 0.95, base_text, transform=ax.transAxes, ha='right', va='top',
                fontsize=9, color='C0', fontweight='bold')
        ax.text(0.98, 0.88, shift_text, transform=ax.transAxes, ha='right', va='top',
                fontsize=9, color=color, fontweight='bold')

        ax.set_xlabel('Logit Entropy (nats)')
        ax.set_ylabel('Count')
        ax.set_title(f'Next-token entropy: base vs {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        tag = label.split()[0]  # 'v1' or 'v2'
        save_path = os.path.join(args.results_dir, f'entropy_histogram_{tag}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {save_path}")

    # Save data
    data_path = os.path.join(args.results_dir, 'entropy_data.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(entropies, f)
    print(f"Saved: {data_path}")


if __name__ == '__main__':
    main()
