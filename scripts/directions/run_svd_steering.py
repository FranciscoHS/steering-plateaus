#!/usr/bin/env python3
"""
SVD decomposition of the English-Mandarin steering direction across all layers.

Extracts paired activations for 50 English/Chinese prompt pairs, forms the
difference matrix D, and computes (uncentered) SVD at every layer. This tells
us where the language direction crystallizes and whether it's 1D or multi-dimensional.

Usage:
    python scripts/run_svd_steering.py --model qwen3-0.6b
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

from src.model import load_model
from data.bilingual_prompts import ENGLISH_PROMPTS, CHINESE_PROMPTS


def extract_all_layers(model, tokenizer, prompts, device):
    """Extract last-token activations at ALL layers for a list of prompts.

    Returns dict mapping layer_idx -> [n_prompts, d_model] tensor (CPU).
    Layer 0 = after block 0, layer L = after block L.
    """
    n_layers = None
    all_acts = None  # will be [n_layers, n_prompts, d_model]

    for i, prompt in enumerate(prompts):
        tokens = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        # hidden_states: tuple of (n_layers+1) tensors [1, seq_len, d_model]
        # index 0 = embedding output, index L+1 = after block L
        hidden_states = outputs.hidden_states

        if all_acts is None:
            # n_layers = number of transformer blocks (exclude embedding layer)
            n_layers = len(hidden_states) - 1
            d_model = hidden_states[0].shape[-1]
            all_acts = torch.zeros(n_layers, len(prompts), d_model)

        for layer in range(n_layers):
            # hidden_states[layer+1] = output after block `layer`
            all_acts[layer, i, :] = hidden_states[layer + 1][0, -1, :].cpu()

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(prompts)} prompts processed")

    return all_acts, n_layers


def compute_svd_diagnostics(D):
    """Compute SVD diagnostics for a difference matrix D [N, d].

    Returns dict with:
        singular_values: all singular values
        variance_explained: fraction of ||D||_F^2 explained by each SV
        cumulative_variance: cumulative fraction explained
        mean_direction: normalized mean of rows
        pc1: first right singular vector (top direction)
        cosine_pc1_mean: cosine similarity between PC1 and mean direction
    """
    # Mean direction (= normalized steering vector)
    mean_dir = D.mean(dim=0)
    mean_norm = mean_dir.norm()
    mean_unit = mean_dir / mean_norm if mean_norm > 1e-10 else mean_dir

    # SVD (uncentered — directly on D)
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    # S: singular values, Vh: [min(N,d), d] right singular vectors

    total_var = (S ** 2).sum()
    var_explained = (S ** 2) / total_var
    cumulative_var = torch.cumsum(var_explained, dim=0)

    pc1 = Vh[0]  # top right singular vector
    cosine_pc1_mean = (pc1 @ mean_unit).abs().item()  # abs because sign is arbitrary

    return {
        'singular_values': S.numpy(),
        'variance_explained': var_explained.numpy(),
        'cumulative_variance': cumulative_var.numpy(),
        'mean_direction': mean_unit.numpy(),
        'mean_norm': mean_norm.item(),
        'pc1': pc1.numpy(),
        'cosine_pc1_mean': cosine_pc1_mean,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-0.6b')
    parser.add_argument('--results_dir', type=str, default=None)
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = f'results/svd_steering/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    # Extract activations at all layers
    print(f"\nExtracting English activations ({len(ENGLISH_PROMPTS)} prompts)...")
    en_acts, n_layers = extract_all_layers(model, tokenizer, ENGLISH_PROMPTS, device)
    print(f"Extracting Chinese activations ({len(CHINESE_PROMPTS)} prompts)...")
    zh_acts, _ = extract_all_layers(model, tokenizer, CHINESE_PROMPTS, device)
    print(f"Model has {n_layers} layers, d_model={d_model}")

    # SVD at every layer
    print(f"\nComputing SVD at all {n_layers} layers...")
    results_by_layer = {}
    for layer in range(n_layers):
        D = en_acts[layer] - zh_acts[layer]  # [50, d_model]
        diag = compute_svd_diagnostics(D)
        results_by_layer[layer] = diag

        # Print summary for this layer
        ve = diag['variance_explained']
        print(f"  Layer {layer:2d}: PC1 explains {ve[0]*100:.1f}%, "
              f"top-5 cumul {diag['cumulative_variance'][min(4, len(ve)-1)]*100:.1f}%, "
              f"cos(PC1,mean)={diag['cosine_pc1_mean']:.4f}, "
              f"mean_norm={diag['mean_norm']:.1f}")

    # Save full results
    results = {
        'args': vars(args),
        'model': args.model,
        'n_layers': n_layers,
        'd_model': d_model,
        'n_pairs': len(ENGLISH_PROMPTS),
        'results_by_layer': results_by_layer,
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {results_path}")

    # --- Plots ---

    layers = list(range(n_layers))

    # 1. PC1 variance explained across layers
    pc1_ve = [results_by_layer[l]['variance_explained'][0] for l in layers]
    top5_ve = [results_by_layer[l]['cumulative_variance'][min(4, len(results_by_layer[l]['cumulative_variance'])-1)] for l in layers]
    cosines = [results_by_layer[l]['cosine_pc1_mean'] for l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(layers, [v * 100 for v in pc1_ve], 'o-', markersize=4)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Variance explained (%)')
    axes[0].set_title('PC1 variance explained')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(layers, [v * 100 for v in top5_ve], 's-', markersize=4, color='tab:orange')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Cumulative variance (%)')
    axes[1].set_title('Top-5 PCs cumulative variance')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(layers, cosines, 'd-', markersize=4, color='tab:green')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('|cos(PC1, mean)|')
    axes[2].set_title('PC1 alignment with mean direction')
    axes[2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(args.results_dir, 'summary_across_layers.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved summary_across_layers.png")

    # 2. Full singular value spectrum at a few representative layers
    representative = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    for l in representative:
        svs = results_by_layer[l]['singular_values']
        ax.plot(range(1, len(svs) + 1), svs / svs[0], 'o-', markersize=3, label=f'Layer {l}')
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Normalized singular value ($\\sigma_k / \\sigma_1$)')
    ax.set_title('Singular value spectrum (normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    fig.savefig(os.path.join(args.results_dir, 'sv_spectrum.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved sv_spectrum.png")

    # 3. Cumulative variance plot for same representative layers
    fig, ax = plt.subplots(figsize=(8, 5))
    for l in representative:
        cv = results_by_layer[l]['cumulative_variance']
        ax.plot(range(1, len(cv) + 1), cv * 100, 'o-', markersize=3, label=f'Layer {l}')
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Cumulative variance explained (%)')
    ax.set_title('Cumulative variance explained')
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(args.results_dir, 'cumulative_variance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved cumulative_variance.png")

    # 4. Mean direction norm across layers (strength of mean language signal)
    mean_norms = [results_by_layer[l]['mean_norm'] for l in layers]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, mean_norms, 'o-', markersize=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('||mean difference||')
    ax.set_title('Norm of mean (en - zh) difference across layers')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(args.results_dir, 'mean_norm.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved mean_norm.png")

    print("\nDone!")


if __name__ == '__main__':
    main()
