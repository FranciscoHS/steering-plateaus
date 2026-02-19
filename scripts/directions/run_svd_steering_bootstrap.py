#!/usr/bin/env python3
"""
Bootstrap stability analysis for SVD of English-Mandarin steering direction.

Subsamples pairs to check whether the SVD structure (variance explained,
PC1 direction) is stable or an artifact of small N.

Usage:
    python scripts/run_svd_steering_bootstrap.py --model qwen3-0.6b
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
    """Extract last-token activations at ALL layers for a list of prompts."""
    all_acts = None
    n_layers = None

    for i, prompt in enumerate(prompts):
        tokens = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        if all_acts is None:
            n_layers = len(hidden_states) - 1
            d_model = hidden_states[0].shape[-1]
            all_acts = torch.zeros(n_layers, len(prompts), d_model)

        for layer in range(n_layers):
            all_acts[layer, i, :] = hidden_states[layer + 1][0, -1, :].cpu()

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(prompts)} prompts processed")

    return all_acts, n_layers


def svd_diagnostics(D):
    """Compute SVD diagnostics for difference matrix D [N, d]."""
    mean_dir = D.mean(dim=0)
    mean_norm = mean_dir.norm()
    mean_unit = mean_dir / mean_norm if mean_norm > 1e-10 else mean_dir

    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    total_var = (S ** 2).sum()
    var_explained = (S ** 2) / total_var

    pc1 = Vh[0]
    cosine_pc1_mean = (pc1 @ mean_unit).abs().item()

    return {
        'var_explained': var_explained.numpy(),
        'pc1': pc1,
        'cosine_pc1_mean': cosine_pc1_mean,
        'mean_unit': mean_unit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-0.6b')
    parser.add_argument('--n_bootstrap', type=int, default=20,
                        help='Number of bootstrap subsamples')
    parser.add_argument('--subsample_size', type=int, default=25,
                        help='Number of pairs per subsample')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    N = len(ENGLISH_PROMPTS)

    if args.results_dir is None:
        args.results_dir = f'results/svd_steering/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model and extract activations
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']

    print(f"\nExtracting English activations ({N} prompts)...")
    en_acts, n_layers = extract_all_layers(model, tokenizer, ENGLISH_PROMPTS, device)
    print(f"Extracting Chinese activations ({N} prompts)...")
    zh_acts, _ = extract_all_layers(model, tokenizer, CHINESE_PROMPTS, device)
    print(f"Model has {n_layers} layers, d_model={d_model}")

    # Full-data SVD (reference)
    print("\nComputing full-data SVD...")
    full_results = {}
    for layer in range(n_layers):
        D_full = en_acts[layer] - zh_acts[layer]
        full_results[layer] = svd_diagnostics(D_full)

    # Bootstrap subsampling
    print(f"\nBootstrap: {args.n_bootstrap} subsamples of {args.subsample_size} pairs each")
    boot_pc1_ve = np.zeros((n_layers, args.n_bootstrap))
    boot_cosine_pc1_mean = np.zeros((n_layers, args.n_bootstrap))
    boot_cosine_pc1_fullpc1 = np.zeros((n_layers, args.n_bootstrap))
    # For anomalous layers: check if PC1 directions are consistent across subsamples
    boot_pc1_pairwise_cosines = {layer: [] for layer in range(n_layers)}

    for b in range(args.n_bootstrap):
        idx = np.random.choice(N, size=args.subsample_size, replace=False)
        for layer in range(n_layers):
            D_sub = en_acts[layer][idx] - zh_acts[layer][idx]
            diag = svd_diagnostics(D_sub)
            boot_pc1_ve[layer, b] = diag['var_explained'][0]
            boot_cosine_pc1_mean[layer, b] = diag['cosine_pc1_mean']
            # Cosine between subsample PC1 and full-data PC1
            cos_full = (diag['pc1'] @ full_results[layer]['pc1']).abs().item()
            boot_cosine_pc1_fullpc1[layer, b] = cos_full

        if (b + 1) % 5 == 0:
            print(f"  {b+1}/{args.n_bootstrap} bootstrap samples done")

    # Print summary for interesting layers
    print("\n=== Bootstrap summary ===")
    for layer in range(n_layers):
        full_ve = full_results[layer]['var_explained'][0]
        full_cos = full_results[layer]['cosine_pc1_mean']
        ve_mean = boot_pc1_ve[layer].mean()
        ve_std = boot_pc1_ve[layer].std()
        cos_mean = boot_cosine_pc1_mean[layer].mean()
        cos_std = boot_cosine_pc1_mean[layer].std()
        pc1_stab = boot_cosine_pc1_fullpc1[layer].mean()
        pc1_stab_std = boot_cosine_pc1_fullpc1[layer].std()
        print(f"  Layer {layer:2d}: full_ve={full_ve*100:.1f}% "
              f"boot_ve={ve_mean*100:.1f}+/-{ve_std*100:.1f}% | "
              f"full_cos={full_cos:.3f} "
              f"boot_cos={cos_mean:.3f}+/-{cos_std:.3f} | "
              f"PC1_stability={pc1_stab:.3f}+/-{pc1_stab_std:.3f}")

    # Save
    results = {
        'args': vars(args),
        'n_layers': n_layers,
        'full_results': {l: {'var_explained': full_results[l]['var_explained'],
                             'cosine_pc1_mean': full_results[l]['cosine_pc1_mean']}
                         for l in range(n_layers)},
        'boot_pc1_ve': boot_pc1_ve,
        'boot_cosine_pc1_mean': boot_cosine_pc1_mean,
        'boot_cosine_pc1_fullpc1': boot_cosine_pc1_fullpc1,
    }
    with open(os.path.join(args.results_dir, 'bootstrap_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # --- Plots ---

    layers = list(range(n_layers))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. PC1 variance explained: full vs bootstrap
    ax = axes[0]
    full_ve = [full_results[l]['var_explained'][0] * 100 for l in layers]
    ax.plot(layers, full_ve, 'o-', markersize=4, label='Full (N=50)', zorder=3)
    boot_mean = boot_pc1_ve.mean(axis=1) * 100
    boot_std = boot_pc1_ve.std(axis=1) * 100
    ax.fill_between(layers, boot_mean - boot_std, boot_mean + boot_std,
                     alpha=0.3, color='tab:orange')
    ax.plot(layers, boot_mean, 's-', markersize=3, color='tab:orange',
            label=f'Bootstrap (N={args.subsample_size})')
    ax.set_xlabel('Layer')
    ax.set_ylabel('PC1 variance explained (%)')
    ax.set_title('PC1 variance: full vs bootstrap')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. cos(PC1, mean): full vs bootstrap
    ax = axes[1]
    full_cos = [full_results[l]['cosine_pc1_mean'] for l in layers]
    ax.plot(layers, full_cos, 'o-', markersize=4, label='Full (N=50)', zorder=3)
    boot_cos_mean = boot_cosine_pc1_mean.mean(axis=1)
    boot_cos_std = boot_cosine_pc1_mean.std(axis=1)
    ax.fill_between(layers, boot_cos_mean - boot_cos_std, boot_cos_mean + boot_cos_std,
                     alpha=0.3, color='tab:orange')
    ax.plot(layers, boot_cos_mean, 's-', markersize=3, color='tab:orange',
            label=f'Bootstrap (N={args.subsample_size})')
    ax.set_xlabel('Layer')
    ax.set_ylabel('|cos(PC1, mean)|')
    ax.set_title('PC1-mean alignment: full vs bootstrap')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. PC1 stability: cosine between subsample PC1 and full-data PC1
    ax = axes[2]
    stab_mean = boot_cosine_pc1_fullpc1.mean(axis=1)
    stab_std = boot_cosine_pc1_fullpc1.std(axis=1)
    ax.fill_between(layers, stab_mean - stab_std, stab_mean + stab_std,
                     alpha=0.3, color='tab:green')
    ax.plot(layers, stab_mean, 'd-', markersize=4, color='tab:green')
    ax.set_xlabel('Layer')
    ax.set_ylabel('|cos(PC1_boot, PC1_full)|')
    ax.set_title('PC1 direction stability across subsamples')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(args.results_dir, 'bootstrap_stability.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved bootstrap_stability.png")
    print("Done!")


if __name__ == '__main__':
    main()
