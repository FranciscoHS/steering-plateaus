#!/usr/bin/env python3
"""
Compare Jacobian singular value spectra: original activations vs shifted (a_i + delta).

For each activation, compute top-k singular values of J = d(logits)/d(activation)
at the original point and at the shifted point, then compare.

Usage:
    python scripts/jacobian/compare_spectra.py
    python scripts/jacobian/compare_spectra.py --n_act 20 --k 50
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, get_logits_at_point, _get_blocks
from src.data import load_fineweb_fixed_length
from src.analysis import compute_top_singular_values


def extract_contexts_and_activations(model, token_ids_list, layer, device, batch_size=64):
    """Extract prefix (context) and last-token activation at given layer."""
    all_contexts = []
    all_activations = []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].float().cpu())
        all_activations.append(hidden[:, -1, :].float().cpu())
    contexts = torch.cat(all_contexts, 0)
    activations = torch.cat(all_activations, 0)
    return contexts, activations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=20,
                        help='Number of activations to compute spectra for')
    parser.add_argument('--k', type=int, default=50,
                        help='Number of top singular values to compute')
    parser.add_argument('--v1_results', type=str, default='results/delta_opt/results_r66.pkl')
    parser.add_argument('--v2_results', type=str, default='results/delta_opt_v2/results_r66.pkl')
    parser.add_argument('--which', type=str, default='both', choices=['v1', 'v2', 'both'])
    parser.add_argument('--results_dir', type=str, default='results/jacobian_spectra')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load deltas
    deltas = {}
    for label, path in [('v1', args.v1_results), ('v2', args.v2_results)]:
        if args.which in (label, 'both') and os.path.exists(path):
            with open(path, 'rb') as f:
                results = pickle.load(f)
            deltas[label] = results['delta']
            print(f"Loaded {label}: norm={results['delta'].norm().item():.1f}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    for key in deltas:
        deltas[key] = deltas[key].to(device)

    # Extract activations
    print(f"\nLoading {args.n_act} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print("Extracting activations...")
    contexts, activations = extract_contexts_and_activations(
        model, token_ids_list, args.layer, device
    )
    print(f"Activations: {activations.shape}, mean norm: {activations.norm(dim=-1).mean():.1f}")

    # Compute spectra
    all_spectra = {'base': []}
    for label in deltas:
        all_spectra[label] = []

    t0 = time.time()
    for i in range(args.n_act):
        ctx = contexts[i:i+1].to(device)
        act = activations[i].to(device)

        # Base spectrum
        point = act.reshape(1, 1, -1)
        svs_base, _ = compute_top_singular_values(model, ctx, point, layer=args.layer, k=args.k)
        all_spectra['base'].append(svs_base)

        # Shifted spectra
        for label, delta in deltas.items():
            shifted = (act + delta).reshape(1, 1, -1)
            svs_shifted, _ = compute_top_singular_values(model, ctx, shifted, layer=args.layer, k=args.k)
            all_spectra[label].append(svs_shifted)

        elapsed = time.time() - t0
        print(f"  [{i+1}/{args.n_act}] ({elapsed:.0f}s) "
              f"base top-5: {svs_base[:5].round(1)}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")

    # Save
    save_path = os.path.join(args.results_dir, 'spectra.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({'args': vars(args), 'spectra': all_spectra}, f)
    print(f"Saved: {save_path}")

    # Print summary stats
    for label, spectra_list in all_spectra.items():
        arr = np.stack(spectra_list)  # [n_act, k]
        mean_spectrum = arr.mean(axis=0)
        print(f"\n{label} — mean top-10 SVs: {mean_spectrum[:10].round(1)}")
        print(f"  Frobenius (from top-{args.k}): {np.sqrt((arr**2).sum(axis=1)).mean():.1f}")

    # Plot: mean spectrum comparison
    for label in deltas:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        base_arr = np.stack(all_spectra['base'])
        shift_arr = np.stack(all_spectra[label])

        # Left: mean spectrum (log scale)
        ax = axes[0]
        base_mean = base_arr.mean(axis=0)
        shift_mean = shift_arr.mean(axis=0)
        base_std = base_arr.std(axis=0)
        shift_std = shift_arr.std(axis=0)
        x = np.arange(1, args.k + 1)
        ax.plot(x, base_mean, 'b-', linewidth=2, label='Base')
        ax.fill_between(x, base_mean - base_std, base_mean + base_std, color='blue', alpha=0.15)
        ax.plot(x, shift_mean, 'r-', linewidth=2, label=f'{label} shifted')
        ax.fill_between(x, shift_mean - shift_std, shift_mean + shift_std, color='red', alpha=0.15)
        ax.set_xlabel('Singular value index')
        ax.set_ylabel('Singular value')
        ax.set_yscale('log')
        ax.set_title(f'Mean spectrum (top {args.k})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: ratio (shifted / base) per index
        ax = axes[1]
        ratios = shift_arr / (base_arr + 1e-10)
        ratio_mean = ratios.mean(axis=0)
        ratio_std = ratios.std(axis=0)
        ax.plot(x, ratio_mean, 'k-', linewidth=2)
        ax.fill_between(x, ratio_mean - ratio_std, ratio_mean + ratio_std, color='gray', alpha=0.2)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Singular value index')
        ax.set_ylabel('Ratio (shifted / base)')
        ax.set_title(f'SV ratio: {label} shifted / base')
        ax.grid(True, alpha=0.3)

        fig.suptitle(f'Jacobian spectrum: base vs {label} shifted (layer {args.layer})', fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(args.results_dir, f'spectrum_{label}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
