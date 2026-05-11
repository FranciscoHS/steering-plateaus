#!/usr/bin/env python3
"""
Extra steering (directions 6-10) and variance-explained plot for case studies.

Usage:
    python scripts/dictionary_learning/case_study_extra.py \
        --decomp_pkl results/case_study/eiffel_v2/decomp_results.pkl \
        --steer_prompts "The Eiffel tower is in" "München liegt in"

    python scripts/dictionary_learning/case_study_extra.py \
        --decomp_pkl results/case_study/code_v2/decomp_results.pkl \
        --steer_prompts "def fibonacci(n)" "My favorite thing to do is"
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
from scripts.steering.spherical_steering_generate import generate_with_spherical_steering


def plot_variance_explained(magnitudes, act_norm, save_path, title):
    """Plot cumulative variance explained (Σα² / ||a||²) vs number of directions."""
    alphas_sq = np.array([m**2 for m in magnitudes])
    total_sq = act_norm**2
    cumulative = np.cumsum(alphas_sq) / total_sq

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Cumulative variance explained
    ax = axes[0]
    ax.plot(range(1, len(magnitudes)+1), cumulative, 'o-', color='C0', markersize=5)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of directions')
    ax.set_ylabel('Cumulative Σα² / ||a||²')
    ax.set_title('Cumulative variance explained')
    ax.set_xlim(0.5, len(magnitudes)+0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Per-direction α²
    ax = axes[1]
    ax.bar(range(1, len(magnitudes)+1), alphas_sq / total_sq, color='C0', alpha=0.7)
    ax.set_xlabel('Direction')
    ax.set_ylabel('α² / ||a||²')
    ax.set_title('Per-direction variance fraction')
    ax.set_xlim(0.5, len(magnitudes)+0.5)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decomp_pkl', required=True)
    parser.add_argument('--steer_prompts', nargs='+', required=True)
    parser.add_argument('--dir_start', type=int, default=6, help='First direction index (1-based)')
    parser.add_argument('--dir_end', type=int, default=10, help='Last direction index (1-based)')
    parser.add_argument('--steer_angles', type=str, default='0,20,30,40')
    parser.add_argument('--n_tokens', type=int, default=50)
    args = parser.parse_args()

    with open(args.decomp_pkl, 'rb') as f:
        res = pickle.load(f)

    decomp_args = res['args']
    directions = res['directions']
    magnitudes = res['magnitudes']
    target_prompt = res['target_prompt']
    act_norm = res['target_activation'].norm().item()

    output_dir = os.path.dirname(args.decomp_pkl)
    angles = [float(a) for a in args.steer_angles.split(',')]

    # ── Variance explained plot ──
    plot_path = os.path.join(output_dir, 'variance_explained.png')
    plot_variance_explained(magnitudes, act_norm, plot_path,
                           f"Decomposition: '{target_prompt}' ({decomp_args['model']} layer {decomp_args['layer']})")

    # ── Steering with directions dir_start..dir_end ──
    print(f"\nLoading model: {decomp_args['model']}")
    model, tokenizer, device = load_model(decomp_args['model'])
    layer = decomp_args['layer']

    all_prompts = [target_prompt] + args.steer_prompts
    prompt_labels = ['target'] + [f'steer_{i+1}' for i in range(len(args.steer_prompts))]

    d_start = args.dir_start - 1  # 0-based
    d_end = min(args.dir_end, len(directions))

    print(f"Steering with directions {args.dir_start}-{d_end}, angles={angles}")

    output_lines = []
    output_lines.append(f"Case study: '{target_prompt}' — directions {args.dir_start}-{d_end}")
    output_lines.append(f"Model: {decomp_args['model']}, layer {layer}")
    output_lines.append(f"Angles: {angles}, tokens: {args.n_tokens}")
    output_lines.append('=' * 70)

    for d_idx in range(d_start, d_end):
        direction = directions[d_idx].to(device).float()
        direction = direction / direction.norm()

        output_lines.append(f"\nDIRECTION {d_idx+1} (α={magnitudes[d_idx]:.2f}, "
                          f"response={res['responses'][d_idx]:.4f})")
        output_lines.append('-' * 50)

        for prompt, label in zip(all_prompts, prompt_labels):
            output_lines.append(f"\n  Prompt ({label}): {prompt}")
            for angle in angles:
                text, ids = generate_with_spherical_steering(
                    model, tokenizer, prompt, direction, angle,
                    layer, device, args.n_tokens, steer_mode='rolling'
                )
                output_lines.append(f"    {angle:3.0f}°: {text}")

    output_text = '\n'.join(output_lines)
    txt_path = os.path.join(output_dir, f'completions_d{args.dir_start}-{d_end}.txt')
    with open(txt_path, 'w') as f:
        f.write(output_text)

    print(output_text)
    print(f"\nSaved to {txt_path}")


if __name__ == '__main__':
    main()
