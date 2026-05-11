#!/usr/bin/env python3
"""
Steer with response-decomposition directions and generate completions.

Loads top-k directions from response_decomp results, steers the target
activation's prompt + a few other prompts at various angles, and saves
all completions to a text file.

Usage:
    python scripts/delta_optimization/steer_response_dirs.py
    python scripts/delta_optimization/steer_response_dirs.py --results_pkl results/response_decomp_k30/results.pkl --n_dirs 3
"""

import argparse
import math
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.steering.spherical_steering_generate import generate_with_spherical_steering


def main():
    parser = argparse.ArgumentParser(description='Steer with response decomp directions')
    parser.add_argument('--results_pkl', default='results/response_decomp_k30/results.pkl')
    parser.add_argument('--n_dirs', type=int, default=3, help='Number of top directions to steer with')
    parser.add_argument('--angles', type=str, default='0,15,30,60',
                        help='Comma-separated steering angles in degrees')
    parser.add_argument('--n_tokens', type=int, default=30)
    parser.add_argument('--extra_prompts', type=str, nargs='+', default=None,
                        help='Additional prompts to steer (default: 2 FineWeb prompts)')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()

    # Load decomposition results
    with open(args.results_pkl, 'rb') as f:
        res = pickle.load(f)

    decomp_args = res['args']
    directions = res['directions'][:args.n_dirs]
    magnitudes = res['magnitudes'][:args.n_dirs]

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.results_pkl), 'steering')
    os.makedirs(args.output_dir, exist_ok=True)

    angles = [float(a) for a in args.angles.split(',')]

    print(f"Config: model={decomp_args['model']}, layer={decomp_args['layer']}")
    print(f"  {args.n_dirs} directions, angles={angles}")

    # Load model
    model, tokenizer, device = load_model(decomp_args['model'])
    layer = decomp_args['layer']

    # Get prompts: the original target prompt + 2 others
    token_ids_list = load_fineweb_fixed_length(
        decomp_args['n_response_acts'] + 1, tokenizer,
        seq_len=decomp_args['seq_len'], seed=decomp_args['seed']
    )

    # Decode the target prompt
    target_ids = token_ids_list[decomp_args['sample_idx']]
    target_prompt = tokenizer.decode(target_ids, skip_special_tokens=True)

    # Get 2 other prompts from a different seed
    other_ids_list = load_fineweb_fixed_length(
        5, tokenizer, seq_len=decomp_args['seq_len'], seed=999
    )
    other_prompts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in other_ids_list[:2]]

    if args.extra_prompts:
        prompts = [target_prompt] + args.extra_prompts
        prompt_labels = ['target'] + [f'custom_{i}' for i in range(len(args.extra_prompts))]
    else:
        prompts = [target_prompt] + other_prompts
        prompt_labels = ['target', 'other_1', 'other_2']

    # Generate completions
    output_lines = []
    output_lines.append(f"Response-decomposition steering: {decomp_args['model']} layer {layer}")
    output_lines.append(f"Directions: top {args.n_dirs}, angles: {angles}")
    output_lines.append(f"Tokens per completion: {args.n_tokens}")
    output_lines.append('=' * 70)

    for d_idx in range(args.n_dirs):
        direction = directions[d_idx].to(device).float()
        direction = direction / direction.norm()

        output_lines.append(f"\n{'='*70}")
        output_lines.append(f"DIRECTION {d_idx+1} (α={magnitudes[d_idx]:.2f})")
        output_lines.append(f"{'='*70}")

        for p_idx, (prompt, label) in enumerate(zip(prompts, prompt_labels)):
            output_lines.append(f"\n--- Prompt ({label}): {prompt[:80]}... ---")

            for angle in angles:
                text, ids = generate_with_spherical_steering(
                    model, tokenizer, prompt, direction, angle,
                    layer, device, args.n_tokens, steer_mode='rolling'
                )
                output_lines.append(f"  {angle:3.0f}°: {text}")

    # Save
    output_text = '\n'.join(output_lines)
    txt_path = os.path.join(args.output_dir, 'completions.txt')
    with open(txt_path, 'w') as f:
        f.write(output_text)

    print(output_text)
    print(f"\nSaved to {txt_path}")


if __name__ == '__main__':
    main()
