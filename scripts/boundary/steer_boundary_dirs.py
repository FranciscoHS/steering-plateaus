#!/usr/bin/env python3
"""
Steer activations along boundary MELBO directions to see what they do.

Two modes:
  --from_boundary: First rotate to the boundary (along walk_dir by median
    boundary angle), then steer from there along bnd_MELBO directions.
  (default): Steer directly from original activations.

Usage:
    python scripts/boundary/steer_boundary_dirs.py --model qwen3-0.6b --layer 14 --from_lang en
    python scripts/boundary/steer_boundary_dirs.py --model qwen3-0.6b --layer 14 --from_lang en --from_boundary
"""

import argparse
import os
import pickle
import sys
import numpy as np
sys.path.insert(0, '.')

import torch
from src.model import load_model, _get_blocks
from data.bilingual_prompts import CHINESE_PROMPTS, ENGLISH_PROMPTS


def _spherical_rotate(a, direction, cos_t, sin_t):
    """Spherical rotation: rotate a toward direction by angle (given cos/sin).

    a: [..., d], direction: [d] unit vector. Returns same shape as a.
    Preserves norm of a.
    """
    R = a.norm(dim=-1, keepdim=True)
    a_unit = a / (R + 1e-10)
    proj = (a_unit * direction).sum(dim=-1, keepdim=True)
    v_perp = direction.unsqueeze(0) - proj * a_unit
    v_perp = v_perp / (v_perp.norm(dim=-1, keepdim=True) + 1e-10)
    return R * (cos_t * a_unit + sin_t * v_perp)


def generate_with_spherical_steering(model, tokenizer, prompt, steer_direction,
                                     angle_deg, steer_layer, device, n_tokens=20,
                                     pre_direction=None, pre_angle_deg=0.0):
    """Generate tokens with spherical steering applied at every step.

    If pre_direction and pre_angle_deg are given, first rotates toward
    pre_direction by pre_angle_deg, then rotates from there toward
    steer_direction by angle_deg. This implements "steer from boundary".

    No KV cache — full sequence reprocessed at each step for correctness.
    Steers all positions from last prompt token onward (rolling mode).
    """
    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    direction = steer_direction.to(device)

    do_pre = pre_direction is not None and pre_angle_deg > 0
    if do_pre:
        cos_pre = np.cos(np.radians(pre_angle_deg))
        sin_pre = np.sin(np.radians(pre_angle_deg))
        pre_dir = pre_direction.to(device)

    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    prompt_len = input_ids.shape[1]
    first_steer_pos = prompt_len - 1

    generated_ids = []

    for step in range(n_tokens):
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            steer_slice = hidden[0, first_steer_pos:, :]
            if steer_slice.shape[0] == 0:
                return output

            # Step 1: rotate to boundary (if requested)
            if do_pre:
                steer_slice = _spherical_rotate(steer_slice, pre_dir, cos_pre, sin_pre)

            # Step 2: rotate along steer direction
            steered = _spherical_rotate(steer_slice, direction, cos_theta, sin_theta)
            hidden[0, first_steer_pos:, :] = steered

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        layers = _get_blocks(model)
        handle = layers[steer_layer].register_forward_hook(steering_hook)

        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)

        handle.remove()

        logits = outputs.logits[0, -1, :]
        next_token_id = logits.argmax().item()
        generated_ids.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([
            input_ids,
            torch.tensor([[next_token_id]], device=device)
        ], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--from_lang', default='en', choices=['en', 'zh'],
                        help='Starting language for steering prompts')
    parser.add_argument('--from_boundary', action='store_true',
                        help='First rotate to boundary, then steer from there')
    parser.add_argument('--bnd_melbo_path', default=None)
    parser.add_argument('--orig_melbo_path', default=None)
    parser.add_argument('--angles', type=str, default='0,5,10,15,20,30',
                        help='Comma-separated steering angles in degrees')
    parser.add_argument('--n_tokens', type=int, default=30)
    parser.add_argument('--n_prompts', type=int, default=10)
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    if args.bnd_melbo_path is None:
        args.bnd_melbo_path = f'results/boundary_melbo/{args.model}/from_{args.from_lang}/optimization_results.pkl'
    if args.orig_melbo_path is None:
        args.orig_melbo_path = f'results/melbo_constrained/{args.model}/optimization_results.pkl'
    if args.results_dir is None:
        args.results_dir = f'results/boundary_melbo/{args.model}/from_{args.from_lang}'
    os.makedirs(args.results_dir, exist_ok=True)

    angles = [float(a) for a in args.angles.split(',')]

    # Choose prompts based on language
    if args.from_lang == 'en':
        prompts = ENGLISH_PROMPTS[:args.n_prompts]
        src_label = 'English'
    else:
        prompts = CHINESE_PROMPTS[:args.n_prompts]
        src_label = 'Mandarin'

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    # Load directions
    print(f"Loading boundary MELBO from {args.bnd_melbo_path}")
    with open(args.bnd_melbo_path, 'rb') as f:
        bnd_saved = pickle.load(f)
    bnd_dirs = bnd_saved['bnd_melbo_directions']

    print(f"Loading original MELBO from {args.orig_melbo_path}")
    with open(args.orig_melbo_path, 'rb') as f:
        orig_saved = pickle.load(f)
    melbo_1 = orig_saved['melbo_directions'][0]
    mean_dir = orig_saved['mean_dir']

    # Ensure MELBO_1 points English-ward
    if (melbo_1 @ mean_dir) < 0:
        melbo_1 = -melbo_1

    # Set up pre-rotation for boundary mode
    pre_direction = None
    pre_angle_deg = 0.0
    if args.from_boundary:
        median_boundary = float(np.median(bnd_saved['per_prompt_boundary_deg']))
        # Walk direction: same as used in boundary finding
        if args.from_lang == 'en':
            pre_direction = -melbo_1  # toward Mandarin
        else:
            pre_direction = melbo_1   # toward English
        pre_angle_deg = median_boundary
        print(f"\nFrom-boundary mode: pre-rotating {pre_angle_deg:.1f} deg along walk direction")

    # Directions to test: MELBO_1 (calibration) and bnd_MELBO dirs (both signs)
    directions = [
        ('MELBO_1 (->En)', melbo_1),
        ('MELBO_1 (->Zh)', -melbo_1),
        ('bnd_MELBO_1 (+)', bnd_dirs[0]),
        ('bnd_MELBO_1 (-)', -bnd_dirs[0]),
        ('bnd_MELBO_2 (+)', bnd_dirs[1]),
        ('bnd_MELBO_2 (-)', -bnd_dirs[1]),
    ]

    mode_label = f"from boundary ({pre_angle_deg:.0f} deg)" if args.from_boundary else "from original"
    print(f"\nSteering {mode_label}, {len(prompts)} {src_label} prompts")
    print(f"Layer: {args.layer}, angles: {angles}, tokens: {args.n_tokens}")
    print(f"Directions: {[d[0] for d in directions]}")

    output_lines = []
    output_lines.append(f"Mode: {mode_label}")

    for p_idx, prompt in enumerate(prompts):
        header = f"\n{'='*70}\nPrompt [{p_idx}]: {prompt}\n{'='*70}"
        print(header)
        output_lines.append(header)

        for dir_name, direction in directions:
            dir_header = f"\n  --- {dir_name} ---"
            print(dir_header)
            output_lines.append(dir_header)

            for angle in angles:
                text = generate_with_spherical_steering(
                    model, tokenizer, prompt, direction, angle,
                    args.layer, device, n_tokens=args.n_tokens,
                    pre_direction=pre_direction, pre_angle_deg=pre_angle_deg)
                line = f"  {angle:5.1f} deg: {text[:150]}"
                print(line)
                output_lines.append(line)

    # Save text output
    suffix = '_from_boundary' if args.from_boundary else ''
    output_path = os.path.join(args.results_dir, f'steering_outputs{suffix}.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"\nSaved outputs to {output_path}")
    print("Done!")


if __name__ == '__main__':
    main()
