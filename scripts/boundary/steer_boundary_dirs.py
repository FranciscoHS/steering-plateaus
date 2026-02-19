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


def generate_with_multi_pre_rotation(model, tokenizer, prompt, steer_direction,
                                     angle_deg, steer_layer, device, n_tokens=20,
                                     pre_steps=None):
    """Generate tokens with compound spherical steering.

    pre_steps: list of (direction, angle_deg) pairs to apply sequentially before
    the final steer_direction rotation. This chains boundary walks.
    """
    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    direction = steer_direction.to(device)

    pre_steps_dev = []
    if pre_steps:
        for d, a in pre_steps:
            pre_steps_dev.append((d.to(device), np.cos(np.radians(a)), np.sin(np.radians(a))))

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

            # Chain pre-rotations
            for pre_dir, cos_p, sin_p in pre_steps_dev:
                steer_slice = _spherical_rotate(steer_slice, pre_dir, cos_p, sin_p)

            # Final steer direction
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
    parser.add_argument('--iterative', action='store_true',
                        help='Steer with all directions from iterative boundary MELBO')
    parser.add_argument('--iterative_path', default=None)
    parser.add_argument('--bnd_melbo_path', default=None)
    parser.add_argument('--orig_melbo_path', default=None)
    parser.add_argument('--angles', type=str, default='0,5,10,15,20,30',
                        help='Comma-separated steering angles in degrees')
    parser.add_argument('--n_tokens', type=int, default=30)
    parser.add_argument('--n_prompts', type=int, default=5)
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    if args.bnd_melbo_path is None:
        args.bnd_melbo_path = f'results/boundary_melbo/{args.model}/from_{args.from_lang}/optimization_results.pkl'
    if args.orig_melbo_path is None:
        args.orig_melbo_path = f'results/melbo_constrained/{args.model}/optimization_results.pkl'
    if args.iterative_path is None:
        args.iterative_path = f'results/iterative_boundary_melbo/{args.model}/results.pkl'
    if args.results_dir is None:
        args.results_dir = f'results/iterative_boundary_melbo/{args.model}'
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

    # Load original MELBO
    print(f"Loading original MELBO from {args.orig_melbo_path}")
    with open(args.orig_melbo_path, 'rb') as f:
        orig_saved = pickle.load(f)
    melbo_1 = orig_saved['melbo_directions'][0]
    mean_dir = orig_saved['mean_dir']
    if (melbo_1 @ mean_dir) < 0:
        melbo_1 = -melbo_1

    if args.iterative:
        _run_iterative_steering(model, tokenizer, device, args, prompts,
                                src_label, angles, melbo_1)
    else:
        _run_single_steering(model, tokenizer, device, args, prompts,
                             src_label, angles, melbo_1)


def _run_single_steering(model, tokenizer, device, args, prompts, src_label, angles, melbo_1):
    """Original single-iteration steering mode."""
    print(f"Loading boundary MELBO from {args.bnd_melbo_path}")
    with open(args.bnd_melbo_path, 'rb') as f:
        bnd_saved = pickle.load(f)
    bnd_dirs = bnd_saved['bnd_melbo_directions']

    pre_direction = None
    pre_angle_deg = 0.0
    if args.from_boundary:
        median_boundary = float(np.median(bnd_saved['per_prompt_boundary_deg']))
        if args.from_lang == 'en':
            pre_direction = -melbo_1
        else:
            pre_direction = melbo_1
        pre_angle_deg = median_boundary
        print(f"\nFrom-boundary mode: pre-rotating {pre_angle_deg:.1f} deg along walk direction")

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

    output_lines = [f"Mode: {mode_label}"]

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

    suffix = '_from_boundary' if args.from_boundary else ''
    output_path = os.path.join(args.results_dir, f'steering_outputs{suffix}.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"\nSaved outputs to {output_path}")


def _run_iterative_steering(model, tokenizer, device, args, prompts, src_label, angles, melbo_1):
    """Steer with directions from all iterations, from original + from boundary."""
    print(f"Loading iterative results from {args.iterative_path}")
    with open(args.iterative_path, 'rb') as f:
        iter_results = pickle.load(f)

    iterations = iter_results['iterations']

    # Build the chain of walk directions + boundary angles for compound rotation
    # iter0: walked along -MELBO_1, boundary at angle_0
    # iter1: walked along best_bnd_MELBO from iter0, boundary at angle_1
    # So to reach iter1's boundary, you chain: rotate(-MELBO_1, angle_0) then rotate(walk_1, angle_1)
    walk_chain = []  # list of (direction, median_boundary_angle)
    for it in iterations:
        walk_label = it['walk_label']
        bnd_angles = it['boundary_angles_deg']
        valid = [a for a in bnd_angles if a is not None]
        median_bnd = float(np.median(valid))

        # Reconstruct walk direction
        if it['iteration'] == 0:
            if args.from_lang == 'en':
                walk_dir = -melbo_1  # toward Zh
            else:
                walk_dir = melbo_1   # toward En
        else:
            # Walk dir was the best bnd_MELBO from previous iteration
            prev_it = iterations[it['iteration'] - 1]
            prev_kls = prev_it['bnd_melbo_kls']
            best_idx = int(np.argmax(prev_kls))
            walk_dir = prev_it['bnd_melbo_directions'][best_idx]

        walk_chain.append((walk_dir, median_bnd))

    # For each iteration, collect directions to steer with
    for mode_name, from_boundary in [('from_original', False), ('from_boundary', True)]:
        output_lines = [f"Mode: {mode_name}"]
        print(f"\n{'#'*70}")
        print(f"# {mode_name}")
        print(f"{'#'*70}")

        for it_idx, it in enumerate(iterations):
            # Skip iter0 — we've already steered with those
            if it_idx == 0:
                continue

            bnd_dirs = it['bnd_melbo_directions']
            directions = []
            for j, d in enumerate(bnd_dirs):
                directions.append((f'iter{it_idx}_bndM{j+1} (+)', d))
                directions.append((f'iter{it_idx}_bndM{j+1} (-)', -d))

            # Pre-rotation steps to reach this iteration's boundary
            pre_steps = None
            if from_boundary:
                # Chain all walks up to and including this iteration's boundary
                pre_steps = walk_chain[:it_idx + 1]

            pre_label = ""
            if pre_steps:
                chain_desc = " -> ".join([f"{a:.0f}deg" for _, a in pre_steps])
                pre_label = f" (chain: {chain_desc})"

            it_header = f"\n--- Iteration {it_idx} directions{pre_label} ---"
            print(it_header)
            output_lines.append(it_header)

            for p_idx, prompt in enumerate(prompts):
                header = f"\n{'='*70}\nPrompt [{p_idx}]: {prompt}\n{'='*70}"
                print(header)
                output_lines.append(header)

                for dir_name, direction in directions:
                    dir_header = f"\n  --- {dir_name} ---"
                    print(dir_header)
                    output_lines.append(dir_header)

                    for angle in angles:
                        text = generate_with_multi_pre_rotation(
                            model, tokenizer, prompt, direction, angle,
                            args.layer, device, n_tokens=args.n_tokens,
                            pre_steps=pre_steps)
                        line = f"  {angle:5.1f} deg: {text[:150]}"
                        print(line)
                        output_lines.append(line)

        output_path = os.path.join(args.results_dir, f'steering_outputs_{mode_name}.txt')
        with open(output_path, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"\nSaved to {output_path}")

    print("Done!")


if __name__ == '__main__':
    main()
