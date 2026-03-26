#!/usr/bin/env python3
"""Steer with a direction found by small-angle MELBO and generate completions.

Loads a direction from iterative_melbo_hf results, then generates text
at various steering angles using spherical (norm-preserving) steering.

Usage:
    python scripts/melbo/steer_smallangle_direction.py \
        --results_pkl results/iterative_melbo_hf/qwen3-1.7b_L7_meas-2_angle0.1/results.pkl \
        --angles 0,5,10,15
"""

import argparse
import os
import pickle
import sys
import numpy as np
sys.path.insert(0, '.')

import torch
from src.model import load_model, _get_blocks


def generate_with_spherical_steering(model, tokenizer, prompt, direction,
                                     angle_deg, steer_layer, device,
                                     n_tokens=50):
    """Generate tokens with spherical steering at each step."""
    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    direction = direction.to(device)

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

            R = steer_slice.norm(dim=-1, keepdim=True)
            a_unit = steer_slice / (R + 1e-10)
            proj = (a_unit * direction).sum(dim=-1, keepdim=True)
            v_perp = direction.unsqueeze(0) - proj * a_unit
            v_perp = v_perp / (v_perp.norm(dim=-1, keepdim=True) + 1e-10)
            steered = R * (cos_theta * a_unit + sin_theta * v_perp)
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

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_pkl', type=str, required=True,
                        help='Path to results.pkl from iterative_melbo_hf')
    parser.add_argument('--dir_idx', type=str, default='0',
                        help='Direction index or comma-separated indices (e.g. 0,1,2)')
    parser.add_argument('--angles', type=str, default='0,5,10,15',
                        help='Comma-separated steering angles in degrees')
    parser.add_argument('--n_tokens', type=int, default=50)
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom prompt(s), comma-separated. If not set, uses defaults.')
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    args = parser.parse_args()

    angles = [float(a) for a in args.angles.split(',')]

    # Load direction
    print(f"Loading results from {args.results_pkl}")
    with open(args.results_pkl, 'rb') as f:
        data = pickle.load(f)

    dir_indices = [int(x) for x in args.dir_idx.split(',')]
    directions = {i: data['directions'][i] for i in dir_indices}
    melbo_args = data['args']
    model_name = melbo_args['model']
    steer_layer = melbo_args['inject_layer']
    opt_angle = melbo_args['opt_angle']
    print(f"Directions {dir_indices}: optimized at {opt_angle}° on {model_name} layer {steer_layer}")

    # Load model
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    print(f"Loading model: {model_name}")
    model, tokenizer, device = load_model(model_name, dtype=model_dtype)

    # Prompts
    if args.prompt is not None:
        prompts = [p.strip() for p in args.prompt.split(',')]
    else:
        prompts = [
            "The capital of France is",
            "In recent years, artificial intelligence has",
            "Once upon a time, in a small village,",
            "The entrance to Simbol",
        ]

    # Generate
    results_dir = os.path.dirname(args.results_pkl)
    outpath = os.path.join(results_dir, 'steering_generations.txt')

    lines = []
    for di in dir_indices:
        direction = directions[di]
        dir_header = f"\n{'#'*70}\n# Direction {di}\n{'#'*70}"
        print(dir_header)
        lines.append(dir_header)
        for prompt in prompts:
            header = f"\n{'='*70}\nPrompt: \"{prompt}\"\n{'='*70}"
            print(header)
            lines.append(header)
            for angle in angles:
                text = generate_with_spherical_steering(
                    model, tokenizer, prompt, direction, angle,
                    steer_layer, device, n_tokens=args.n_tokens)
                line = f"  {angle:5.1f}°: {text}"
                print(line)
                lines.append(line)

    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nSaved to {outpath}")


if __name__ == '__main__':
    main()
