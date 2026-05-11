#!/usr/bin/env python3
"""
Steer with the v9-optimized direction at various angles.

Loads a direction from a v9 results pkl, picks a few FineWeb prompts,
and generates text with spherical steering at specified angles.

Usage:
    python scripts/delta_optimization/steer_v9_direction.py \
        --direction_pkl results/delta_opt_v9_frob001/results_10deg_kl1.0.pkl \
        --angles 0,15,30,45,60
"""

import argparse
import os
import pickle
import sys
import numpy as np
sys.path.insert(0, '.')

import torch

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length


def generate_with_spherical_steering(model, tokenizer, prompt, steer_direction,
                                     angle_deg, steer_layer, device,
                                     n_tokens=30):
    """Generate tokens with spherical steering at each step (rolling mode)."""
    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    direction = steer_direction.to(device)

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
    parser.add_argument('--direction_pkl', required=True,
                        help='Path to v9 results pkl containing "direction" key')
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--angles', type=str, default='0,15,30,45,60')
    parser.add_argument('--n_prompts', type=int, default=5)
    parser.add_argument('--n_tokens', type=int, default=30)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    angles = [float(a) for a in args.angles.split(',')]

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(args.direction_pkl), 'steering')
    os.makedirs(args.results_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load direction
    print(f"Loading direction from {args.direction_pkl}")
    with open(args.direction_pkl, 'rb') as f:
        data = pickle.load(f)
    direction = data['direction'].float()
    direction = direction / direction.norm()
    print(f"  Direction shape: {direction.shape}")

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    # Get prompts (same FineWeb sequences used in optimization)
    token_ids_list = load_fineweb_fixed_length(
        args.n_prompts, tokenizer, seq_len=args.seq_len, seed=args.seed
    )

    # Generate
    all_results = []
    output_lines = []

    for i in range(args.n_prompts):
        prompt_text = tokenizer.decode(token_ids_list[i], skip_special_tokens=True)
        header = f"\n--- Prompt {i+1}: \"{prompt_text}\" ---"
        print(header)
        output_lines.append(header)

        for angle in angles:
            text = generate_with_spherical_steering(
                model, tokenizer, prompt_text, direction,
                angle, args.layer, device, n_tokens=args.n_tokens
            )
            line = f"  {angle:5.1f} deg: {text}"
            print(line)
            output_lines.append(line)
            all_results.append({
                'prompt': prompt_text,
                'prompt_idx': i,
                'angle_deg': angle,
                'generated_text': text,
            })

    # Save
    results_path = os.path.join(args.results_dir, 'steering_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({'args': vars(args), 'results': all_results}, f)

    txt_path = os.path.join(args.results_dir, 'steering_results.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(output_lines) + '\n')
    print(f"\nSaved: {txt_path}")
    print(f"Saved: {results_path}")


if __name__ == '__main__':
    main()
