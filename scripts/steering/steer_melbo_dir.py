#!/usr/bin/env python3
"""Steer generation along a MELBO direction using spherical steering.

Usage:
    python scripts/steering/steer_melbo_dir.py --dir_idx 76 --prompt "This movie was absolutely" \
        --angles 0,10,15,20,25,30 --checkpoint results/iterative_melbo_hf/.../checkpoint.pkl
"""
import argparse
import os
import pickle
import sys
import numpy as np
sys.path.insert(0, '.')

import torch
from src.model import load_model, _get_blocks


def generate_with_spherical_steering(model, tokenizer, prompt, steer_direction,
                                     angle_deg, steer_layer, device, n_tokens=30):
    """Generate tokens with spherical steering at last-token position."""
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
        input_ids = torch.cat([input_ids,
                               torch.tensor([[next_token_id]], device=device)], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=7)
    parser.add_argument('--dir_idx', type=int, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prompt', type=str, default='This movie was absolutely')
    parser.add_argument('--angles', type=str, default='0,10,15,20,25,30')
    parser.add_argument('--n_tokens', type=int, default=30)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    angles = [float(a) for a in args.angles.split(',')]

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    directions = ckpt['directions']
    print(f"  {len(directions)} directions available")

    direction = directions[args.dir_idx]
    if isinstance(direction, np.ndarray):
        direction = torch.from_numpy(direction).float()
    direction = direction / direction.norm()
    print(f"  Using direction {args.dir_idx}")

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)

    # Run steering
    prompts = [p.strip() for p in args.prompt.split('|')]

    lines = []
    for prompt in prompts:
        lines.append(f"\n{'='*60}")
        lines.append(f"PROMPT: {prompt}")
        for angle in angles:
            text = generate_with_spherical_steering(
                model, tokenizer, prompt, direction, angle,
                args.layer, device, n_tokens=args.n_tokens
            )
            line = f"  {angle:4.0f}deg: {prompt}{text}"
            print(line)
            lines.append(line)

    # Save log
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"\nSaved to {args.output}")

    print("\nDone")


if __name__ == '__main__':
    main()
