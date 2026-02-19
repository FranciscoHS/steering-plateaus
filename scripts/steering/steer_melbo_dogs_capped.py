#!/usr/bin/env python3
"""
Spherical steering with capped MELBO directions found on dog breed prompts.

Generates text with norm-preserving rotation to qualitatively check
what each direction actually does.

Usage:
    python scripts/steer_melbo_dogs_capped.py [--model qwen3-0.6b] [--layer 14]
"""

import argparse
import os
import pickle
import sys
import numpy as np
sys.path.insert(0, '.')

import torch
from src.model import load_model, _get_blocks
from data.animal_prompts import DOGS


def generate_with_spherical_steering(model, tokenizer, prompt, steer_direction,
                                     angle_deg, steer_layer, device,
                                     n_tokens=50, temperature=0.7):
    """Generate tokens with spherical (norm-preserving) steering.

    Reprocesses full sequence at each step (no KV cache) for correctness.
    Steers all positions from last prompt token onward ('rolling' mode).
    """
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

            steer_slice = hidden[0, first_steer_pos:, :]  # [n_steer, d]
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
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
        else:
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
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--results_path', default=None)
    parser.add_argument('--n_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.results_path is None:
        args.results_path = 'results/melbo_dogs_capped/qwen3-0.6b_target1.0/results.pkl'

    out_dir = 'results/melbo_dogs_capped/qwen3-0.6b_target1.0'
    out_path = os.path.join(out_dir, 'steering_generations.txt')

    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    print(f"Loading directions from {args.results_path}")
    with open(args.results_path, 'rb') as f:
        results = pickle.load(f)

    directions = results['directions']
    print(f"Loaded {len(directions)} directions")

    # Angles: both positive and negative (negative = steer in opposite direction)
    angles = [-30, -15, -5, 0, 5, 15, 30]

    prompts = [
        "The weather today is",
        "Once upon a time",
        "The best way to cook",
        "In the year 2050,",
    ]

    lines = []
    def out(s=''):
        print(s)
        lines.append(s)

    out(f"Spherical steering with capped MELBO directions (target=1.0)")
    out(f"Model: {args.model}, layer: {args.layer}, n_tokens: {args.n_tokens}, temp: {args.temperature}")
    out(f"Angles: {angles}")
    out(f"Prompts: {prompts}")
    out()

    for d_idx, direction in enumerate(directions):
        direction = direction.to(device)
        # Normalize
        direction = direction / direction.norm()

        out(f"{'='*70}")
        out(f"DIRECTION D_{d_idx+1}")
        out(f"{'='*70}")

        for prompt in prompts:
            out(f"\n  Prompt: \"{prompt}\"")
            for angle in angles:
                text = generate_with_spherical_steering(
                    model, tokenizer, prompt, direction, float(angle),
                    args.layer, device, args.n_tokens,
                    temperature=args.temperature)
                if angle == 0:
                    out(f"    [{angle:+4d} deg (baseline)] {text[:150]}")
                else:
                    out(f"    [{angle:+4d} deg           ] {text[:150]}")
        out()

    # Write to file
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nSaved to {out_path}")
    print("Done!")


if __name__ == '__main__':
    main()
