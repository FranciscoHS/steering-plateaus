#!/usr/bin/env python3
"""
Small-magnitude steering test: compare PC1, MELBO_1, and mean at mag 1-5.

Single English prompt steered toward Mandarin (negative magnitudes).
Saves output to a text file.

Usage:
    python scripts/test_small_mag_steering.py --model qwen3-0.6b
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
from src.model import load_model


def generate_with_steering(model, tokenizer, prompt, direction, magnitude,
                           steer_layer, device, max_new_tokens=50,
                           greedy=False):
    """Generate text with additive steering at a given layer."""
    steer_vec = (magnitude * direction).to(device)

    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + steer_vec
            return (hidden,) + output[1:]
        else:
            return output + steer_vec

    handle = model.model.layers[steer_layer].register_forward_hook(steering_hook)

    tokens = tokenizer(prompt, return_tensors='pt').to(device)
    torch.manual_seed(42)
    with torch.no_grad():
        if greedy:
            output = model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        else:
            output = model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_k=0,
                top_p=1.0,
            )

    handle.remove()

    generated = tokenizer.decode(output[0][tokens['input_ids'].shape[1]:],
                                 skip_special_tokens=True)
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--results_path', default=None)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--output_file', default='results/small_mag_steering.txt')
    args = parser.parse_args()

    if args.results_path is None:
        args.results_path = f'results/melbo_constrained/{args.model}/results.pkl'

    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    print(f"Loading directions from {args.results_path}")
    with open(args.results_path, 'rb') as f:
        results = pickle.load(f)

    mean_dir = results['mean_dir']
    pc1 = results['pc1']
    melbo_1 = results['melbo_directions'][0]

    # Ensure all point English-ward (positive cosine with mean)
    if (pc1 @ mean_dir) < 0:
        pc1 = -pc1
    if (melbo_1 @ mean_dir) < 0:
        melbo_1 = -melbo_1

    directions = {
        'mean': mean_dir,
        'PC1': pc1,
        'MELBO_1': melbo_1,
    }

    prompt = "The weather today is"
    magnitudes = [-5, -6, -7, -8]

    lines = []
    lines.append(f"Small-magnitude steering test (fine-grained 5-8)")
    lines.append(f"Prompt: \"{prompt}\"")
    lines.append(f"Layer: {args.layer}")
    lines.append(f"Magnitudes: {magnitudes} (negative = toward Mandarin)")
    lines.append(f"Max new tokens: {args.max_new_tokens}")
    lines.append("")

    for greedy in [True, False]:
        mode = "GREEDY" if greedy else "TEMP=0.3"
        lines.append(f"{'='*60}")
        lines.append(f"  {mode}")
        lines.append(f"{'='*60}")
        lines.append("")

        # Baseline
        text = generate_with_steering(
            model, tokenizer, prompt, mean_dir, 0.0,
            args.layer, device, args.max_new_tokens, greedy=greedy)
        lines.append(f"Baseline (mag=0): {text}")
        lines.append("")

        for mag in magnitudes:
            lines.append(f"--- mag={mag} ---")
            for name, direction in directions.items():
                text = generate_with_steering(
                    model, tokenizer, prompt, direction, float(mag),
                    args.layer, device, args.max_new_tokens, greedy=greedy)
                lines.append(f"  {name:8s}: {text}")
            lines.append("")

    output = "\n".join(lines)
    print(output)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        f.write(output)
    print(f"\nSaved to {args.output_file}")


if __name__ == '__main__':
    main()
