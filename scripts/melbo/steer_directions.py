#!/usr/bin/env python3
"""Steer model generation along given directions at various magnitudes.

Injects a steering vector at a specified layer and generates text.

Usage:
    python scripts/melbo/steer_directions.py --model qwen3-1.7b --inject_layer 7 \
        --directions spectral_K3.pt --magnitudes 0 5 10 20 50
"""

import argparse, os, sys
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F

from src.model import load_model, _get_blocks


def generate_steered(model, tokenizer, device, prompt, inject_layer, steering_vec,
                     magnitude, max_new_tokens=100, temperature=0.7):
    """Generate text with a steering vector added at inject_layer."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    blocks = _get_blocks(model)

    def steering_hook(module, input, output):
        # output is typically a tuple (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + magnitude * steering_vec.to(hidden.device, hidden.dtype)
            return (hidden,) + output[1:]
        else:
            return output + magnitude * steering_vec.to(output.device, output.dtype)

    handle = blocks[inject_layer].register_forward_hook(steering_hook)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )

    handle.remove()

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--inject_layer', type=int, default=7)
    parser.add_argument('--directions', type=str, required=True,
                        help='Path to .pt file with directions [K, d_model]')
    parser.add_argument('--magnitudes', type=float, nargs='+', default=[0, 5, 10, 20, 50])
    parser.add_argument('--prompts', type=str, nargs='+',
                        default=[
                            'The weather today is',
                            'I think the best way to',
                            'Once upon a time,',
                            'The president announced that',
                        ])
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None,
                        help='Output file (default: stdout + results_dir/rollouts.txt)')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = 'results/melbo_multiseed/steer_rollouts'
    os.makedirs(args.results_dir, exist_ok=True)

    output_path = args.output or os.path.join(args.results_dir, 'rollouts.txt')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)

    directions = torch.load(args.directions, weights_only=True)
    n_dirs = len(directions)
    print(f"Loaded {n_dirs} directions from {args.directions}")

    lines = []

    for d_idx in range(n_dirs):
        d_vec = directions[d_idx].to(device)
        d_vec = F.normalize(d_vec, dim=0)

        for mag in args.magnitudes:
            for prompt in args.prompts:
                torch.manual_seed(args.seed)
                text = generate_steered(
                    model, tokenizer, device, prompt,
                    args.inject_layer, d_vec, mag,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature)

                header = f"=== Dir {d_idx} | mag={mag} | prompt=\"{prompt}\" ==="
                print(header)
                print(text)
                print()
                lines.append(header)
                lines.append(text)
                lines.append('')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved rollouts to {output_path}")


if __name__ == '__main__':
    main()
