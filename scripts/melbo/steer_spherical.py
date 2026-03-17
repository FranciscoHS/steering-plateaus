#!/usr/bin/env python3
"""Spherical steering during generation along given directions.

Rotates activations on the sphere (norm-preserving) at specified angles.

Usage:
    python scripts/melbo/steer_spherical.py --model qwen3-1.7b --inject_layer 7 \
        --directions spectral_K3.pt --angles 0 5 10 15 20
"""

import argparse, os, sys
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F

from src.model import load_model, _get_blocks


def generate_spherical(model, tokenizer, device, prompt, inject_layer, direction,
                       angle_deg, n_tokens=50):
    """Generate text with spherical steering (norm-preserving rotation).

    At each generation step, rotates the activation at inject_layer toward
    direction by angle_deg degrees, preserving the activation norm.
    Greedy decoding, no KV cache (full reprocessing each step).
    """
    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    d_vec = direction.to(device)

    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    prompt_len = input_ids.shape[1]
    first_steer_pos = prompt_len - 1

    generated_ids = []
    blocks = _get_blocks(model)

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

            # Project direction orthogonal to each activation
            proj = (a_unit * d_vec).sum(dim=-1, keepdim=True)
            v_perp = d_vec.unsqueeze(0) - proj * a_unit
            v_perp = v_perp / (v_perp.norm(dim=-1, keepdim=True) + 1e-10)

            steered = R * (cos_theta * a_unit + sin_theta * v_perp)
            hidden[0, first_steer_pos:, :] = steered

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        handle = blocks[inject_layer].register_forward_hook(steering_hook)
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
        handle.remove()

        next_id = outputs.logits[0, -1, :].argmax().item()
        generated_ids.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([
            input_ids,
            torch.tensor([[next_id]], device=device)
        ], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--inject_layer', type=int, default=7)
    parser.add_argument('--directions', type=str, required=True)
    parser.add_argument('--angles', type=float, nargs='+', default=[0, 5, 10, 15, 20])
    parser.add_argument('--prompts', type=str, nargs='+',
                        default=[
                            'The weather today is',
                            'I think the best way to',
                            'Once upon a time,',
                            'The president announced that',
                        ])
    parser.add_argument('--n_tokens', type=int, default=50)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = 'results/melbo_multiseed/steer_spherical'
    os.makedirs(args.results_dir, exist_ok=True)
    output_path = os.path.join(args.results_dir, 'rollouts.txt')

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)

    directions = torch.load(args.directions, weights_only=True)
    n_dirs = len(directions)
    print(f"Loaded {n_dirs} directions from {args.directions}")

    # Generate all rollouts, organized as results[d_idx][prompt] = {angle: text}
    results = {}
    for d_idx in range(n_dirs):
        d_vec = F.normalize(directions[d_idx], dim=0)
        results[d_idx] = {}

        for prompt in args.prompts:
            results[d_idx][prompt] = {}
            for angle in args.angles:
                text = generate_spherical(
                    model, tokenizer, device, prompt,
                    args.inject_layer, d_vec, angle,
                    n_tokens=args.n_tokens)
                results[d_idx][prompt][angle] = text
                print(f"Dir {d_idx} | {angle}° | {prompt[:30]}... -> {text[:60]}...")

    # Write grouped by (direction, prompt), with all angles on consecutive lines
    lines = []
    for d_idx in range(n_dirs):
        lines.append(f"{'='*60}")
        lines.append(f"Direction {d_idx}")
        lines.append(f"{'='*60}")
        lines.append('')
        for prompt in args.prompts:
            lines.append(f"=== Prompt: '{prompt}' ===")
            for angle in args.angles:
                text = results[d_idx][prompt][angle]
                # Escape newlines for compact display
                text_escaped = repr(text)[1:-1]  # strip outer quotes from repr
                lines.append(f"  {angle:5.1f} deg: {text_escaped}")
            lines.append('')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved rollouts to {output_path}")


if __name__ == '__main__':
    main()
