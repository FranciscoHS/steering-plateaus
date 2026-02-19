#!/usr/bin/env python3
"""
Steering validation with random direction controls.

For each code-adjacent prompt, compare:
  - Unsteered baseline
  - Steering along each language direction
  - Steering along several random directions

This tests whether the language directions produce qualitatively different
behavior from random perturbation.

Usage:
    python scripts/directions/validate_steering_with_random.py [--model qwen3-0.6b]
"""

import sys
import argparse
import os
import pickle
import numpy as np

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import load_model


def steer_and_generate(model, tokenizer, device, input_ids, layer, direction,
                       magnitude_deg, max_new_tokens=60):
    """Generate text with spherical steering at the given layer."""
    direction_unit = (direction / direction.norm()).to(device)

    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        a = hidden[:, -1, :]
        a_norm = a.norm(dim=-1, keepdim=True)
        a_unit = a / (a_norm + 1e-10)
        proj = (a_unit * direction_unit).sum(dim=-1, keepdim=True)
        u_perp = direction_unit.unsqueeze(0) - proj * a_unit
        u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)

        angle = np.radians(magnitude_deg)
        a_steered = a_norm * (np.cos(angle) * a_unit + np.sin(angle) * u_perp)
        hidden = hidden.clone()
        hidden[:, -1, :] = a_steered

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    blocks = model.model.layers if hasattr(model, 'model') else model.transformer.h
    hook_handle = blocks[layer].register_forward_hook(steering_hook)

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    finally:
        hook_handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--magnitude', type=float, default=20.0)
    parser.add_argument('--n_random', type=int, default=5,
                        help='Number of random directions to test')
    parser.add_argument('--max_tokens', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load directions
    results_dir = f'results/programming_contrastive/{args.model}'
    with open(os.path.join(results_dir, 'contrastive_results.pkl'), 'rb') as f:
        results = pickle.load(f)
    languages = results['languages']
    f_hat = results['f_hat']
    d_model = f_hat[languages[0]].shape[0]

    # Generate random directions
    random_dirs = []
    for i in range(args.n_random):
        d = torch.randn(d_model)
        d = d / d.norm()
        random_dirs.append(d)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    prompts = [
        "The following code",
        "Write a function that",
        "Here is a simple",
    ]

    for prompt_text in prompts:
        input_ids = tokenizer(prompt_text, return_tensors='pt')['input_ids']
        print(f"\n{'='*70}")
        print(f"PROMPT: \"{prompt_text}\"  (steering at {args.magnitude}°)")
        print(f"{'='*70}")

        # Baseline
        with torch.no_grad():
            outputs = model.generate(
                input_ids.to(device),
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  [   baseline] {baseline.replace(chr(10), chr(92)+'n')[:150]}")

        # Language directions
        for lang in languages:
            comp = steer_and_generate(
                model, tokenizer, device, input_ids, args.layer,
                f_hat[lang], args.magnitude, args.max_tokens)
            print(f"  [{lang:>11s}] {comp.replace(chr(10), chr(92)+'n')[:150]}")

        # Random directions
        for i, d in enumerate(random_dirs):
            comp = steer_and_generate(
                model, tokenizer, device, input_ids, args.layer,
                d, args.magnitude, args.max_tokens)
            print(f"  [  random_{i+1}] {comp.replace(chr(10), chr(92)+'n')[:150]}")

    print("\nDone!")


if __name__ == '__main__':
    main()
