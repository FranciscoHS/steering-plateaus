#!/usr/bin/env python3
"""
Steer GPT-2 by adding the optimized delta to layer 6 activations.

For a set of prompts, compare:
- Normal model output
- Steered model output (delta added at layer 6, last token)

Usage:
    python scripts/steer_with_delta.py
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
from src.model import load_model


def generate_with_steering(model, tokenizer, prompt, delta, layer, device,
                           max_new_tokens=30, magnitude=1.0):
    """
    Generate text with and without steering.

    Adds magnitude * delta to the residual stream at the given layer,
    last token position, at each generation step.

    Returns:
        normal_text: string (unsteered completion)
        steered_text: string (steered completion)
    """
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

    # Normal generation
    with torch.no_grad():
        normal_out = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0
        )
    normal_text = tokenizer.decode(normal_out[0], skip_special_tokens=True)

    # Steered generation: hook that adds delta at layer L, last position
    steering_vec = (magnitude * delta).to(device)

    def steering_hook(module, input, output):
        # GPT-2 block output is a tuple: (hidden_states, present, ...)
        hidden = output[0]
        # Add delta to last token position
        hidden[:, -1, :] = hidden[:, -1, :] + steering_vec
        return (hidden,) + output[1:]

    # Register hook on the target layer
    hook_handle = model.transformer.h[layer].register_forward_hook(steering_hook)

    with torch.no_grad():
        steered_out = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0
        )
    steered_text = tokenizer.decode(steered_out[0], skip_special_tokens=True)

    hook_handle.remove()

    return normal_text, steered_text


def main():
    parser = argparse.ArgumentParser(description='Steer GPT-2 with optimized delta')
    parser.add_argument('--results_file', type=str, default='results/delta_opt/results_r66.pkl')
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--max_new_tokens', type=int, default=30)
    args = parser.parse_args()

    # Load delta
    print("Loading results...")
    with open(args.results_file, 'rb') as f:
        results = pickle.load(f)
    delta = results['delta']
    print(f"Delta norm: {delta.norm().item():.1f}")

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    delta = delta.to(device)

    # Test prompts
    prompts = [
        "The capital of France is",
        "In the year 2050,",
        "The most important thing about",
        "Scientists recently discovered that",
        "Once upon a time, there was",
        "The weather today is",
        "To solve this problem, we need to",
        "The cat sat on the",
        "According to recent studies,",
        "My favorite food is",
    ]

    # Try different magnitudes
    magnitudes = [0.5, 1.0, 2.0, 5.0]

    for mag in magnitudes:
        print(f"\n{'='*60}")
        print(f"Magnitude: {mag} (effective norm: {mag * delta.norm().item():.1f})")
        print(f"{'='*60}")

        for prompt in prompts:
            normal, steered = generate_with_steering(
                model, tokenizer, prompt, delta, args.layer, device,
                max_new_tokens=args.max_new_tokens, magnitude=mag
            )
            # Show only the completion (after the prompt)
            normal_completion = normal[len(prompt):]
            steered_completion = steered[len(prompt):]

            print(f"\nPrompt: {repr(prompt)}")
            print(f"  Normal:  {repr(normal_completion.strip())}")
            print(f"  Steered: {repr(steered_completion.strip())}")


if __name__ == '__main__':
    main()
