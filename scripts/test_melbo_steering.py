#!/usr/bin/env python3
"""
Qualitative steering test: generate text steered along MELBO_1, PC1, and mean.

Tests whether the high-impact directions found by MELBO/SVD actually steer
the model's output language (English <-> Chinese).

Additive steering: at layer L, hidden_states += magnitude * direction.
Positive magnitude = English-ward (since D = en - zh).

Usage:
    python scripts/test_melbo_steering.py --model qwen3-0.6b
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
from src.model import load_model


def generate_with_steering(model, tokenizer, prompt, direction, magnitude,
                           steer_layer, device, max_new_tokens=50):
    """Generate text with additive steering at a given layer.

    Adds magnitude * direction to the residual stream after steer_layer
    at every generation step.
    """
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
    with torch.no_grad():
        output = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
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

    # Check signs: mean points en-ward by construction.
    # Ensure PC1 and MELBO_1 also point en-ward (positive cosine with mean).
    if (pc1 @ mean_dir) < 0:
        pc1 = -pc1
    if (melbo_1 @ mean_dir) < 0:
        melbo_1 = -melbo_1

    directions = {
        'mean': mean_dir,
        'PC1': pc1,
        'MELBO_1': melbo_1,
    }

    # Prompts: mix of English and Chinese
    prompts_en = [
        "The weather today is",
        "Once upon a time",
        "The capital of France is",
        "I think the best way to",
    ]
    prompts_zh = [
        "今天的天气",
        "从前有一个",
        "法国的首都是",
        "我认为最好的方法是",
    ]

    # Magnitudes: negative = Chinese-ward, positive = English-ward
    magnitudes = [-20, -10, 0, 10, 20]

    print(f"\nSteering at layer {args.layer}")
    print(f"Mean act norm at this layer: ~56.7")
    print(f"Directions: {list(directions.keys())}")
    print(f"Magnitudes: {magnitudes}")

    # English prompts: can we push toward Chinese?
    print(f"\n{'#'*70}")
    print("ENGLISH PROMPTS (negative mag = push toward Chinese)")
    print(f"{'#'*70}")

    for prompt in prompts_en:
        print(f"\nPrompt: \"{prompt}\"")
        for mag in magnitudes:
            if mag == 0:
                text = generate_with_steering(
                    model, tokenizer, prompt, mean_dir, 0.0,
                    args.layer, device, args.max_new_tokens)
                print(f"  [baseline]          {text[:120]}")
            else:
                for name, direction in directions.items():
                    text = generate_with_steering(
                        model, tokenizer, prompt, direction, float(mag),
                        args.layer, device, args.max_new_tokens)
                    print(f"  [{name:8s} mag={mag:+3d}] {text[:120]}")

    # Chinese prompts: can we push toward English?
    print(f"\n{'#'*70}")
    print("CHINESE PROMPTS (positive mag = push toward English)")
    print(f"{'#'*70}")

    for prompt in prompts_zh:
        print(f"\nPrompt: \"{prompt}\"")
        for mag in magnitudes:
            if mag == 0:
                text = generate_with_steering(
                    model, tokenizer, prompt, mean_dir, 0.0,
                    args.layer, device, args.max_new_tokens)
                print(f"  [baseline]          {text[:120]}")
            else:
                for name, direction in directions.items():
                    text = generate_with_steering(
                        model, tokenizer, prompt, direction, float(mag),
                        args.layer, device, args.max_new_tokens)
                    print(f"  [{name:8s} mag={mag:+3d}] {text[:120]}")

    print("\nDone!")


if __name__ == '__main__':
    main()
