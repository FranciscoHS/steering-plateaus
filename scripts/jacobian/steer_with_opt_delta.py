#!/usr/bin/env python3
"""
Steer GPT-2 with the optimized delta directions from Jacobian norm optimization.

Uses the same FineWeb prompts that were used during optimization (seed=42, seq_len=5)
to check whether the direction produces interpretable behavior changes.

Tests both v1 (minimize Jacobian) and v2 (match Jacobian) directions.

Usage:
    python scripts/jacobian/steer_with_opt_delta.py
    python scripts/jacobian/steer_with_opt_delta.py --n_prompts 50 --magnitudes 0.25 0.5 1.0 2.0
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model, forward_from_layer
from src.data import load_fineweb_fixed_length


def generate_with_steering(model, tokenizer, input_ids, delta, layer, device,
                           max_new_tokens=30, magnitude=1.0):
    """
    Generate text with delta added at layer L, last token position, at each step.

    Args:
        input_ids: [1, seq_len] tensor
        delta: [d_model] tensor
        layer: which transformer block to hook

    Returns:
        text: decoded completion string
    """
    input_ids = input_ids.to(device)
    steering_vec = (magnitude * delta).to(device)

    def steering_hook(module, input, output):
        hidden = output[0]
        hidden[:, -1, :] = hidden[:, -1, :] + steering_vec
        return (hidden,) + output[1:]

    hook_handle = model.transformer.h[layer].register_forward_hook(steering_hook)

    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    hook_handle.remove()
    return text


def get_top_predictions(model, tokenizer, input_ids, delta, layer, device,
                        magnitude=0.0, top_k=10):
    """
    Get top-k next token predictions with optional steering at layer L.

    Instead of generating, just does a single forward pass and returns the
    top predicted tokens. Useful for seeing the immediate effect of steering
    on the next-token distribution.
    """
    input_ids = input_ids.to(device)

    if magnitude != 0.0:
        steering_vec = (magnitude * delta).to(device)

        def steering_hook(module, input, output):
            hidden = output[0]
            hidden[:, -1, :] = hidden[:, -1, :] + steering_vec
            return (hidden,) + output[1:]

        hook_handle = model.transformer.h[layer].register_forward_hook(steering_hook)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # last token logits
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.log()).sum().item()
        topk = torch.topk(probs, top_k)

    if magnitude != 0.0:
        hook_handle.remove()

    tokens = [tokenizer.decode([idx]) for idx in topk.indices]
    probs_list = topk.values.tolist()

    return tokens, probs_list, entropy


def main():
    parser = argparse.ArgumentParser(description='Steer with optimized delta on training prompts')
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_prompts', type=int, default=20,
                        help='Number of optimization prompts to test')
    parser.add_argument('--magnitudes', type=float, nargs='+',
                        default=[0.1, 0.25, 0.5, 1.0],
                        help='Steering magnitudes (multiplied by delta)')
    parser.add_argument('--max_new_tokens', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42,
                        help='Same seed as optimization to get same prompts')
    parser.add_argument('--v1_results', type=str, default='results/delta_opt/results_r66.pkl')
    parser.add_argument('--v2_results', type=str, default='results/delta_opt_v2/results_r66.pkl')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['v1', 'v2', 'both', 'predictions'],
                        help='Which delta to test, or "predictions" for next-token analysis')
    args = parser.parse_args()

    # Load deltas
    deltas = {}
    for label, path in [('v1 (minimize)', args.v1_results), ('v2 (match)', args.v2_results)]:
        if args.mode == 'both' or label.startswith(args.mode) or args.mode == 'predictions':
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    results = pickle.load(f)
                deltas[label] = results['delta']
                print(f"Loaded {label}: norm={results['delta'].norm().item():.1f}, r={results['r']:.1f}")
            else:
                print(f"Warning: {path} not found, skipping {label}")

    if not deltas:
        print("No deltas loaded!")
        return

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    for key in deltas:
        deltas[key] = deltas[key].to(device)

    # Load the same FineWeb sequences used during optimization
    print(f"\nLoading FineWeb sequences (seed={args.seed}, seq_len={args.seq_len})...")
    n_total = 200  # optimization used 100 act + 100 ctx
    token_ids_list = load_fineweb_fixed_length(n_total, tokenizer, seq_len=args.seq_len, seed=args.seed)

    # The first 100 are the activation sequences (the ones we optimized over)
    opt_sequences = token_ids_list[:100]

    # Decode and show the prompts
    print(f"\nFirst {args.n_prompts} optimization prompts:")
    for i in range(args.n_prompts):
        text = tokenizer.decode(opt_sequences[i])
        print(f"  [{i:3d}] {repr(text)}")

    if args.mode == 'predictions':
        # Next-token prediction analysis: show how top predictions change
        print(f"\n{'='*80}")
        print(f"NEXT-TOKEN PREDICTION ANALYSIS")
        print(f"{'='*80}")

        for label, delta in deltas.items():
            print(f"\n--- {label} (norm={delta.norm().item():.1f}) ---")

            for i in range(args.n_prompts):
                input_ids = opt_sequences[i].unsqueeze(0)
                prompt_text = tokenizer.decode(opt_sequences[i])

                # Baseline
                tokens_base, probs_base, entropy_base = get_top_predictions(
                    model, tokenizer, input_ids, delta, args.layer, device, magnitude=0.0
                )

                print(f"\n[{i:3d}] {repr(prompt_text)}")
                print(f"  base (H={entropy_base:.2f}): {tokens_base[:5]} {[f'{p:.3f}' for p in probs_base[:5]]}")

                for mag in args.magnitudes:
                    tokens_s, probs_s, entropy_s = get_top_predictions(
                        model, tokenizer, input_ids, delta, args.layer, device, magnitude=mag
                    )
                    print(f"  mag={mag:<4} (H={entropy_s:.2f}): {tokens_s[:5]} {[f'{p:.3f}' for p in probs_s[:5]]}")

    else:
        # Generation mode: generate completions with and without steering
        for label, delta in deltas.items():
            print(f"\n{'='*80}")
            print(f"STEERING WITH: {label} (norm={delta.norm().item():.1f})")
            print(f"{'='*80}")

            for mag in args.magnitudes:
                effective_norm = mag * delta.norm().item()
                print(f"\n--- magnitude={mag} (effective norm={effective_norm:.1f}) ---")

                for i in range(args.n_prompts):
                    input_ids = opt_sequences[i].unsqueeze(0)
                    prompt_text = tokenizer.decode(opt_sequences[i])

                    # Normal generation
                    with torch.no_grad():
                        normal_out = model.generate(
                            input_ids.to(device), max_new_tokens=args.max_new_tokens,
                            do_sample=False, temperature=1.0
                        )
                    normal_text = tokenizer.decode(normal_out[0], skip_special_tokens=True)

                    # Steered generation
                    steered_text = generate_with_steering(
                        model, tokenizer, input_ids, delta, args.layer, device,
                        max_new_tokens=args.max_new_tokens, magnitude=mag
                    )

                    normal_completion = normal_text[len(prompt_text):]
                    steered_completion = steered_text[len(prompt_text):]

                    if normal_completion != steered_completion:
                        print(f"  [{i:3d}] {repr(prompt_text)}")
                        print(f"    normal:  {repr(normal_completion.strip()[:80])}")
                        print(f"    steered: {repr(steered_completion.strip()[:80])}")


if __name__ == '__main__':
    main()
