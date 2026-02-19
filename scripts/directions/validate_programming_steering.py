#!/usr/bin/env python3
"""
Validate programming language steering directions by generating text.

Loads recovered feature directions from run_programming_contrastive.py,
steers FineWeb activations along each direction, and generates text to
check if steering toward f̂_python produces Python-like output, etc.

Usage:
    python scripts/directions/validate_programming_steering.py [--model qwen3-0.6b] [--layer 14]
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
from src.data import load_fineweb_fixed_length


def steer_and_generate(model, tokenizer, device, input_ids, layer, direction,
                       magnitude, max_new_tokens=60):
    """Generate text with spherical steering at the given layer.

    Uses a hook to rotate the residual stream toward `direction` at every
    forward pass (so each generated token is steered).
    """
    direction_unit = direction / direction.norm()
    direction_dev = direction_unit.to(device)

    def steering_hook(module, input, output):
        # output is (hidden_states, ...) or just hidden_states depending on model
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Spherical steering: rotate toward direction
        a = hidden[:, -1, :]  # last token position
        a_norm = a.norm(dim=-1, keepdim=True)
        a_unit = a / (a_norm + 1e-10)

        # Component along direction
        proj = (a_unit * direction_dev).sum(dim=-1, keepdim=True)
        # Perpendicular component of direction w.r.t. activation
        u_perp = direction_dev.unsqueeze(0) - proj * a_unit
        u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)

        # Rotate by angle
        angle = np.radians(magnitude)
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        a_steered = a_norm * (cos_t * a_unit + sin_t * u_perp)

        hidden = hidden.clone()
        hidden[:, -1, :] = a_steered

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    # Register hook on the target layer's output
    # For HF models, we hook on the layer module itself
    blocks = model.model.layers if hasattr(model, 'model') else model.transformer.h
    hook_handle = blocks[layer].register_forward_hook(steering_hook)

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
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
    parser.add_argument('--magnitude', type=float, default=20.0,
                        help='Steering angle in degrees')
    parser.add_argument('--n_prompts', type=int, default=5,
                        help='Number of FineWeb prompts to test')
    parser.add_argument('--max_tokens', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default=None,
                        help='Path to contrastive results (default: auto)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load results
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = f'results/programming_contrastive/{args.model}'

    results_path = os.path.join(results_dir, 'contrastive_results.pkl')
    print(f"Loading directions from {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    languages = results['languages']
    f_hat = results['f_hat']
    print(f"Languages: {languages}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    # Load some FineWeb prompts
    print(f"\nLoading {args.n_prompts} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_prompts, tokenizer, seq_len=8, seed=args.seed)

    # Generate unsteered baseline + steered for each language
    print(f"\nSteering angle: {args.magnitude}°")
    print(f"Max tokens: {args.max_tokens}")

    for prompt_idx in range(args.n_prompts):
        input_ids = token_ids_list[prompt_idx].unsqueeze(0)
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        print(f"\n{'='*70}")
        print(f"PROMPT {prompt_idx+1}: \"{prompt_text}\"")
        print(f"{'='*70}")

        # Unsteered baseline
        with torch.no_grad():
            outputs = model.generate(
                input_ids.to(device),
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        baseline_display = baseline.replace('\n', '\\n')[:150]
        print(f"  [baseline] {baseline_display}")

        # Steered for each language
        for lang in languages:
            direction = f_hat[lang]
            completion = steer_and_generate(
                model, tokenizer, device, input_ids, args.layer,
                direction, args.magnitude, args.max_tokens)
            comp_display = completion.replace('\n', '\\n')[:150]
            print(f"  [{lang:>12s}] {comp_display}")

    # Also try with a code-adjacent prompt
    print(f"\n{'='*70}")
    print("BONUS: Steering a code-adjacent prompt")
    print(f"{'='*70}")
    code_prompts = [
        "Write a function that",
        "The following code",
        "Here is a simple",
    ]
    for prompt_text in code_prompts:
        input_ids = tokenizer(prompt_text, return_tensors='pt')['input_ids']
        print(f"\nPROMPT: \"{prompt_text}\"")

        # Baseline
        with torch.no_grad():
            outputs = model.generate(
                input_ids.to(device),
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  [baseline] {baseline.replace(chr(10), chr(92)+'n')[:150]}")

        for lang in languages:
            direction = f_hat[lang]
            completion = steer_and_generate(
                model, tokenizer, device, input_ids, args.layer,
                direction, args.magnitude, args.max_tokens)
            print(f"  [{lang:>12s}] {completion.replace(chr(10), chr(92)+'n')[:150]}")

    print("\nDone!")


if __name__ == '__main__':
    main()
