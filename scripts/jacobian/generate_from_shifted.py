#!/usr/bin/env python3
"""
Generate text from shifted activations (a_i + delta) vs originals.

Instead of steering at every step, we:
1. Run the prompt through layers 0-6 to get the real activation
2. Add delta to the last-token activation at layer 6 (once)
3. Continue the forward pass from layer 6 onward to get the first token
4. Then generate normally from there (no more intervention)

This tests whether the shifted point itself is coherent, rather than
whether continuous steering in that direction works.

Usage:
    python scripts/jacobian/generate_from_shifted.py
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model, forward_from_layer, _get_blocks
from src.data import load_fineweb_fixed_length


def generate_from_patched_activation(model, tokenizer, input_ids, delta, layer,
                                      device, n_tokens=20):
    """
    Patch the layer-6 last-token activation once, then generate freely.

    1. Forward pass through the full model with a hook that captures + patches
       the layer-L activation on the FIRST forward pass only.
    2. For subsequent tokens, no patching — normal generation.

    Returns generated text (excluding prompt).
    """
    input_ids = input_ids.to(device)
    first_pass = [True]  # mutable flag

    def patch_hook(module, input, output):
        if first_pass[0]:
            hidden = output[0]
            hidden[0, -1, :] = hidden[0, -1, :] + delta
            first_pass[0] = False
            return (hidden,) + output[1:]
        return output

    layers = _get_blocks(model)
    handle = layers[layer].register_forward_hook(patch_hook)

    current_ids = input_ids.clone()
    generated_ids = []

    for step in range(n_tokens):
        with torch.no_grad():
            outputs = model(current_ids, use_cache=False)

        next_id = outputs.logits[0, -1, :].argmax().item()
        generated_ids.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

        current_ids = torch.cat([
            current_ids,
            torch.tensor([[next_id]], device=device)
        ], dim=1)

    handle.remove()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def get_first_token_predictions(model, tokenizer, input_ids, delta, layer,
                                 device, top_k=10):
    """
    Single forward pass: patch layer-L last-token activation, get next-token predictions.
    """
    input_ids = input_ids.to(device)

    def patch_hook(module, input, output):
        hidden = output[0]
        hidden[0, -1, :] = hidden[0, -1, :] + delta
        return (hidden,) + output[1:]

    layers = _get_blocks(model)
    handle = layers[layer].register_forward_hook(patch_hook)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)

    handle.remove()

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * probs.log()).sum().item()
    topk = torch.topk(probs, top_k)

    tokens = [tokenizer.decode([idx]) for idx in topk.indices]
    probs_list = topk.values.tolist()
    return tokens, probs_list, entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_prompts', type=int, default=20)
    parser.add_argument('--n_tokens', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--v1_results', type=str, default='results/delta_opt/results_r66.pkl')
    parser.add_argument('--v2_results', type=str, default='results/delta_opt_v2/results_r66.pkl')
    parser.add_argument('--which', type=str, default='both', choices=['v1', 'v2', 'both'])
    args = parser.parse_args()

    # Load deltas (full vectors, not unit)
    deltas = {}
    for label, path in [('v1', args.v1_results), ('v2', args.v2_results)]:
        if args.which in (label, 'both') and os.path.exists(path):
            with open(path, 'rb') as f:
                results = pickle.load(f)
            delta = results['delta']
            deltas[label] = delta
            print(f"Loaded {label}: norm={delta.norm().item():.1f}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    for key in deltas:
        deltas[key] = deltas[key].to(device)

    # Load FineWeb sequences
    print(f"\nLoading FineWeb sequences (seed={args.seed}, seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(200, tokenizer, seq_len=args.seq_len, seed=args.seed)
    opt_sequences = token_ids_list[:100]

    # Zero delta for baseline
    d_model = deltas[list(deltas.keys())[0]].shape[0]
    zero_delta = torch.zeros(d_model, device=device)

    for label, delta in deltas.items():
        print(f"\n{'='*80}")
        print(f"DIRECTION: {label} (norm={delta.norm().item():.1f})")
        print(f"One-shot patch at layer {args.layer}, then free generation")
        print(f"{'='*80}")

        for i in range(args.n_prompts):
            input_ids = opt_sequences[i].unsqueeze(0)
            prompt_text = tokenizer.decode(opt_sequences[i])

            # Baseline: no patch
            baseline = generate_from_patched_activation(
                model, tokenizer, input_ids, zero_delta, args.layer,
                device, n_tokens=args.n_tokens
            )

            # Shifted: add delta once
            shifted = generate_from_patched_activation(
                model, tokenizer, input_ids, delta, args.layer,
                device, n_tokens=args.n_tokens
            )

            # Also get first-token predictions for both
            tok_base, prob_base, h_base = get_first_token_predictions(
                model, tokenizer, input_ids, zero_delta, args.layer, device
            )
            tok_shift, prob_shift, h_shift = get_first_token_predictions(
                model, tokenizer, input_ids, delta, args.layer, device
            )

            print(f"\n[{i:3d}] {repr(prompt_text)}")
            top3_base = ' '.join(f'{repr(t)}:{p:.3f}' for t, p in zip(tok_base[:3], prob_base[:3]))
            top3_shift = ' '.join(f'{repr(t)}:{p:.3f}' for t, p in zip(tok_shift[:3], prob_shift[:3]))
            print(f"  base   (H={h_base:.2f}) [{top3_base}]")
            print(f"    -> {repr(baseline[:100])}")
            print(f"  shift  (H={h_shift:.2f}) [{top3_shift}]")
            print(f"    -> {repr(shifted[:100])}")


if __name__ == '__main__':
    main()
