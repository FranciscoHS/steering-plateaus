#!/usr/bin/env python3
"""
Cross-group feature combination experiment.

Tests whether combining features from different groups (natural language + programming
language) produces combined behavior. E.g., does Russian + Go → Russian Go code?

Loads pre-computed feature vectors from:
- results/allpairs_contrastive/qwen3-4b/directions.pkl  (natural language features)
- results/programming_contrastive/qwen3-4b/directions.pkl  (programming language features)

Usage:
    python scripts/steering/run_crossgroup_steering.py [--angle 15] [--n_tokens 80]
"""

import sys
import os
import pickle
import argparse
import numpy as np

sys.path.insert(0, '.')

import torch
from src.model import load_model, _get_blocks


def generate_steered(model, tokenizer, device, prompt, steer_direction, angle_deg,
                     n_tokens, temperature, layer):
    """Generate text with spherical steering at a given layer.

    Steers all positions from the last prompt token onward.
    """
    if angle_deg == 0:
        tokens = tokenizer(prompt, return_tensors='pt')
        input_ids = tokens['input_ids'].to(device)
        generated_ids = []
        for step in range(n_tokens):
            with torch.no_grad():
                outputs = model(input_ids, use_cache=False)
            logits = outputs.logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            generated_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    d = steer_direction

    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    prompt_len = input_ids.shape[1]
    first_steer_pos = prompt_len - 1

    generated_ids = []
    for step in range(n_tokens):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            steer_slice = hidden[0, first_steer_pos:, :]
            if steer_slice.shape[0] == 0:
                return output
            R = steer_slice.norm(dim=-1, keepdim=True)
            a_unit = steer_slice / (R + 1e-10)
            proj = (a_unit * d).sum(dim=-1, keepdim=True)
            v_perp = d.unsqueeze(0) - proj * a_unit
            v_perp = v_perp / (v_perp.norm(dim=-1, keepdim=True) + 1e-10)
            steered = R * (cos_theta * a_unit + sin_theta * v_perp)
            hidden[0, first_steer_pos:, :] = steered
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        layers = _get_blocks(model)
        handle = layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
        handle.remove()

        logits = outputs.logits[0, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        generated_ids.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-4b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--angles', nargs='+', type=float, default=[10, 15, 20, 25, 30])
    parser.add_argument('--n_tokens', type=int, default=80)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--n_samples', type=int, default=3,
                        help='Number of samples per (prompt, direction, angle)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bfloat16', action='store_true')
    parser.add_argument('--nat_langs', nargs='+', default=None,
                        help='Natural language(s) to test (default: all from source)')
    parser.add_argument('--prog_lang', default='c', help='Programming language feature to use')
    parser.add_argument('--lang_source', default='5lang',
                        choices=['5lang', 'allpairs', '4lang'],
                        help='Source of natural language features')
    args = parser.parse_args()

    # Load feature vectors
    if args.lang_source == '5lang':
        lang_path = f'results/allpairs_contrastive/{args.model}/5lang_contrastive.pkl'
        with open(lang_path, 'rb') as f:
            lang_data = pickle.load(f)
        all_nat_feats = lang_data['features_5']
    elif args.lang_source == '4lang':
        lang_path = f'results/allpairs_contrastive/{args.model}/4lang_no_german.pkl'
        with open(lang_path, 'rb') as f:
            lang_data = pickle.load(f)
        all_nat_feats = lang_data['features_4']
    else:
        lang_path = f'results/allpairs_contrastive/{args.model}/directions.pkl'
        with open(lang_path, 'rb') as f:
            lang_data = pickle.load(f)
        all_nat_feats = lang_data['features']

    prog_path = f'results/programming_contrastive/{args.model}/directions.pkl'
    with open(prog_path, 'rb') as f:
        prog_data = pickle.load(f)
    prog_feat = prog_data['features'][args.prog_lang]  # unit-normalized

    nat_langs = args.nat_langs if args.nat_langs else list(all_nat_feats.keys())
    print(f"Source: {args.lang_source}, nat languages: {nat_langs}, prog: {args.prog_lang}")

    # Print cosines
    for nl in nat_langs:
        cos = (all_nat_feats[nl] @ prog_feat).item()
        print(f"  cos({nl}, {args.prog_lang}) = {cos:.4f}")

    # Load model
    print(f"\nLoading model: {args.model}")
    load_dtype = torch.bfloat16 if args.bfloat16 else None
    model, tokenizer, device = load_model(args.model, dtype=load_dtype)
    print(f"Device: {device}")

    prog_feat = prog_feat.to(device)
    prog_label = args.prog_lang.capitalize()

    # Build steering configs: baseline + each nat_lang in both signs
    steer_configs = {'baseline': None}
    for nl in nat_langs:
        nf = all_nat_feats[nl].to(device)
        nl_label = nl.upper()
        steer_configs[f'+{nl_label}'] = nf
        steer_configs[f'-{nl_label}'] = -nf

    prompts = [
        "The following code",
        "In this article, we will",
        "The capital of France is",
    ]

    # Run experiment
    print(f"\nPrompts: {len(prompts)}")
    print(f"Directions: {list(steer_configs.keys())}")
    print(f"Angles: {args.angles}")
    print(f"Samples per config: {args.n_samples}")
    print(f"Tokens per generation: {args.n_tokens}")
    print()

    for prompt in prompts:
        print(f"{'='*80}")
        print(f"PROMPT: \"{prompt}\"")
        print(f"{'='*80}")

        for dir_name, direction in steer_configs.items():
            angles = [0] if dir_name == 'baseline' else args.angles

            for angle in angles:
                print(f"\n--- {dir_name} @ {angle}° ---")
                for sample_idx in range(args.n_samples):
                    seed = args.seed + sample_idx * 1000 + int(angle * 10) + hash(dir_name) % 100
                    torch.manual_seed(seed)

                    text = generate_steered(
                        model, tokenizer, device, prompt,
                        direction, angle, args.n_tokens, args.temperature, args.layer)

                    print(f"  [{sample_idx+1}] {text[:200]}")

    print("\nDone!")


if __name__ == '__main__':
    main()
