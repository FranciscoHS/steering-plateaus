#!/usr/bin/env python3
"""
Spherical steering with the optimized delta directions on the optimization prompts.

Norm-preserving rotation toward the delta direction, sweeping angles 0-60 degrees.
Tests both v1 (minimize Jacobian) and v2 (match Jacobian) directions.

For each prompt × angle, shows:
- Top-5 next-token predictions (single forward pass, fast)
- Generated completions (at selected angles)

Usage:
    python scripts/delta_optimization/steer_delta_spherical.py
    python scripts/delta_optimization/steer_delta_spherical.py --mode generate --angles 0,10,20,30,40,50,60
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length


def spherical_steer_next_token(model, tokenizer, input_ids, steer_direction,
                                angle_deg, steer_layer, device, top_k=5):
    """
    Single forward pass with spherical steering, return top-k predictions.

    Rotates the last-token activation at steer_layer toward steer_direction
    by angle_deg degrees on the constant-norm sphere.
    """
    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    direction = steer_direction.to(device)

    input_ids = input_ids.to(device)

    def steering_hook(module, input, output):
        hidden = output[0]
        a = hidden[0, -1, :]  # [d_model]
        R = a.norm()
        a_unit = a / (R + 1e-10)

        # Project direction orthogonal to activation
        proj = (a_unit * direction).sum()
        v_perp = direction - proj * a_unit
        v_perp = v_perp / (v_perp.norm() + 1e-10)

        # Spherical rotation
        steered = R * (cos_theta * a_unit + sin_theta * v_perp)
        hidden[0, -1, :] = steered
        return (hidden,) + output[1:]

    layers = _get_blocks(model)
    handle = layers[steer_layer].register_forward_hook(steering_hook)

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


def generate_with_spherical_steering(model, tokenizer, input_ids, steer_direction,
                                      angle_deg, steer_layer, device, n_tokens=30,
                                      temperature=1.0):
    """
    Generate tokens with spherical steering at each step.
    Steers only the last token position. No KV cache.
    """
    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    direction = steer_direction.to(device)

    current_ids = input_ids.to(device)
    generated_ids = []

    for step in range(n_tokens):
        def steering_hook(module, input, output):
            hidden = output[0]
            a = hidden[0, -1, :]
            R = a.norm()
            a_unit = a / (R + 1e-10)

            proj = (a_unit * direction).sum()
            v_perp = direction - proj * a_unit
            v_perp = v_perp / (v_perp.norm() + 1e-10)

            steered = R * (cos_theta * a_unit + sin_theta * v_perp)
            hidden[0, -1, :] = steered
            return (hidden,) + output[1:]

        layers = _get_blocks(model)
        handle = layers[steer_layer].register_forward_hook(steering_hook)

        with torch.no_grad():
            outputs = model(current_ids, use_cache=False)

        handle.remove()

        logits = outputs.logits[0, -1, :]
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        if temperature > 0:
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
        else:
            next_id = logits.argmax().item()
        generated_ids.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

        current_ids = torch.cat([
            current_ids,
            torch.tensor([[next_id]], device=device)
        ], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True), generated_ids


def main():
    parser = argparse.ArgumentParser(description='Spherical steering with optimized delta')
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_prompts', type=int, default=20)
    parser.add_argument('--angles', type=str,
                        default=','.join(str(i) for i in range(0, 65, 5)),
                        help='Comma-separated angles in degrees')
    parser.add_argument('--n_tokens', type=int, default=30)
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (0 = greedy)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for FineWeb loading (42 = same prompts as optimization)')
    parser.add_argument('--v1_results', type=str, default='results/delta_opt/results_r66.pkl')
    parser.add_argument('--v2_results', type=str, default='results/delta_opt_v2/results_r66.pkl')
    parser.add_argument('--v4_results', type=str, default='results/delta_opt_v4/results_50.0deg.pkl')
    parser.add_argument('--v5_results', type=str, default='results/delta_opt_v5/results_50.0deg.pkl')
    parser.add_argument('--which', type=str, default='both',
                        choices=['v1', 'v2', 'v4', 'v5', 'both', 'all'])
    parser.add_argument('--negate', action='store_true',
                        help='Negate the delta direction (steer in opposite direction)')
    parser.add_argument('--mode', type=str, default='predictions',
                        choices=['predictions', 'generate', 'both'],
                        help='predictions: top-k at each angle; generate: full completions')
    parser.add_argument('--results_dir', type=str, default='results/delta_steer_spherical')
    args = parser.parse_args()

    angles = [float(a) for a in args.angles.split(',')]
    os.makedirs(args.results_dir, exist_ok=True)

    # Load deltas and normalize to unit vectors
    deltas = {}
    all_versions = [('v1', args.v1_results), ('v2', args.v2_results),
                    ('v4', args.v4_results), ('v5', args.v5_results)]
    for label, path in all_versions:
        if args.which in (label, 'both', 'all') and os.path.exists(path):
            with open(path, 'rb') as f:
                results = pickle.load(f)
            if 'direction' in results:
                delta_unit = results['direction']
                delta_unit = delta_unit / delta_unit.norm()
            else:
                delta = results['delta']
                delta_unit = delta / delta.norm()
            if args.negate:
                delta_unit = -delta_unit
            deltas[label] = delta_unit
            sign_str = " (NEGATED)" if args.negate else ""
            print(f"Loaded {label}{sign_str}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    for key in deltas:
        deltas[key] = deltas[key].to(device)

    # Load FineWeb sequences
    is_opt_prompts = (args.seed == 42)
    prompt_desc = "optimization prompts" if is_opt_prompts else f"fresh prompts (seed={args.seed})"
    print(f"\nLoading FineWeb sequences ({prompt_desc}, seq_len={args.seq_len})...")
    n_load = 200 if is_opt_prompts else args.n_prompts
    token_ids_list = load_fineweb_fixed_length(n_load, tokenizer, seq_len=args.seq_len, seed=args.seed)
    opt_sequences = token_ids_list[:100] if is_opt_prompts else token_ids_list

    # Show prompts
    print(f"\nFirst {args.n_prompts} {prompt_desc}:")
    for i in range(args.n_prompts):
        print(f"  [{i:3d}] {repr(tokenizer.decode(opt_sequences[i]))}")

    all_results = {}

    for label, delta_unit in deltas.items():
        neg_str = " NEGATED" if args.negate else ""
        print(f"\n{'='*80}")
        print(f"DIRECTION: {label}{neg_str} (minimize={label=='v1'}, match={label=='v2'})")
        print(f"Spherical steering, angles: {angles}, {prompt_desc}")
        print(f"{'='*80}")

        dir_results = []

        # --- Predictions mode ---
        if args.mode in ('predictions', 'both'):
            print(f"\n--- Next-token predictions ---")
            for i in range(args.n_prompts):
                input_ids = opt_sequences[i].unsqueeze(0)
                prompt_text = tokenizer.decode(opt_sequences[i])

                print(f"\n[{i:3d}] {repr(prompt_text)}")
                for angle in angles:
                    tokens, probs, entropy = spherical_steer_next_token(
                        model, tokenizer, input_ids, delta_unit, angle,
                        args.layer, device
                    )
                    top3_str = '  '.join(f'{repr(t)}:{p:.3f}' for t, p in zip(tokens[:3], probs[:3]))
                    print(f"  {angle:5.1f}° H={entropy:.2f}  {top3_str}")

                    dir_results.append({
                        'prompt_idx': i,
                        'prompt': prompt_text,
                        'angle_deg': angle,
                        'top_tokens': tokens,
                        'top_probs': probs,
                        'entropy': entropy,
                    })

        # --- Generation mode ---
        if args.mode in ('generate', 'both'):
            gen_angles = angles
            print(f"\n--- Generation (angles: {gen_angles}) ---")
            for i in range(args.n_prompts):
                input_ids = opt_sequences[i].unsqueeze(0)
                prompt_text = tokenizer.decode(opt_sequences[i])

                print(f"\n[{i:3d}] {repr(prompt_text)}")
                for angle in gen_angles:
                    text, ids = generate_with_spherical_steering(
                        model, tokenizer, input_ids, delta_unit, angle,
                        args.layer, device, n_tokens=args.n_tokens,
                        temperature=args.temperature
                    )
                    print(f"  {angle:5.1f}°: {repr(text)}")

        all_results[label] = dir_results

    # Save
    save_path = os.path.join(args.results_dir, 'results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({'args': vars(args), 'results': all_results}, f)
    print(f"\nSaved: {save_path}")


if __name__ == '__main__':
    main()
