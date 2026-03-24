#!/usr/bin/env python3
"""
Steering experiment for manifold v2: steer each of the 5 FineWeb activations
with (a) its MELBO direction and (b) the -mean (zh-en) direction.
Measure CJK fraction in completions at various angles.

Usage:
    python scripts/melbo/steer_manifold_v2_cjk.py \
        --melbo_results results/melbo_manifold_v2/qwen3-0.6b/results.pkl
"""

import sys
import pickle
import json
import os
import time
import numpy as np
sys.path.insert(0, '.')

import torch
from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length

LAYER = 14
N_TOKENS = 50
TEMPERATURE = 0.7
SEED = 42
ANGLES = [0, 5, 10, 15, 20, 25, 30, 45]


def is_cjk_char(ch):
    cp = ord(ch)
    if 0x4E00 <= cp <= 0x9FFF: return True
    if 0x3400 <= cp <= 0x4DBF: return True
    if 0x20000 <= cp <= 0x2A6DF: return True
    if 0xF900 <= cp <= 0xFAFF: return True
    if 0x3000 <= cp <= 0x303F: return True
    if 0xFF00 <= cp <= 0xFFEF: return True
    return False


def cjk_fraction(text):
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) == 0:
        return 0.0
    return sum(1 for ch in chars if is_cjk_char(ch)) / len(chars)


def generate_steered(model, tokenizer, device, input_ids, steer_direction, angle_deg,
                     n_tokens, temperature, layer):
    """Generate with spherical steering from given input_ids."""
    if angle_deg == 0:
        generated_ids = []
        ids = input_ids.clone()
        for step in range(n_tokens):
            with torch.no_grad():
                outputs = model(ids, use_cache=False)
            logits = outputs.logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            generated_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    d = steer_direction
    prompt_len = input_ids.shape[1]
    first_steer_pos = prompt_len - 1

    generated_ids = []
    ids = input_ids.clone()
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
            outputs = model(ids, use_cache=False)
        handle.remove()

        logits = outputs.logits[0, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        generated_ids.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--melbo_results', required=True)
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = os.path.join(
            os.path.dirname(args.melbo_results), 'steering_cjk')
    os.makedirs(args.results_dir, exist_ok=True)

    # Load MELBO results
    print(f"Loading MELBO results from {args.melbo_results}")
    with open(args.melbo_results, 'rb') as f:
        melbo_data = pickle.load(f)
    act_results = melbo_data['act_results']
    mean_dir = melbo_data['mean_dir']
    test_indices = melbo_data['test_indices']
    n_acts = len(act_results)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    neg_mean = (-mean_dir).to(device)

    # Load the same FineWeb sequences used in the manifold v2 experiment
    print(f"Loading FineWeb sequences (same seed as v2)...")
    token_ids_list = load_fineweb_fixed_length(50, tokenizer, seq_len=5, seed=SEED)

    total = n_acts * 2 * len(ANGLES)
    print(f"Activations: {n_acts}, Directions: 2 (MELBO + -mean), Angles: {ANGLES}")
    print(f"Total generations: {total}")

    all_results = []
    count = 0
    t0 = time.time()

    for act_i, ar in enumerate(act_results):
        act_idx = ar['act_idx']
        input_ids = token_ids_list[act_idx].unsqueeze(0).to(device)
        prompt_text = tokenizer.decode(token_ids_list[act_idx], skip_special_tokens=True)
        melbo_dir = ar['melbo_direction'].to(device)

        print(f"\nActivation {act_i+1}/{n_acts} (idx {act_idx}): \"{prompt_text}\"")

        for dir_name, direction in [('MELBO', melbo_dir), ('-mean', neg_mean)]:
            for angle in ANGLES:
                gen_seed = SEED + act_idx * 1000 + angle * 10 + hash(dir_name) % 100
                torch.manual_seed(gen_seed)

                text = generate_steered(
                    model, tokenizer, device, input_ids, direction, angle,
                    N_TOKENS, TEMPERATURE, LAYER)

                frac = cjk_fraction(text)
                all_results.append({
                    'act_i': act_i,
                    'act_idx': act_idx,
                    'prompt': prompt_text,
                    'direction': dir_name,
                    'angle': angle,
                    'text': text,
                    'cjk_fraction': frac,
                    'cos_melbo_mean': ar['cos_with_mean'],
                })

                count += 1

    elapsed = time.time() - t0
    print(f"\nAll {count} generations complete in {elapsed:.0f}s")

    # Save
    results_path = os.path.join(args.results_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary per activation
    print(f"\n{'='*70}")
    print("CJK FRACTION BY ACTIVATION, DIRECTION, AND ANGLE")
    print(f"{'='*70}")

    for act_i in range(n_acts):
        act_idx = act_results[act_i]['act_idx']
        cos_m = act_results[act_i]['cos_with_mean']
        print(f"\nAct {act_i+1} (idx {act_idx}, cos(MELBO,mean)={cos_m:.3f}):")
        print(f"  {'Direction':12s}", end='')
        for angle in ANGLES:
            print(f"  {angle:>5d}°", end='')
        print()

        for dir_name in ['MELBO', '-mean']:
            print(f"  {dir_name:12s}", end='')
            for angle in ANGLES:
                matches = [r['cjk_fraction'] for r in all_results
                           if r['act_i'] == act_i and r['direction'] == dir_name
                           and r['angle'] == angle]
                print(f"  {matches[0]:5.1%}", end='')
            print()

    print(f"\nSaved: {results_path}")
    print("Done!")


if __name__ == '__main__':
    main()
