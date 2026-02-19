#!/usr/bin/env python3
"""
Systematic steering comparison: MELBO_1 vs -mean vs MELBO_1⊥mean.

Measures fraction of Mandarin/CJK characters in generated text across
different steering angles and directions.

Usage:
    python scripts/run_steering_mandarin_fraction.py
"""

import sys
import pickle
import json
import os
import time
import unicodedata
import numpy as np
sys.path.insert(0, '.')

import torch
from src.model import load_model, _get_blocks

LAYER = 14
N_TOKENS = 50
TEMPERATURE = 0.7
SEED = 42

# Diverse English prompts across topics
PROMPTS = [
    # Everyday / conversational
    "The weather today is",
    "I went to the store and",
    "My favorite food is",
    "Last weekend I decided to",
    "The best thing about summer is",
    # Narrative / fiction
    "Once upon a time, there",
    "She opened the door and",
    "The old man sat quietly",
    "In the dark forest, a",
    "He picked up the letter and",
    # Factual / knowledge
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest ocean on Earth is",
    "Photosynthesis is the process by which",
    "The first person to walk on the moon",
    # Science / academic
    "In recent years, scientists have",
    "The theory of relativity states that",
    "Machine learning algorithms are used to",
    "Climate change has led to",
    "The human brain contains approximately",
    # Abstract / opinion
    "The meaning of life is",
    "Democracy is important because",
    "Education should focus on",
    "The internet has changed how",
    "Happiness can be found in",
    # Technical / instructions
    "To make a cup of coffee",
    "The first step in solving this",
    "When writing a good essay",
    "A healthy diet should include",
    "The most efficient way to",
]

ANGLES = [0, 5, 10, 25, 45]

DIRECTIONS_TO_TEST = ['MELBO_1', '-mean', 'MELBO_1_orth_mean']


def is_cjk_char(ch):
    """Check if character is CJK (Chinese/Japanese/Korean)."""
    cp = ord(ch)
    # CJK Unified Ideographs
    if 0x4E00 <= cp <= 0x9FFF:
        return True
    # CJK Unified Ideographs Extension A
    if 0x3400 <= cp <= 0x4DBF:
        return True
    # CJK Unified Ideographs Extension B
    if 0x20000 <= cp <= 0x2A6DF:
        return True
    # CJK Compatibility Ideographs
    if 0xF900 <= cp <= 0xFAFF:
        return True
    # CJK punctuation and symbols commonly used in Chinese text
    if 0x3000 <= cp <= 0x303F:
        return True
    # Fullwidth forms (，。！etc)
    if 0xFF00 <= cp <= 0xFFEF:
        return True
    return False


def cjk_fraction(text):
    """Fraction of non-whitespace characters that are CJK."""
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) == 0:
        return 0.0
    cjk_count = sum(1 for ch in chars if is_cjk_char(ch))
    return cjk_count / len(chars)


def generate_steered(model, tokenizer, device, prompt, steer_direction, angle_deg,
                     n_tokens, temperature, layer):
    """Generate with spherical steering, temperature sampling."""
    if angle_deg == 0:
        # No steering, just generate normally
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
    results_dir = 'results/steering_mandarin_fraction/qwen3-0.6b'
    os.makedirs(results_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model, tokenizer, device = load_model('qwen3-0.6b')

    # Load directions
    with open('results/melbo_constrained/qwen3-0.6b/optimization_results.pkl', 'rb') as f:
        saved = pickle.load(f)

    mean_dir = saved['mean_dir'].to(device)
    melbo1 = saved['melbo_directions'][0].to(device)

    neg_mean = -mean_dir
    melbo1_orth = melbo1 - (melbo1 @ mean_dir) * mean_dir
    melbo1_orth = melbo1_orth / melbo1_orth.norm()

    directions = {
        'MELBO_1': melbo1,
        '-mean': neg_mean,
        'MELBO_1_orth_mean': melbo1_orth,
    }

    print(f"cos(MELBO_1, mean) = {(melbo1 @ mean_dir).item():.4f}")
    print(f"cos(MELBO_1_orth, mean) = {(melbo1_orth @ mean_dir).item():.6f}")
    print(f"Prompts: {len(PROMPTS)}, Angles: {ANGLES}, Directions: {list(directions.keys())}")
    print(f"Total generations: {len(PROMPTS) * len(ANGLES) * len(directions)}")

    # Run all generations
    all_results = []
    total = len(PROMPTS) * len(ANGLES) * len(DIRECTIONS_TO_TEST)
    count = 0
    t0 = time.time()

    for prompt_idx, prompt in enumerate(PROMPTS):
        for angle in ANGLES:
            for dir_name in DIRECTIONS_TO_TEST:
                # Deterministic seed per (prompt, angle, direction)
                gen_seed = SEED + prompt_idx * 1000 + angle * 10 + hash(dir_name) % 100
                torch.manual_seed(gen_seed)

                direction = directions[dir_name]
                text = generate_steered(
                    model, tokenizer, device, prompt, direction, angle,
                    N_TOKENS, TEMPERATURE, LAYER)

                frac = cjk_fraction(text)

                result = {
                    'prompt_idx': prompt_idx,
                    'prompt': prompt,
                    'direction': dir_name,
                    'angle': angle,
                    'text': text,
                    'cjk_fraction': frac,
                    'n_chars': len([ch for ch in text if not ch.isspace()]),
                }
                all_results.append(result)

                count += 1
                if count % 15 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / count * (total - count)
                    print(f"  [{count}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"\nAll {total} generations complete in {elapsed:.0f}s")

    # Save raw results
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {results_dir}/results.json")

    # Compute and print summary statistics
    print(f"\n{'='*70}")
    print("MEAN CJK FRACTION BY DIRECTION AND ANGLE")
    print(f"{'='*70}")
    print(f"{'Direction':20s}", end='')
    for angle in ANGLES:
        print(f"  {angle:>5d}°", end='')
    print()
    print("-" * 60)

    for dir_name in DIRECTIONS_TO_TEST:
        print(f"{dir_name:20s}", end='')
        for angle in ANGLES:
            fracs = [r['cjk_fraction'] for r in all_results
                     if r['direction'] == dir_name and r['angle'] == angle]
            mean_frac = np.mean(fracs)
            print(f"  {mean_frac:5.1%}", end='')
        print()

    # Also print std
    print(f"\n{'='*70}")
    print("STD OF CJK FRACTION")
    print(f"{'='*70}")
    print(f"{'Direction':20s}", end='')
    for angle in ANGLES:
        print(f"  {angle:>5d}°", end='')
    print()
    print("-" * 60)

    for dir_name in DIRECTIONS_TO_TEST:
        print(f"{dir_name:20s}", end='')
        for angle in ANGLES:
            fracs = [r['cjk_fraction'] for r in all_results
                     if r['direction'] == dir_name and r['angle'] == angle]
            std_frac = np.std(fracs)
            print(f"  {std_frac:5.1%}", end='')
        print()

    # Median
    print(f"\n{'='*70}")
    print("MEDIAN CJK FRACTION")
    print(f"{'='*70}")
    print(f"{'Direction':20s}", end='')
    for angle in ANGLES:
        print(f"  {angle:>5d}°", end='')
    print()
    print("-" * 60)

    for dir_name in DIRECTIONS_TO_TEST:
        print(f"{dir_name:20s}", end='')
        for angle in ANGLES:
            fracs = [r['cjk_fraction'] for r in all_results
                     if r['direction'] == dir_name and r['angle'] == angle]
            med_frac = np.median(fracs)
            print(f"  {med_frac:5.1%}", end='')
        print()

    print("\nDone!")


if __name__ == '__main__':
    main()
