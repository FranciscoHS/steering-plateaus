#!/usr/bin/env python3
"""
Steering evaluation for genre directions.

Tests whether genre directions (from run_register_contrastive.py) actually
shift model output toward the target genre. Uses neutral prompts and sweeps
steering angles.

Evaluation: qualitative (printed completions) + simple heuristics (avg word
length, sentence complexity) as genre proxies.

Usage:
    python scripts/steering/run_steering_registers.py [--model qwen3-0.6b]
"""

import sys
import argparse
import pickle
import json
import os
import time
import numpy as np
sys.path.insert(0, '.')

import torch
from src.model import load_model, _get_blocks

LAYER = 14
N_TOKENS = 50
TEMPERATURE = 0.7
SEED = 42

ANGLES = [0, 5, 10, 15, 20, 30, 45]

# Neutral prompts that don't strongly suggest any register
NEUTRAL_PROMPTS = [
    "The situation with the economy is",
    "When people think about the future they",
    "The most important thing about education is",
    "Technology has changed the way we",
    "In many countries the weather has been",
    "The relationship between food and health is",
    "Cities around the world are becoming more",
    "The way people communicate has changed because",
    "One of the biggest challenges today is",
    "Many experts believe that the solution to",
]


def compute_register_heuristics(text):
    """Simple heuristics as register proxies.

    These are rough indicators, not classifiers:
    - avg_word_length: longer words suggest formal/academic
    - long_word_fraction: fraction of words with 8+ chars (academic indicator)
    - short_word_fraction: fraction of words with <=3 chars
    - avg_sentence_length: longer sentences suggest formal/academic
    - contraction_count: contractions suggest casual/slang
    - exclamation_count: exclamations suggest casual/slang
    """
    words = text.split()
    if not words:
        return {}

    word_lengths = [len(w.strip('.,!?;:"\'-')) for w in words]
    word_lengths = [l for l in word_lengths if l > 0]
    if not word_lengths:
        return {}

    # Sentence count (rough)
    sentences = max(1, text.count('.') + text.count('!') + text.count('?'))

    # Contractions
    contraction_words = {"i'm", "i've", "i'll", "i'd", "you're", "you've",
                         "you'll", "you'd", "he's", "she's", "it's", "we're",
                         "we've", "we'll", "they're", "they've", "they'll",
                         "isn't", "aren't", "wasn't", "weren't", "don't",
                         "doesn't", "didn't", "won't", "wouldn't", "couldn't",
                         "shouldn't", "can't", "couldn't", "that's", "there's",
                         "here's", "what's", "who's", "let's", "gonna", "wanna",
                         "gotta", "kinda", "sorta", "ain't", "y'all",
                         "tbh", "imo", "ngl", "fr", "rn", "lowkey", "highkey",
                         "literally", "totally", "basically", "honestly", "like"}
    text_lower = text.lower()
    words_lower = text_lower.split()
    contraction_count = sum(1 for w in words_lower
                           if w.strip('.,!?;:') in contraction_words)

    return {
        'avg_word_length': np.mean(word_lengths),
        'long_word_fraction': sum(1 for l in word_lengths if l >= 8) / len(word_lengths),
        'short_word_fraction': sum(1 for l in word_lengths if l <= 3) / len(word_lengths),
        'avg_sentence_length': len(words) / sentences,
        'contraction_count': contraction_count,
        'exclamation_count': text.count('!'),
        'n_words': len(words),
    }


def generate_steered(model, tokenizer, device, prompt, steer_direction, angle_deg,
                     n_tokens, temperature, layer):
    """Generate text with spherical perturbation steering at given layer."""
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
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=LAYER)
    parser.add_argument('--n_tokens', type=int, default=N_TOKENS)
    parser.add_argument('--temperature', type=float, default=TEMPERATURE)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--n_prompts', type=int, default=None,
                        help='Number of neutral prompts to use (default: all)')
    args = parser.parse_args()

    results_dir = f'results/genre_contrastive/{args.model}'

    # Load directions
    directions_path = os.path.join(results_dir, 'direction_results.pkl')
    if not os.path.exists(directions_path):
        print(f"Error: {directions_path} not found.")
        print("Run run_register_contrastive.py first.")
        sys.exit(1)

    print("Loading directions...")
    with open(directions_path, 'rb') as f:
        saved = pickle.load(f)

    registers = saved['registers']
    print(f"Registers: {registers}")

    print("Loading model...")
    model, tokenizer, device = load_model(args.model)
    torch.manual_seed(args.seed)

    # Build steering configs: steer toward each register using all-pair estimators
    steer_configs = {}
    for reg in registers:
        steer_configs[f'f_allpair_{reg}'] = saved['f_allpair'][reg]['normalized'].to(device)
        steer_configs[f'f_star_{reg}'] = saved['f_star'][reg]['normalized'].to(device)

    # Add MELBO directions if available
    if saved.get('melbo') and saved['melbo'].get('directions'):
        for i, d in enumerate(saved['melbo']['directions']):
            steer_configs[f'MELBO_joint_{i+1}'] = d.to(device)

    prompts = NEUTRAL_PROMPTS
    if args.n_prompts:
        prompts = prompts[:args.n_prompts]

    dir_names = list(steer_configs.keys())
    total = len(prompts) * len(ANGLES) * len(dir_names)
    print(f"Neutral prompts: {len(prompts)}, Angles: {ANGLES}, Directions: {len(dir_names)}")
    print(f"Total generations: {total}")

    all_results = []
    count = 0
    t0 = time.time()

    for prompt_idx, prompt in enumerate(prompts):
        for angle in ANGLES:
            for dir_name in dir_names:
                gen_seed = args.seed + prompt_idx * 1000 + angle * 10 + hash(dir_name) % 100
                torch.manual_seed(gen_seed)

                direction = steer_configs[dir_name]
                text = generate_steered(
                    model, tokenizer, device, prompt, direction, angle,
                    args.n_tokens, args.temperature, args.layer)

                heuristics = compute_register_heuristics(text)

                result = {
                    'prompt_idx': prompt_idx,
                    'prompt': prompt,
                    'direction': dir_name,
                    'angle': angle,
                    'text': text,
                    'heuristics': heuristics,
                }
                all_results.append(result)

                count += 1
                if count % 50 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / count * (total - count)
                    print(f"  [{count}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"\nAll {total} generations complete in {elapsed:.0f}s")

    with open(os.path.join(results_dir, 'steering_registers.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {results_dir}/steering_registers.json")

    # Print summary: for each register direction, show heuristics by angle
    print(f"\n{'='*80}")
    print("SUMMARY: Register heuristics by direction and angle")
    print(f"{'='*80}")

    # Focus on all-pair estimators for the summary
    for reg in registers:
        dir_name = f'f_allpair_{reg}'
        print(f"\n  Steering toward: {reg} (via all-pair estimator)")
        print(f"    {'Angle':>6s}  {'AvgWLen':>8s}  {'LongFrac':>9s}  {'Contrxn':>8s}  {'SentLen':>8s}")
        for angle in ANGLES:
            results_at = [r for r in all_results
                         if r['direction'] == dir_name and r['angle'] == angle]
            if not results_at or not results_at[0].get('heuristics'):
                continue
            avg_wl = np.mean([r['heuristics'].get('avg_word_length', 0) for r in results_at])
            long_f = np.mean([r['heuristics'].get('long_word_fraction', 0) for r in results_at])
            contr = np.mean([r['heuristics'].get('contraction_count', 0) for r in results_at])
            sent_l = np.mean([r['heuristics'].get('avg_sentence_length', 0) for r in results_at])
            print(f"    {angle:>5d}°  {avg_wl:>8.2f}  {long_f:>8.1%}  {contr:>8.1f}  {sent_l:>8.1f}")

    # Print example generations for qualitative inspection
    print(f"\n{'='*80}")
    print("EXAMPLE GENERATIONS (prompt 0, angles 0° and 30°)")
    print(f"{'='*80}")

    for angle in [0, 30]:
        print(f"\n  Angle = {angle}°:")
        for reg in registers:
            dir_name = f'f_allpair_{reg}'
            results_at = [r for r in all_results
                         if r['direction'] == dir_name and r['angle'] == angle
                         and r['prompt_idx'] == 0]
            if results_at:
                text = results_at[0]['text']
                prompt = results_at[0]['prompt']
                print(f"    [{reg:10s}] {prompt} || {text[:80]}")

    print("\nDone!")


if __name__ == '__main__':
    main()
