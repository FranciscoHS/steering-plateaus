#!/usr/bin/env python3
"""
Steer along the top-k MELBO directions found on diverse activations.

For each direction, generate text at several steering angles (both positive
and negative) from a set of diverse prompts. This helps interpret what
behavioral feature each direction corresponds to.

Usage:
    python scripts/run_steer_melbo_diverse.py
"""

import sys, os, pickle, time
import numpy as np
sys.path.insert(0, '.')

import torch
from src.model import load_model, _get_blocks

LAYER = 14
N_TOKENS = 50
TEMPERATURE = 0.7
SEED = 42

# Diverse prompts — mix of topics and styles to see what changes
PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "The best way to cook pasta is",
    "In a recent study, scientists found that",
    "My favorite thing about summer is",
    "The meaning of life is",
    "To solve this math problem, first",
    "Dear customer, thank you for your",
]

# Steering angles: positive and negative
ANGLES = [-30, -20, -10, -5, 0, 5, 10, 20, 30]


def generate_steered(model, tokenizer, device, prompt, steer_direction, angle_deg,
                     n_tokens, temperature, layer):
    """Generate with spherical steering at every token."""
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

    cos_theta = np.cos(np.radians(abs(angle_deg)))
    sin_theta = np.sin(np.radians(abs(angle_deg)))
    sign = 1 if angle_deg > 0 else -1
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
            v_perp = sign * d.unsqueeze(0) - proj * a_unit
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
    results_dir = 'results/melbo_diverse/qwen3-0.6b'

    # Load MELBO directions
    print("Loading MELBO directions...")
    with open(os.path.join(results_dir, 'results.pkl'), 'rb') as f:
        melbo = pickle.load(f)
    directions = melbo['directions']
    kls = melbo['kls']
    n_dirs = len(directions)
    print(f"Loaded {n_dirs} directions")
    for i, kl in enumerate(kls):
        print(f"  D{i+1}: KL={kl:.4f}")

    # Load model
    print("\nLoading model...")
    model, tokenizer, device = load_model('qwen3-0.6b')

    # Move directions to device
    directions_dev = [d.to(device) for d in directions]

    # Generate
    all_results = []
    total = len(PROMPTS) * len(ANGLES) * n_dirs
    count = 0
    t0 = time.time()

    for dir_idx in range(n_dirs):
        print(f"\n{'='*60}")
        print(f"MELBO_{dir_idx+1} (KL={kls[dir_idx]:.4f})")
        print(f"{'='*60}")

        for prompt in PROMPTS:
            for angle in ANGLES:
                torch.manual_seed(SEED + hash(prompt) % 10000 + angle * 10 + dir_idx)

                text = generate_steered(
                    model, tokenizer, device, prompt,
                    directions_dev[dir_idx], angle,
                    N_TOKENS, TEMPERATURE, LAYER)

                result = {
                    'direction': dir_idx,
                    'prompt': prompt,
                    'angle': angle,
                    'text': text,
                }
                all_results.append(result)

                count += 1

            # Print this prompt's results for this direction
            baseline = [r for r in all_results
                       if r['direction'] == dir_idx and r['prompt'] == prompt and r['angle'] == 0]
            if baseline:
                print(f"\nPrompt: {prompt}")
                print(f"  0°: {baseline[0]['text'][:120]}")
                for angle in [-20, -10, 10, 20]:
                    matches = [r for r in all_results
                              if r['direction'] == dir_idx and r['prompt'] == prompt and r['angle'] == angle]
                    if matches:
                        print(f"  {angle:+d}°: {matches[0]['text'][:120]}")

        elapsed = time.time() - t0
        rate = count / elapsed if elapsed > 0 else 0
        remaining = (total - count) / rate if rate > 0 else 0
        print(f"\n[{count}/{total}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

    # Save
    save_path = os.path.join(results_dir, 'steering_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'results': all_results,
            'prompts': PROMPTS,
            'angles': ANGLES,
            'n_directions': n_dirs,
            'kls': kls,
        }, f)
    print(f"\nSaved: {save_path}")

    # Print full summary
    print(f"\n{'='*60}")
    print("FULL SUMMARY")
    print(f"{'='*60}")
    for dir_idx in range(n_dirs):
        print(f"\n--- MELBO_{dir_idx+1} (KL={kls[dir_idx]:.4f}) ---")
        for prompt in PROMPTS[:4]:  # First 4 prompts for summary
            print(f"\nPrompt: {prompt}")
            for angle in ANGLES:
                matches = [r for r in all_results
                          if r['direction'] == dir_idx and r['prompt'] == prompt and r['angle'] == angle]
                if matches:
                    print(f"  {angle:+d}°: {matches[0]['text'][:100]}")

    print("\nDone!")


if __name__ == '__main__':
    main()
