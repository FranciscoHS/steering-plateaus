#!/usr/bin/env python3
"""
Fine-grained steering sweep for all capped MELBO directions.
Positive direction only, 7-15 degrees in 1-degree steps.
Multiple samples per configuration to see variety.
"""

import sys, os, pickle, time
import numpy as np
sys.path.insert(0, '.')

import torch
from src.model import load_model, _get_blocks

LAYER = 14
N_TOKENS = 50
TEMPERATURE = 0.7
N_SAMPLES = 3  # samples per prompt/angle/direction

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

ANGLES = [0, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def generate_steered(model, tokenizer, device, prompt, steer_direction, angle_deg,
                     n_tokens, temperature, layer, seed):
    """Generate with spherical steering at every token."""
    torch.manual_seed(seed)

    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    d = steer_direction

    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    prompt_len = input_ids.shape[1]
    first_steer_pos = prompt_len - 1

    generated_ids = []
    for step in range(n_tokens):
        if angle_deg == 0:
            with torch.no_grad():
                outputs = model(input_ids, use_cache=False)
        else:
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                s = hidden[0, first_steer_pos:, :]
                if s.shape[0] == 0:
                    return output
                R = s.norm(dim=-1, keepdim=True)
                a_u = s / (R + 1e-10)
                proj = (a_u * d).sum(dim=-1, keepdim=True)
                v_perp = d.unsqueeze(0) - proj * a_u
                v_perp = v_perp / (v_perp.norm(dim=-1, keepdim=True) + 1e-10)
                hidden[0, first_steer_pos:, :] = R * (cos_theta * a_u + sin_theta * v_perp)
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
    results_dir = 'results/melbo_diverse_capped/qwen3-0.6b_target1.0'

    print("Loading capped MELBO directions...")
    with open(os.path.join(results_dir, 'results.pkl'), 'rb') as f:
        melbo = pickle.load(f)
    directions = melbo['directions']
    rewards = melbo['rewards']
    n_dirs = len(directions)

    print("Loading model...")
    model, tokenizer, device = load_model('qwen3-0.6b')
    directions_dev = [d.to(device) for d in directions]

    total = n_dirs * len(PROMPTS) * len(ANGLES) * N_SAMPLES
    count = 0
    t0 = time.time()

    out_path = os.path.join(results_dir, 'fine_sweep.txt')
    with open(out_path, 'w') as f:
        f.write(f"Fine-grained steering sweep: capped MELBO directions\n")
        f.write(f"Model: qwen3-0.6b, layer {LAYER}\n")
        f.write(f"Angles: {ANGLES} (positive direction only)\n")
        f.write(f"Temperature: {TEMPERATURE}, {N_SAMPLES} samples per config\n")
        f.write(f"{'=' * 80}\n\n")

        for dir_idx in range(n_dirs):
            f.write(f"--- D{dir_idx+1} (reward={rewards[dir_idx]:.4f}) ---\n\n")
            print(f"Direction {dir_idx+1}/{n_dirs}...")

            for prompt in PROMPTS:
                f.write(f"Prompt: {prompt}\n")
                for angle in ANGLES:
                    for s in range(N_SAMPLES):
                        seed = 42 + hash(prompt) % 10000 + angle * 100 + dir_idx * 7 + s * 31
                        text = generate_steered(
                            model, tokenizer, device, prompt,
                            directions_dev[dir_idx], angle,
                            N_TOKENS, TEMPERATURE, LAYER, seed)
                        text = text.replace('\n', ' ')
                        label = f"+{angle:2d}" if s == 0 else "   "
                        f.write(f"  {label} [{s+1}]: {text[:150]}\n")
                        count += 1

                f.write("\n")

            elapsed = time.time() - t0
            rate = count / elapsed if elapsed > 0 else 0
            remaining = (total - count) / rate if rate > 0 else 0
            print(f"  [{count}/{total}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")
            f.write("\n")
            f.flush()

    print(f"\nSaved: {out_path}")
    print("Done!")


if __name__ == '__main__':
    main()
