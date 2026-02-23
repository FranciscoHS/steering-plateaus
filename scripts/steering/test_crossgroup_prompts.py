#!/usr/bin/env python3
"""
Quick scan of many prompts with RU+C combined steering at 30°.
Goal: find the prompt where the compositional effect is clearest.
"""

import sys
import os
import pickle
import numpy as np
import torch

sys.path.insert(0, '/workspace/steering-plateaus')
from src.model import load_model, _get_blocks


def generate_steered(model, tokenizer, device, prompt, directions, angle_deg,
                     n_tokens=80, temperature=0.7, layer=14):
    """Generate with spherical steering. directions = list of unit vectors, applied sequentially."""
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

            cos_t = np.cos(np.radians(angle_deg))
            sin_t = np.sin(np.radians(angle_deg))

            # Apply each direction sequentially
            h = steer_slice.clone()
            for d in directions:
                R = h.norm(dim=-1, keepdim=True)
                a_unit = h / (R + 1e-10)
                proj = (a_unit * d).sum(dim=-1, keepdim=True)
                v_perp = d.unsqueeze(0) - proj * a_unit
                v_perp = v_perp / (v_perp.norm(dim=-1, keepdim=True) + 1e-10)
                h = R * (cos_t * a_unit + sin_t * v_perp)
            hidden[0, first_steer_pos:, :] = h
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
    model, tokenizer, device = load_model('qwen3-4b')

    # Load features
    with open("results/allpairs_contrastive/qwen3-4b/directions.pkl", "rb") as f:
        lang_data = pickle.load(f)
    with open("results/programming_contrastive/qwen3-4b/directions_6lang_with_python.pkl", "rb") as f:
        prog_data = pickle.load(f)

    ru_vec = lang_data["features"]["ru"].to(device)
    c_vec = prog_data["features"]["c"].to(device)
    cos_rc = (ru_vec @ c_vec).item()
    print(f"cos(ru, c) = {cos_rc:.4f}")

    prompts = [
        "Explain how to implement a linked list",
        "Here is a simple sorting algorithm:",
        "// This function calculates",
        "def solve(n):",
        "The error in this code is",
        "To read a file in",
        "A recursive function that",
        "Convert the following pseudocode to code:",
        "Implement binary search",
        "The most efficient way to",
        "Write a program that",
        "How do I fix this bug?",
    ]

    angle = 30
    conditions = [
        ("baseline", []),
        ("+RU", [ru_vec]),
        ("+C", [c_vec]),
        ("+RU+C", [ru_vec, c_vec]),
    ]

    results = []
    torch.manual_seed(42)

    for prompt in prompts:
        results.append(f"\n{'='*70}")
        results.append(f"PROMPT: \"{prompt}\"")
        results.append(f"{'='*70}")

        for cond_name, dirs in conditions:
            a = 0 if cond_name == "baseline" else angle
            out = generate_steered(model, tokenizer, device, prompt, dirs, a)
            text = out[:300].replace('\n', ' | ')
            results.append(f"  {cond_name:>8}: {text}")

    output = "\n".join(results)
    print(output)

    with open("results/crossgroup_prompt_search.txt", "w") as f:
        f.write(output)
    print("\nSaved to results/crossgroup_prompt_search.txt")


if __name__ == '__main__':
    main()
