#!/usr/bin/env python3
"""
Steer with per-prompt MELBO directions to check interpretability.

For 3 prompts, take their top-1 direction and generate 20 tokens
at angles 0, 10, 20, 30, 40, 50 degrees. Steering = perturb at
layer 14 on every forward pass.
"""

import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model


def make_steering_hook(direction, angle_rad):
    """Return a hook that spherically perturbs the last-token activation."""
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    def hook_fn(module, input, output):
        # output is (hidden_states, ...) or just hidden_states depending on model
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Perturb last token position only
        a = hidden[:, -1, :]  # [batch, d]
        a_norm = a.norm(dim=-1, keepdim=True)
        a_unit = a / a_norm

        proj = (direction.unsqueeze(0) * a_unit).sum(dim=-1, keepdim=True)
        u_perp = direction.unsqueeze(0) - proj * a_unit
        u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)

        a_pert = a_norm * (cos_t * a_unit + sin_t * u_perp)
        hidden = hidden.clone()
        hidden[:, -1, :] = a_pert

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    return hook_fn


def generate_steered(model, tokenizer, prompt, direction, angle_deg, layer,
                     device, n_tokens=20):
    """Generate with steering hook active at every forward pass."""
    angle_rad = np.deg2rad(angle_deg)
    tokens = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = tokens['input_ids']

    # Register hook on the target layer's output
    # GPT-2: model.transformer.h[layer] is the block
    block = model.transformer.h[layer]
    hook = block.register_forward_hook(make_steering_hook(direction, angle_rad))

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=n_tokens,
                do_sample=False,  # greedy
                temperature=1.0,
            )
        # Decode only the new tokens
        new_ids = output_ids[0, input_ids.shape[1]:]
        return tokenizer.decode(new_ids)
    finally:
        hook.remove()


def main():
    results_path = 'results/melbo_python_individual/gpt2-large_target1.0/results.pkl'
    print(f"Loading results from {results_path}...")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    prompts = results['prompts']
    all_directions = results['all_directions']

    print("Loading model...")
    model, tokenizer, device = load_model('gpt2-large')
    layer = 14

    # 3 prompts: one code, one English-about-Python, one programming
    chosen = [0, 4, 9]
    angles = [0, 10, 20, 30, 40, 50]

    for p_idx in chosen:
        prompt = prompts[p_idx]
        direction = all_directions[p_idx][0].to(device)  # top-1 direction

        print(f"\n{'='*60}")
        print(f"Prompt {p_idx}: {prompt}")
        print(f"{'='*60}")

        for angle_deg in angles:
            if angle_deg == 0:
                # Unsteered baseline
                tokens = tokenizer(prompt, return_tensors='pt').to(device)
                with torch.no_grad():
                    output_ids = model.generate(
                        tokens['input_ids'],
                        max_new_tokens=20,
                        do_sample=False,
                    )
                new_ids = output_ids[0, tokens['input_ids'].shape[1]:]
                continuation = tokenizer.decode(new_ids)
            else:
                continuation = generate_steered(
                    model, tokenizer, prompt, direction, angle_deg, layer,
                    device, n_tokens=20)
            print(f"  {angle_deg:3d} deg: {continuation}")


if __name__ == '__main__':
    main()
