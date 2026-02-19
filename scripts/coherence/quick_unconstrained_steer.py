#!/usr/bin/env python3
"""Quick qualitative check: generate text while steering along unconstrained MELBO direction."""

import sys
import pickle
import numpy as np
sys.path.insert(0, '.')

import torch
from src.model import load_model, _get_blocks

# Load model
print("Loading model...")
model, tokenizer, device = load_model('qwen3-0.6b')

# Load unconstrained direction
with open('results/melbo_unconstrained/qwen3-0.6b/optimization_results.pkl', 'rb') as f:
    unc = pickle.load(f)
direction = unc['unc_directions'][0].to(device)
print(f"Unconstrained_1 norm: {direction.norm():.4f}")

LAYER = 14
N_TOKENS = 100

prompts = [
    "The weather today is",
    "Once upon a time, there",
    "The capital of France is",
    "In recent years, scientists have",
    "She opened the door and",
]

TEMPERATURE = 0.7

def generate_steered(prompt, steer_direction, angle_deg, n_tokens=100):
    """Generate with spherical steering, rolling mode, with temperature sampling."""
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
        handle = layers[LAYER].register_forward_hook(hook)
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
        handle.remove()

        logits = outputs.logits[0, -1, :] / TEMPERATURE
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        generated_ids.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


angles = [0, 5, 8, 12]

print(f"\n{'='*70}")
print(f"UNCONSTRAINED MELBO_1 STEERING (layer {LAYER}, {N_TOKENS} tokens, temp={TEMPERATURE})")
print(f"{'='*70}")

for prompt in prompts:
    print(f"\nPrompt: \"{prompt}\"")
    for angle in angles:
        text = generate_steered(prompt, direction, angle, N_TOKENS)
        print(f"  {angle:3d} deg: {text}")

print("\nDone!")
