#!/usr/bin/env python3
"""
Qualitative steering comparison:
  - MELBO_1 (full direction)
  - -mean (negative diff-of-means)
  - MELBO_1 ⊥ mean (orthogonal component — "feature minus noise"?)

Spherical steering at layer 14 with temperature sampling.
"""

import sys
import pickle
import numpy as np
sys.path.insert(0, '.')

import torch
from src.model import load_model, _get_blocks

# Load model
print("Loading model...")
model, tokenizer, device = load_model('qwen3-0.6b')

# Load directions
with open('results/melbo_constrained/qwen3-0.6b/optimization_results.pkl', 'rb') as f:
    saved = pickle.load(f)

mean_dir = saved['mean_dir'].to(device)
melbo1 = saved['melbo_directions'][0].to(device)

# Compute orthogonal component
neg_mean = -mean_dir
melbo1_orth = melbo1 - (melbo1 @ mean_dir) * mean_dir
melbo1_orth = melbo1_orth / melbo1_orth.norm()

print(f"cos(MELBO_1, mean) = {(melbo1 @ mean_dir).item():.4f}")
print(f"cos(MELBO_1_orth, mean) = {(melbo1_orth @ mean_dir).item():.6f} (should be ~0)")
print(f"cos(MELBO_1_orth, MELBO_1) = {(melbo1_orth @ melbo1).item():.4f}")
print(f"||orth component|| before normalization = {(melbo1 - (melbo1 @ mean_dir) * mean_dir).norm().item():.4f}")

LAYER = 14
N_TOKENS = 100
TEMPERATURE = 0.7

prompts = [
    "The weather today is",
    "Once upon a time, there",
    "The capital of France is",
    "In recent years, scientists have",
    "She opened the door and",
]

directions = {
    'MELBO_1':        melbo1,
    '-mean':          neg_mean,
    'MELBO_1⊥mean':  melbo1_orth,
}


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


angles = [5, 8, 12]

# Set seed for reproducibility
torch.manual_seed(42)

output_lines = []

def log(s=''):
    print(s)
    output_lines.append(s)

log(f"ORTHOGONAL STEERING COMPARISON (layer {LAYER}, {N_TOKENS} tokens, temp={TEMPERATURE})")
log(f"{'='*80}")

for prompt in prompts:
    log(f"\nPrompt: \"{prompt}\"")
    for angle in angles:
        log(f"\n  {angle} deg:")
        for dir_name, direction in directions.items():
            # Reset seed per generation for fair comparison
            torch.manual_seed(42 + hash(prompt + str(angle) + dir_name) % 10000)
            text = generate_steered(prompt, direction, angle, N_TOKENS)
            log(f"    {dir_name:15s}: {text}")

log(f"\n{'='*80}")
log("Done!")

# Save
outpath = 'results/orth_comparison/qwen3-0.6b/steering_output.txt'
with open(outpath, 'w') as f:
    f.write('\n'.join(output_lines))
print(f"\nSaved to {outpath}")
