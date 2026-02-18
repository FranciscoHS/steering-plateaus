#!/usr/bin/env python3
"""
Steer FineWeb prompts along MELBO_1 (red) and MELBO_2 (yellow)
in both +/- directions up to 60 degrees. 5 prompts, qualitative.
"""
import os
import pickle
import sys
import torch
import numpy as np
sys.path.insert(0, '.')
from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length

# Load robust results
with open('results/melbo_animal/qwen3-0.6b_robust_r20_s500/results.pkl', 'rb') as f:
    r = pickle.load(f)
melbo_dirs = r['melbo_directions']

# Load model
model, tokenizer, device = load_model('qwen3-0.6b')
layer = 14

# Directions
d1 = melbo_dirs[0].to(device)
d1 = d1 / d1.norm()
d2 = melbo_dirs[1].to(device)
d2 = d2 / d2.norm()

# Load FineWeb — 5 prompts, 20 tokens each
token_ids_list = load_fineweb_fixed_length(5, tokenizer, seq_len=20, seed=123)
prompts = [tokenizer.decode(tids, skip_special_tokens=True) for tids in token_ids_list]

# Output file
out_dir = 'results/melbo_animal/qwen3-0.6b_robust_r20_s500'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'steering_generations_v2.txt')
out_f = open(out_path, 'w')


def log(msg):
    print(msg)
    out_f.write(msg + '\n')
    out_f.flush()


def generate_with_spherical_steering(prompt, direction, angle_deg, n_tokens=30):
    """Generate tokens with spherical steering at the steer layer."""
    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))

    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    prompt_len = input_ids.shape[1]
    first_steer_pos = prompt_len - 1

    generated_ids = []

    for step in range(n_tokens):
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            steer_slice = hidden[0, first_steer_pos:, :]
            if steer_slice.shape[0] == 0:
                return output

            R = steer_slice.norm(dim=-1, keepdim=True)
            a_unit = steer_slice / (R + 1e-10)
            proj = (a_unit * direction).sum(dim=-1, keepdim=True)
            v_perp = direction.unsqueeze(0) - proj * a_unit
            v_perp = v_perp / (v_perp.norm(dim=-1, keepdim=True) + 1e-10)
            steered = R * (cos_theta * a_unit + sin_theta * v_perp)
            hidden[0, first_steer_pos:, :] = steered

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        layers = _get_blocks(model)
        handle = layers[layer].register_forward_hook(steering_hook)
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
        handle.remove()

        logits = outputs.logits[0, -1, :]
        next_token_id = logits.argmax().item()
        generated_ids.append(next_token_id)
        if next_token_id == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([
            input_ids, torch.tensor([[next_token_id]], device=device)
        ], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# Show full prompts
log("=== PROMPTS ===")
for i, p in enumerate(prompts):
    log(f"  [{i}] \"{p}\"")

angles = [0, 15, 25, 35, 45, 60]

for dir_name, direction in [("MELBO_1 (red)", d1), ("MELBO_2 (yellow)", d2)]:
    for sign, sign_label in [(1.0, "+"), (-1.0, "-")]:
        d = sign * direction
        log(f"\n{'='*70}")
        log(f"{dir_name}, {sign_label}dir")
        log(f"{'='*70}")
        for angle in angles:
            log(f"\n--- {angle} deg ---")
            for i, prompt in enumerate(prompts):
                cont = generate_with_spherical_steering(prompt, d, angle)
                log(f"  [{i}] {cont}")

out_f.close()
print(f"\nSaved to {out_path}")
print("Done!")
