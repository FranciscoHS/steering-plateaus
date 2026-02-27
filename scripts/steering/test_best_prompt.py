#!/usr/bin/env python3
"""Get full-length outputs for the best compositional prompt."""

import sys
import os
import pickle
import numpy as np
import torch

sys.path.insert(0, '/workspace/steering-plateaus')
from src.model import load_model, _get_blocks


def generate_steered(model, tokenizer, device, prompt, directions, angle_deg,
                     n_tokens=150, temperature=0.7, layer=14):
    """Generate with spherical steering."""
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

    with open("results/allpairs_contrastive/qwen3-4b/directions.pkl", "rb") as f:
        lang_data = pickle.load(f)
    with open("results/programming_contrastive/qwen3-4b/directions_6lang_with_python.pkl", "rb") as f:
        prog_data = pickle.load(f)

    ru_vec = lang_data["features"]["ru"].to(device)
    c_vec = prog_data["features"]["c"].to(device)

    prompts = [
        "// This function calculates",
        "Here is a simple sorting algorithm:",
    ]

    conditions = [
        ("baseline", []),
        ("+RU only", [ru_vec]),
        ("+C only", [c_vec]),
        ("+RU+C combined", [ru_vec, c_vec]),
    ]

    angle = 30
    torch.manual_seed(42)

    lines = []
    for prompt in prompts:
        lines.append(f"\n{'='*70}")
        lines.append(f"PROMPT: \"{prompt}\"")
        lines.append(f"{'='*70}")
        for cond_name, dirs in conditions:
            a = 0 if cond_name == "baseline" else angle
            # Run 2 samples
            for s in range(2):
                torch.manual_seed(42 + s * 1000)
                out = generate_steered(model, tokenizer, device, prompt, dirs, a, n_tokens=150)
                lines.append(f"\n--- {cond_name} @ {a} (sample {s+1}) ---")
                lines.append(out)

    output = "\n".join(lines)
    print(output)
    with open("results/crossgroup_best_prompts.txt", "w") as f:
        f.write(output)
    print("\nSaved.")


if __name__ == '__main__':
    main()
