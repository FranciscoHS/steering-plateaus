#!/usr/bin/env python3
"""
Cross-prompt steering: take one prompt's MELBO direction and steer all others.

Tests whether directions found for one prompt transfer — evidence they're
real features rather than prompt-specific artifacts.
"""

import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model


def make_steering_hook(direction, angle_rad):
    """Hook that spherically perturbs the last-token activation."""
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        a = hidden[:, -1, :]
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
                     device, n_tokens=20, temperature=None):
    """Generate with steering hook active at every forward pass."""
    angle_rad = np.deg2rad(angle_deg)
    tokens = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = tokens['input_ids']

    block = model.transformer.h[layer]
    hook = block.register_forward_hook(make_steering_hook(direction, angle_rad))

    gen_kwargs = dict(max_new_tokens=n_tokens, pad_token_id=tokenizer.eos_token_id)
    if temperature is not None:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs['do_sample'] = False

    try:
        with torch.no_grad():
            output_ids = model.generate(input_ids, **gen_kwargs)
        new_ids = output_ids[0, input_ids.shape[1]:]
        return tokenizer.decode(new_ids)
    finally:
        hook.remove()


def generate_unsteered(model, tokenizer, prompt, device, n_tokens=20,
                       temperature=None):
    """Normal generation without steering."""
    tokens = tokenizer(prompt, return_tensors='pt').to(device)
    gen_kwargs = dict(max_new_tokens=n_tokens, pad_token_id=tokenizer.eos_token_id)
    if temperature is not None:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs['do_sample'] = False
    with torch.no_grad():
        output_ids = model.generate(tokens['input_ids'], **gen_kwargs)
    new_ids = output_ids[0, tokens['input_ids'].shape[1]:]
    return tokenizer.decode(new_ids)


def main():
    results_path = 'results/melbo_python_individual/gpt2-large_target1.0_10dirs/results.pkl'
    print(f"Loading results from {results_path}...")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    prompts = results['prompts']
    all_directions = results['all_directions']

    print("Loading model...")
    model, tokenizer, device = load_model('gpt2-large')
    layer = 14

    # Check overlap of P0_D3 with comma activation
    d_p0_d3 = all_directions[0][3].to(device)
    d_p3_d3 = all_directions[3][3].to(device)

    comma_tokens = tokenizer(',', return_tensors='pt').to(device)
    block = model.transformer.h[layer]
    comma_act = None
    def capture_hook(module, input, output):
        nonlocal comma_act
        h = output[0] if isinstance(output, tuple) else output
        comma_act = h[:, -1, :].clone()
    hook = block.register_forward_hook(capture_hook)
    with torch.no_grad():
        model(**comma_tokens)
    hook.remove()
    comma_unit = comma_act[0] / comma_act[0].norm()
    print(f"|cos|(P0_D3, comma_act) = {abs(torch.dot(d_p0_d3, comma_unit).item()):.4f}")
    print(f"|cos|(P3_D3, comma_act) = {abs(torch.dot(d_p3_d3, comma_unit).item()):.4f}")

    # Steering with temperature
    test_prompts = {0: prompts[0], 3: prompts[3]}
    test_dirs = {
        'P0_D3': d_p0_d3,
        'P3_D3': d_p3_d3,
    }
    angles = [0, 20, 60]

    lines = []
    for dir_name, direction in test_dirs.items():
        lines.append(f"{'='*70}")
        lines.append(f"STEERING WITH: {dir_name}  (temp=0.7, 3 samples each)")
        lines.append(f"{'='*70}")
        for p_idx, prompt in test_prompts.items():
            lines.append(f"\n  P{p_idx}: {prompt}")
            for angle_deg in angles:
                for sample in range(3):
                    if angle_deg == 0:
                        cont = generate_unsteered(model, tokenizer, prompt,
                                                  device, n_tokens=40,
                                                  temperature=0.7)
                    else:
                        cont = generate_steered(
                            model, tokenizer, prompt, direction, angle_deg,
                            layer, device, n_tokens=40, temperature=0.7)
                    tag = f"    {angle_deg:3d} deg [{sample+1}]:"
                    lines.append(f"{tag} {cont}")
        lines.append("")

    out = '\n'.join(lines)
    print(out)
    out_path = os.path.join(os.path.dirname(results_path),
                            'steer_matched_pair_temp.txt')
    with open(out_path, 'w') as f:
        f.write(out)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
