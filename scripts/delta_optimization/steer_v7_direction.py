#!/usr/bin/env python3
"""Steer along the v7 optimized direction at various angles, generate 20 tokens."""

import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model, _get_blocks

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--n_samples', type=int, default=3)
parser.add_argument('--n_tokens', type=int, default=20)
parser.add_argument('--angles', type=float, nargs='+', default=[0, 15, 30, 60])
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

with open(args.checkpoint, 'rb') as f:
    cp = pickle.load(f)

run_args = cp['args']
direction = cp['opt_state']['direction']
direction = direction / direction.norm()

print('Loading model...', flush=True)
model, tokenizer, device = load_model(run_args['model'])
direction = direction.to(device)
layer = run_args['layer']

# Pick random samples
np.random.seed(args.seed)
indices = np.random.choice(run_args['n_act'], args.n_samples, replace=False)

# Recover original token sequences
from src.data import load_fineweb_fixed_length
token_ids_list = load_fineweb_fixed_length(
    run_args['n_act'] + run_args['n_ctx'], tokenizer,
    seq_len=run_args['seq_len'], seed=run_args['seed']
)


def steer_hook_fn(mod, inp, out, sin_t, cos_t, d):
    """Spherical steering on last token at block output."""
    if isinstance(out, tuple):
        hidden = out[0]
    else:
        hidden = out

    a = hidden[:, -1, :].clone()
    R = a.norm(dim=-1, keepdim=True)
    a_hat = a / R
    proj = (d * a_hat).sum(dim=-1, keepdim=True)
    v = d - proj * a_hat
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-10)
    steered = cos_t * a + sin_t * R * v

    hidden = hidden.clone()
    hidden[:, -1, :] = steered

    if isinstance(out, tuple):
        return (hidden,) + out[1:]
    return hidden


angles = args.angles
output_lines = []

for idx in indices:
    prompt_tokens = token_ids_list[idx].to(device)
    prompt_text = tokenizer.decode(prompt_tokens)
    output_lines.append(f"=== Sample {idx} ===")
    output_lines.append(f"Prompt: {repr(prompt_text)}")

    for angle in angles:
        theta = np.deg2rad(float(angle))
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        input_ids = prompt_tokens.unsqueeze(0)
        blocks = _get_blocks(model)

        for gen_step in range(args.n_tokens):
            handle = blocks[layer].register_forward_hook(
                lambda mod, inp, out, _s=sin_t, _c=cos_t, _d=direction:
                    steer_hook_fn(mod, inp, out, _s, _c, _d)
            )
            with torch.no_grad():
                logits = model(input_ids).logits[:, -1, :]
                next_token = logits.argmax(dim=-1, keepdim=True)
            handle.remove()
            input_ids = torch.cat([input_ids, next_token], dim=1)

        generated = tokenizer.decode(input_ids[0, len(prompt_tokens):])
        output_lines.append(f"  {angle:3.0f} deg: {repr(generated)}")

    output_lines.append("")

for line in output_lines:
    print(line, flush=True)

with open(args.output, 'w') as f:
    f.write('\n'.join(output_lines) + '\n')
print(f"Saved: {args.output}", flush=True)
