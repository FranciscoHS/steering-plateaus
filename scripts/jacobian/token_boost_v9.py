#!/usr/bin/env python3
"""
Show which tokens get the largest probability boost from steering.

For each prompt, compares unsteered vs steered (spherical) logits at
the last token position and reports top gained/lost tokens.

Usage:
    python scripts/jacobian/token_boost_v9.py \
        --direction_pkl results/delta_opt_v9_10act/results_10deg_kl1.0.pkl \
        --angle 30
"""

import argparse
import os
import pickle
import sys
import numpy as np
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length


def get_steered_logits(model, tokenizer, prompt, steer_direction, angle_deg,
                       steer_layer, device):
    """Get logits for next token with spherical steering at the last prompt token only."""
    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    direction = steer_direction.to(device)

    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)

    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Steer only last token
        a = hidden[0, -1, :]
        R = a.norm()
        a_unit = a / (R + 1e-10)
        proj = (a_unit * direction).sum()
        v_perp = direction - proj * a_unit
        v_perp = v_perp / (v_perp.norm() + 1e-10)
        hidden[0, -1, :] = R * (cos_theta * a_unit + sin_theta * v_perp)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    layers = _get_blocks(model)
    handle = layers[steer_layer].register_forward_hook(steering_hook)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)

    handle.remove()
    return outputs.logits[0, -1, :].float().cpu()


def get_unsteered_logits(model, tokenizer, prompt, device):
    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
    return outputs.logits[0, -1, :].float().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--direction_pkl', required=True)
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--angle', type=float, default=30)
    parser.add_argument('--n_prompts', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--top_k', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(args.direction_pkl), 'token_boost')
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Loading direction from {args.direction_pkl}")
    with open(args.direction_pkl, 'rb') as f:
        data = pickle.load(f)
    direction = data['direction'].float()
    direction = direction / direction.norm()

    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    token_ids_list = load_fineweb_fixed_length(
        args.n_prompts, tokenizer, seq_len=args.seq_len, seed=args.seed
    )

    output_lines = []

    for i in range(args.n_prompts):
        prompt_text = tokenizer.decode(token_ids_list[i], skip_special_tokens=True)

        logits_orig = get_unsteered_logits(model, tokenizer, prompt_text, device)
        logits_steer = get_steered_logits(model, tokenizer, prompt_text, direction,
                                          args.angle, args.layer, device)

        probs_orig = F.softmax(logits_orig, dim=-1)
        probs_steer = F.softmax(logits_steer, dim=-1)
        prob_diff = probs_steer - probs_orig

        # Top gained
        top_gained_idx = prob_diff.argsort(descending=True)[:args.top_k]
        # Top lost
        top_lost_idx = prob_diff.argsort()[:args.top_k]

        header = f"\n--- Prompt {i+1}: \"{prompt_text}\" ---"
        print(header)
        output_lines.append(header)

        # Show original top-5
        orig_top5 = probs_orig.argsort(descending=True)[:5]
        line = f"  Original top-5: " + ", ".join(
            f"{tokenizer.decode([t])!r} ({probs_orig[t]:.3f})" for t in orig_top5
        )
        print(line)
        output_lines.append(line)

        steered_top5 = probs_steer.argsort(descending=True)[:5]
        line = f"  Steered top-5:  " + ", ".join(
            f"{tokenizer.decode([t])!r} ({probs_steer[t]:.3f})" for t in steered_top5
        )
        print(line)
        output_lines.append(line)

        kl = (probs_orig * (probs_orig.log() - probs_steer.log())).sum().item()
        line = f"  KL(orig || steered) = {kl:.4f}"
        print(line)
        output_lines.append(line)

        line = f"  Top gained:"
        print(line)
        output_lines.append(line)
        for j, idx in enumerate(top_gained_idx):
            tok = tokenizer.decode([idx])
            line = f"    {tok!r:20s}  {probs_orig[idx]:.4f} -> {probs_steer[idx]:.4f}  (diff +{prob_diff[idx]:.4f})"
            print(line)
            output_lines.append(line)

        line = f"  Top lost:"
        print(line)
        output_lines.append(line)
        for j, idx in enumerate(top_lost_idx):
            tok = tokenizer.decode([idx])
            line = f"    {tok!r:20s}  {probs_orig[idx]:.4f} -> {probs_steer[idx]:.4f}  (diff {prob_diff[idx]:.4f})"
            print(line)
            output_lines.append(line)

    txt_path = os.path.join(args.results_dir, f'token_boost_{args.angle}deg.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(output_lines) + '\n')
    print(f"\nSaved: {txt_path}")


if __name__ == '__main__':
    main()
