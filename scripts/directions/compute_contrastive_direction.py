#!/usr/bin/env python3
"""Compute a contrastive steering vector from bilingual prompt pairs.

Takes the mean difference of activations between two languages to produce
a single steering direction. Saves in the pickle format expected by
sweep_collective_threshold.py.

Usage:
    python scripts/directions/compute_contrastive_direction.py \
        --model qwen3-1.7b --layer 7 \
        --outdir results/contrastive_directions/qwen3-1.7b_L7_mandarin
"""
import argparse, os, sys, pickle
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F

from src.model import load_model
from data.bilingual_prompts_hq import PROMPT_PAIRS


def get_acts(model, tokenizer, device, prompts, layer):
    """Extract final-token activations at given layer for a list of prompts."""
    acts = []
    for p in prompts:
        toks = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**toks, output_hidden_states=True)
        acts.append(out.hidden_states[layer + 1][:, -1, :].squeeze(0))
    return torch.stack(acts).cpu()


def main():
    parser = argparse.ArgumentParser(description='Compute contrastive steering vector')
    parser.add_argument('--model', required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    args = parser.parse_args()

    # Load model
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)

    # Extract activations for both languages
    en_prompts = [p[0] for p in PROMPT_PAIRS]
    zh_prompts = [p[1] for p in PROMPT_PAIRS]
    print(f"Extracting activations for {len(PROMPT_PAIRS)} prompt pairs at layer {args.layer}...")

    en_acts = get_acts(model, tokenizer, device, en_prompts, args.layer)
    zh_acts = get_acts(model, tokenizer, device, zh_prompts, args.layer)
    print(f"  English acts: {en_acts.shape}, Mandarin acts: {zh_acts.shape}")

    # Contrastive direction: mean(Mandarin) - mean(English)
    diff = (zh_acts - en_acts).mean(dim=0)  # [d_model]
    diff_norm = diff.norm().item()
    direction = F.normalize(diff, dim=0)
    print(f"  ||mean diff|| = {diff_norm:.2f}")

    # Save
    os.makedirs(args.outdir, exist_ok=True)
    save_data = {
        'directions': direction.unsqueeze(0),  # [1, d_model]
        'diff_norm': diff_norm,
        'n_pairs': len(PROMPT_PAIRS),
        'layer': args.layer,
        'model': args.model,
    }
    pkl_path = os.path.join(args.outdir, 'contrastive_direction.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved: {pkl_path}")


if __name__ == '__main__':
    main()
