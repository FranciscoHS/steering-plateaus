#!/usr/bin/env python3
"""Extract all-pairwise contrastive (diff-of-means) directions at a chosen layer.

Reproduces Phases 1-3 of run_allpairs_contrastive.py / run_programming_contrastive.py
(least-squares estimator f̂_i = (1/k) Σ_{j≠i} v̄_ij over matched prompt sets), without
the perturbation sweeps — so the same directions can be re-extracted at a different
layer cheaply.

Saves the directions.pkl format the alignment/sweep tooling expects:
    {'features': {name: unit vec}, 'features_raw': {...}, 'layer': L, 'model': name}

Usage:
    python scripts/directions/extract_allpairs_directions.py \\
        --model qwen3-1.7b --layer 10 --prompt_set multilingual \\
        --out results/allpairs_contrastive/qwen3-1.7b/directions_L10.pkl
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F

from src.model import load_model


def get_acts(model, tokenizer, device, prompts, layer):
    """Final-token activations at resid_post of `layer` (hidden_states[layer+1])."""
    acts = []
    for p in prompts:
        toks = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**toks, output_hidden_states=True)
        acts.append(out.hidden_states[layer + 1][:, -1, :].squeeze(0).float().cpu())
    return torch.stack(acts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--prompt_set', choices=['multilingual', 'programming'],
                    required=True)
    ap.add_argument('--bfloat16', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if args.prompt_set == 'multilingual':
        from data.multilingual_prompts import PROMPT_SETS, LANGUAGES
    else:
        from data.programming_prompts import PROMPT_SETS, LANGUAGES
    langs = list(LANGUAGES)
    k = len(langs)
    print(f'{args.prompt_set}: {k} languages, {len(PROMPT_SETS)} matched prompt sets')

    model, tokenizer, device = load_model(args.model)

    print(f'Extracting activations at layer {args.layer}...')
    acts = {}
    for lang in langs:
        prompts = [ps[lang] for ps in PROMPT_SETS]
        acts[lang] = get_acts(model, tokenizer, device, prompts, args.layer)
        print(f'  {lang}: {tuple(acts[lang].shape)}', flush=True)

    # pairwise mean differences v̄_ij, then f̂_i = (1/k) Σ_{j≠i} v̄_ij
    pairwise = {}
    for i, li in enumerate(langs):
        for j, lj in enumerate(langs):
            if i < j:
                pairwise[(li, lj)] = (acts[li] - acts[lj]).mean(dim=0)

    features_raw, features = {}, {}
    for lang in langs:
        f_hat = torch.zeros_like(acts[lang][0])
        for other in langs:
            if other == lang:
                continue
            if (lang, other) in pairwise:
                f_hat += pairwise[(lang, other)]
            else:
                f_hat -= pairwise[(other, lang)]
        f_hat = f_hat / k
        features_raw[lang] = f_hat
        features[lang] = F.normalize(f_hat, dim=-1)
        print(f'  f̂_{lang}: ||raw|| = {f_hat.norm().item():.4f}')

    total = sum(features_raw.values())
    print(f'  ||Σ f̂_i|| (should be ~0): {total.norm().item():.6f}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump({'features': features, 'features_raw': features_raw,
                     'languages': langs, 'layer': args.layer, 'model': args.model,
                     'prompt_set': args.prompt_set,
                     'n_prompts': len(PROMPT_SETS)}, f)
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
