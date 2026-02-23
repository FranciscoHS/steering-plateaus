#!/usr/bin/env python3
"""
Compute 5-language contrastive features including English: {en, es, zh, ru, ar}.

Tests hypothesis that including English in the set produces a cleaner Spanish
feature, since Spanish no longer needs to carry the "European-ness" signal
to distinguish itself from {zh, ru, ar}.

Extracts English activations from the same prompt sets used in the 10-language
run, then computes 5-language all-pairwise features.
"""

import sys, os, pickle
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import load_model, _get_blocks
from data.multilingual_prompts import PROMPT_SETS

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='qwen3-4b')
parser.add_argument('--layer', type=int, default=14)
parser.add_argument('--bfloat16', action='store_true')
args = parser.parse_args()

SUBSET = ['en', 'es', 'zh', 'ru', 'ar']

results_dir = f'results/allpairs_contrastive/{args.model}'

# Load model
print(f"Loading model: {args.model}")
load_dtype = torch.bfloat16 if args.bfloat16 else None
model, tokenizer, device = load_model(args.model, dtype=load_dtype)

# Extract activations for all languages in subset
def get_acts(prompts, layer):
    acts = []
    for p in prompts:
        toks = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**toks, output_hidden_states=True)
        acts.append(out.hidden_states[layer + 1][:, -1, :].squeeze(0))
    return torch.stack(acts).cpu()

print(f"\nExtracting activations at layer {args.layer}...")
acts = {}
for lang in SUBSET:
    prompts = [ps[lang] for ps in PROMPT_SETS]
    acts[lang] = get_acts(prompts, args.layer)
    print(f"  {lang}: {acts[lang].shape}")

# Compute all pairwise mean differences
print("\nComputing pairwise differences...")
k = len(SUBSET)
pairwise_means = {}
for i, li in enumerate(SUBSET):
    for j, lj in enumerate(SUBSET):
        if i < j:
            diff = (acts[li] - acts[lj]).mean(dim=0)
            pairwise_means[(li, lj)] = diff
            print(f"  {li}-{lj}: ||v̄|| = {diff.norm().item():.2f}")

# All-pairwise estimator: f̂_i = (1/k) Σ_{j≠i} v̄_{ij}
print("\nComputing features...")
features_raw = {}
features = {}
for lang in SUBSET:
    f_hat = torch.zeros_like(acts[lang][0])
    for other in SUBSET:
        if other == lang:
            continue
        if (lang, other) in pairwise_means:
            f_hat += pairwise_means[(lang, other)]
        else:
            f_hat -= pairwise_means[(other, lang)]
    f_hat = f_hat / k
    features_raw[lang] = f_hat
    features[lang] = F.normalize(f_hat.unsqueeze(0), dim=-1).squeeze(0)
    print(f"  {lang}: ||f̂|| = {f_hat.norm().item():.4f}")

# Zero-sum check
total = sum(features_raw.values())
print(f"\n  Σ f̂_i norm (should be ~0): {total.norm().item():.6f}")

# Pairwise cosines
print("\n  Pairwise cosines:")
for i in range(len(SUBSET)):
    for j in range(i + 1, len(SUBSET)):
        cos = (features[SUBSET[i]] @ features[SUBSET[j]]).item()
        print(f"    {SUBSET[i]} vs {SUBSET[j]}: {cos:+.4f}")

# Compare with 5-lang (no English) features
print("\n  Comparing with 5-lang (no English) features:")
with open(os.path.join(results_dir, '5lang_contrastive.pkl'), 'rb') as f:
    data_5 = pickle.load(f)
features_5_no_en = data_5['features_5']
for lang in ['es', 'zh', 'ru', 'ar']:
    f_new = features[lang].float()
    f_old = features_5_no_en[lang].float()
    print(f"    {lang}: cos(new, old) = {(f_new @ f_old).item():.4f}")

# Save
save_data = {
    'features': {k: v.cpu() for k, v in features.items()},
    'features_raw': {k: v.cpu() for k, v in features_raw.items()},
    'pairwise_means': {k: v.cpu() for k, v in pairwise_means.items()},
    'subset': SUBSET,
    'layer': args.layer,
    'model': args.model,
    'note': 'English included to test whether it improves Spanish feature',
}
out_path = os.path.join(results_dir, '5lang_with_english.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(save_data, f)
print(f"\nSaved: {out_path}")
print("Done!")
