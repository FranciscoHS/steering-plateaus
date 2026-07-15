#!/usr/bin/env python3
"""
Extract contrastive (diff-of-means) directions from matched prompt pairs.

Many concepts in data/ are stored as `PROMPT_PAIRS = [(A, B), ...]` where A and B
differ only along one concept (e.g. gender: male vs female; tense: past vs future).
The contrastive direction at layer L is the difference of the last-token mean
activations:

    d_concept = normalize( mean_b(act_B) - mean_a(act_A) )           (B - A)

These are concepts a small model like gpt2-xl actually represents well (unlike, say,
distinguishing programming languages), so they make a meaningful structured baseline
for perturbation sweeps.

Output pkl format matches what sweep_diff_of_activations.py / run_response_direction_sweep.py
expect:
    {'features': {Concept: vec[d_model]}, 'layer': L, 'model': name, 'sign': 'B_minus_A',
     'n_pairs': {Concept: n}}

Usage:
    python scripts/directions/extract_paired_directions.py --model gpt2-xl --layer 2
"""

import argparse
import importlib
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F

from src.model import load_model

# concept name -> (module, PROMPT_PAIRS attribute). Direction sign is B - A.
DEFAULT_CONCEPTS = {
    'Gender':    'data.gender_prompts',      # male  -> female
    'Tense':     'data.tense_prompts',       # past  -> future
    'TensePP':   'data.tense_present_prompts',
    'Number':    'data.number_prompts',      # singular -> plural
    'Sentiment': 'data.sentiment_prompts',   # negative -> positive
    'Age':       'data.age_prompts',
    'Person':    'data.person_prompts',
    'Status':    'data.status_prompts',
    'Era':       'data.era_prompts',
    'Wealth':    'data.wealth_prompts',
    'Certainty': 'data.certainty_prompts',
    'Honesty':   'data.honesty_prompts',
    'Health':    'data.health_prompts',
}


def last_token_acts(model, tokenizer, device, prompts, layer):
    """Layer-L last-token residual stream for a list of prompts -> [n, d_model]."""
    acts = []
    for p in prompts:
        tok = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**tok, output_hidden_states=True)
        acts.append(out.hidden_states[layer + 1][0, -1].float().cpu())
    return torch.stack(acts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='gpt2-xl')
    ap.add_argument('--layer', type=int, default=2)
    ap.add_argument('--concepts', type=str, default=None,
                    help='comma-separated subset of: ' + ','.join(DEFAULT_CONCEPTS))
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    concepts = (args.concepts.split(',') if args.concepts else list(DEFAULT_CONCEPTS))
    if args.out is None:
        args.out = f'results/paired_directions/{args.model}_L{args.layer}/directions.pkl'
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"Loading {args.model}...")
    model, tokenizer, device = load_model(args.model)

    features, n_pairs = {}, {}
    for name in concepts:
        mod = importlib.import_module(DEFAULT_CONCEPTS[name])
        pairs = mod.PROMPT_PAIRS
        a_prompts = [a for a, b in pairs]
        b_prompts = [b for a, b in pairs]
        a_acts = last_token_acts(model, tokenizer, device, a_prompts, args.layer)
        b_acts = last_token_acts(model, tokenizer, device, b_prompts, args.layer)
        diff = b_acts.mean(0) - a_acts.mean(0)
        # diagnostic: how consistent is the per-pair difference vs the mean diff?
        per_pair = F.normalize(b_acts - a_acts, dim=-1)
        align = (per_pair @ F.normalize(diff, dim=0)).mean().item()
        features[name] = F.normalize(diff, dim=0)
        n_pairs[name] = len(pairs)
        print(f"  {name:10s} n={len(pairs):2d}  ||diff||={diff.norm():.2f}  "
              f"mean per-pair alignment to mean-diff = {align:.3f}")

    out = {'features': features, 'layer': args.layer, 'model': args.model,
           'sign': 'B_minus_A', 'n_pairs': n_pairs}
    with open(args.out, 'wb') as f:
        pickle.dump(out, f)
    print(f"\nSaved {len(features)} directions -> {args.out}")


if __name__ == '__main__':
    main()
