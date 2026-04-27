#!/usr/bin/env python3
"""Test Formal direction on informal prompts to check if it actually formalises."""
import os, sys, pickle, json
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model
from scripts.perturbation.generate_composition_L2 import (
    make_steering_hook, generate)
import scripts.perturbation.generate_composition_L2 as gcl

DIRS_PATH = 'results/pair_exploration_L2_v2/directions_L2.pkl'
OUT_PATH = 'results/pair_exploration_L2_v2/generations_formal_informal_prompts.json'
TXT_PATH = 'results/pair_exploration_L2_v2/generations_formal_informal_prompts.txt'

PROMPTS = [
    "yo so my buddy was like",
    "lol you won't believe what happened today",
    "dude i'm so tired bro",
    "ngl that was kinda lit, like",
    "she's literally the worst, omg",
]

ANGLES_DEG = [0.0, 20.0, 30.0]   # baseline + two steering levels
N_GENS = 3
gcl.LAYER = 2


def main():
    print("Loading Gemma-2-9B...", flush=True)
    model, tok, dev = load_model('gemma-2-9b', dtype=torch.bfloat16)
    with open(DIRS_PATH, 'rb') as f:
        D = pickle.load(f)['directions']
    d_formal = F.normalize(D['Formal'].float(), dim=0)

    results = {}
    for prompt in PROMPTS:
        gcl.PROMPT = prompt
        results[prompt] = {}
        print(f"\n=== prompt: {prompt!r} ===", flush=True)
        for ang in ANGLES_DEG:
            alpha = float(np.deg2rad(ang))
            hook = (make_steering_hook(d_formal, None, alpha, 0.0)
                    if ang > 0 else None)
            outs = generate(model, tok, dev, hook, N_GENS,
                            hash((prompt, ang)) % 10000)
            results[prompt][f'{ang:.0f}'] = outs
            for k, o in enumerate(outs):
                print(f"  [{ang:>4.0f}° s{k}] {o[:140]}", flush=True)

    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(TXT_PATH, 'w') as f:
        for prompt, by_ang in results.items():
            f.write('=' * 70 + '\n')
            f.write(f'PROMPT: {prompt!r}\n')
            f.write('=' * 70 + '\n')
            for ang, samples in by_ang.items():
                f.write(f'\n-- Formal angle = {ang}° --\n')
                for k, o in enumerate(samples):
                    f.write(f'sample {k}: {prompt}{o}\n')
            f.write('\n')
    print(f"\nSaved {OUT_PATH} and {TXT_PATH}", flush=True)


if __name__ == '__main__':
    main()
