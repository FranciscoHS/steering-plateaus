#!/usr/bin/env python3
"""Formal × Portuguese composition at 30° each, on informal English prompts."""
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
OUT_PATH = 'results/pair_exploration_L2_v2/generations_formal_pt_30deg.json'
TXT_PATH = 'results/pair_exploration_L2_v2/generations_formal_pt_30deg.txt'

PROMPTS = [
    "lol you won't believe what happened today",
    "she's literally the worst, omg",
]

ANGLE = 30.0
N_GENS = 5
gcl.LAYER = 2


def main():
    print("Loading Gemma-2-9B...", flush=True)
    model, tok, dev = load_model('gemma-2-9b', dtype=torch.bfloat16)
    with open(DIRS_PATH, 'rb') as f:
        D = pickle.load(f)['directions']
    d_formal = F.normalize(D['Formal'].float(), dim=0)
    d_pt = F.normalize(D['Portuguese'].float(), dim=0)

    alpha = float(np.deg2rad(ANGLE))

    results = {}
    for prompt in PROMPTS:
        gcl.PROMPT = prompt
        results[prompt] = {}
        print(f"\n=== prompt: {prompt!r} (angle={ANGLE}°) ===", flush=True)
        conds = [
            ('baseline',      None,    None,  0.0,    0.0),
            ('formal_only',   d_formal, None, alpha, 0.0),
            ('pt_only',       None,    d_pt,  0.0,   alpha),
            ('formal_and_pt', d_formal, d_pt, alpha, alpha),
        ]
        for cname, dp, dg, ap, ag in conds:
            hook = (make_steering_hook(dp, dg, ap, ag)
                    if dp is not None or dg is not None else None)
            outs = generate(model, tok, dev, hook, N_GENS,
                            hash((prompt, cname)) % 10000)
            results[prompt][cname] = {
                'angle_formal_deg': float(np.rad2deg(ap)),
                'angle_pt_deg': float(np.rad2deg(ag)),
                'samples': outs,
            }
            for k, o in enumerate(outs):
                print(f"  [{cname} {k}] {o[:140]}", flush=True)

    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(TXT_PATH, 'w') as f:
        for prompt, by_cond in results.items():
            f.write('=' * 70 + '\n')
            f.write(f'PROMPT: {prompt!r}  (angle each = {ANGLE}°)\n')
            f.write('=' * 70 + '\n')
            for cname, info in by_cond.items():
                f.write(f'\n-- [{cname}]  '
                        f'formal={info["angle_formal_deg"]}°  '
                        f'pt={info["angle_pt_deg"]}° --\n')
                for k, o in enumerate(info['samples']):
                    f.write(f'sample {k}: {prompt}{o}\n')
            f.write('\n')
    print(f"\nSaved {OUT_PATH} and {TXT_PATH}", flush=True)


if __name__ == '__main__':
    main()
