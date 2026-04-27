#!/usr/bin/env python3
"""Formal × Portuguese composition at 20°+20° and 25°+25°, on informal prompts."""
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
OUT_PATH = 'results/pair_exploration_L2_v2/generations_formal_pt_2025.json'
TXT_PATH = 'results/pair_exploration_L2_v2/generations_formal_pt_2025.txt'

PROMPTS = [
    "lol you won't believe what happened today",
    "she's literally the worst, omg",
]
ANGLE_PAIRS = [(20.0, 20.0), (25.0, 25.0)]
N_GENS = 5
gcl.LAYER = 2


def main():
    print("Loading Gemma-2-9B...", flush=True)
    model, tok, dev = load_model('gemma-2-9b', dtype=torch.bfloat16)
    with open(DIRS_PATH, 'rb') as f:
        D = pickle.load(f)['directions']
    d_formal = F.normalize(D['Formal'].float(), dim=0)
    d_pt = F.normalize(D['Portuguese'].float(), dim=0)

    results = {}
    for prompt in PROMPTS:
        gcl.PROMPT = prompt
        results[prompt] = {}
        print(f"\n=== prompt: {prompt!r} ===", flush=True)
        for af, ap in ANGLE_PAIRS:
            tag = f'formal{af:.0f}_pt{ap:.0f}'
            af_r = float(np.deg2rad(af))
            ap_r = float(np.deg2rad(ap))
            hook = make_steering_hook(d_formal, d_pt, af_r, ap_r)
            outs = generate(model, tok, dev, hook, N_GENS,
                            hash((prompt, tag)) % 10000)
            results[prompt][tag] = {
                'angle_formal_deg': af, 'angle_pt_deg': ap,
                'samples': outs,
            }
            for k, o in enumerate(outs):
                print(f"  [{tag} {k}] {o[:140]}", flush=True)

    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(TXT_PATH, 'w') as f:
        for prompt, by_cond in results.items():
            f.write('=' * 70 + '\n')
            f.write(f'PROMPT: {prompt!r}\n')
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
