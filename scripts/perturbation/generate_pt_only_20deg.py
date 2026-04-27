#!/usr/bin/env python3
"""PT-only at 20° on 'she's literally the worst, omg' — fill the missing row."""
import os, sys, pickle, json
sys.path.insert(0, '.')
import numpy as np
import torch, torch.nn.functional as F

from src.model import load_model
from scripts.perturbation.generate_composition_L2 import (
    make_steering_hook, generate)
import scripts.perturbation.generate_composition_L2 as gcl

PROMPT = "she's literally the worst, omg"
ANGLE = 20.0
N_GENS = 5
OUT = 'results/pair_exploration_L2_v2/generations_pt_only_20deg_shes.json'
TXT = 'results/pair_exploration_L2_v2/generations_pt_only_20deg_shes.txt'

gcl.LAYER = 2
gcl.PROMPT = PROMPT


def main():
    print("Loading...", flush=True)
    model, tok, dev = load_model('gemma-2-9b', dtype=torch.bfloat16)
    with open('results/pair_exploration_L2_v2/directions_L2.pkl', 'rb') as f:
        D = pickle.load(f)['directions']
    d_pt = F.normalize(D['Portuguese'].float(), dim=0)
    alpha = float(np.deg2rad(ANGLE))
    hook = make_steering_hook(None, d_pt, 0.0, alpha)
    outs = generate(model, tok, dev, hook, N_GENS, hash(('pt_only_20', PROMPT)) % 10000)
    with open(OUT, 'w') as f:
        json.dump({'prompt': PROMPT, 'angle_deg': ANGLE, 'samples': outs}, f,
                  indent=2, ensure_ascii=False)
    with open(TXT, 'w') as f:
        f.write(f'PROMPT: {PROMPT!r}  PT-only @ {ANGLE}°\n\n')
        for k, o in enumerate(outs):
            f.write(f'sample {k}: {PROMPT}{o}\n\n')
    for k, o in enumerate(outs):
        print(f'  [{k}] {o[:140]}')
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
