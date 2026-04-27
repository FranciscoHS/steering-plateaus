#!/usr/bin/env python3
"""Composition generations for Formal × Language pairs at L2."""
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
OUT_DIR = 'results/pair_exploration_L2_v2'
PAIRS = [
    ('Formal', 'Portuguese'),
    ('Formal', 'Italian'),
    ('Formal', 'Spanish'),
    ('Formal', 'German'),
    ('Formal', 'French'),
    ('Formal', 'Japanese'),
]
ANGLE = 20.0
N_GENS = 5
PROMPT = "They elected a new chairman"

gcl.LAYER = 2
gcl.PROMPT = PROMPT


def main():
    print("Loading Gemma-2-9B...", flush=True)
    model, tok, dev = load_model('gemma-2-9b', dtype=torch.bfloat16)
    with open(DIRS_PATH, 'rb') as f:
        D = pickle.load(f)['directions']
    norm = {k: F.normalize(v.float(), dim=0) for k, v in D.items()}

    alpha = float(np.deg2rad(ANGLE))

    for a, b in PAIRS:
        out_path = os.path.join(OUT_DIR, f'generations_{a}_{b}.json')
        if os.path.exists(out_path):
            print(f"skip {a}×{b}"); continue
        print(f"\n=== {a} × {b} ({ANGLE}° each) ===", flush=True)
        da, db = norm[a], norm[b]
        conds = [('baseline', None, None, 0.0, 0.0),
                 ('a_only', da, None, alpha, 0.0),
                 ('b_only', None, db, 0.0, alpha),
                 ('a_and_b', da, db, alpha, alpha)]
        result = {'pair': [a, b], 'prompt': PROMPT, 'layer': 2,
                  'angle_deg': ANGLE, 'conditions': {}}
        for cname, dp, dg, ap, ag in conds:
            hook = (make_steering_hook(dp, dg, ap, ag)
                    if dp is not None or dg is not None else None)
            outs = generate(model, tok, dev, hook, N_GENS,
                            hash((a, b, cname)) % 10000)
            result['conditions'][cname] = {
                'angle_a_deg': float(np.rad2deg(ap)),
                'angle_b_deg': float(np.rad2deg(ag)),
                'samples': outs,
            }
            for k, o in enumerate(outs):
                print(f"  [{cname} {k}] {o[:120]}", flush=True)
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        # text dump too
        txt = out_path.replace('.json', '.txt')
        with open(txt, 'w') as f:
            f.write(f"Pair: {a} × {b}\nPrompt: {PROMPT}\n\n")
            for cname, info in result['conditions'].items():
                f.write('=' * 70 + '\n')
                f.write(f"[{cname}]  angle_{a}={info['angle_a_deg']}°  "
                        f"angle_{b}={info['angle_b_deg']}°\n")
                f.write('=' * 70 + '\n')
                for k, o in enumerate(info['samples']):
                    f.write(f"\n-- sample {k} --\n{PROMPT}{o}\n")
                f.write('\n')
        print(f"saved {out_path}", flush=True)


if __name__ == '__main__':
    main()
