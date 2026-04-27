#!/usr/bin/env python3
"""Run composition generations on the top-ranked pairs from pair_exploration_L2.

Each pair generates 5 samples × 4 conditions (baseline / +A / +B / +A+B).
Starting prompt "They elected a new chairman", same as the PT × Gender table.
Output: results/pair_exploration_L2/generations_<pair>.{json,txt}
"""
import os, sys, pickle, json
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks
from scripts.perturbation.generate_composition_L2 import (
    make_steering_hook, generate)

PAIRS = [
    ('Chinese',    'Gender',  'd_Chinese',    'd_Gender',    20.0, 20.0),
    ('Arabic',     'Gender',  'd_Arabic',     'd_Gender',    20.0, 20.0),
]
# Also Dutch × Gender (worst threshold-stability) for contrast
PAIRS_EXTRA = [
    ('Dutch',      'Gender',  'd_Dutch',      'd_Gender',    20.0, 20.0),
]

PROMPT = "They elected a new chairman"
LAYER = 2
N_GENS = 5
DIR_PKL = 'results/pair_exploration_L2/directions_L2.pkl'
OUT_DIR = 'results/pair_exploration_L2'


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--include_dutch', action='store_true')
    args = ap.parse_args()

    pairs = list(PAIRS)
    if args.include_dutch:
        pairs += PAIRS_EXTRA

    print("Loading Gemma-2-9B...", flush=True)
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)

    with open(DIR_PKL, 'rb') as f:
        stored = pickle.load(f)
    direction_tensor = {n: F.normalize(d.float(), dim=0)
                        for n, d in stored['directions'].items()}

    from scripts.perturbation import generate_composition_L2 as gcl
    gcl.LAYER = LAYER
    gcl.PROMPT = PROMPT

    for name_a, name_b, _, _, angle_a, angle_b in pairs:
        print(f"\n=== {name_a} × {name_b} "
              f"(angles {angle_a}°, {angle_b}°) ===", flush=True)
        d_a = direction_tensor[name_a]
        d_b = direction_tensor[name_b]
        alpha_a = float(np.deg2rad(angle_a))
        alpha_b = float(np.deg2rad(angle_b))

        conditions = [
            ('baseline',    None,   None,        0.0, 0.0),
            ('a_only',      d_a,    None,        alpha_a, 0.0),
            ('b_only',      None,   d_b,         0.0, alpha_b),
            ('a_and_b',     d_a,    d_b,         alpha_a, alpha_b),
        ]

        all_results = {}
        for cname, dp, dg, a_pt, a_gen in conditions:
            hook_fn = (make_steering_hook(dp, dg, a_pt, a_gen)
                       if dp is not None or dg is not None else None)
            outs = generate(model, tokenizer, device, hook_fn,
                            N_GENS, hash((name_a, name_b, cname)) % 1000)
            all_results[cname] = {
                'angle_a_deg': float(np.rad2deg(a_pt)),
                'angle_b_deg': float(np.rad2deg(a_gen)),
                'samples': outs,
            }
            for k, o in enumerate(outs):
                print(f"  [{cname} {k}] {o[:100]}...", flush=True)

        out_path = os.path.join(OUT_DIR, f'generations_{name_a}_{name_b}.json')
        with open(out_path, 'w') as f:
            json.dump({
                'pair': [name_a, name_b],
                'prompt': PROMPT,
                'layer': LAYER,
                'conditions': all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved {out_path}", flush=True)

        # Pretty text dump too
        txt_path = os.path.join(OUT_DIR, f'generations_{name_a}_{name_b}.txt')
        with open(txt_path, 'w') as f:
            f.write(f"Pair: {name_a} × {name_b}\nPrompt: {PROMPT}\n\n")
            for cname, info in all_results.items():
                f.write('=' * 70 + '\n')
                f.write(f"[{cname}]  "
                        f"angle_{name_a}={info['angle_a_deg']}°  "
                        f"angle_{name_b}={info['angle_b_deg']}°\n")
                f.write('=' * 70 + '\n')
                for k, o in enumerate(info['samples']):
                    f.write(f"\n-- sample {k} --\n{PROMPT}{o}\n")
                f.write('\n')
        print(f"Saved {txt_path}", flush=True)


if __name__ == '__main__':
    main()
