#!/usr/bin/env python3
"""Sweep python_only generations across multiple angles at L2.

Same prompt, same hook, same sampling as generate_gender_python_L2.py — but
only the Python direction is active (Gender = 0°), and we loop over a list
of angles.

Output: results/new_directions_L2/generations_Python_only_<deg>.{json,txt}
"""
import os, sys, pickle, json, argparse
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks
from scripts.perturbation.generate_composition_L2 import make_steering_hook
from scripts.perturbation.generate_gender_python_L2 import generate

DEFAULT_PROMPT = "They elected a new chairman"
LAYER = 2
N_GENS_PER_ANGLE = 5
DIR_PKL = 'results/new_directions_L2/directions_L2.pkl'
OUT_DIR = 'results/new_directions_L2'
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--angles', type=float, nargs='+',
                    default=[35.0, 40.0, 60.0])
    ap.add_argument('--n_gens', type=int, default=N_GENS_PER_ANGLE)
    ap.add_argument('--prompt', type=str, default=DEFAULT_PROMPT)
    ap.add_argument('--tag', type=str, default='',
                    help='filename suffix to differentiate prompt runs')
    args = ap.parse_args()
    PROMPT = args.prompt
    tag = ('_' + args.tag) if args.tag else ''

    print('Loading Gemma-2-9B...', flush=True)
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)

    with open(DIR_PKL, 'rb') as f:
        cached = pickle.load(f)['directions']
    d_python = F.normalize(cached['Python'].float(), dim=0)

    # patch the global LAYER used by generate()
    import scripts.perturbation.generate_gender_python_L2 as g
    g.LAYER = LAYER
    g.PROMPT = PROMPT

    for ang in args.angles:
        ar = float(np.deg2rad(ang))
        print(f'\n[python_only] angle={ang:.1f}°', flush=True)
        hook = make_steering_hook(None, d_python, 0.0, ar)
        outs = generate(model, tokenizer, device, hook, args.n_gens,
                        SEED + hash(('py', ang)) % 1000)
        for k, o in enumerate(outs):
            print(f'  sample {k}: {o[:100]}...', flush=True)
        out_json = os.path.join(OUT_DIR, f'generations_Python_only_{int(ang)}{tag}.json')
        out_txt = os.path.join(OUT_DIR, f'generations_Python_only_{int(ang)}{tag}.txt')
        with open(out_json, 'w') as f:
            json.dump({'prompt': PROMPT, 'layer': LAYER,
                       'angle_python_deg': ang,
                       'samples': outs}, f, indent=2, ensure_ascii=False)
        # also save raw text
        with open(out_txt, 'w') as f:
            f.write(f'Prompt: {PROMPT!r}\nLayer: {LAYER}\n'
                    f'angle_python={ang:.1f}°\n\n')
            for k, o in enumerate(outs):
                f.write(f'-- sample {k} --\n{PROMPT}{o}\n\n')
        print(f'  saved {out_json}, {out_txt}', flush=True)


if __name__ == '__main__':
    main()
