#!/usr/bin/env python3
"""Steering generations with a randomly sampled unit direction in residual
stream. Same hook + angles as steer_decomp_generate.py, for baseline."""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from src.model import load_model, _get_blocks

from scripts.superellipse.steer_decomp_generate import generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_pkl', required=True,
                    help='decomposition results.pkl (used only to read model '
                         'name, layer, and hidden dim)')
    ap.add_argument('--prompts', nargs='+', default=['I love to'])
    ap.add_argument('--angles', type=float, nargs='+',
                    default=[15.0, 30.0, 45.0])
    ap.add_argument('--max_new_tokens', type=int, default=100)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--gen_seed', type=int, default=0,
                    help='generation sampling seed (same as decomp script)')
    ap.add_argument('--random_seeds', type=int, nargs='+', default=[0, 1, 2],
                    help='one random direction per seed')
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--results_dir', default=None)
    args = ap.parse_args()

    with open(args.results_pkl, 'rb') as f:
        res = pickle.load(f)
    hidden = res['features'].shape[-1]
    model_name = res['args']['model']
    layer = res['args']['layer']

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(args.results_pkl),
                                        'steering_random')
    os.makedirs(args.results_dir, exist_ok=True)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {model_name} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(model_name,
                                          dtype=dtype_map[args.dtype])
    print(f'Layer: {layer}, hidden: {hidden}', flush=True)

    # Sample random unit directions, one per seed.
    rand_dirs = {}
    for s in args.random_seeds:
        g = torch.Generator(device='cpu').manual_seed(s)
        v = torch.randn(hidden, generator=g)
        v = v / v.norm()
        rand_dirs[s] = v.float()

    runs = []
    for pi, prompt in enumerate(args.prompts):
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        print(f'\n[#{pi}] {prompt!r}', flush=True)

        base = generate(model, tokenizer, input_ids, layer, device,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, gen_seed=args.gen_seed)
        runs.append({'prompt_idx': pi, 'prompt': prompt, 'feature': 'unsteered',
                     'random_seed': None, 'angle_deg': 0.0, 'generated': base})
        print(f'  unsteered: {base!r}', flush=True)

        for s in args.random_seeds:
            d = rand_dirs[s]
            label = f'rand{s}'
            for ang in args.angles:
                g = generate(model, tokenizer, input_ids, layer, device,
                             direction=d, angle_rad=np.deg2rad(ang),
                             max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature,
                             gen_seed=args.gen_seed)
                runs.append({'prompt_idx': pi, 'prompt': prompt,
                             'feature': label, 'random_seed': s,
                             'angle_deg': ang, 'generated': g})
                print(f'  {label} @ {ang:.0f}deg: {g!r}', flush=True)

    out_json = os.path.join(args.results_dir,
                            'steering_random_generations.json')
    with open(out_json, 'w') as f:
        json.dump({'args': vars(args), 'model': model_name, 'layer': layer,
                   'runs': runs}, f, indent=2)

    out_txt = os.path.join(args.results_dir,
                           'steering_random_generations.txt')
    with open(out_txt, 'w') as f:
        f.write(f'Steering with random unit directions ({model_name}, '
                f'layer {layer})\n')
        f.write(f'random_seeds: {args.random_seeds}\n')
        f.write(f'angles: {args.angles} deg, '
                f'max_new_tokens={args.max_new_tokens}, '
                f'temperature={args.temperature}, gen_seed={args.gen_seed}\n\n')
        cur_p = None
        for r in runs:
            if r['prompt_idx'] != cur_p:
                cur_p = r['prompt_idx']
                f.write(f'\n{"=" * 80}\n[#{r["prompt_idx"]}] PROMPT: '
                        f'{r["prompt"]!r}\n{"=" * 80}\n')
            tag = ('UNSTEERED' if r['feature'] == 'unsteered'
                   else f'{r["feature"]} @ {r["angle_deg"]:.0f}deg')
            f.write(f'\n--- {tag} ---\n{r["generated"]}\n')
    print(f'\nSaved: {out_json}', flush=True)
    print(f'Saved: {out_txt}', flush=True)


if __name__ == '__main__':
    main()
