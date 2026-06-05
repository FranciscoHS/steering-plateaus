#!/usr/bin/env python3
"""Steering generations with the learned single feature direction d_f.

Single-direction variant of steer_triple_generate.py. Geodesic-rotates the
last-token residual at the learning layer by angle theta along d_f, sustained
at every decode step:
    a' = R (cos theta * a_hat + sin theta * d_perp)
with d_perp = direction orthogonalized against a_hat.

Usage:
    python scripts/superellipse/steer_df_generate.py \\
        --results_pkl results/feature_dir_vs_contrastive/gemma-2-9b_L15_a30/results.pkl \\
        --angles 15 30 60 --max_new_tokens 100
"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, '.')
import numpy as np
import torch

from src.model import load_model, _get_blocks


DEFAULT_PROMPTS = [
    "My favorite thing to do is",
    "Today I will definitely be",
    "My favorite food is",
    "The best thing about summer is",
]


def make_steer_hook(direction, angle_rad, device):
    """Forward hook: geodesic-rotate the last-token residual by angle along direction."""
    cos_t, sin_t = float(np.cos(angle_rad)), float(np.sin(angle_rad))

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        a = h[:, -1, :]                                  # [B, d]
        R = a.norm(dim=-1, keepdim=True)
        a_hat = a / R
        d = direction.to(device=a.device, dtype=a.dtype)
        proj = (a_hat * d).sum(-1, keepdim=True)
        d_perp = d.unsqueeze(0) - proj * a_hat
        d_perp = d_perp / (d_perp.norm(dim=-1, keepdim=True) + 1e-10)
        steered = R * (cos_t * a_hat + sin_t * d_perp)
        h = h.clone()
        h[:, -1, :] = steered.to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    return hook


@torch.no_grad()
def generate(model, tokenizer, input_ids, layer, device, direction=None,
             angle_rad=0.0, max_new_tokens=100, temperature=0.7, gen_seed=0):
    handle = None
    if direction is not None and angle_rad > 0:
        block = _get_blocks(model)[layer]
        handle = block.register_forward_hook(
            make_steer_hook(direction, angle_rad, device))
    gen_kwargs = dict(max_new_tokens=max_new_tokens,
                      pad_token_id=tokenizer.eos_token_id)
    if temperature and temperature > 0:
        torch.manual_seed(gen_seed)
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)
    try:
        out = model.generate(input_ids, **gen_kwargs)
    finally:
        if handle is not None:
            handle.remove()
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_pkl', required=True,
                    help='learn_feature_dir_vs_contrastive results.pkl')
    ap.add_argument('--prompts', nargs='+', default=DEFAULT_PROMPTS)
    ap.add_argument('--angles', type=float, nargs='+', default=[15.0, 30.0, 60.0])
    ap.add_argument('--max_new_tokens', type=int, default=100)
    ap.add_argument('--temperature', type=float, default=0.7,
                    help='>0 = single sample; 0 = greedy')
    ap.add_argument('--gen_seed', type=int, default=0)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--results_dir', default=None)
    args = ap.parse_args()

    with open(args.results_pkl, 'rb') as f:
        res = pickle.load(f)
    d_f = res['d_f'].float()
    d_f = d_f / d_f.norm()
    model_name = res['args']['model']
    layer = res['args']['layer']

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(args.results_pkl),
                                        'steering')
    os.makedirs(args.results_dir, exist_ok=True)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {model_name} ({args.dtype})...')
    model, tokenizer, device = load_model(model_name, dtype=dtype_map[args.dtype])
    print(f'Layer: {layer}, d_f norm: {d_f.norm().item():.4f}')

    runs = []
    for pi, prompt in enumerate(args.prompts):
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        print(f'\n[#{pi}] {prompt!r}')

        # Unsteered
        base = generate(model, tokenizer, input_ids, layer, device,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, gen_seed=args.gen_seed)
        runs.append({'prompt_idx': pi, 'prompt': prompt, 'angle_deg': 0.0,
                     'generated': base})
        print(f'  unsteered: {base!r}')

        for ang in args.angles:
            g = generate(model, tokenizer, input_ids, layer, device,
                         direction=d_f, angle_rad=np.deg2rad(ang),
                         max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, gen_seed=args.gen_seed)
            runs.append({'prompt_idx': pi, 'prompt': prompt, 'angle_deg': ang,
                         'generated': g})
            print(f'  d_f @ {ang:.0f}deg: {g!r}')

    out_json = os.path.join(args.results_dir, 'steering_df_generations.json')
    with open(out_json, 'w') as f:
        json.dump({'args': vars(args), 'model': model_name, 'layer': layer,
                   'runs': runs}, f, indent=2)

    out_txt = os.path.join(args.results_dir, 'steering_df_generations.txt')
    with open(out_txt, 'w') as f:
        f.write(f'Steering with learned d_f ({model_name}, layer {layer})\n')
        f.write(f'angles: {args.angles} deg, '
                f'max_new_tokens={args.max_new_tokens}, '
                f'temperature={args.temperature}, gen_seed={args.gen_seed}\n\n')
        cur = None
        for r in runs:
            if r['prompt_idx'] != cur:
                cur = r['prompt_idx']
                f.write(f'\n{"="*80}\n'
                        f'[#{r["prompt_idx"]}] PROMPT: {r["prompt"]!r}\n'
                        f'{"="*80}\n')
            tag = ('UNSTEERED' if r['angle_deg'] == 0.0
                   else f'd_f @ {r["angle_deg"]:.0f}deg')
            f.write(f'\n--- {tag} ---\n{r["generated"]}\n')

    print(f'\nSaved: {out_json}')
    print(f'Saved: {out_txt}')


if __name__ == '__main__':
    main()
