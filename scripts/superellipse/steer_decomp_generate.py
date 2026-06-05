#!/usr/bin/env python3
"""Steering generations with each of the K decomposition features d_k.

One prompt by default; one unsteered baseline + (K x len(angles)) steered
generations. Sustained geodesic last-token steering at the perturb layer
(same hook as steer_df_generate.py).
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from src.model import load_model, _get_blocks


def make_steer_hook(direction, angle_rad, device):
    cos_t, sin_t = float(np.cos(angle_rad)), float(np.sin(angle_rad))

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        a = h[:, -1, :]
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
    return tokenizer.decode(out[0, input_ids.shape[1]:],
                            skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_pkl', required=True,
                    help='decomposition results.pkl with features (K, h)')
    ap.add_argument('--prompts', nargs='+', default=['I love to'])
    ap.add_argument('--angles', type=float, nargs='+',
                    default=[15.0, 30.0, 45.0])
    ap.add_argument('--max_new_tokens', type=int, default=100)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--gen_seed', type=int, default=0)
    ap.add_argument('--feature_indices', type=int, nargs='+', default=None,
                    help='which feature rows to use (default: all)')
    ap.add_argument('--feature_signs', type=float, nargs='+', default=None,
                    help='per-feature sign multiplier (default: all +1); '
                         'length must match --feature_indices')
    ap.add_argument('--model_name', default=None,
                    help='override model name (defaults to args["model"] from pkl)')
    ap.add_argument('--layer', type=int, default=None,
                    help='override perturb layer (defaults to args["layer"] from pkl)')
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--results_dir', default=None)
    args = ap.parse_args()

    with open(args.results_pkl, 'rb') as f:
        res = pickle.load(f)
    feats = res['features']
    if not isinstance(feats, torch.Tensor):
        feats = torch.from_numpy(feats)
    feats = feats.float()
    K = feats.shape[0]
    if args.feature_indices is None:
        args.feature_indices = list(range(K))
    feats = feats[args.feature_indices]
    feats = feats / feats.norm(dim=-1, keepdim=True)
    if args.feature_signs is not None:
        if len(args.feature_signs) != len(args.feature_indices):
            raise ValueError('feature_signs length must match feature_indices')
        signs = torch.tensor(args.feature_signs, dtype=feats.dtype).view(-1, 1)
        feats = feats * signs
    sign_for = (lambda i: args.feature_signs[i] if args.feature_signs else 1.0)
    labels = [f'{"+" if sign_for(j) > 0 else "-"}d{i+1}'
              for j, i in enumerate(args.feature_indices)]
    model_name = args.model_name or res['args']['model']
    layer = args.layer if args.layer is not None else res['args']['layer']

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(args.results_pkl),
                                        'steering_decomp')
    os.makedirs(args.results_dir, exist_ok=True)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {model_name} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(model_name,
                                          dtype=dtype_map[args.dtype])
    print(f'Layer: {layer}, K_use: {len(labels)} -> {labels}', flush=True)

    runs = []
    for pi, prompt in enumerate(args.prompts):
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        print(f'\n[#{pi}] {prompt!r}', flush=True)

        base = generate(model, tokenizer, input_ids, layer, device,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, gen_seed=args.gen_seed)
        runs.append({'prompt_idx': pi, 'prompt': prompt, 'feature': 'unsteered',
                     'angle_deg': 0.0, 'generated': base})
        print(f'  unsteered: {base!r}', flush=True)

        for fi, (label, d) in enumerate(zip(labels, feats)):
            for ang in args.angles:
                g = generate(model, tokenizer, input_ids, layer, device,
                             direction=d, angle_rad=np.deg2rad(ang),
                             max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature,
                             gen_seed=args.gen_seed)
                runs.append({'prompt_idx': pi, 'prompt': prompt,
                             'feature': label, 'angle_deg': ang,
                             'generated': g})
                print(f'  {label} @ {ang:.0f}deg: {g!r}', flush=True)

    out_json = os.path.join(args.results_dir,
                            'steering_decomp_generations.json')
    with open(out_json, 'w') as f:
        json.dump({'args': vars(args), 'model': model_name, 'layer': layer,
                   'runs': runs}, f, indent=2)

    out_txt = os.path.join(args.results_dir,
                           'steering_decomp_generations.txt')
    with open(out_txt, 'w') as f:
        f.write(f'Steering with decomposition features ({model_name}, '
                f'layer {layer})\n')
        f.write(f'features: {labels}\n')
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
