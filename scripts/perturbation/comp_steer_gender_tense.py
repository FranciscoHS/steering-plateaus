#!/usr/bin/env python3
"""Compositional steering test: Gender + Tense on Gemma-2-9B.

For each baseline prompt (male + past-tense), generate completions under
4 conditions:
  1. baseline    — no steering
  2. +Gender     — rotate last-position L2 activation 20° toward Gender
  3. +Tense      — rotate 20° toward Tense
  4. +Gender+Tense — rotate 20° toward (Gender + Tense) / ||Gender + Tense||

Steering is applied by rotating ONLY the last-position residual stream
output of layer 2 (model.model.layers[2]) on its own activation sphere.
The rotation is applied only on the prompt-forward (seq_len > 1); during
token-by-token generation it is bypassed.

Output: results/comp_steering/gender_tense_L2_20deg.json
"""
import os, sys, json, math, argparse
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F

from src.model import load_model

OUT_DIR = 'results/comp_steering'
DIR_PKL = 'results/fig4_robustness/directions_gemma_L2.pkl'
LAYER = 2
THETA_DEG = 20.0
N_SAMPLES = 3
MAX_NEW = 40
TEMPERATURE = 0.8
TOP_P = 0.9

PROMPTS = [
    "Yesterday, my brother walked",
    "Last week, my father said",
    "Earlier today, the king ordered",
    "An hour ago, the actor left",
    "Last summer, my uncle traveled",
]

CONDITIONS = ['baseline', '+Gender', '+Tense', '+Gender+Tense']


def load_dirs():
    import pickle
    with open(DIR_PKL, 'rb') as f:
        all_dirs = pickle.load(f)['directions']
    g = F.normalize(all_dirs['Gender'].float(), dim=0)
    t = F.normalize(all_dirs['Tense'].float(), dim=0)
    return g, t


def make_steer_hook(steer_dir, theta_rad):
    """Hook that rotates the last-position residual stream by theta_rad
    on the activation sphere toward steer_dir. Fires only on the
    prompt-forward (seq_len > 1) of the *first* call."""
    state = {'fired': False}

    def hook(module, args, output):
        if state['fired']:
            return output
        # output may be a tensor or (tensor, ...) tuple depending on layer impl
        if isinstance(output, tuple):
            x = output[0]
            rest = output[1:]
        else:
            x = output
            rest = None
        if x.shape[1] <= 1:
            return output  # token-by-token forward, skip
        # Rotate last position
        a = x[:, -1, :].float()                     # (1, D)
        R = a.norm(dim=-1, keepdim=True)            # (1, 1)
        a_unit = a / R                              # (1, D)
        d = steer_dir.to(a.device).unsqueeze(0)     # (1, D)
        d_o = d - (d * a_unit).sum(-1, keepdim=True) * a_unit
        d_o = d_o / d_o.norm(dim=-1, keepdim=True)
        s = math.sin(theta_rad); c = math.cos(theta_rad)
        new_a = R * (c * a_unit + s * d_o)
        x_new = x.clone()
        x_new[:, -1, :] = new_a.to(x.dtype)
        state['fired'] = True
        if rest is None:
            return x_new
        return (x_new,) + rest

    return hook


def get_steer_for_condition(cond, g, t):
    if cond == 'baseline':
        return None
    if cond == '+Gender':
        return g
    if cond == '+Tense':
        return t
    if cond == '+Gender+Tense':
        return F.normalize(g + t, dim=0)
    raise ValueError(cond)


def generate_one(model, tokenizer, device, prompt, cond, g, t):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    block = model.model.layers[LAYER]
    handle = None
    steer = get_steer_for_condition(cond, g, t)
    if steer is not None:
        handle = block.register_forward_hook(
            make_steer_hook(steer, math.radians(THETA_DEG)))
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if handle is not None:
            handle.remove()
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'gender_tense_L2_20deg.json'))
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print(f'loading gemma-2-9b ...')
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)
    g, t = load_dirs()
    print(f'directions loaded: Gender (D={g.numel()}), Tense')

    results = []
    for prompt in PROMPTS:
        for cond in CONDITIONS:
            for k in range(N_SAMPLES):
                text = generate_one(model, tokenizer, device, prompt, cond, g, t)
                results.append({
                    'prompt': prompt, 'condition': cond, 'sample': k, 'text': text,
                })
                print(f'[{cond:18s}] [s={k}] {text!r}'[:200], flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'layer': LAYER, 'theta_deg': THETA_DEG, 'n_samples': N_SAMPLES,
            'max_new': MAX_NEW, 'temperature': TEMPERATURE, 'top_p': TOP_P,
            'prompts': PROMPTS, 'conditions': CONDITIONS,
            'results': results,
        }, f, indent=2)
    print(f'saved → {args.out}')


if __name__ == '__main__':
    main()
