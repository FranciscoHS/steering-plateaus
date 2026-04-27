#!/usr/bin/env python3
"""Compositional steering test v2: Gender + Tense on Gemma-2-9B with
per-axis spherical angles.

For each baseline prompt (male + past-tense), generate completions under
each (theta_gender, theta_tense) condition. Per-axis interpretation:
  point = R · (C · â + sin(θ_g)·ĝ_⊥ + sin(θ_t)·t̂_⊥)
where C = sqrt(1 − sin(θ_g)² − sin(θ_t)²), ĝ_⊥ is Gender orthogonalized to
â, and t̂_⊥ is Tense orthogonalized to both â and ĝ_⊥.

Conditions = list of (θ_g_deg, θ_t_deg):
  (0,0)   baseline
  (30,0)  +Gender 30°
  (0,30)  +Tense  30°
  (30,30) +Gender+Tense 30°/30°
  (60,0)  +Gender 60°
  (0,60)  +Tense  60°
  (60,60) +Gender+Tense 60°/60°

Output: results/comp_steering/gender_tense_L2_30_60.json
"""
import os, sys, json, math, argparse
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F

from src.model import load_model

OUT_DIR = 'results/comp_steering'
DIR_PKL = 'results/fig4_robustness/directions_gemma_L2.pkl'
LAYER = 2
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

# (theta_deg, mode) where mode ∈ {'baseline','gender','tense','composite'}
CONDITIONS = [
    (0,  'baseline'),
    (30, 'gender'),
    (30, 'tense'),
    (30, 'composite'),
    (60, 'gender'),
    (60, 'tense'),
    (60, 'composite'),
]


def load_dirs():
    import pickle
    with open(DIR_PKL, 'rb') as f:
        all_dirs = pickle.load(f)['directions']
    g = F.normalize(all_dirs['Gender'].float(), dim=0)
    t = F.normalize(all_dirs['Tense'].float(), dim=0)
    return g, t


def make_steer_hook(steer_dir, theta_rad):
    """Hook that rotates last position by theta_rad on the sphere toward
    steer_dir (a unit vector). Fires only on the prompt-forward of the
    first call."""
    state = {'fired': False}

    def hook(module, args, output):
        if state['fired']:
            return output
        if isinstance(output, tuple):
            x = output[0]; rest = output[1:]
        else:
            x = output; rest = None
        if x.shape[1] <= 1:
            return output
        a = x[:, -1, :].float()                     # (1, D)
        R = a.norm(dim=-1, keepdim=True)
        a_unit = a / R
        d = steer_dir.to(a.device).unsqueeze(0)
        d_o = d - (d * a_unit).sum(-1, keepdim=True) * a_unit
        d_o = d_o / d_o.norm(dim=-1, keepdim=True)
        s = math.sin(theta_rad); c = math.cos(theta_rad)
        new_a = R * (c * a_unit + s * d_o)
        x_new = x.clone()
        x_new[:, -1, :] = new_a.to(x.dtype)
        state['fired'] = True
        if rest is None: return x_new
        return (x_new,) + rest

    return hook


def cond_label(theta, mode):
    if mode == 'baseline': return 'baseline'
    return {'gender': '+Gender', 'tense': '+Tense', 'composite': '+Composite'}[mode] + f' ({theta}°)'


def get_steer_for_mode(mode, g, t):
    if mode == 'gender':    return g
    if mode == 'tense':     return t
    if mode == 'composite': return F.normalize(g + t, dim=0)
    raise ValueError(mode)


def generate_one(model, tokenizer, device, prompt, theta, mode, g, t):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    block = model.model.layers[LAYER]
    handle = None
    if mode != 'baseline':
        steer = get_steer_for_mode(mode, g, t)
        handle = block.register_forward_hook(
            make_steer_hook(steer, math.radians(theta)))
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
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'gender_tense_L2_30_60.json'))
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print('loading gemma-2-9b ...', flush=True)
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)
    g, t = load_dirs()
    print(f'directions loaded: Gender (D={g.numel()}), Tense', flush=True)

    results = []
    for prompt in PROMPTS:
        for theta, mode in CONDITIONS:
            lbl = cond_label(theta, mode)
            for k in range(N_SAMPLES):
                text = generate_one(
                    model, tokenizer, device, prompt, theta, mode, g, t)
                results.append({
                    'prompt': prompt, 'condition': lbl,
                    'theta': theta, 'mode': mode,
                    'sample': k, 'text': text,
                })
                print(f'[{lbl:20s}] [s={k}] {text!r}'[:200], flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'layer': LAYER, 'n_samples': N_SAMPLES,
            'max_new': MAX_NEW, 'temperature': TEMPERATURE, 'top_p': TOP_P,
            'prompts': PROMPTS,
            'conditions': [{'theta': a, 'mode': b, 'label': cond_label(a,b)}
                           for a,b in CONDITIONS],
            'results': results,
        }, f, indent=2)
    print(f'saved → {args.out}')


if __name__ == '__main__':
    main()
