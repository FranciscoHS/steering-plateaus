#!/usr/bin/env python3
"""Generate Gender × Python compositional steering samples at L2.

Mirrors generate_composition_L2.py exactly but for the (Gender, Python) pair
instead of (PT, Gender). Same prompt, same angle (20° each), same sampling.

4 conditions: baseline / gender_only / python_only / gender+python.
"""
import os, sys, pickle, json, argparse
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks
from scripts.perturbation.generate_composition_L2 import make_steering_hook

# --- Config (matches generate_composition_L2.py) ---
PROMPT = "They elected a new chairman"
LAYER = 2
ANGLE_GENDER_DEG = 20.0
ANGLE_PYTHON_DEG = 20.0
N_GENS_PER_COND = 5
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.9
TOP_P = 0.95
SEED = 42

DIR_PKL = 'results/new_directions_L2/directions_L2.pkl'
OUT_DIR = 'results/new_directions_L2'


def generate(model, tokenizer, device, hook_fn, n_gens, seed_base):
    blocks = _get_blocks(model)
    handle = blocks[LAYER].register_forward_hook(hook_fn) if hook_fn else None
    inp = tokenizer(PROMPT, return_tensors='pt').to(device)
    outs = []
    try:
        for k in range(n_gens):
            torch.manual_seed(seed_base + k)
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )
            text = tokenizer.decode(
                out[0, inp.input_ids.shape[1]:], skip_special_tokens=True)
            outs.append(text)
    finally:
        if handle is not None:
            handle.remove()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_gens', type=int, default=N_GENS_PER_COND)
    ap.add_argument('--angle_gender', type=float, default=ANGLE_GENDER_DEG)
    ap.add_argument('--angle_python', type=float, default=ANGLE_PYTHON_DEG)
    ap.add_argument('--tag', type=str, default='',
                    help='suffix for output filenames')
    args = ap.parse_args()

    tag = args.tag or f'g{int(args.angle_gender)}_py{int(args.angle_python)}'
    out_json = os.path.join(OUT_DIR, f'generations_Gender_Python_{tag}.json')
    out_txt = os.path.join(OUT_DIR, f'generations_Gender_Python_{tag}.txt')

    print("Loading Gemma-2-9B...", flush=True)
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)

    with open(DIR_PKL, 'rb') as f:
        cached = pickle.load(f)['directions']
    d_gender = F.normalize(cached['Gender'].float(), dim=0)
    d_python = F.normalize(cached['Python'].float(), dim=0)
    print(f"  cos(Gender, Python) = {(d_gender @ d_python).item():.4f}",
          flush=True)

    ag = float(np.deg2rad(args.angle_gender))
    ap_ = float(np.deg2rad(args.angle_python))

    # NB: make_steering_hook signature uses (pt, gen) names but is generic.
    # First slot = Gender, second slot = Python.
    conditions = [
        ('baseline',      None,      None,     0.0, 0.0),
        ('gender_only',   d_gender,  None,     ag, 0.0),
        ('python_only',   None,      d_python, 0.0, ap_),
        ('gender_python', d_gender,  d_python, ag, ap_),
    ]

    all_results = {}
    for name, d1, d2, a1, a2 in conditions:
        print(f"\n[{name}] angle_gender={np.rad2deg(a1):.1f}° "
              f"angle_python={np.rad2deg(a2):.1f}°", flush=True)
        hook_fn = (make_steering_hook(d1, d2, a1, a2)
                   if d1 is not None or d2 is not None else None)
        outs = generate(model, tokenizer, device, hook_fn,
                        args.n_gens, SEED + hash(name) % 1000)
        all_results[name] = {
            'angle_gender_deg': float(np.rad2deg(a1)),
            'angle_python_deg': float(np.rad2deg(a2)),
            'samples': outs,
        }
        for k, o in enumerate(outs):
            print(f"  sample {k}: {o[:100]}...", flush=True)

    with open(out_json, 'w') as f:
        json.dump({
            'prompt': PROMPT,
            'layer': LAYER,
            'model': 'gemma-2-9b',
            'max_new_tokens': MAX_NEW_TOKENS,
            'temperature': TEMPERATURE, 'top_p': TOP_P,
            'conditions': all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_json}")

    with open(out_txt, 'w') as f:
        f.write(f"Prompt: {PROMPT!r}\n")
        f.write(f"Model: gemma-2-9b · perturb L{LAYER}\n")
        f.write(f"max_new_tokens={MAX_NEW_TOKENS}, temp={TEMPERATURE}, "
                f"top_p={TOP_P}\n\n")
        for name, info in all_results.items():
            f.write('=' * 70 + '\n')
            f.write(f"[{name}]  angle_gender={info['angle_gender_deg']:.1f}°  "
                    f"angle_python={info['angle_python_deg']:.1f}°\n")
            f.write('=' * 70 + '\n')
            for k, o in enumerate(info['samples']):
                f.write(f"\n-- sample {k} --\n{PROMPT}{o}\n")
            f.write('\n')
    print(f"Saved: {out_txt}")


if __name__ == '__main__':
    main()
