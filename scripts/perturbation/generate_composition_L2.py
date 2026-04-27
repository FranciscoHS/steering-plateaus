#!/usr/bin/env python3
"""Generate PT × Gender compositional steering samples at layer 2.

Four conditions:
  1. baseline (no steering)
  2. +PT direction only, 20 deg
  3. +Gender direction only, 20 deg
  4. PT+Gender, 20 deg along each axis (so combined total ~29 deg on sphere)

Starting prompt: "They elected a new chairman."
N generations per condition (different sampling seeds), saved to json/txt.
"""
import os, sys, pickle, json, argparse
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks

# --- Config ---
PROMPT = "They elected a new chairman"
LAYER = 2
ANGLE_PT_DEG = 20.0
ANGLE_GEN_DEG = 20.0
N_GENS_PER_COND = 5
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.9
TOP_P = 0.95
SEED = 42

OUT_DIR = 'results/paper_L2_gemma2_9b'
DIR_PKL = os.path.join(OUT_DIR, 'directions_L2.pkl')
OUT_JSON = os.path.join(OUT_DIR, 'generations_composition_L2.json')
OUT_TXT = os.path.join(OUT_DIR, 'generations_composition_L2.txt')


def make_steering_hook(direction_pt, direction_gen, alpha_pt, alpha_gen):
    """Return a forward-hook for the L`LAYER` residual stream that applies
    spherical rotation to the **last** position each call.

    alpha_pt / alpha_gen are radians along PT / Gender respectively; either
    may be zero (no contribution). Both directions are orthogonalized
    (against the activation's unit direction and each other) at call time.
    """
    def hook(module, inputs, output):
        # output: [B, T, D] or tuple whose first element is that
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        # apply to final position only
        a = hs[:, -1, :].float()  # [B, D]
        R = a.norm(dim=-1, keepdim=True)  # [B, 1]
        a_unit = a / (R + 1e-12)

        # Orthogonalize PT and Gender against a_unit, and Gender against PT_perp
        def orth(d, basis):
            # subtract projections onto each basis vector (all unit vectors)
            x = d
            for b in basis:
                x = x - (x * b).sum(dim=-1, keepdim=True) * b
            n = x.norm(dim=-1, keepdim=True)
            return x / (n + 1e-12), n > 1e-6

        basis = [a_unit]
        pt_perp = torch.zeros_like(a_unit)
        gen_perp = torch.zeros_like(a_unit)
        if direction_pt is not None:
            dp = direction_pt.to(a.device, a.dtype).unsqueeze(0).expand_as(a_unit)
            pt_perp, _ = orth(dp, basis)
            basis = basis + [pt_perp]
        if direction_gen is not None:
            dg = direction_gen.to(a.device, a.dtype).unsqueeze(0).expand_as(a_unit)
            gen_perp, _ = orth(dg, basis)

        # Compose on-sphere perturbation:
        # p = R * ( c*a_unit + sin(alpha_pt)*pt_perp + sin(alpha_gen)*gen_perp )
        s_pt = float(np.sin(alpha_pt))
        s_gen = float(np.sin(alpha_gen))
        c = float(np.sqrt(max(0.0, 1.0 - s_pt**2 - s_gen**2)))

        new_a = R * (c * a_unit + s_pt * pt_perp + s_gen * gen_perp)
        new_a = new_a.to(hs.dtype)
        hs[:, -1, :] = new_a

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs
    return hook


def generate(model, tokenizer, device, hook_fn, n_gens, seed_base):
    """Register hook on L`LAYER`, generate n_gens samples with varied seeds."""
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
    ap.add_argument('--angle_pt', type=float, default=ANGLE_PT_DEG)
    ap.add_argument('--angle_gen', type=float, default=ANGLE_GEN_DEG)
    args = ap.parse_args()

    print("Loading Gemma-2-9B...", flush=True)
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)

    with open(DIR_PKL, 'rb') as f:
        dirs = pickle.load(f)
    d_pt = F.normalize(dirs['d_portuguese'].float(), dim=0)
    d_gen = F.normalize(dirs['d_gender'].float(), dim=0)
    print(f"  cos(PT, gender) = {(d_pt @ d_gen).item():.4f}", flush=True)

    alpha_pt_r = float(np.deg2rad(args.angle_pt))
    alpha_gen_r = float(np.deg2rad(args.angle_gen))

    conditions = [
        ('baseline',   None,   None,        0.0, 0.0),
        ('pt_only',    d_pt,   None,        alpha_pt_r, 0.0),
        ('gender_only', None,  d_gen,       0.0, alpha_gen_r),
        ('pt_gender',  d_pt,   d_gen,       alpha_pt_r, alpha_gen_r),
    ]

    all_results = {}
    for name, dp, dg, a_pt, a_gen in conditions:
        print(f"\n[{name}] angle_pt={np.rad2deg(a_pt):.1f}° "
              f"angle_gen={np.rad2deg(a_gen):.1f}°", flush=True)
        if dp is None and dg is None:
            hook_fn = None
        else:
            hook_fn = make_steering_hook(
                direction_pt=dp, direction_gen=dg,
                alpha_pt=a_pt, alpha_gen=a_gen)
        outs = generate(model, tokenizer, device, hook_fn,
                        args.n_gens, SEED + hash(name) % 1000)
        all_results[name] = {
            'angle_pt_deg': float(np.rad2deg(a_pt)),
            'angle_gen_deg': float(np.rad2deg(a_gen)),
            'samples': outs,
        }
        for k, o in enumerate(outs):
            print(f"  sample {k}: {o[:100]}...", flush=True)

    with open(OUT_JSON, 'w') as f:
        json.dump({
            'prompt': PROMPT,
            'layer': LAYER,
            'model': 'gemma-2-9b',
            'max_new_tokens': MAX_NEW_TOKENS,
            'temperature': TEMPERATURE,
            'top_p': TOP_P,
            'conditions': all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_JSON}")

    with open(OUT_TXT, 'w') as f:
        f.write(f"Prompt: {PROMPT!r}\n")
        f.write(f"Model: gemma-2-9b · perturb L{LAYER}\n")
        f.write(f"max_new_tokens={MAX_NEW_TOKENS}, temp={TEMPERATURE}, "
                f"top_p={TOP_P}\n\n")
        for name, info in all_results.items():
            f.write('=' * 70 + '\n')
            f.write(f"[{name}]  angle_pt={info['angle_pt_deg']:.1f}°  "
                    f"angle_gen={info['angle_gen_deg']:.1f}°\n")
            f.write('=' * 70 + '\n')
            for k, o in enumerate(info['samples']):
                f.write(f"\n-- sample {k} --\n{PROMPT}{o}\n")
            f.write('\n')
    print(f"Saved: {OUT_TXT}")


if __name__ == '__main__':
    main()
