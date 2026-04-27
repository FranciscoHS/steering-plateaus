"""Compositional steering generations in the exp-map chart (E3).

For each prompt, build the anchor activation a at perturb_layer L (last
token), orthonormalize d1, d2 at a_hat to obtain d1_perp, d2_perp, and
generate completions under each (alpha_1, alpha_2) tangent-coord
condition by replacing the last-token activation with

    new_a = ||a|| (cos r a_hat + sin r v / r),
       v  = alpha_1 d1_perp + alpha_2 d2_perp,
       r  = sqrt(alpha_1^2 + alpha_2^2).

Polar form (the version we agreed on for paper text):
    new_a = ||a|| (cos alpha a_hat + sin alpha d(phi)),
    d(phi) = cos phi * d1_perp + sin phi * d2_perp.
A condition (alpha, phi) maps to (alpha_1, alpha_2) = (alpha cos phi,
alpha sin phi), so the user may pass either via --conditions or
--polar.

Output: results/exp_map/data/comp_steering/<target>_L<L>_<d1>__<d2>.json

Usage:
  python scripts/exp_map/comp_steer.py --target gemma --d1 Gender --d2 Tense \
      --conditions "0,0;30,0;0,30;30,30;60,0;0,60;60,60"
  python scripts/exp_map/comp_steer.py --target gemma --d1 Gender --d2 Tense \
      --polar "0,0;30,0;30,45;30,90;60,0;60,45;60,90"
"""
from __future__ import annotations
import os, sys, json, argparse, time, math
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks
from scripts.exp_map.lib import registry, directions as dirlib
from scripts.exp_map.lib.parametrize import (
    orthonormalize, exp_map_2d, composite_direction, eps_thresholds, CHART,
)

OUT_DIR = "results/exp_map/data/comp_steering"
os.makedirs(OUT_DIR, exist_ok=True)


DEFAULT_PROMPTS = [
    "Yesterday, my brother walked",
    "Last week, my father said",
    "Earlier today, the king ordered",
    "The man approached and",
    "Once upon a time, the prince",
    "He looked at his wife and",
]

DEFAULT_CONDITIONS = [(0, 0), (30, 0), (0, 30), (30, 30),
                      (60, 0), (0, 60), (60, 60)]


def parse_pairs(s: str) -> list[tuple[float, float]]:
    out = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk: continue
        a, b = chunk.split(",")
        out.append((float(a), float(b)))
    return out


def polar_to_cart(polar_deg: list[tuple[float, float]]
                   ) -> list[tuple[float, float]]:
    """Convert (alpha_deg, phi_deg) -> (alpha_1_deg, alpha_2_deg) per the
    composite-direction parametrization.
    alpha_1 = alpha cos phi,   alpha_2 = alpha sin phi  (degrees).
    """
    out = []
    for a_deg, phi_deg in polar_deg:
        phi = math.radians(phi_deg)
        out.append((a_deg * math.cos(phi), a_deg * math.sin(phi)))
    return out


def make_steer_hook_prefix(perturb_layer: int, new_a: torch.Tensor):
    """Prefix-only hook: replace last-token residual with new_a only on the
    prefix forward (T > 1). T == 1 generation steps pass through unchanged.
    Matches comp_steer_gender_tense_v2.make_steer_hook."""
    fired = {"n": 0}

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        if hidden.dim() != 3:
            return output
        if hidden.shape[1] <= 1:
            return output
        hidden = hidden.clone()
        hidden[:, -1, :] = new_a.to(hidden.dtype).to(hidden.device)
        fired["n"] += 1
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    return hook, fired


def make_steer_hook_persistent(d1: torch.Tensor, d2: torch.Tensor,
                                alpha1_rad: float, alpha2_rad: float):
    """Persistent rotation hook: every forward (prefix AND generation steps),
    rotate the last-token residual at the perturb layer by tangent coords
    (alpha1_rad, alpha2_rad) along d1, d2 (re-orthonormalized against the
    current step's last-token unit vector). Matches the intervention style
    of generate_composition_L2.make_steering_hook (the one that produces
    a clear breakdown at large angles)."""
    import math
    fired = {"n": 0}
    r2 = alpha1_rad * alpha1_rad + alpha2_rad * alpha2_rad

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]; rest = output[1:]
        else:
            hs = output; rest = None
        if hs.dim() != 3:
            return output
        a32 = hs[:, -1, :].float()
        R = a32.norm(dim=-1, keepdim=True)
        a_unit = a32 / R.clamp(min=1e-12)
        d1d = d1.to(a_unit.device, a_unit.dtype).unsqueeze(0).expand_as(a_unit)
        d1p = d1d - (d1d * a_unit).sum(-1, keepdim=True) * a_unit
        d1p = d1p / d1p.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        d2d = d2.to(a_unit.device, a_unit.dtype).unsqueeze(0).expand_as(a_unit)
        d2p = d2d - (d2d * a_unit).sum(-1, keepdim=True) * a_unit
        d2p = d2p - (d2p * d1p).sum(-1, keepdim=True) * d1p
        d2p = d2p / d2p.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        if r2 < 1e-24:
            new_a = R * a_unit
        else:
            r = math.sqrt(r2)
            v = alpha1_rad * d1p + alpha2_rad * d2p
            new_a = R * (math.cos(r) * a_unit + (math.sin(r) / r) * v)
        hs = hs.clone()
        hs[:, -1, :] = new_a.to(hs.dtype)
        fired["n"] += 1
        if rest is None: return hs
        return (hs,) + rest

    return hook, fired


def get_block(model, layer: int):
    return _get_blocks(model)[layer]


@torch.no_grad()
def anchor_activation(model, tokenizer, device, prompt: str, layer: int):
    tok = tokenizer(prompt, return_tensors="pt").to(device)
    outs = model(**tok, output_hidden_states=True)
    return outs.hidden_states[layer + 1][:, -1, :].squeeze(0).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--d1", default="Gender")
    ap.add_argument("--d2", default="Tense")
    ap.add_argument("--prompts", default=None,
                    help="path to a text file of prompts, one per line; "
                         "default = a small built-in list")
    ap.add_argument("--conditions", default=None,
                    help="(alpha_1_deg, alpha_2_deg) tuples, semicolon-sep")
    ap.add_argument("--polar", default=None,
                    help="(alpha_deg, phi_deg) tuples, semicolon-sep; "
                         "converted to (alpha_1, alpha_2) internally")
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--composite_only", action="store_true",
                    help="instead of (alpha_1, alpha_2) grid, sweep alpha "
                         "along the orthogonalized-sum composite (D6).")
    ap.add_argument("--composite_alphas",
                    default="0,15,30,45,60",
                    help="comma-separated alpha (deg) values for "
                         "--composite_only mode")
    ap.add_argument("--tag", default=None,
                    help="optional suffix appended to the output filename "
                         "to avoid clobbering prior runs")
    ap.add_argument("--prompt", default=None,
                    help="single prompt string (overrides --prompts and "
                         "DEFAULT_PROMPTS)")
    ap.add_argument("--output_format", default="json",
                    choices=["json", "txt", "both"])
    ap.add_argument("--mode", default="persistent",
                    choices=["prefix", "persistent"],
                    help="prefix: replace last-token residual only on the "
                         "prefix forward (matches comp_steer_gender_tense_v2). "
                         "persistent: re-rotate the last-token residual at "
                         "every forward (prefix + every generation step), "
                         "matching generate_composition_L2 (default).")
    args = ap.parse_args()

    if args.composite_only:
        a_list = [float(x) for x in args.composite_alphas.split(",")]
        # alpha along (d1_perp + d2_perp) / sqrt(2): same as
        # (alpha cos 45, alpha sin 45)
        conditions = [(a / math.sqrt(2), a / math.sqrt(2)) for a in a_list]
        cond_label_kind = "composite_alpha_deg"
    elif args.polar:
        conditions = polar_to_cart(parse_pairs(args.polar))
        cond_label_kind = "polar_deg"
    elif args.conditions:
        conditions = parse_pairs(args.conditions)
        cond_label_kind = "alpha1_alpha2_deg"
    else:
        conditions = DEFAULT_CONDITIONS
        cond_label_kind = "alpha1_alpha2_deg"

    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts:
        with open(args.prompts) as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    cfg = registry.TARGETS[args.target]
    print(f"[{time.strftime('%H:%M:%S')}] {cfg['model']} L{args.layer}", flush=True)
    model, tokenizer, device = load_model(cfg["model"], dtype=torch.bfloat16)

    dirs_blob = dirlib.extract_all(
        model, tokenizer, device, args.target, args.layer,
        names=[args.d1, args.d2])
    d1 = dirs_blob["directions"][args.d1].float().to(device)
    d2 = dirs_blob["directions"][args.d2].float().to(device)
    sign1 = dirs_blob["signs"][args.d1]
    sign2 = dirs_blob["signs"][args.d2]
    print(f"  d1={args.d1} ({sign1})  d2={args.d2} ({sign2})", flush=True)

    block = get_block(model, args.layer)
    torch.manual_seed(args.seed)

    out = {
        "chart": CHART,
        "eps_thresholds": eps_thresholds(),
        "model": args.target, "model_full": cfg["model"],
        "perturb_layer": args.layer,
        "directions": (args.d1, args.d2),
        "direction_signs": {args.d1: sign1, args.d2: sign2},
        "condition_kind": cond_label_kind,
        "conditions_alpha1_alpha2_deg": [(float(a), float(b))
                                          for a, b in conditions],
        "mode": args.mode,
        "n_samples": args.n_samples,
        "max_new": args.max_new,
        "temperature": args.temperature, "top_p": args.top_p,
        "seed": args.seed,
        "results": [],
    }

    for pi, prompt in enumerate(prompts):
        a = anchor_activation(model, tokenizer, device, prompt, args.layer)
        try:
            d1p, d2p = orthonormalize(a / a.norm(), d1, d2)
        except Exception as e:
            print(f"  prompt {pi}: skip ({e})"); continue
        # Record composite info too (D6).
        comp = composite_direction(a / a.norm(), d1, d2)
        # Build all perturbed activations up front.
        new_acts = {}
        for (a1_deg, a2_deg) in conditions:
            r1 = math.radians(a1_deg); r2 = math.radians(a2_deg)
            new_acts[(a1_deg, a2_deg)] = exp_map_2d(
                a, d1p, d2p,
                torch.tensor(r1, dtype=a.dtype, device=device),
                torch.tensor(r2, dtype=a.dtype, device=device))

        prompt_block = {"prompt": prompt, "anchor_norm": float(a.norm()),
                         "conditions": []}

        for (a1_deg, a2_deg) in conditions:
            new_a = new_acts[(a1_deg, a2_deg)]
            polar_alpha = math.degrees(math.sqrt(
                math.radians(a1_deg) ** 2 + math.radians(a2_deg) ** 2))
            polar_phi = (math.degrees(math.atan2(math.radians(a2_deg),
                                                 math.radians(a1_deg)))
                         if (a1_deg != 0 or a2_deg != 0) else 0.0)

            samples = []
            for _ in range(args.n_samples):
                if args.mode == "prefix":
                    hook, fired = make_steer_hook_prefix(args.layer, new_a)
                else:
                    hook, fired = make_steer_hook_persistent(
                        d1, d2,
                        math.radians(a1_deg), math.radians(a2_deg))
                handle = block.register_forward_hook(hook)
                try:
                    tok = tokenizer(prompt, return_tensors="pt").to(device)
                    gen = model.generate(
                        **tok, max_new_tokens=args.max_new,
                        do_sample=True, temperature=args.temperature,
                        top_p=args.top_p, pad_token_id=tokenizer.eos_token_id)
                finally:
                    handle.remove()
                samples.append(tokenizer.decode(
                    gen[0, tok.input_ids.shape[1]:], skip_special_tokens=True))

            prompt_block["conditions"].append({
                "alpha1_deg": float(a1_deg),
                "alpha2_deg": float(a2_deg),
                "alpha_polar_deg": polar_alpha,
                "phi_polar_deg": polar_phi,
                "samples": samples,
            })

        out["results"].append(prompt_block)
        print(f"  prompt {pi+1}/{len(prompts)} done", flush=True)

    suffix = f"_{args.tag}" if args.tag else ""
    base = os.path.join(OUT_DIR, f"{args.target}_L{args.layer}_"
                         f"{args.d1}__{args.d2}{suffix}")
    if args.output_format in ("json", "both"):
        with open(base + ".json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {base}.json")
    if args.output_format in ("txt", "both"):
        lines = []
        lines.append(f"# {out['model_full']}  L={out['perturb_layer']}")
        lines.append(f"# d1={out['directions'][0]} ({out['direction_signs'][out['directions'][0]]})")
        lines.append(f"# d2={out['directions'][1]} ({out['direction_signs'][out['directions'][1]]})")
        lines.append(f"# mode={out['mode']}")
        lines.append(f"# n_samples={out['n_samples']} max_new={out['max_new']} "
                     f"temp={out['temperature']} top_p={out['top_p']} seed={out['seed']}")
        lines.append("")
        for r in out["results"]:
            lines.append(f"PROMPT: {r['prompt']!r}")
            for c in r["conditions"]:
                lines.append(f"  alpha1={c['alpha1_deg']:>5.1f}  alpha2={c['alpha2_deg']:>5.1f}  "
                             f"(alpha={c['alpha_polar_deg']:>5.1f}  phi={c['phi_polar_deg']:>5.1f})")
                for i, s in enumerate(c["samples"]):
                    lines.append(f"    [{i}] {s.strip()}")
                lines.append("")
            lines.append("")
        with open(base + ".txt", "w") as f:
            f.write("\n".join(lines))
        print(f"saved {base}.txt")


if __name__ == "__main__":
    main()
