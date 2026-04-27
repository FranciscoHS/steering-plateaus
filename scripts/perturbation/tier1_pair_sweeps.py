#!/usr/bin/env python3
"""Tier-1 dir×dir 2D pair sweeps.

20 representative direction × direction pairs spanning SEM × SEM,
SEM × new-single-axis, new × new, and SEM × programming. 41×41 spherical
grid (0..MAX_ANGLE) on N_ACT FineWeb starting activations at perturb_layer,
measured at penultimate layer.

For each pair we save:
  results/fig4_robustness/v2_T1_<target>_<a>_<b>.pkl
  {
    'angles_deg': (n_steps,),
    'l2': (n_act, n_steps, n_steps),    # spherical-grid L²
    'direction_labels': (a, b),
    'layer': perturb_layer, 'measure_layer': measure_layer,
    'model': target,
  }

Direction extension: any direction in PAIRS that isn't in the existing
direction cache is computed from training prompts and added to the cache.

Usage:
  python scripts/perturbation/tier1_pair_sweeps.py --target gemma --layer 2
"""
import os, sys, pickle, argparse, time, importlib
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks, forward_from_layer_to_layer
from src.data import load_fineweb_fixed_length

OUT_DIR = 'results/fig4_robustness'

TARGETS = {
    'gemma':   dict(model='gemma-2-9b'),
    'qwen':    dict(model='qwen3-1.7b'),
    'llama':   dict(model='llama-3.1-8b'),
    'mistral': dict(model='mistral-7b-v0.3'),
    'aya':     dict(model='aya-expanse-8b'),
    'yi':      dict(model='yi-1.5-9b'),
}

# 20 Tier-1 pairs
PAIRS = [
    # SEM × SEM (3)
    ('Gender', 'Sentiment'),
    ('Gender', 'Formal'),
    ('Sentiment', 'Formal'),
    # SEM × new-single-axis (6)
    ('Gender', 'Tense'),
    ('Sentiment', 'Honesty'),
    ('Formal', 'Refusal'),
    ('Gender', 'Status'),
    ('Sentiment', 'Era'),
    ('Formal', 'Person'),
    # new × new (6)
    ('Age', 'Era'),
    ('Tense', 'TensePresent'),
    ('Number', 'Person'),
    ('Honesty', 'HonestyShort'),
    ('Refusal', 'Honesty'),
    ('Status', 'Era'),
    # SEM × programming (5)
    ('Gender', 'Python'),
    ('Sentiment', 'Rust'),
    ('Formal', 'JavaScript'),
    ('Sentiment', 'Haskell'),
    ('Gender', 'TypeScript'),
]

# Direction extractors for the 10 new single-axis directions not in the cache
NEW_DIRS = [
    # name, module, a_attr, b_attr  (direction = b_mean - a_mean)
    ('Age',          'data.age_prompts',           'YOUNG_PROMPTS',        'OLD_PROMPTS'),
    ('Era',          'data.era_prompts',           'MEDIEVAL_PROMPTS',     'MODERN_PROMPTS'),
    ('Tense',        'data.tense_prompts',         'PAST_PROMPTS',         'FUTURE_PROMPTS'),
    ('TensePresent', 'data.tense_present_prompts', 'PRESENT_PROMPTS',      'PAST_PROMPTS'),
    ('Number',       'data.number_prompts',        'SINGULAR_PROMPTS',     'PLURAL_PROMPTS'),
    ('Person',       'data.person_prompts',        'FIRST_PROMPTS',        'THIRD_PROMPTS'),
    ('Status',       'data.status_prompts',        'ROYAL_PROMPTS',        'COMMONER_PROMPTS'),
    ('Honesty',      'data.honesty_prompts',       'HONEST_PROMPTS',       'DISHONEST_PROMPTS'),
    ('HonestyShort', 'data.honesty_short_prompts', 'HONEST_SHORT_PROMPTS', 'DISHONEST_SHORT_PROMPTS'),
    ('Refusal',      'data.refusal_prompts',       'COMPLY_PROMPTS',       'REFUSE_PROMPTS'),
]

N_ACT = 30
SEQ_LEN = 5
SEED = 42
N_STEPS = 41
MAX_ANGLE = 30.0
BATCH = 64


def ts(): return time.strftime('%H:%M:%S')


def safe_pkl(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f: pickle.dump(obj, f)
    os.replace(tmp, path)


def compute_dom(model, tokenizer, device, layer, a_prompts, b_prompts):
    """direction = (b_mean − a_mean) normalized."""
    def _acts(prompts):
        out = []
        for p in prompts:
            tok = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                o = model(**tok, output_hidden_states=True)
            out.append(o.hidden_states[layer + 1][:, -1, :].squeeze(0).float().cpu())
        return torch.stack(out)
    return F.normalize((_acts(b_prompts) - _acts(a_prompts)).mean(0), dim=0)


def fineweb_ctx_acts(model, tokenizer, device, layer, target):
    cache_dir = f'/workspace/fineweb_cache_{target}'
    os.makedirs(cache_dir, exist_ok=True)
    token_lists = load_fineweb_fixed_length(
        N_ACT, tokenizer, seq_len=SEQ_LEN, seed=SEED, cache_dir=cache_dir)
    contexts, activations = [], []
    for tids in token_lists:
        tids = tids.unsqueeze(0).to(device)
        with torch.no_grad():
            outs = model(tids, output_hidden_states=True)
        h = outs.hidden_states[layer + 1]
        contexts.append(h[:, :-1, :].float().cpu())
        activations.append(h[:, -1, :].squeeze(0).float().cpu())
    return contexts, torch.stack(activations)


def sweep_2d(model, ctx, act, d1, d2, layer, measure_layer, device,
             angles_rad, model_dtype):
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=torch.float32)
    R = a_d.norm(); a_unit = a_d / R
    d1_d = d1.to(device=device, dtype=torch.float32)
    d2_d = d2.to(device=device, dtype=torch.float32)
    d1_o = F.normalize(d1_d - (d1_d @ a_unit) * a_unit, dim=0)
    d2_o = d2_d - (d2_d @ a_unit) * a_unit
    d2_o = F.normalize(d2_o - (d2_o @ d1_o) * d1_o, dim=0)
    n = len(angles_rad)
    s = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)
    S1, S2 = torch.meshgrid(s, s, indexing='ij')
    C = torch.sqrt(torch.clamp(1.0 - S1**2 - S2**2, min=0.0))
    pts = (R * (C.reshape(-1, 1) * a_unit.unsqueeze(0)
                + S1.reshape(-1, 1) * d1_o.unsqueeze(0)
                + S2.reshape(-1, 1) * d2_o.unsqueeze(0))).to(model_dtype)
    n_pts = n * n
    out = np.zeros(n_pts, dtype=np.float32)
    with torch.no_grad():
        ref = forward_from_layer_to_layer(
            model, ctx_d, a_d[None, None].to(model_dtype),
            layer, measure_layer).float()
        for st in range(0, n_pts, BATCH):
            ed = min(st + BATCH, n_pts)
            pf = forward_from_layer_to_layer(
                model, ctx_d, pts[st:ed].unsqueeze(1),
                layer, measure_layer).float()
            out[st:ed] = (pf - ref).norm(dim=-1).cpu().numpy()
    return out.reshape(n, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(TARGETS))
    ap.add_argument('--layer', type=int, default=2)
    args = ap.parse_args()
    cfg = TARGETS[args.target]

    print(f'[{ts()}] loading {cfg["model"]}', flush=True)
    model, tokenizer, device = load_model(cfg['model'], dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    perturb_layer = args.layer
    model_dtype = next(model.parameters()).dtype
    print(f'  n_layers={n_layers}  perturb={perturb_layer}  measure={measure_layer}',
          flush=True)

    # Load + extend direction cache
    dir_pkl = os.path.join(OUT_DIR,
                           f'directions_{args.target}_L{perturb_layer}.pkl')
    if not os.path.exists(dir_pkl):
        from scripts.perturbation.fig4_robustness_v2 import extract_all_directions
        print(f'[{ts()}] building base direction cache → {dir_pkl}', flush=True)
        all_dirs = extract_all_directions(model, tokenizer, device,
                                           perturb_layer, dir_pkl)
    else:
        with open(dir_pkl, 'rb') as f:
            all_dirs = pickle.load(f)['directions']

    needed = sorted({n for pair in PAIRS for n in pair})
    missing = [n for n in needed if n not in all_dirs]
    new_added = []
    for name, mod_path, a_attr, b_attr in NEW_DIRS:
        if name in needed and name not in all_dirs:
            mod = importlib.import_module(mod_path)
            a = getattr(mod, a_attr); b = getattr(mod, b_attr)
            print(f'[{ts()}] extending dirs: computing {name}', flush=True)
            all_dirs[name] = compute_dom(model, tokenizer, device,
                                          perturb_layer, a, b)
            new_added.append(name)
    if new_added:
        safe_pkl(dir_pkl, {'directions': all_dirs, 'layer': perturb_layer})
        print(f'[{ts()}] cache updated with {new_added}', flush=True)
    still_missing = [n for n in needed if n not in all_dirs]
    if still_missing:
        print(f'WARN missing dirs: {still_missing}');

    print(f'[{ts()}] extracting FineWeb activations', flush=True)
    contexts, activations = fineweb_ctx_acts(
        model, tokenizer, device, perturb_layer, args.target)

    angles = np.linspace(0, MAX_ANGLE, N_STEPS)
    angles_rad = np.deg2rad(angles)

    for a, b in PAIRS:
        if a not in all_dirs or b not in all_dirs:
            print(f'  SKIP {a}×{b} (missing dir)'); continue
        out_path = os.path.join(
            OUT_DIR, f'v2_T1_{args.target}_{a}_{b}.pkl')
        if os.path.exists(out_path):
            print(f'  exists: {out_path}'); continue
        d1 = F.normalize(all_dirs[a].float(), dim=0)
        d2 = F.normalize(all_dirs[b].float(), dim=0)
        l2 = np.zeros((N_ACT, N_STEPS, N_STEPS), dtype=np.float32)
        t0 = time.time()
        for ai in range(N_ACT):
            l2[ai] = sweep_2d(
                model, contexts[ai], activations[ai], d1, d2,
                perturb_layer, measure_layer, device, angles_rad, model_dtype)
        out = {
            'angles_deg': angles, 'l2': l2,
            'direction_labels': (a, b),
            'layer': perturb_layer, 'measure_layer': measure_layer,
            'model': args.target,
        }
        safe_pkl(out_path, out)
        print(f'  [{ts()}] saved {a}×{b}  {time.time()-t0:.1f}s', flush=True)

    print(f'[{ts()}] all pairs done for {args.target}', flush=True)


if __name__ == '__main__':
    main()
