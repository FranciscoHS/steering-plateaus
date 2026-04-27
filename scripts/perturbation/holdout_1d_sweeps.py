#!/usr/bin/env python3
"""1D perturbation sweeps from prompts matched to each contrastive direction.

For each (direction, holdout source) pair, extract activations from the
holdout prompts and sweep along the direction (perturb_layer → measure_layer)
from 0° to MAX_ANGLE. Per source, also run N_RAND random unit vectors as a
baseline (so each direction's response is comparable to random sweeps from
the same starting activations).

Available holdout sources (one per direction we have data for):
  Gender    ← data.male_prompts_holdout.MALE_PROMPTS_HOLDOUT
  Sentiment ← data.positive_prompts_holdout.POSITIVE_PROMPTS_HOLDOUT

Output: results/fig4_robustness/v2_M1d_<target>_L<layer>.pkl
  {
    'angles_deg': (n_steps,),
    'layer': perturb_layer, 'measure_layer': measure_layer,
    'sources': {
       <direction_name>: {
           'l2_dir':  (n_act, n_steps),       # along this direction
           'l2_rand': (n_rand, n_act, n_steps),
           'holdout_module': 'data.male_prompts_holdout',
       }, ...
    },
    'random_seed': SEED, 'n_act': N_ACT,
  }

Usage:
  python scripts/perturbation/holdout_1d_sweeps.py --target gemma --layer 2
"""
import os, sys, pickle, argparse, time
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks, forward_from_layer_to_layer

OUT_DIR = 'results/fig4_robustness'

TARGETS = {
    'gemma':   dict(model='gemma-2-9b',      perturb_layer=2),
    'qwen':    dict(model='qwen3-1.7b',      perturb_layer=7),
    'llama':   dict(model='llama-3.1-8b',    perturb_layer=4),
    'mistral': dict(model='mistral-7b-v0.3', perturb_layer=4),
    'aya':     dict(model='aya-expanse-8b',  perturb_layer=4),
    'yi':      dict(model='yi-1.5-9b',       perturb_layer=6),
}

# (direction_name, module path, attribute)
SOURCES = [
    ('Gender',    'data.male_prompts_holdout',     'MALE_PROMPTS_HOLDOUT'),
    ('Sentiment', 'data.positive_prompts_holdout', 'POSITIVE_PROMPTS_HOLDOUT'),
]

N_ACT = 30
N_STEPS = 51        # 0..75° in 1.5° increments
MAX_ANGLE = 75.0
N_RAND = 10
SEED = 42
BATCH = 64


def ts(): return time.strftime('%H:%M:%S')


def extract_activations_from_prompts(model, tokenizer, device, layer, prompts):
    """Return (list of context tensors [1, T-1, D], (n, D) activations) at
    final position, hidden_states[layer+1]. Variable T per prompt."""
    contexts, activations = [], []
    for p in prompts:
        tok = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            outs = model(**tok, output_hidden_states=True)
        h = outs.hidden_states[layer + 1]
        contexts.append(h[:, :-1, :].float().cpu())
        activations.append(h[:, -1, :].squeeze(0).float().cpu())
    return contexts, torch.stack(activations)


def sweep_1d(model, ctx_d, a_dev, d_dev, layer, measure_layer,
             angles_rad, model_dtype, device):
    R = a_dev.norm(); a_unit = a_dev / R
    d_o = F.normalize(d_dev - (d_dev @ a_unit) * a_unit, dim=0)
    s = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)
    c = torch.sqrt(torch.clamp(1.0 - s**2, min=0.0))
    pts = (R * (c.unsqueeze(-1) * a_unit.unsqueeze(0)
                + s.unsqueeze(-1) * d_o.unsqueeze(0))).to(model_dtype)
    n = len(angles_rad)
    out = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        ref = forward_from_layer_to_layer(
            model, ctx_d, a_dev[None, None].to(model_dtype),
            layer, measure_layer).float()
        for st in range(0, n, BATCH):
            ed = min(st + BATCH, n)
            pf = forward_from_layer_to_layer(
                model, ctx_d, pts[st:ed].unsqueeze(1),
                layer, measure_layer).float()
            out[st:ed] = (pf - ref).norm(dim=-1).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(TARGETS))
    ap.add_argument('--layer', type=int, default=None,
                    help='Override perturb layer (default: per-model)')
    args = ap.parse_args()
    cfg = TARGETS[args.target]
    perturb_layer = args.layer if args.layer is not None else cfg['perturb_layer']

    suffix = f'_L{perturb_layer}'
    out_path = os.path.join(OUT_DIR,
                            f'v2_M1d_{args.target}{suffix}.pkl')
    if os.path.exists(out_path):
        print(f'{out_path} exists, exiting'); return

    print(f'[{ts()}] loading {cfg["model"]}', flush=True)
    model, tokenizer, device = load_model(cfg['model'], dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    model_dtype = next(model.parameters()).dtype
    print(f'  n_layers={n_layers}  perturb={perturb_layer}  measure={measure_layer}',
          flush=True)

    # Direction cache (will build if missing)
    dir_pkl = os.path.join(OUT_DIR,
                           f'directions_{args.target}_L{perturb_layer}.pkl')
    if not os.path.exists(dir_pkl):
        from scripts.perturbation.fig4_robustness_v2 import extract_all_directions
        print(f'[{ts()}] building direction cache → {dir_pkl}', flush=True)
        all_dirs = extract_all_directions(model, tokenizer, device,
                                           perturb_layer, dir_pkl)
    else:
        with open(dir_pkl, 'rb') as f:
            all_dirs = pickle.load(f)['directions']

    angles = np.linspace(0, MAX_ANGLE, N_STEPS)
    angles_rad = np.deg2rad(angles)
    rng = np.random.RandomState(SEED)

    sources_out = {}
    for dir_name, mod_path, attr in SOURCES:
        if dir_name not in all_dirs:
            print(f'  SKIP {dir_name}: not in dirs cache'); continue
        import importlib
        mod = importlib.import_module(mod_path)
        prompts = getattr(mod, attr)[:N_ACT]
        print(f'[{ts()}] extracting {len(prompts)} holdout acts for {dir_name}',
              flush=True)
        contexts, activations = extract_activations_from_prompts(
            model, tokenizer, device, perturb_layer, prompts)
        d_model = activations.shape[-1]
        d_cpu = F.normalize(all_dirs[dir_name].float(), dim=0)
        d_dev = d_cpu.to(device)

        l2_dir = np.zeros((N_ACT, N_STEPS), dtype=np.float32)
        t0 = time.time()
        for ai in range(N_ACT):
            ctx_d = contexts[ai].to(device=device, dtype=model_dtype)
            a_dev = activations[ai].to(device=device, dtype=torch.float32)
            l2_dir[ai] = sweep_1d(model, ctx_d, a_dev, d_dev, perturb_layer,
                                   measure_layer, angles_rad, model_dtype, device)
        print(f'  [{ts()}] dir sweep  {time.time()-t0:.1f}s', flush=True)

        rand_dirs = F.normalize(
            torch.tensor(rng.randn(N_RAND, d_model), dtype=torch.float32), dim=-1)
        l2_rand = np.zeros((N_RAND, N_ACT, N_STEPS), dtype=np.float32)
        t0 = time.time()
        for ri in range(N_RAND):
            d_dev_r = rand_dirs[ri].to(device)
            for ai in range(N_ACT):
                ctx_d = contexts[ai].to(device=device, dtype=model_dtype)
                a_dev = activations[ai].to(device=device, dtype=torch.float32)
                l2_rand[ri, ai] = sweep_1d(model, ctx_d, a_dev, d_dev_r,
                                            perturb_layer, measure_layer,
                                            angles_rad, model_dtype, device)
        print(f'  [{ts()}] random sweeps {time.time()-t0:.1f}s', flush=True)

        sources_out[dir_name] = {
            'l2_dir': l2_dir,
            'l2_rand': l2_rand,
            'holdout_module': mod_path,
            'holdout_attr': attr,
        }

    out = {
        'angles_deg': angles,
        'layer': perturb_layer,
        'measure_layer': measure_layer,
        'model': args.target,
        'sources': sources_out,
        'random_seed': SEED,
        'n_act': N_ACT,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, 'wb') as f: pickle.dump(out, f)
    print(f'[{ts()}] saved → {out_path}', flush=True)


if __name__ == '__main__':
    main()
