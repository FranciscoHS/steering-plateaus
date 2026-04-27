#!/usr/bin/env python3
"""Quick 1D random-direction sweep for a single model.

Generates N_DIRS random unit vectors at the model's perturb_layer, sweeps each
from 0° to MAX_ANGLE, records L² at the measure layer for each (context, angle).
Output: results/fig4_robustness/v2_B1d_<target>.pkl
        with 'l2' shape (n_dirs, n_contexts, n_angles)

Usage:
  python scripts/perturbation/quick_random_baseline_1d.py --target aya
"""
import os, sys, pickle, argparse, time
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks, forward_from_layer_to_layer
from src.data import load_fineweb_fixed_length

OUT_DIR = 'results/fig4_robustness'

TARGETS = {
    'gemma':   dict(model='gemma-2-9b',      perturb_layer=2),
    'qwen':    dict(model='qwen3-1.7b',      perturb_layer=7),
    'llama':   dict(model='llama-3.1-8b',    perturb_layer=4),
    'mistral': dict(model='mistral-7b-v0.3', perturb_layer=4),
    'aya':     dict(model='aya-expanse-8b',  perturb_layer=4),
    'yi':      dict(model='yi-1.5-9b',       perturb_layer=6),
}
N_DIRS = 10
N_ACT = 30
SEQ_LEN = 5
SEED = 42
N_STEPS = 41
MAX_ANGLE = 30.0
BATCH = 64


def ts(): return time.strftime('%H:%M:%S')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(TARGETS))
    args = ap.parse_args()
    cfg = TARGETS[args.target]

    out_path = os.path.join(OUT_DIR, f'v2_B1d_{args.target}.pkl')
    if os.path.exists(out_path):
        print(f'{out_path} exists, exiting'); return

    print(f'[{ts()}] loading {cfg["model"]}')
    model, tokenizer, device = load_model(cfg['model'], dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    perturb_layer = cfg['perturb_layer']
    measure_layer = n_layers - 2
    print(f'  n_layers={n_layers}  perturb_layer={perturb_layer}  measure_layer={measure_layer}')

    # FineWeb activations at perturb_layer
    cache_dir = f'/workspace/fineweb_cache_{args.target}'
    os.makedirs(cache_dir, exist_ok=True)
    token_lists = load_fineweb_fixed_length(
        N_ACT, tokenizer, seq_len=SEQ_LEN, seed=SEED, cache_dir=cache_dir)
    contexts, activations = [], []
    for tids in token_lists:
        tids = tids.unsqueeze(0).to(device)
        with torch.no_grad():
            outs = model(tids, output_hidden_states=True)
        h = outs.hidden_states[perturb_layer + 1]
        contexts.append(h[:, :-1, :].float().cpu())
        activations.append(h[:, -1, :].squeeze(0).float().cpu())
    activations = torch.stack(activations)
    print(f'[{ts()}] {N_ACT} activations, mean norm: {activations.norm(dim=-1).mean():.2f}')

    rng = np.random.RandomState(SEED)
    d_model = activations.shape[-1]
    rand_dirs = F.normalize(torch.tensor(rng.randn(N_DIRS, d_model), dtype=torch.float32), dim=-1)

    angles = np.linspace(0, MAX_ANGLE, N_STEPS)
    angles_rad = np.deg2rad(angles)
    model_dtype = next(model.parameters()).dtype

    out_l2 = np.zeros((N_DIRS, N_ACT, N_STEPS), dtype=np.float32)
    for di in range(N_DIRS):
        d = rand_dirs[di].to(device)
        t0 = time.time()
        for ai in range(N_ACT):
            ctx_d = contexts[ai].to(device=device, dtype=model_dtype)
            a = activations[ai].to(device=device, dtype=torch.float32)
            R = a.norm(); a_unit = a / R
            d_o = F.normalize(d.float() - (d.float() @ a_unit) * a_unit, dim=0)
            s = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)
            c = torch.sqrt(torch.clamp(1.0 - s**2, min=0.0))
            pts = (R * (c.unsqueeze(-1) * a_unit.unsqueeze(0) + s.unsqueeze(-1) * d_o.unsqueeze(0))).to(model_dtype)
            with torch.no_grad():
                ref = forward_from_layer_to_layer(
                    model, ctx_d, a[None,None].to(model_dtype), perturb_layer, measure_layer).float()
                for st in range(0, N_STEPS, BATCH):
                    ed = min(st + BATCH, N_STEPS)
                    pf = forward_from_layer_to_layer(
                        model, ctx_d, pts[st:ed].unsqueeze(1), perturb_layer, measure_layer).float()
                    out_l2[di, ai, st:ed] = (pf - ref).norm(dim=-1).cpu().numpy()
        print(f'  dir {di+1}/{N_DIRS}  {time.time()-t0:.1f}s')

    out = {
        'angles_deg': angles,
        'l2': out_l2,
        'layer': perturb_layer, 'measure_layer': measure_layer,
        'model': args.target, 'n_dirs': N_DIRS, 'n_act': N_ACT,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, 'wb') as f: pickle.dump(out, f)
    print(f'[{ts()}] saved → {out_path}')


if __name__ == '__main__':
    main()
