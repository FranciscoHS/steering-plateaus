#!/usr/bin/env python3
"""1D perturbation sweeps to 75° for all non-natural-language contrastive
directions, plus random baseline.

Loads the cached `directions_<target>_L<layer>.pkl` (built by
fig4_robustness_v2.py:extract_all_directions). Filters out natural-language
directions (Portuguese, Chinese, ..., Arabic) and runs a 1D spherical sweep
along each remaining direction (SEM + programming) for N_ACT FineWeb contexts.
Also runs N_RAND random unit vectors as baseline.

Output: results/fig4_robustness/v2_C1d75_<target>.pkl
  {
    'angles_deg': (n_steps,),
    'layer': perturb_layer, 'measure_layer': measure_layer,
    'directions': { name: l2 (n_act, n_steps) },     # contrastive
    'random':     l2 (n_rand, n_act, n_steps),       # random baseline
    'random_seed': SEED, 'n_act': N_ACT,
  }

Usage:
  python scripts/perturbation/extended_1d_sweeps_75deg.py --target aya
  python scripts/perturbation/extended_1d_sweeps_75deg.py --target llama
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
NATURAL_LANGS = {'English', 'Portuguese', 'Chinese', 'Spanish', 'French',
                 'German', 'Dutch', 'Italian', 'Japanese', 'Russian', 'Arabic'}

N_ACT = 30
SEQ_LEN = 5
SEED = 42
N_STEPS = 51        # 0..75° in 1.5° increments
MAX_ANGLE = 75.0
N_RAND = 10
BATCH = 64


def ts(): return time.strftime('%H:%M:%S')


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


def sweep_1d(model, ctx_d, a_dev, d_dev, layer, measure_layer,
             angles_rad, model_dtype, device):
    """Spherical 1D sweep at one (context, activation) pair along one direction.
    Returns L² (n_steps,)."""
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
                    help='Override perturb layer (default: per-model from TARGETS)')
    args = ap.parse_args()
    cfg = TARGETS[args.target]

    perturb_layer = args.layer if args.layer is not None else cfg['perturb_layer']
    suffix = f'_L{perturb_layer}' if args.layer is not None else ''
    out_path = os.path.join(OUT_DIR, f'v2_C1d75_{args.target}{suffix}.pkl')
    if os.path.exists(out_path):
        print(f'{out_path} exists, exiting'); return

    print(f'[{ts()}] loading {cfg["model"]}', flush=True)
    model, tokenizer, device = load_model(cfg['model'], dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    print(f'  n_layers={n_layers}  perturb={perturb_layer}  measure={measure_layer}',
          flush=True)

    # Load (or build) direction cache
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
    contrastive = {k: v for k, v in all_dirs.items() if k not in NATURAL_LANGS}
    print(f'[{ts()}] {len(contrastive)} non-language dirs: {list(contrastive)}',
          flush=True)

    # FineWeb contexts + activations at perturb_layer
    print(f'[{ts()}] extracting {N_ACT} FineWeb activations', flush=True)
    contexts, activations = fineweb_ctx_acts(
        model, tokenizer, device, perturb_layer, args.target)
    print(f'  mean act norm: {activations.norm(dim=-1).mean():.2f}',
          flush=True)

    angles = np.linspace(0, MAX_ANGLE, N_STEPS)
    angles_rad = np.deg2rad(angles)
    model_dtype = next(model.parameters()).dtype

    # Contrastive sweeps
    out_dirs = {}
    for di, (name, d) in enumerate(contrastive.items()):
        d_cpu = F.normalize(d.float(), dim=0)
        l2 = np.zeros((N_ACT, N_STEPS), dtype=np.float32)
        t0 = time.time()
        for ai in range(N_ACT):
            ctx_d = contexts[ai].to(device=device, dtype=model_dtype)
            a_dev = activations[ai].to(device=device, dtype=torch.float32)
            d_dev = d_cpu.to(device)
            l2[ai] = sweep_1d(model, ctx_d, a_dev, d_dev, perturb_layer,
                              measure_layer, angles_rad, model_dtype, device)
        out_dirs[name] = l2
        print(f'  [{ts()}] {di+1}/{len(contrastive)} {name:12s}  {time.time()-t0:.1f}s',
              flush=True)

    # Random baseline
    rng = np.random.RandomState(SEED)
    d_model = activations.shape[-1]
    rand_dirs = F.normalize(
        torch.tensor(rng.randn(N_RAND, d_model), dtype=torch.float32), dim=-1)
    rand_l2 = np.zeros((N_RAND, N_ACT, N_STEPS), dtype=np.float32)
    for ri in range(N_RAND):
        d_dev = rand_dirs[ri].to(device)
        t0 = time.time()
        for ai in range(N_ACT):
            ctx_d = contexts[ai].to(device=device, dtype=model_dtype)
            a_dev = activations[ai].to(device=device, dtype=torch.float32)
            rand_l2[ri, ai] = sweep_1d(model, ctx_d, a_dev, d_dev,
                                        perturb_layer, measure_layer,
                                        angles_rad, model_dtype, device)
        print(f'  [{ts()}] rand {ri+1}/{N_RAND}  {time.time()-t0:.1f}s',
              flush=True)

    out = {
        'angles_deg': angles,
        'layer': perturb_layer,
        'measure_layer': measure_layer,
        'model': args.target,
        'directions': out_dirs,
        'random': rand_l2,
        'random_seed': SEED,
        'n_act': N_ACT,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, 'wb') as f: pickle.dump(out, f)
    print(f'[{ts()}] saved → {out_path}', flush=True)


if __name__ == '__main__':
    main()
