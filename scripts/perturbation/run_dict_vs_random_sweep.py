#!/usr/bin/env python3
"""Perturbation sweep: high-p dictionary atoms vs random directions, at the layer the
atoms were optimized at (L14), out to a wide angle.

Reuses the sweep machinery from run_feature_vs_random_sweep.py but takes its
"named" directions from a dictionary.pt (the soft_p / high-responsiveness dictionary)
instead of a handful of contrastive pkls. Output format matches that script so the
shape-collapse plot reads it directly:

    angles_deg : [n_angles]
    l2['random']   : [n_act, n_random_dirs, n_angles]
    l2['atom{j}']  : [n_act, n_angles]   (one per sampled dictionary atom)
    (kl, cos stored the same way)

Usage (on the GPU pod):
    python -u scripts/perturbation/run_dict_vs_random_sweep.py \
        --model qwen3-1.7b --layer 14 \
        --dict_path results/dictionary/qwen3-1.7b_L14_K20_randcontrast_stepresample_softp/dictionary.pt \
        --n_act 50 --n_random_dirs 10 --max_angle 90 --n_angles 60 \
        --results_dir results/dict_vs_random_sweep/qwen3-1.7b_L14_softp
"""
import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import torch

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.perturbation.run_feature_vs_random_sweep import (
    project_orthogonal, sweep_single_direction, sweep_random_directions,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--n_atoms', type=int, default=None,
                    help='number of dictionary atoms to sweep (default: all)')
    ap.add_argument('--atom_seed', type=int, default=0,
                    help='if subsampling atoms, RNG seed for the choice')
    ap.add_argument('--n_act', type=int, default=50)
    ap.add_argument('--n_random_dirs', type=int, default=10)
    ap.add_argument('--max_angle', type=float, default=90.0)
    ap.add_argument('--n_angles', type=int, default=60)
    ap.add_argument('--seq_len', type=int, default=128)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--results_dir', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/dict_vs_random_sweep/{args.model}_L{args.layer}'
    os.makedirs(args.results_dir, exist_ok=True)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)
    n_angles = len(angles_deg)

    print(f"Loading model: {args.model}", flush=True)
    model, tokenizer, device = load_model(args.model)
    if args.measure_layer is not None and args.measure_layer < 0:
        args.measure_layer = len(_get_blocks(model)) + args.measure_layer

    # --- dictionary atoms ---
    blob = torch.load(args.dict_path, map_location='cpu', weights_only=False)
    D = blob['dictionary'].float()
    D = D / D.norm(dim=-1, keepdim=True)
    K = D.shape[0]
    if args.n_atoms is not None and args.n_atoms < K:
        g = torch.Generator().manual_seed(args.atom_seed)
        idx = torch.randperm(K, generator=g)[:args.n_atoms].tolist()
    else:
        idx = list(range(K))
    atoms = {f'atom{j}': D[j] for j in idx}
    print(f"Dictionary {args.dict_path}: {K} atoms, sweeping {len(atoms)} of them", flush=True)
    print(f"Config: layer={args.layer}, n_act={args.n_act}, n_random={args.n_random_dirs}, "
          f"angles=[0,{args.max_angle}] ({n_angles} pts)", flush=True)

    # --- activations at `layer` ---
    print(f"Loading {args.n_act} FineWeb sequences (seq_len={args.seq_len})...", flush=True)
    token_ids_list = load_fineweb_fixed_length(args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    all_contexts, all_activations = [], []
    bs = 4
    for i in range(0, len(token_ids_list), bs):
        batch = torch.stack(token_ids_list[i:i + bs]).to(device)
        with torch.no_grad():
            out = model(batch, output_hidden_states=True)
        hidden = out.hidden_states[args.layer + 1]
        all_contexts.append(hidden[:, :-1, :].float().cpu())
        all_activations.append(hidden[:, -1, :].float().cpu())
    contexts = torch.cat(all_contexts, 0)
    activations = torch.cat(all_activations, 0)
    print(f"Activations: {tuple(activations.shape)}, mean norm {activations.norm(dim=-1).mean():.2f}", flush=True)

    # --- storage ---
    res_l2 = {'random': np.zeros((args.n_act, args.n_random_dirs, n_angles))}
    res_kl = {'random': np.zeros((args.n_act, args.n_random_dirs, n_angles))}
    res_cos = {'random': np.zeros((args.n_act, args.n_random_dirs, n_angles))}
    for name in atoms:
        res_l2[name] = np.zeros((args.n_act, n_angles))
        res_kl[name] = np.zeros((args.n_act, n_angles))
        res_cos[name] = np.zeros((args.n_act, n_angles))

    ckpt_path = os.path.join(args.results_dir, 'checkpoint.pkl')

    def save(tag):
        out = {'args': vars(args), 'angles_deg': angles_deg, 'atom_idx': idx,
               'l2': res_l2, 'kl': res_kl, 'cos': res_cos, 'done_acts': tag}
        with open(ckpt_path, 'wb') as f:
            pickle.dump(out, f)

    t0 = time.time()
    for i in range(args.n_act):
        rk, rl, rc = sweep_random_directions(
            model, contexts[i], activations[i], args.n_random_dirs,
            angles_rad, args.layer, device, measure_layer=args.measure_layer)
        res_kl['random'][i], res_l2['random'][i], res_cos['random'][i] = rk, rl, rc
        for name, d in atoms.items():
            vp = project_orthogonal(d, activations[i])
            k, l, c = sweep_single_direction(
                model, contexts[i], activations[i], vp,
                angles_rad, args.layer, device, measure_layer=args.measure_layer)
            res_kl[name][i], res_l2[name][i], res_cos[name][i] = k, l, c
        if (i + 1) % 5 == 0 or i == args.n_act - 1:
            save(i + 1)
            print(f"  act {i+1}/{args.n_act}  ({time.time()-t0:.0f}s)", flush=True)

    final_path = os.path.join(args.results_dir, 'dict_vs_random_sweep.pkl')
    with open(final_path, 'wb') as f:
        pickle.dump({'args': vars(args), 'angles_deg': angles_deg, 'atom_idx': idx,
                     'l2': res_l2, 'kl': res_kl, 'cos': res_cos}, f)
    print(f"saved: {final_path}", flush=True)


if __name__ == '__main__':
    main()
