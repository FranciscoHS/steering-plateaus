#!/usr/bin/env python3
"""1D random-direction perturbation sweeps: Jacobian-optimized points vs real.

For each anchor — the freely Jacobian-optimized Gaussian points from
intrinsic_dim_jac_optimized.py, and real activations on the same carrier
contexts — perturb along random unit directions in the exp-map chart and
record L2 distance at the measurement layer vs angle. One combined plot:
optimized anchors in one color, real anchors in another (per-anchor median
over random directions).

Contexts are the pod's current FineWeb-Edu stream (same recipe as the
optimization run, but note the stream order differs from the April run, so
the optimized points ride on different contexts than they were optimized
for — same caveat as the full-circle heatmaps).

Usage:
    python scripts/jacobian/sweep_1d_jacopt_vs_real.py \\
        --out results/fullcircle_heatmaps/jacopt_L2_1d
"""
import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks
from scripts.superellipse.canonical.sweep_core import sweep_1d_one_anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opt_data', default='results/intrinsic_dim/jac_optimized_data.pt')
    ap.add_argument('--source_layer', type=int, default=2)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--n_anchors', type=int, default=5)
    ap.add_argument('--n_dirs', type=int, default=5,
                    help='random directions per anchor')
    ap.add_argument('--dir_seed', type=int, default=0)
    ap.add_argument('--max_angle', type=float, default=90.0)
    ap.add_argument('--n_steps', type=int, default=61)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    opt = torch.load(args.opt_data, map_location='cpu', weights_only=False)
    X_opt = opt['X_opt'].float()

    print('Loading Qwen/Qwen3-1.7B (float32)', flush=True)
    model, tokenizer, device = load_model('Qwen/Qwen3-1.7B',
                                          dtype=torch.float32)
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    d_model = X_opt.shape[1]
    print(f'Perturb L{args.source_layer} -> measure L{measure_layer}',
          flush=True)

    print('Streaming FineWeb-Edu for contexts...', flush=True)
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceFW/FineWeb-Edu', 'sample-10BT',
                      split='train', streaming=True)
    ds = ds.shuffle(seed=42)
    texts = []
    for ex in ds:
        if len(ex['text']) >= args.seq_len * 3:
            texts.append(ex['text'])
        if len(texts) >= args.n_anchors:
            break
    tokens = tokenizer(texts, return_tensors='pt', padding='max_length',
                       truncation=True, max_length=args.seq_len).to(device)
    with torch.no_grad():
        out = model(**tokens, output_hidden_states=True)
    hidden = out.hidden_states[args.source_layer + 1]
    contexts = hidden[:, :-1, :].float().cpu()
    real_acts = hidden[:, -1, :].float().cpu()

    g = torch.Generator().manual_seed(args.dir_seed)
    dirs = torch.randn(args.n_dirs, d_model, generator=g)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    angles_deg = np.linspace(-args.max_angle, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)

    model_dtype = next(model.parameters()).dtype
    curves = {'opt': [], 'real': []}   # [n_anchors, n_dirs, n_steps]
    for name, acts in [('opt', X_opt[:args.n_anchors]),
                       ('real', real_acts)]:
        for i in range(args.n_anchors):
            ctx = contexts[i:i + 1].to(device=device, dtype=model_dtype)
            a = acts[i].to(device)
            per_dir = []
            t0 = time.time()
            with torch.no_grad():
                for d in dirs:
                    l2 = sweep_1d_one_anchor(
                        model, ctx, a, d.to(device), args.source_layer,
                        measure_layer, device, angles_rad)
                    per_dir.append(l2)
            curves[name].append(np.stack(per_dir))
            print(f'{name} anchor{i}: ||a||={a.norm():.1f} '
                  f'{time.time() - t0:.0f}s', flush=True)
        curves[name] = np.stack(curves[name])

    with open(os.path.join(args.out, 'sweep1d_curves.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'angles_deg': angles_deg,
                     'curves': curves}, f)
    print(f"saved {os.path.join(args.out, 'sweep1d_curves.pkl')}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {'opt': 'tab:red', 'real': 'tab:blue'}
    labels = {'opt': 'Jacobian-optimized point', 'real': 'real activation'}
    for name in ['real', 'opt']:
        med = np.median(curves[name], axis=1)   # [n_anchors, n_steps]
        for i in range(med.shape[0]):
            ax.plot(angles_deg, med[i], color=colors[name], alpha=0.5, lw=1.2,
                    label=labels[name] if i == 0 else None)
    ax.set_xlabel('perturbation angle (deg)')
    ax.set_ylabel(f'L2 distance at layer {measure_layer}')
    ax.set_title(f'1D random-direction sweeps, qwen3-1.7b '
                 f'L{args.source_layer}->{measure_layer}\n'
                 f'{args.n_anchors} anchors x median over {args.n_dirs} '
                 f'random dirs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(args.out, 'sweep1d_opt_vs_real.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'saved {path}', flush=True)


if __name__ == '__main__':
    main()
