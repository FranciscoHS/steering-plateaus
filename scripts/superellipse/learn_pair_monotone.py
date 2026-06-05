#!/usr/bin/env python3
"""
Learn a superellipse pair (d1, d2) with a monotonicity / inside-the-box penalty.

Same reward as learn_superellipse_dict.py stage 1 -- minimize the equal-mix
ratio R_mix/(R1+R2) -- but with an added penalty that forbids the optimizer
from reaching R_mix~0 by *cancellation* (points outside the per-feature-threshold
square), which is beyond the L^p model and breaks the superellipse fit.

The model R = F(sum_i w_i |f_i.v|^p) is a sum of non-negative terms, so adding
a component cannot reduce the response: every iso-response point lies at or
inside the square. Concretely, the mix at angle alpha has the same projection
onto d1 as a pure-d1 perturbation at alpha' (sin alpha' = sin alpha / sqrt(2)),
plus a d2 component, so the model requires
    R_mix(alpha) >= max(R1(alpha'), R2(alpha')).
We penalize the dimensionless shortfall
    L_box = mean ReLU(1 - R_mix(alpha) / max(R1(alpha'), R2(alpha'))).

Total loss = mean[R_mix/(R1+R2)] + lam_box * L_box + lam_orth * (d1.d2)^2.

Usage:
    python scripts/superellipse/learn_pair_monotone.py \
        --model gemma-2-9b --layer 15 --n_act 10 --angles 30 \
        --lam_box 1.0 --lam_orth 1.0 --dtype bfloat16
"""

import argparse
import math
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from src.data import load_fineweb_fixed_length

# Reuse geometry + response helpers verbatim.
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation, pair_responses,
)


def monotone_pair_loss(model, acts, ctxs, f_refs, d1, d2, angles_rad,
                       layer, measure_layer):
    """Mean ratio loss + mean box-violation penalty over anchors & angles.

    Returns (ratio_loss, box_loss, (mR1, mR2, mRmix, mViol)) where mViol is the
    mean raw box shortfall for logging.
    """
    eps = 1e-8
    ratio_sum = 0.0
    box_sum = 0.0
    r1_sum = r2_sum = rmix_sum = 0.0
    count = 0
    for i in range(len(acts)):
        for ar in angles_rad:
            r1, r2, rmix = pair_responses(
                model, acts[i], ctxs[i], f_refs[i], d1, d2,
                ar, layer, measure_layer)
            # matched-projection angle: sin(a') = sin(a)/sqrt(2)
            ar_p = math.asin(math.sin(ar) / math.sqrt(2.0))
            r1p, r2p, _ = pair_responses(
                model, acts[i], ctxs[i], f_refs[i], d1, d2,
                ar_p, layer, measure_layer)
            ratio = rmix / (r1 + r2 + eps)
            r_ax = torch.maximum(r1p, r2p)
            box = torch.relu(1.0 - rmix / (r_ax + eps))
            ratio_sum = ratio_sum + ratio
            box_sum = box_sum + box
            r1_sum += r1.item(); r2_sum += r2.item(); rmix_sum += rmix.item()
            count += 1
    ratio_loss = ratio_sum / count
    box_loss = box_sum / count
    return ratio_loss, box_loss, (r1_sum / count, r2_sum / count,
                                  rmix_sum / count, box_loss.item())


def optimize_pair(model, acts, ctxs, f_refs, angles_rad, layer, measure_layer,
                  device, d_model, n_steps, lr, lam_box, lam_orth, seed):
    """Optimize (d1, d2) jointly on the sphere with the monotone box penalty."""
    g = torch.Generator(device=device).manual_seed(seed)
    new = [torch.randn(d_model, generator=g, device=device) for _ in range(2)]
    new = [(v / v.norm()).requires_grad_(True) for v in new]
    optimizer = torch.optim.Adam(new, lr=lr)

    history = []
    for step in range(n_steps):
        optimizer.zero_grad()
        d1, d2 = new
        ratio_loss, box_loss, (mr1, mr2, mrmix, mviol) = monotone_pair_loss(
            model, acts, ctxs, f_refs, d1, d2, angles_rad, layer, measure_layer)
        d1u = d1 / (d1.norm() + 1e-10)
        d2u = d2 / (d2.norm() + 1e-10)
        orth = (d1u @ d2u) ** 2
        loss = ratio_loss + lam_box * box_loss + lam_orth * orth
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for v in new:
                v.data.div_(v.data.norm())

        history.append({
            'step': step, 'loss': loss.item(), 'ratio': ratio_loss.item(),
            'box': box_loss.item(), 'orth': orth.item(),
            'R1': mr1, 'R2': mr2, 'Rmix': mrmix,
        })
        if step % 10 == 0 or step == n_steps - 1:
            print(f"  step {step:4d}  L={loss.item():.4f}  ratio={ratio_loss.item():.4f}  "
                  f"box={box_loss.item():.4f}  orth={orth.item():.4f}  "
                  f"R1={mr1:.1f} R2={mr2:.1f} Rmix={mrmix:.1f}")

    pair = torch.stack([v.detach() for v in new])
    return pair, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gemma-2-9b')
    parser.add_argument('--layer', type=int, default=15)
    parser.add_argument('--measure_layer', type=int, default=None)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_act', type=int, default=10)
    parser.add_argument('--angles', type=float, nargs='+', default=[30.0])
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--lam_box', type=float, default=1.0,
                        help='Weight of the monotonicity (inside-the-box) penalty')
    parser.add_argument('--lam_orth', type=float, default=1.0,
                        help='Weight of the (d1.d2)^2 orthogonality penalty')
    parser.add_argument('--data_source', type=str, default='fineweb')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--init_seed', type=int, default=None,
                        help='Separate seed for direction init (default: --seed)')
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    init_seed = args.init_seed if args.init_seed is not None else args.seed

    angle_tag = '-'.join(f'{a:g}' for a in args.angles)
    if args.results_dir is None:
        args.results_dir = (f'results/pair_monotone/'
                            f'{args.model}_L{args.layer}_a{angle_tag}_box{args.lam_box:g}')
    os.makedirs(args.results_dir, exist_ok=True)

    angles_rad = [np.deg2rad(a) for a in args.angles]

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    measure_layer = args.measure_layer if args.measure_layer is not None else n_layers - 2
    print(f"Device: {device}, d_model: {d_model}, n_layers: {n_layers}, "
          f"perturb_layer: {args.layer}, measure_layer: {measure_layer}")
    print(f"Config: n_act={args.n_act}, angles={args.angles} deg, n_steps={args.n_steps}, "
          f"lr={args.lr}, lam_box={args.lam_box}, lam_orth={args.lam_orth}, "
          f"init_seed={init_seed}")

    print(f"Loading {args.n_act} {args.data_source} sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.layer} activations...")
    all_contexts, all_activations = extract_context_and_activation(
        model, token_ids_list, args.layer, device)
    model_dtype_actual = next(model.parameters()).dtype
    acts = [all_activations[i].to(device=device, dtype=torch.float32)
            for i in range(args.n_act)]
    ctxs = [all_contexts[i].unsqueeze(0).to(device=device, dtype=model_dtype_actual)
            for i in range(args.n_act)]
    print(f"Mean activation norm: {torch.stack(acts).norm(dim=-1).mean().item():.2f}")

    f_refs = []
    with torch.no_grad():
        for i in range(args.n_act):
            point = acts[i].reshape(1, 1, d_model).to(model_dtype_actual)
            f_ref = forward_from_layer_to_layer(
                model, ctxs[i], point, args.layer, measure_layer).squeeze(0).float()
            f_refs.append(f_ref)

    print(f"\n=== Optimizing pair (d1, d2) with monotonicity penalty ===")
    t0 = time.time()
    pair, history = optimize_pair(
        model, acts, ctxs, f_refs, angles_rad, args.layer, measure_layer,
        device, d_model, args.n_steps, args.lr, args.lam_box, args.lam_orth,
        init_seed)
    print(f"Done in {time.time()-t0:.1f}s")

    cos12 = float((pair[0] @ pair[1]).item())
    print(f"cos(d1, d2) = {cos12:.4f}")
    print(f"Final: ratio={history[-1]['ratio']:.4f}  box={history[-1]['box']:.4f}  "
          f"Rmix={history[-1]['Rmix']:.1f}  R1={history[-1]['R1']:.1f}")

    results = {
        'args': vars(args),
        'directions': pair.cpu(),
        'measure_layer': measure_layer,
        'angles_deg': args.angles,
        'history': history,
        'cos_d1_d2': cos12,
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # plot loss curves
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    steps = [h['step'] for h in history]
    ax[0].plot(steps, [h['loss'] for h in history], label='total')
    ax[0].plot(steps, [h['ratio'] for h in history], '--', label='ratio')
    ax[0].plot(steps, [h['box'] for h in history], ':', label='box penalty')
    ax[0].plot(steps, [h['orth'] for h in history], '-.', label='orth')
    ax[0].axhline(0.5, color='gray', ls=':', alpha=0.6)
    ax[0].set_xlabel('step'); ax[0].set_ylabel('loss'); ax[0].legend(fontsize=8)
    ax[0].set_title('Optimization'); ax[0].grid(alpha=0.3)
    ax[1].plot(steps, [h['R1'] for h in history], label='R1')
    ax[1].plot(steps, [h['R2'] for h in history], label='R2')
    ax[1].plot(steps, [h['Rmix'] for h in history], label='Rmix')
    ax[1].set_xlabel('step'); ax[1].set_ylabel('mean L2 response')
    ax[1].legend(fontsize=8); ax[1].set_title('Responses'); ax[1].grid(alpha=0.3)
    fig.suptitle(f'{args.model} L{args.layer}->{measure_layer}, '
                 f'lam_box={args.lam_box}, cos(d1,d2)={cos12:.3f}')
    plt.tight_layout()
    png = os.path.join(args.results_dir, 'summary.png')
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved results: {os.path.join(args.results_dir, 'results.pkl')}")
    print(f"Saved plot: {png}")


if __name__ == '__main__':
    main()
