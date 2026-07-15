#!/usr/bin/env python3
"""
Cheap proxy-p evaluation of a trained SAE's decoder atoms.

Loads an SAE checkpoint, samples random PAIRS of decoder atoms, and computes the
single-angle superellipse surrogate L = R_mix/(R_d + R_c) for each pair over a
set of anchor activations (perturb at `layer`, measure L2 at `measure_layer`).

L ≈ 0.5  -> no interference (p≈2);  L < 0.5 -> super-additive (p>2, what we want).
So a *lower* L distribution = higher-p directions. Compare a culled SAE vs a
control by running this on both and looking at the L distribution (median AND the
high-L tail, which is where culling should bite).

This is the FAST screen. If positive, follow up with the full superellipse fit.

Usage:
    python -u scripts/dictionary_learning/eval_sae_proxy_p.py \
        --sae results/trained_sae/l10_doubled_k32_2M/sae_weights.pt \
        --model Qwen/Qwen3-1.7B --layer 10 \
        --n_pairs 4000 --n_anchor 32 --out results/pcull_eval/baseline_L.pkl
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from src.autoencoders import load_sae

# Reuse anchor builder + scoring geometry from the trainer.
import importlib.util
_p = os.path.join(os.path.dirname(__file__), 'train_sae_pcull.py')
_spec = importlib.util.spec_from_file_location('pcull', _p)
pcull = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pcull)


@torch.no_grad()
def eval_pairs(model, sae, pair_a, pair_b, anchors, layer, measure_layer,
               angle_deg, chunk_pairs=2048):
    """L per pair (mean over anchors). pair_a, pair_b: [P] long. Returns L [P]."""
    import math
    device = sae.W_dec.device
    W = F.normalize(sae.W_dec.detach(), dim=-1)
    D = W[pair_a]
    C = W[pair_b]
    cos_t = math.cos(math.radians(angle_deg))
    sin_t = math.sin(math.radians(angle_deg))
    L_sum = torch.zeros(pair_a.shape[0], device=device)
    for a, ctx, f_ref in zip(*anchors):
        a_norm = a.norm()
        a_hat = a / (a_norm + 1e-10)
        L = pcull._pair_ratios_one_anchor(
            model, a, a_hat, a_norm, ctx, f_ref, D, C,
            cos_t, sin_t, layer, measure_layer, chunk_pairs)
        L_sum += L
    return (L_sum / len(anchors[0])).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sae', required=True, help='Path to sae_weights.pt')
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--layer', type=int, default=10)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--n_pairs', type=int, default=4000)
    ap.add_argument('--n_anchor', type=int, default=32)
    ap.add_argument('--angle', type=float, default=30.0)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--anchor_seed', type=int, default=999)
    ap.add_argument('--pair_seed', type=int, default=0)
    ap.add_argument('--chunk_pairs', type=int, default=2048)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model, tokenizer, device = load_model(args.model)
    n_layers = len(_get_blocks(model))
    measure_layer = args.measure_layer if args.measure_layer is not None else n_layers - 2
    print(f"model={args.model} layer={args.layer} measure_layer={measure_layer}")

    sae = load_sae(args.sae, device)
    d_sae = sae.d_sae
    print(f"SAE d_sae={d_sae}, k={sae.k}")

    anchors = pcull.build_anchor_pool(
        model, tokenizer, args.layer, args.n_anchor, args.seq_len,
        args.anchor_seed, device, measure_layer)

    g = torch.Generator().manual_seed(args.pair_seed)
    pair_a = torch.randint(0, d_sae, (args.n_pairs,), generator=g)
    pair_b = torch.randint(0, d_sae, (args.n_pairs,), generator=g)
    keep = pair_a != pair_b
    pair_a, pair_b = pair_a[keep].to(device), pair_b[keep].to(device)

    L = eval_pairs(model, sae, pair_a, pair_b, anchors, args.layer,
                   measure_layer, args.angle, args.chunk_pairs).numpy()

    q = np.quantile(L, [0.1, 0.25, 0.5, 0.75, 0.9])
    summary = {
        'sae': args.sae, 'n_pairs': int(L.shape[0]), 'n_anchor': args.n_anchor,
        'angle': args.angle, 'layer': args.layer, 'measure_layer': measure_layer,
        'L_mean': float(L.mean()), 'L_median': float(q[2]),
        'L_p10': float(q[0]), 'L_p25': float(q[1]), 'L_p75': float(q[3]),
        'L_p90': float(q[4]), 'frac_super': float((L < 0.5).mean()),
    }
    with open(args.out, 'wb') as f:
        pickle.dump({'L': L, 'summary': summary, 'args': vars(args)}, f)
    print("\n=== proxy-p summary ===")
    for kk, vv in summary.items():
        print(f"  {kk}: {vv}")
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
