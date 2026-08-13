#!/usr/bin/env python3
"""Slerp interpolation from real activations to jac-opt points, 50 pairs.

Scales interp_real_to_jacopt6.py up: for i = 0..49, A = the real last-token
activation of FineWeb sample i (seq_len 5, seed 42), B = jac-opt point i
(the point whose optimization carrier was paired with that sample index in
the sweeps). Context: sample i's own context. Metric: relative logit
distance d_A/(d_A+d_B), as in interp_jacopt_pairs.py.

Outputs (results/intrinsic_dim/interp_real_to_jacopt/):
  - results.pkl                combined arrays for all pairs
  - rel_dist.png               all curves + median, like the jac-opt pair plot
  - pair_XX.png                individual per-pair plots, like real6_to_jacopt6
"""

import os
import pickle
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/steering-plateaus')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer
from src.data import load_fineweb_fixed_length

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interp_jacopt_pairs import slerp_point

OUT_DIR = 'results/intrinsic_dim/interp_real_to_jacopt'
MODEL = 'qwen3-1.7b'
LAYER = 2
SEQ_LEN = 5
N_PAIRS = 50
N_T = 41


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Loading {MODEL}...', flush=True)
    model, tokenizer, device = load_model(MODEL, dtype=torch.bfloat16)
    model_dtype = next(model.parameters()).dtype

    opt = torch.load('results/intrinsic_dim/jac_optimized_data.pt',
                     map_location='cpu', weights_only=False)
    X_opt = opt['X_opt'].float()[:N_PAIRS]

    token_lists = load_fineweb_fixed_length(
        N_PAIRS, tokenizer, seq_len=SEQ_LEN, seed=42,
        cache_dir='/workspace/fineweb_cache')

    ts = np.linspace(0.0, 1.0, N_T)
    rel = np.zeros((N_PAIRS, N_T), dtype=np.float32)
    abs_dA = np.zeros((N_PAIRS, N_T), dtype=np.float32)
    end_dist = np.zeros(N_PAIRS, dtype=np.float32)
    act_dist = np.zeros(N_PAIRS, dtype=np.float32)
    cos_ab = np.zeros(N_PAIRS, dtype=np.float32)

    for i in range(N_PAIRS):
        tids = token_lists[i].unsqueeze(0).to(device)
        with torch.no_grad():
            outs = model(tids, output_hidden_states=True)
        h = outs.hidden_states[LAYER + 1]
        ctx = h[:, :-1, :].to(model_dtype)
        a_real = h[:, -1, :].squeeze(0).float()
        b_opt = X_opt[i].to(device)
        act_dist[i] = (b_opt - a_real).norm().item()
        cos_ab[i] = torch.nn.functional.cosine_similarity(
            a_real, b_opt, dim=0).item()

        pts = torch.stack([slerp_point(a_real, b_opt, float(t)) for t in ts])
        with torch.no_grad():
            logits = forward_from_layer(
                model, ctx, pts.unsqueeze(1).to(model_dtype), LAYER).float()
        fA, fB = logits[0], logits[-1]
        dA = (logits - fA).norm(dim=-1)
        dB = (logits - fB).norm(dim=-1)
        rel[i] = (dA / (dA + dB + 1e-9)).cpu().numpy()
        abs_dA[i] = dA.cpu().numpy()
        end_dist[i] = (fB - fA).norm().item()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ts, rel[i], color='C0', lw=2)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('interpolation t (slerp)')
        ax.set_ylabel(r'$d_A/(d_A+d_B)$  (logits)')
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'real act {i} -> jac-opt {i}  '
                     f'(same context, endpoint logit dist {end_dist[i]:.0f})')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'pair_{i:02d}.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)
        if (i + 1) % 10 == 0:
            print(f'  pair {i+1}/{N_PAIRS}', flush=True)

    with open(os.path.join(OUT_DIR, 'results.pkl'), 'wb') as f:
        pickle.dump({'t': ts, 'rel_dist': rel, 'abs_dist_from_A': abs_dA,
                     'endpoint_dist': end_dist, 'act_dist': act_dist,
                     'cos_ab': cos_ab, 'model': MODEL, 'layer': LAYER}, f)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(N_PAIRS):
        ax.plot(ts, rel[i], color='C0', alpha=0.15, lw=1)
    ax.plot(ts, np.median(rel, axis=0), color='C0', lw=2.5,
            label=f'median ({N_PAIRS} real->jac-opt pairs)')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='linear reference')
    ax.set_xlabel('interpolation t (slerp)')
    ax.set_ylabel(r'relative distance  $d_A/(d_A+d_B)$  (logits)')
    ax.set_title(f'{MODEL} L{LAYER}: slerp real activation -> jac-opt point '
                 f'(matched index/context)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'rel_dist.png'), dpi=150,
                bbox_inches='tight')
    print(f"Saved {os.path.join(OUT_DIR, 'results.pkl')}")
    print(f"Saved {os.path.join(OUT_DIR, 'rel_dist.png')}")


if __name__ == '__main__':
    main()
