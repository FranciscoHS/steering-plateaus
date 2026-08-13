#!/usr/bin/env python3
"""Slerp interpolation between pairs of REAL activations, rel-dist metric.

Baseline for interp_jacopt_pairs.py / interp_real_to_jacopt_many.py: same
50 (i, j) pairs from the first 30 anchors (same rng seed), but both
endpoints are real last-token activations of FineWeb samples i and j
(seq_len 5, seed 42, qwen3-1.7b L2). Path runs on endpoint A's context,
metric is the relative logit distance d_A/(d_A+d_B).

Output: results/intrinsic_dim/interp_real_pairs/{results.pkl, rel_dist.png}
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

OUT_DIR = 'results/intrinsic_dim/interp_real_pairs'
MODEL = 'qwen3-1.7b'
LAYER = 2
SEQ_LEN = 5
N_ANCHORS = 30
N_PAIRS = 50
N_T = 41
SEED = 0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Loading {MODEL}...', flush=True)
    model, tokenizer, device = load_model(MODEL, dtype=torch.bfloat16)
    model_dtype = next(model.parameters()).dtype

    token_lists = load_fineweb_fixed_length(
        N_ANCHORS, tokenizer, seq_len=SEQ_LEN, seed=42,
        cache_dir='/workspace/fineweb_cache')
    contexts, acts = [], []
    for tids in token_lists:
        with torch.no_grad():
            outs = model(tids.unsqueeze(0).to(device),
                         output_hidden_states=True)
        h = outs.hidden_states[LAYER + 1]
        contexts.append(h[:, :-1, :].cpu())
        acts.append(h[:, -1, :].squeeze(0).float().cpu())
    acts = torch.stack(acts)

    # identical pair selection to interp_jacopt_pairs.py
    rng = np.random.default_rng(SEED)
    pairs = set()
    while len(pairs) < N_PAIRS:
        i, j = rng.choice(N_ANCHORS, size=2, replace=False)
        pairs.add((int(i), int(j)))
    pairs = sorted(pairs)

    ts = np.linspace(0.0, 1.0, N_T)
    rel = np.zeros((len(pairs), N_T), dtype=np.float32)
    abs_dA = np.zeros((len(pairs), N_T), dtype=np.float32)
    end_dist = np.zeros(len(pairs), dtype=np.float32)

    for p, (i, j) in enumerate(pairs):
        ctx = contexts[i].to(device, model_dtype)
        xa = acts[i].to(device)
        xb = acts[j].to(device)
        pts = torch.stack([slerp_point(xa, xb, float(t)) for t in ts])
        with torch.no_grad():
            logits = forward_from_layer(
                model, ctx, pts.unsqueeze(1).to(model_dtype), LAYER).float()
        fA, fB = logits[0], logits[-1]
        dA = (logits - fA).norm(dim=-1)
        dB = (logits - fB).norm(dim=-1)
        rel[p] = (dA / (dA + dB + 1e-9)).cpu().numpy()
        abs_dA[p] = dA.cpu().numpy()
        end_dist[p] = (fB - fA).norm().item()
        if (p + 1) % 10 == 0:
            print(f'  pair {p+1}/{len(pairs)}', flush=True)

    with open(os.path.join(OUT_DIR, 'results.pkl'), 'wb') as f:
        pickle.dump({'pairs': pairs, 't': ts, 'rel_dist': rel,
                     'abs_dist_from_A': abs_dA, 'endpoint_dist': end_dist,
                     'model': MODEL, 'layer': LAYER}, f)

    fig, ax = plt.subplots(figsize=(8, 5))
    for p in range(len(pairs)):
        ax.plot(ts, rel[p], color='C2', alpha=0.15, lw=1)
    ax.plot(ts, np.median(rel, axis=0), color='C2', lw=2.5,
            label=f'median ({len(pairs)} real-real pairs)')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='linear reference')
    ax.set_xlabel('interpolation t (slerp)')
    ax.set_ylabel(r'relative distance  $d_A/(d_A+d_B)$  (logits)')
    ax.set_title(f'{MODEL} L{LAYER}: slerp between real activation pairs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'rel_dist.png'), dpi=150,
                bbox_inches='tight')
    print(f"Saved {os.path.join(OUT_DIR, 'results.pkl')}")
    print(f"Saved {os.path.join(OUT_DIR, 'rel_dist.png')}")


if __name__ == '__main__':
    main()
