#!/usr/bin/env python3
"""Slerp interpolation from a real activation to jac-opt point 6.

A = the real last-token activation of FineWeb sample 6 (seq_len 5, seed 42
— the same sample whose context jac-opt point 6 was paired with in the
sweeps). B = jac-opt point 6. Context: that same sample's context. Metric:
relative logit distance d_A/(d_A+d_B), as in interp_jacopt_pairs.py.

Output: results/intrinsic_dim/interp_jacopt_pairs/real6_to_jacopt6.{png,pkl}
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
from interp_jacopt_pairs import slerp_point

OUT_DIR = 'results/intrinsic_dim/interp_jacopt_pairs'
MODEL = 'qwen3-1.7b'
LAYER = 2
IDX = 6
N_T = 41

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, tokenizer, device = load_model(MODEL, dtype=torch.bfloat16)
    model_dtype = next(model.parameters()).dtype

    token_lists = load_fineweb_fixed_length(
        30, tokenizer, seq_len=5, seed=42,
        cache_dir='/workspace/fineweb_cache')
    tids = token_lists[IDX].unsqueeze(0).to(device)
    with torch.no_grad():
        outs = model(tids, output_hidden_states=True)
    h = outs.hidden_states[LAYER + 1]
    ctx = h[:, :-1, :].to(model_dtype)
    a_real = h[:, -1, :].squeeze(0).float()

    opt = torch.load('results/intrinsic_dim/jac_optimized_data.pt',
                     map_location='cpu', weights_only=False)
    b_opt = opt['X_opt'].float()[IDX].to(device)
    print(f'||real_{IDX}||={a_real.norm():.2f}  '
          f'||jacopt_{IDX}||={b_opt.norm():.2f}  '
          f'cos={torch.nn.functional.cosine_similarity(a_real, b_opt, dim=0):.3f}',
          flush=True)

    ts = np.linspace(0.0, 1.0, N_T)
    pts = torch.stack([slerp_point(a_real, b_opt, float(t)) for t in ts])
    with torch.no_grad():
        logits = forward_from_layer(
            model, ctx, pts.unsqueeze(1).to(model_dtype), LAYER).float()
    fA, fB = logits[0], logits[-1]
    dA = (logits - fA).norm(dim=-1)
    dB = (logits - fB).norm(dim=-1)
    rel = (dA / (dA + dB + 1e-9)).cpu().numpy()
    end = (fB - fA).norm().item()
    print(f'endpoint logit dist: {end:.1f}', flush=True)

    with open(os.path.join(OUT_DIR, 'real6_to_jacopt6.pkl'), 'wb') as f:
        pickle.dump({'t': ts, 'rel_dist': rel, 'endpoint_dist': end,
                     'abs_dA': dA.cpu().numpy(), 'idx': IDX}, f)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, rel, color='C0', lw=2)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('interpolation t (slerp)')
    ax.set_ylabel(r'$d_A/(d_A+d_B)$  (logits)')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f'real act {IDX} -> jac-opt {IDX}  '
                 f'(same context, endpoint logit dist {end:.0f})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'real6_to_jacopt6.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
