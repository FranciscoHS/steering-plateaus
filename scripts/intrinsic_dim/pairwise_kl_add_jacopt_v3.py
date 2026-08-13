#!/usr/bin/env python3
"""Add a 'jac-opt' condition to the pairwise output-KL plot, v3.

Loads the four KL matrices from pairwise_kl/results_v2.pkl and adds the
200 Jacobian-optimized points (jac_optimized_data.pt, raw), patched into
the SAME shared carrier context as the other synthetic sets (FineWeb
sample 0, seq_len 5, seed 42).

Output (v1/v2 preserved): results/intrinsic_dim/pairwise_kl/
  results_v3.pkl, kl_hist_v3.png
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

DATA_DIR = 'results/intrinsic_dim'
OUT_DIR = os.path.join(DATA_DIR, 'pairwise_kl')
MODEL = 'qwen3-1.7b'
LAYER = 2
SEQ_LEN = 5
CTX_IDX = 0
N = 200
BATCH = 50


def main():
    with open(os.path.join(OUT_DIR, 'results_v2.pkl'), 'rb') as f:
        results = pickle.load(f)

    opt = torch.load(os.path.join(DATA_DIR, 'jac_optimized_data.pt'),
                     map_location='cpu', weights_only=False)
    X_opt = opt['X_opt'].float()[:N]

    print(f'Loading {MODEL}...', flush=True)
    model, tokenizer, device = load_model(MODEL, dtype=torch.bfloat16)
    model_dtype = next(model.parameters()).dtype

    token_lists = load_fineweb_fixed_length(
        CTX_IDX + 1, tokenizer, seq_len=SEQ_LEN, seed=42,
        cache_dir='/workspace/fineweb_cache')
    tids = token_lists[CTX_IDX].unsqueeze(0).to(device)
    with torch.no_grad():
        outs = model(tids, output_hidden_states=True)
    ctx = outs.hidden_states[LAYER + 1][:, :-1, :].to(model_dtype)

    logps = []
    for s in range(0, N, BATCH):
        pts = X_opt[s:s + BATCH].to(device)
        with torch.no_grad():
            logits = forward_from_layer(
                model, ctx, pts.unsqueeze(1).to(model_dtype), LAYER).float()
        logps.append(torch.log_softmax(logits, dim=-1))
    logp = torch.cat(logps)
    p = logp.exp()
    self_term = (p * logp).sum(-1, keepdim=True)
    kl = (self_term - p @ logp.T).cpu().numpy()
    results['kl']['jac-opt'] = kl
    off = kl[~np.eye(N, dtype=bool)]
    q25, q75 = np.percentile(off, [25, 75])
    print(f'jac-opt: pairwise KL (nats): median={np.median(off):.3f}  '
          f'IQR={q25:.3f}-{q75:.3f}  min={off.min():.4f}  '
          f'max={off.max():.1f}', flush=True)

    with open(os.path.join(OUT_DIR, 'results_v3.pkl'), 'wb') as f:
        pickle.dump(results, f)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'real': 'C0', 'gauss (fresh)': 'C1', 'gauss rot 25deg': 'C4',
              'real (own ctx, fresh draw)': 'C3', 'jac-opt': 'C2'}
    for name, mat in results['kl'].items():
        o = mat[~np.eye(mat.shape[0], dtype=bool)]
        ax.hist(np.log10(np.maximum(o, 1e-8)), bins=80, density=True,
                alpha=0.5, color=colors.get(name, 'C7'),
                label=f'{name} (median {np.median(o):.2f} nats)')
    ax.set_xlabel(r'$\log_{10}$ pairwise KL$(p_i \| p_j)$  (nats)')
    ax.set_ylabel('density')
    ax.set_title(f'{MODEL} L{LAYER}: pairwise output-KL within sets')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'kl_hist_v3.png'), dpi=150,
                bbox_inches='tight')
    print(f"Saved {os.path.join(OUT_DIR, 'results_v3.pkl')}")
    print(f"Saved {os.path.join(OUT_DIR, 'kl_hist_v3.png')}")


if __name__ == '__main__':
    main()
