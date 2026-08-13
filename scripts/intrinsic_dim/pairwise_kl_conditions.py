#!/usr/bin/env python3
"""Pairwise output-KL within sets: gauss-rotated-25deg vs gauss vs real.

Metric change from pairwise_dist_conditions.py: instead of L2 in activation
space, compare the OUTPUT distributions the activations produce. Every
activation is patched into the same carrier context (FineWeb sample 0,
seq_len 5, seed 42 — the sweep recipe) at qwen3-1.7b L2, and the next-token
distribution is read off. Within each set, KL(p_i || p_j) is computed for
all ordered pairs (n=200 -> 39800 values).

Sets:
  - real            200 rows of X_real (same seed-0 draw as the L2 version)
  - gauss (fresh)   the 200 pre-rotation Gaussian points (X_fresh,
                    cov-matched rows 200:400)
  - gauss rot 25deg X_fresh rotated 25 degrees toward the separating normal
                    w (gauss-start vs jac-opt plane), norm preserved —
                    same construction as make_gauss_rotated_anchors.py,
                    the "plateau-having" synthetic set

Output: results/intrinsic_dim/pairwise_kl/{results.pkl, kl_hist.png}
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
SEED = 0
THETA_DEG = 25.0
BATCH = 50


def rotate_toward(G, w_hat, theta_deg):
    """Rotate each row of G toward w_hat by theta, preserving its norm
    (identical construction to make_gauss_rotated_anchors.py)."""
    norms = G.norm(dim=-1, keepdim=True)
    a_hat = G / norms
    comp = (a_hat @ w_hat).unsqueeze(-1)
    w_perp = w_hat.unsqueeze(0) - comp * a_hat
    w_perp = w_perp / w_perp.norm(dim=-1, keepdim=True)
    t = torch.deg2rad(torch.tensor(theta_deg))
    return norms * (torch.cos(t) * a_hat + torch.sin(t) * w_perp)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    filt = torch.load(os.path.join(DATA_DIR, 'jac_filtered_data.pt'),
                      map_location='cpu', weights_only=False)
    shift = torch.load(os.path.join(DATA_DIR, 'gauss_shifted_anchors.pt'),
                       map_location='cpu', weights_only=False)

    g = torch.Generator().manual_seed(SEED)
    X_real = filt['X_real'].float()
    X_real = X_real[torch.randperm(X_real.shape[0], generator=g)[:N]]
    G = shift['X_fresh'].float()[:N]
    w = shift['w'].float()
    X_rot = rotate_toward(G, w / w.norm(), THETA_DEG)

    sets = {'real': X_real, 'gauss (fresh)': G,
            f'gauss rot {int(THETA_DEG)}deg': X_rot}

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

    results = {'n': N, 'seed': SEED, 'theta_deg': THETA_DEG,
               'ctx_idx': CTX_IDX, 'model': MODEL, 'layer': LAYER, 'kl': {}}
    for name, X in sets.items():
        logps = []
        for s in range(0, N, BATCH):
            pts = X[s:s + BATCH].to(device)
            with torch.no_grad():
                logits = forward_from_layer(
                    model, ctx, pts.unsqueeze(1).to(model_dtype),
                    LAYER).float()
            logps.append(torch.log_softmax(logits, dim=-1))
        logp = torch.cat(logps)                      # [N, V]
        p = logp.exp()
        # KL(i||j) = sum_v p_i (logp_i - logp_j)
        self_term = (p * logp).sum(-1, keepdim=True)  # [N, 1]
        cross = p @ logp.T                            # [N, N]
        kl = (self_term - cross).cpu().numpy()
        off = kl[~np.eye(N, dtype=bool)]
        results['kl'][name] = kl
        q25, q75 = np.percentile(off, [25, 75])
        print(f'{name:18s} pairwise KL (nats): median={np.median(off):.3f}  '
              f'IQR={q25:.3f}-{q75:.3f}  min={off.min():.4f}  '
              f'max={off.max():.1f}', flush=True)

    with open(os.path.join(OUT_DIR, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'real': 'C0', 'gauss (fresh)': 'C1',
              f'gauss rot {int(THETA_DEG)}deg': 'C4'}
    for name, kl in results['kl'].items():
        off = kl[~np.eye(N, dtype=bool)]
        ax.hist(np.log10(np.maximum(off, 1e-8)), bins=80, density=True,
                alpha=0.5, color=colors[name],
                label=f'{name} (median {np.median(off):.2f} nats)')
    ax.set_xlabel(r'$\log_{10}$ pairwise KL$(p_i \| p_j)$  (nats)')
    ax.set_ylabel('density')
    ax.set_title(f'{MODEL} L{LAYER}: pairwise output-KL within sets, '
                 f'carrier ctx {CTX_IDX}, n={N}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'kl_hist.png'), dpi=150,
                bbox_inches='tight')
    print(f"Saved {os.path.join(OUT_DIR, 'results.pkl')}")
    print(f"Saved {os.path.join(OUT_DIR, 'kl_hist.png')}")


if __name__ == '__main__':
    main()
