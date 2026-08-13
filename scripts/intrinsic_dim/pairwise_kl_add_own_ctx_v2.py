#!/usr/bin/env python3
"""Add a 'real (own context)' condition to the pairwise output-KL plot, v2.

v1 (pairwise_kl_add_own_ctx.py) aborted: the FineWeb stream is not
reproducible in order, so index-based context recovery failed (max rel err
1.86). Here the recovery is by MATCHING instead: recompute last-token
activations for all 5000 regenerated texts (April recipe: FineWeb-Edu
sample-10BT, shuffle seed 42, seq_len 10, fp16) and nearest-neighbor match
each of the 200 saved X_real rows (seed-0 draw). Rows with rel err < 2%
keep their matched context; the pairwise-KL condition uses the matched
subset. If fewer than 150 match, the script falls back to a fresh draw of
200 real (activation, context) pairs from the regenerated stream, labeled
accordingly.

Output (originals preserved): results/intrinsic_dim/pairwise_kl/
  results_v2.pkl, kl_hist_v2.png
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

from datasets import load_dataset

from src.model import load_model, forward_from_layer

DATA_DIR = 'results/intrinsic_dim'
OUT_DIR = os.path.join(DATA_DIR, 'pairwise_kl')
MODEL = 'qwen3-1.7b'
LAYER = 2
N = 200
SEED = 0
N_REAL = 5000
SEQ_LEN = 10
BATCH = 64
MATCH_TOL = 0.02
MIN_MATCHES = 150


def kl_matrix(logp):
    p = logp.exp()
    self_term = (p * logp).sum(-1, keepdim=True)
    return (self_term - p @ logp.T).cpu().numpy()


def main():
    with open(os.path.join(OUT_DIR, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)

    filt = torch.load(os.path.join(DATA_DIR, 'jac_filtered_data.pt'),
                      map_location='cpu', weights_only=False)
    X_real_all = filt['X_real']
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(X_real_all.shape[0], generator=g)[:N]
    X_sel = X_real_all[idx].float()

    print('Streaming FineWeb-Edu (shuffle seed 42)...', flush=True)
    ds = load_dataset('HuggingFaceFW/FineWeb-Edu', 'sample-10BT',
                      split='train', streaming=True).shuffle(seed=42)
    texts = []
    for ex in ds:
        if len(ex['text']) >= SEQ_LEN * 3:
            texts.append(ex['text'])
        if len(texts) >= N_REAL:
            break

    print(f'Loading {MODEL} (fp16, as in the April run)...', flush=True)
    model, tokenizer, device = load_model(MODEL, dtype=torch.float16)
    model_dtype = next(model.parameters()).dtype

    print('Recomputing activations for the full regenerated pool...',
          flush=True)
    acts, ctxs = [], []
    for s in range(0, len(texts), BATCH):
        tokens = tokenizer(texts[s:s + BATCH], return_tensors='pt',
                           padding='max_length', truncation=True,
                           max_length=SEQ_LEN).to(device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        hidden = out.hidden_states[LAYER + 1]
        acts.append(hidden[:, -1, :].float().cpu())
        ctxs.append(hidden[:, :-1, :].cpu())
        if (s // BATCH) % 20 == 0:
            print(f'  {s}/{len(texts)}', flush=True)
    acts = torch.cat(acts)
    ctxs = torch.cat(ctxs)

    print('Matching saved rows to regenerated pool...', flush=True)
    d2 = torch.cdist(X_sel, acts)                     # [200, 5000]
    nn_dist, nn_idx = d2.min(dim=1)
    rel = nn_dist / X_sel.norm(dim=-1)
    matched = rel < MATCH_TOL
    n_match = int(matched.sum())
    print(f'matched {n_match}/{N} rows (rel err < {MATCH_TOL})  '
          f'median rel err of matches = '
          f'{rel[matched].median().item() if n_match else float("nan"):.4f}',
          flush=True)

    if n_match >= MIN_MATCHES:
        label = f'real (own ctx, {n_match} matched)'
        X_use = X_sel[matched]
        ctx_use = ctxs[nn_idx[matched]]
    else:
        print('Too few matches — falling back to a fresh draw.', flush=True)
        label = 'real (own ctx, fresh draw)'
        gf = torch.Generator().manual_seed(SEED + 1)
        fresh = torch.randperm(acts.shape[0], generator=gf)[:N]
        X_use = acts[fresh]
        ctx_use = ctxs[fresh]

    logps = []
    for i in range(X_use.shape[0]):
        ctx = ctx_use[i].unsqueeze(0).to(device, model_dtype)
        pt = X_use[i].unsqueeze(0).unsqueeze(1).to(device, model_dtype)
        with torch.no_grad():
            logits = forward_from_layer(model, ctx, pt, LAYER).float()
        logps.append(torch.log_softmax(logits.squeeze(), dim=-1))
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{X_use.shape[0]}', flush=True)
    kl = kl_matrix(torch.stack(logps))
    results['kl'][label] = kl
    m = kl.shape[0]
    off = kl[~np.eye(m, dtype=bool)]
    q25, q75 = np.percentile(off, [25, 75])
    print(f'{label}: pairwise KL (nats): median={np.median(off):.3f}  '
          f'IQR={q25:.3f}-{q75:.3f}  min={off.min():.4f}  '
          f'max={off.max():.1f}', flush=True)

    with open(os.path.join(OUT_DIR, 'results_v2.pkl'), 'wb') as f:
        pickle.dump(results, f)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'real': 'C0', 'gauss (fresh)': 'C1', 'gauss rot 25deg': 'C4',
              label: 'C3'}
    for name, mat in results['kl'].items():
        o = mat[~np.eye(mat.shape[0], dtype=bool)]
        ax.hist(np.log10(np.maximum(o, 1e-8)), bins=80, density=True,
                alpha=0.5, color=colors.get(name, 'C7'),
                label=f'{name} (median {np.median(o):.2f} nats)')
    ax.set_xlabel(r'$\log_{10}$ pairwise KL$(p_i \| p_j)$  (nats)')
    ax.set_ylabel('density')
    ax.set_title(f'{MODEL} L{LAYER}: pairwise output-KL within sets')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'kl_hist_v2.png'), dpi=150,
                bbox_inches='tight')
    print(f"Saved {os.path.join(OUT_DIR, 'results_v2.pkl')}")
    print(f"Saved {os.path.join(OUT_DIR, 'kl_hist_v2.png')}")


if __name__ == '__main__':
    main()
