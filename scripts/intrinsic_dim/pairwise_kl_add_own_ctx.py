#!/usr/bin/env python3
"""Add a 'real (own context)' condition to the pairwise output-KL plot.

Loads the three existing KL matrices from pairwise_kl/results.pkl and adds
a fourth: the same 200 real activations (seed-0 draw), but each evaluated
on ITS OWN prefix context instead of the shared carrier. Pairwise KL then
compares the model's actual next-token distributions at those 200 positions.

jac_filtered_data.pt did not save the contexts, so they are regenerated
with the exact April recipe (FineWeb-Edu sample-10BT streaming, shuffle
seed 42, seq_len 10, fp16). Because stream reproducibility is not
guaranteed, each regenerated last-token activation is checked against the
saved X_real row; the script ABORTS if they do not match.

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


def main():
    with open(os.path.join(OUT_DIR, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)

    filt = torch.load(os.path.join(DATA_DIR, 'jac_filtered_data.pt'),
                      map_location='cpu', weights_only=False)
    X_real_all = filt['X_real']
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(X_real_all.shape[0], generator=g)[:N]
    X_real = X_real_all[idx].float()

    # regenerate the April text stream and re-extract the chosen 200
    # (activation, context) pairs with the exact jac_filtered recipe (fp16)
    print('Streaming FineWeb-Edu (shuffle seed 42)...', flush=True)
    ds = load_dataset('HuggingFaceFW/FineWeb-Edu', 'sample-10BT',
                      split='train', streaming=True).shuffle(seed=42)
    texts = []
    for ex in ds:
        if len(ex['text']) >= SEQ_LEN * 3:
            texts.append(ex['text'])
        if len(texts) >= N_REAL:
            break
    assert len(texts) == X_real_all.shape[0], \
        f'text count {len(texts)} != saved X_real rows {X_real_all.shape[0]}'

    print(f'Loading {MODEL} (fp16, as in the April run)...', flush=True)
    model, tokenizer, device = load_model(MODEL, dtype=torch.float16)
    model_dtype = next(model.parameters()).dtype

    contexts = []
    recomputed = []
    for i in idx.tolist():
        tokens = tokenizer([texts[i]], return_tensors='pt',
                           padding='max_length', truncation=True,
                           max_length=SEQ_LEN).to(device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        hidden = out.hidden_states[LAYER + 1]
        recomputed.append(hidden[:, -1, :].squeeze(0).float().cpu())
        contexts.append(hidden[:, :-1, :].cpu())
    recomputed = torch.stack(recomputed)

    rel_err = ((recomputed - X_real).norm(dim=-1)
               / X_real.norm(dim=-1)).max().item()
    print(f'verification: max relative L2 error vs saved X_real = '
          f'{rel_err:.4f}', flush=True)
    if rel_err > 0.02:
        raise SystemExit('ABORT: regenerated activations do not match '
                         'saved X_real — stream not reproducible, contexts '
                         'would be wrong.')

    logps = []
    for i in range(N):
        ctx = contexts[i].to(device, model_dtype)
        pt = X_real[i].unsqueeze(0).unsqueeze(1).to(device, model_dtype)
        with torch.no_grad():
            logits = forward_from_layer(model, ctx, pt, LAYER).float()
        logps.append(torch.log_softmax(logits.squeeze(), dim=-1))
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{N}', flush=True)
    logp = torch.stack(logps)
    p = logp.exp()
    self_term = (p * logp).sum(-1, keepdim=True)
    kl = (self_term - p @ logp.T).cpu().numpy()
    results['kl']['real (own ctx)'] = kl
    off = kl[~np.eye(N, dtype=bool)]
    q25, q75 = np.percentile(off, [25, 75])
    print(f'real (own ctx)     pairwise KL (nats): median={np.median(off):.3f}  '
          f'IQR={q25:.3f}-{q75:.3f}  min={off.min():.4f}  '
          f'max={off.max():.1f}', flush=True)

    with open(os.path.join(OUT_DIR, 'results_v2.pkl'), 'wb') as f:
        pickle.dump(results, f)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'real': 'C0', 'gauss (fresh)': 'C1', 'gauss rot 25deg': 'C4',
              'real (own ctx)': 'C3'}
    for name, m in results['kl'].items():
        o = m[~np.eye(m.shape[0], dtype=bool)]
        ax.hist(np.log10(np.maximum(o, 1e-8)), bins=80, density=True,
                alpha=0.5, color=colors.get(name, 'C7'),
                label=f'{name} (median {np.median(o):.2f} nats)')
    ax.set_xlabel(r'$\log_{10}$ pairwise KL$(p_i \| p_j)$  (nats)')
    ax.set_ylabel('density')
    ax.set_title(f'{MODEL} L{LAYER}: pairwise output-KL within sets, n={N}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'kl_hist_v2.png'), dpi=150,
                bbox_inches='tight')
    print(f"Saved {os.path.join(OUT_DIR, 'results_v2.pkl')}")
    print(f"Saved {os.path.join(OUT_DIR, 'kl_hist_v2.png')}")


if __name__ == '__main__':
    main()
