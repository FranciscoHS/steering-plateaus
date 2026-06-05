#!/usr/bin/env python3
"""Selectivity / concept-coherence of direction sets.

For each direction d (cosine-scored over a token corpus), take its top-T
activating positions and measure how *concept-like* that set is, independent of
firing frequency:
  - token_entropy : normalized entropy of the token-IDENTITY distribution over
        the top-T positions (in [0,1]; low = fires on few token types, e.g. a
        period-detector, despite possibly firing often).
  - top_token_frac: fraction of the top-T that is the single most common token.
  - ctx_coherence : mean pairwise cosine of the top-T activations AFTER
        projecting out d (removes the trivial 'all project high on d' inflation;
        high = the activating contexts cluster).

Compares direction sets: our dictionary, SAE decoder latents, contrastive
directions (the other high-p candidates), random, and PCA (baselines). The
informative comparison is distributional and relative — where ours land vs
SAE/contrastive vs random/PCA.

Usage:
    python scripts/superellipse/selectivity_analysis.py \\
        --dict_path /tmp/snap1100.pt --sae_ckpt .../sae_L0_50_layer14.pt \\
        --fsec_dirs /workspace/fsec-anonymous/results/directions/dirs_qwen_L14.pkl \\
        --n_sample 400 --top_t 100 --out .../selectivity
"""
import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model
from src.data import load_fineweb_fixed_length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--sae_ckpt', required=True)
    ap.add_argument('--fsec_dirs', required=True,
                    help='fsec dirs_*.pkl with contrastive directions')
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--n_corpus', type=int, default=2000)
    ap.add_argument('--corpus_seq_len', type=int, default=48)
    ap.add_argument('--corpus_seed', type=int, default=123)
    ap.add_argument('--top_t', type=int, default=100)
    ap.add_argument('--n_sample', type=int, default=400,
                    help='# directions sampled per set (contrastive uses all)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    g = torch.Generator().manual_seed(args.seed)

    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])

    # ---- corpus token activations (skip pos 0), keep token id per position ----
    ctok = torch.stack(load_fineweb_fixed_length(
        args.n_corpus, tokenizer, seq_len=args.corpus_seq_len,
        seed=args.corpus_seed))
    Nc, Tc = ctok.shape
    acts_chunks, tok_ids = [], []
    print(f'Capturing corpus acts: {Nc} seqs x {Tc} tok', flush=True)
    for i in range(0, Nc, 32):
        bt = ctok[i:i + 32].to(device)
        with torch.no_grad():
            h = model(bt, output_hidden_states=True).hidden_states[args.layer + 1]
        h = h[:, 1:, :].float().cpu()                  # drop pos 0
        b, Tm1, d = h.shape
        acts_chunks.append(h.reshape(b * Tm1, d))
        tok_ids.append(ctok[i:i + 32, 1:].reshape(-1))
    A = torch.cat(acts_chunks, 0).to(device)           # [M, d]
    tok_ids = torch.cat(tok_ids, 0).numpy()            # [M]
    d_model = A.shape[1]
    A_unit = A / A.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    A_mean = A.mean(0, keepdim=True)
    print(f'  {A.shape[0]} token activations', flush=True)

    # ---- direction sets ----
    def norm_rows(X):
        return X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    sets = {}
    D = norm_rows(torch.load(args.dict_path, map_location='cpu')['dictionary'].float())
    idx = torch.randperm(D.shape[0], generator=g)[:args.n_sample]
    sets['ours'] = D[idx]

    sae = torch.load(args.sae_ckpt, map_location='cpu')
    Wd = sae['W_dec'].float()                           # [d, d_sae]
    Wd = Wd.T if Wd.shape[0] == d_model else Wd         # -> [d_sae, d]
    sidx = torch.randperm(Wd.shape[0], generator=g)[:args.n_sample]
    sets['SAE'] = norm_rows(Wd[sidx])

    fb = pickle.load(open(args.fsec_dirs, 'rb'))
    fdirs = fb['directions'] if 'directions' in fb else fb
    names = sorted(fdirs.keys())
    sets['contrastive'] = norm_rows(torch.stack([fdirs[n].float() for n in names]))

    R = torch.randn(args.n_sample, d_model, generator=g)
    sets['random'] = norm_rows(R)

    # PCA of centered corpus activations
    Ac = (A - A_mean).cpu()
    cov = (Ac.T @ Ac) / Ac.shape[0]
    evals, evecs = torch.linalg.eigh(cov)               # ascending
    sets['PCA'] = evecs[:, -args.n_sample:].T.contiguous()  # top PCs as rows

    # ---- per-direction metrics ----
    T = args.top_t
    results = {}
    for name, Ds in sets.items():
        Ds = Ds.to(device)
        n = Ds.shape[0]
        ent = np.zeros(n); topfrac = np.zeros(n); coh = np.zeros(n)
        for k in range(n):
            d = Ds[k]
            cos = A_unit @ d                            # [M] cosine
            top = torch.topk(cos.abs(), T).indices       # |cos|: both polarities
            ti = tok_ids[top.cpu().numpy()]
            _, counts = np.unique(ti, return_counts=True)
            p = counts / counts.sum()
            ent[k] = (-(p * np.log(p)).sum()) / np.log(T)   # normalized [0,1]
            topfrac[k] = p.max()
            X = A[top]                                  # [T, d]
            X = X - (X @ d).unsqueeze(-1) * d            # project out d
            X = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            G = X @ X.T
            coh[k] = (G.sum() - G.diagonal().sum()).item() / (T * (T - 1))
        results[name] = {'token_entropy': ent, 'top_token_frac': topfrac,
                         'ctx_coherence': coh}
        print(f'{name:12s} n={n:4d}  token_entropy med={np.median(ent):.3f} '
              f'IQR=({np.quantile(ent,.25):.3f},{np.quantile(ent,.75):.3f})  '
              f'top_tok_frac med={np.median(topfrac):.3f}  '
              f'ctx_coh med={np.median(coh):.3f}', flush=True)

    with open(os.path.join(args.out, 'selectivity.pkl'), 'wb') as f:
        pickle.dump({'results': results, 'args': vars(args),
                     'contrastive_names': names}, f)

    # ---- plot ----
    metrics = [('token_entropy', 'token-type entropy (norm) — lower=selective'),
               ('top_token_frac', 'top-token fraction — higher=selective'),
               ('ctx_coherence', 'context coherence — higher=coherent')]
    order = ['ours', 'SAE', 'contrastive', 'random', 'PCA']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (mkey, title) in zip(axes, metrics):
        data = [results[s][mkey] for s in order]
        ax.violinplot(data, showmedians=True)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=30)
        ax.set_title(title, fontsize=10); ax.grid(True, alpha=0.3)
    fig.suptitle(f'{args.model} L{args.layer}  top_t={T}  '
                 f'n_sample={args.n_sample}')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'selectivity.png'), dpi=150,
                bbox_inches='tight')
    print(f'\nSaved {os.path.join(args.out, "selectivity.png")}', flush=True)


if __name__ == '__main__':
    main()
