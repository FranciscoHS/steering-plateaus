#!/usr/bin/env python3
"""
Dead-latent fraction and val recon metrics for a trained TopKSAE.

Streams cache rows through the encoder and counts, per latent, how often it
fires (in the top-k AND value > 0 — index membership alone is not firing,
since topk runs on post-ReLU pre-acts where zeros can occupy slots).

Usage:
    python -u scripts/dictionary_learning/eval_sae_dead_frac.py \
        --sae results/trained_sae_v2/l10_auxk_baseline/sae_weights.pt \
        --cache_path results/trained_autoencoder/l10_cache_10M/cached_activations.pt \
        --n_rows 200000 --out results/sae_v2_eval/baseline_deadfrac.json
"""

import argparse
import json
import os
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F

from src.autoencoders import load_sae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sae', required=True)
    ap.add_argument('--cache_path', required=True)
    ap.add_argument('--n_rows', type=int, default=200000,
                    help='cache rows (from the start) for firing counts')
    ap.add_argument('--val_start', type=int, default=1800000,
                    help='start row of the training val split (0.9 * 2M)')
    ap.add_argument('--val_rows', type=int, default=10000)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sae = load_sae(args.sae, device)
    print(f"SAE d_sae={sae.d_sae}, k={sae.k}")

    with open(args.cache_path + '.meta.json') as f:
        meta = json.load(f)
    mm = np.memmap(args.cache_path, dtype=np.dtype(meta['dtype']), mode='r',
                   shape=tuple(meta['shape']))

    fires = torch.zeros(sae.d_sae, dtype=torch.long, device=device)
    with torch.no_grad():
        for i in range(0, args.n_rows, args.batch):
            batch = torch.from_numpy(np.asarray(mm[i:i + args.batch])).to(device).float()
            top_vals, top_idx, _ = sae.encode(batch)
            fired = top_idx[top_vals > 0]
            fires += torch.bincount(fired, minlength=sae.d_sae)
        dead = int((fires == 0).sum())

        # Val recon metrics on the training val split
        vb = torch.from_numpy(
            np.asarray(mm[args.val_start:args.val_start + args.val_rows])
        ).to(device).float()
        recon, _, _ = sae(vb)
        mse = F.mse_loss(recon, vb).item()
        rel_l2 = ((vb - recon).norm(dim=-1) / vb.norm(dim=-1)).mean().item()
        val_cos = F.cosine_similarity(vb, recon, dim=-1).mean().item()

    res = {'sae': args.sae, 'n_rows': args.n_rows,
           'dead': dead, 'd_sae': sae.d_sae, 'dead_frac': dead / sae.d_sae,
           'val_mse': mse, 'val_rel_l2': rel_l2, 'val_cos': val_cos,
           'fires_median': int(fires.median()), 'fires_mean': float(fires.float().mean())}
    print(json.dumps(res, indent=2))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    fires_path = os.path.splitext(args.out)[0] + '_fires.pt'
    torch.save(fires.cpu(), fires_path)
    print(f"Saved to {args.out} and {fires_path}")


if __name__ == '__main__':
    main()
