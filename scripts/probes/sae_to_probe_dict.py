#!/usr/bin/env python3
"""
Convert a TopKSAE checkpoint into the {'dictionary': [n_atoms, d_model]} format
that run_sparse_probe_suite.py / add_dict_to_suite.py expect for OMP-k probing.

Just extracts the decoder rows (unit-normalized) so the SAE's atoms can be
compared as a dictionary against other dictionaries / random baselines.

Usage:
    python scripts/probes/sae_to_probe_dict.py \
        --sae results/trained_sae_popt/l10_lam1/sae_weights.pt \
        --out results/probe_dicts/popt.pt
"""
import argparse
import os
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F

from src.autoencoders import load_sae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sae', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sae = load_sae(args.sae, 'cpu')
    W = sae.W_dec.detach().float()                  # [d_sae, d_in]
    W = F.normalize(W, dim=-1)
    torch.save({'dictionary': W}, args.out)
    print(f"Saved dictionary {tuple(W.shape)} from {args.sae} -> {args.out}")


if __name__ == '__main__':
    main()
