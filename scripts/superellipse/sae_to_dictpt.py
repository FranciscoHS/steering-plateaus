#!/usr/bin/env python3
"""Convert a Qwen-Scope SAE checkpoint into a contrastive-style dictionary.pt
(normalized decoder directions as rows) so SAE latents can go through the
same tooling as our dictionary atoms (dashboard, alignment, featureness...).

    python scripts/superellipse/sae_to_dictpt.py \\
        --sae_ckpt results/dictionary/qwen3-1.7b_L14_K8000/sae_L0_50_layer14.pt \\
        --out_pt results/sae_qwen_L14_L0_50/dictionary.pt
"""
import argparse
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--sae_ckpt', required=True)
ap.add_argument('--out_pt', required=True)
args = ap.parse_args()

sae = torch.load(args.sae_ckpt, map_location='cpu')
W_dec = sae['W_dec'].float()                       # [d, d_sae] or [d_sae, d]
if W_dec.shape[0] < W_dec.shape[1]:                # ensure rows = latents
    W_dec = W_dec.T
D = W_dec / (W_dec.norm(dim=-1, keepdim=True) + 1e-12)
os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
torch.save({'dictionary': D,
            'source': f'SAE decoder latents from {args.sae_ckpt}'},
           args.out_pt)
print(f'Saved {args.out_pt}  ({D.shape[0]} latents, d={D.shape[1]})')
