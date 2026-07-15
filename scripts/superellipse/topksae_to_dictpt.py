#!/usr/bin/env python3
"""Convert one of our TopKSAE checkpoints (state_dict format) into a
contrastive-style dictionary.pt (normalized decoder rows), mirroring
sae_to_dictpt.py which handles the Qwen-Scope layout.

    python scripts/superellipse/topksae_to_dictpt.py \\
        --sae_ckpt results/trained_sae_popt_v2/l10_lam100/sae_weights.pt \\
        --out_pt results/dtosae_lam100_v2/lam100_dict.pt
"""
import argparse
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--sae_ckpt', required=True)
ap.add_argument('--out_pt', required=True)
args = ap.parse_args()

data = torch.load(args.sae_ckpt, map_location='cpu', weights_only=False)
W_dec = data['state_dict']['W_dec'].float()        # [d_sae, d]
D = W_dec / (W_dec.norm(dim=-1, keepdim=True) + 1e-12)
os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
torch.save({'dictionary': D,
            'source': f'TopKSAE decoder latents from {args.sae_ckpt}'},
           args.out_pt)
print(f'Saved {args.out_pt}  ({D.shape[0]} latents, d={D.shape[1]})')
