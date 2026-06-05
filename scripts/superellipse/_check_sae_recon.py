#!/usr/bin/env python3
"""Sanity-check the TopK-SAE encode/reconstruct convention on real Qwen3 L14
activations before using the decomposition to build synthetic anchors.

Tries both W_enc orientations and a few k values; reports relative recon error.
"""
import sys
sys.path.insert(0, '.')
import torch
from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import extract_context_and_activation

SAE = 'results/dictionary/qwen3-1.7b_L14_K8000/sae_L0_50_layer14.pt'
model, tok, device = load_model('qwen3-1.7b', dtype=torch.bfloat16)
sae = torch.load(SAE, map_location='cpu')
W_enc = sae['W_enc'].float(); W_dec = sae['W_dec'].float()
b_enc = sae['b_enc'].float(); b_dec = sae['b_dec'].float()
print('W_enc', tuple(W_enc.shape), 'W_dec', tuple(W_dec.shape))

toks = load_fineweb_fixed_length(64, tok, seq_len=10, seed=7)
_, acts = extract_context_and_activation(model, toks, 14, device)
a = acts.float().cpu()                                   # [N, d]
N, d = a.shape

def recon_err(k, sub_bdec, use_relu):
    # eval_dictionary_omp convention: pre = a @ W_enc.T + b_enc, topk raw.
    x = (a - b_dec) if sub_bdec else a
    pre = x @ W_enc.T + b_enc                             # [N, d_sae]
    p = torch.relu(pre) if use_relu else pre
    topv, topi = p.topk(k, dim=-1)
    z = torch.zeros_like(pre).scatter(-1, topi, topv)
    rec = z @ W_dec.T + b_dec                            # [N, d]
    err = (rec - a).norm(dim=1) / a.norm(dim=1)
    # variance-explained (FVU vs the mean activation)
    mu = a.mean(0, keepdim=True)
    fvu = ((rec - a) ** 2).sum() / ((a - mu) ** 2).sum()
    return err.mean().item(), fvu.item()

for sub in [False, True]:
    for relu in [False, True]:
        for k in [50, 100]:
            e, fvu = recon_err(k, sub, relu)
            print(f'sub_bdec={sub!s:5s} relu={relu!s:5s} k={k:3d}  '
                  f'rel_err={e:.4f}  FVU={fvu:.4f}')
