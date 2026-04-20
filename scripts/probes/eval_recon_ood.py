#!/usr/bin/env python3
"""Reconstruction quality on OOD samples.

Compares SAE vs deep AE reconstruction at layer 10 last-token, measuring:
  - MSE (per-element, averaged over samples)
  - cosine similarity (mean)
  - relative L2 error
  - KL divergence on next-token logit distribution after in-place swap

Sources: reuses loaders from intrinsic_dim_data_sources.py.
"""

import argparse
import json
import os
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.jacobian.intrinsic_dim_data_sources import SOURCE_LOADERS
from scripts.probes.run_probe_sae_vs_ae import load_sae, load_ae


def _get_blocks(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    raise ValueError(f"Unknown architecture: {type(model)}")


@torch.no_grad()
def run_source(name, model, tokenizer, sae, ae, device, n_pool, seq_len, layer,
               batch_size, min_chars, seed):
    print(f"\n=== {name}: loading {n_pool} texts ===")
    texts = SOURCE_LOADERS[name](n_pool, min_chars, seed)
    print(f"  got {len(texts)} texts")

    blocks = _get_blocks(model)

    raw_list = []
    sae_list = []
    ae_list = []
    logits_base_list = []
    logits_sae_list = []
    logits_ae_list = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(
            batch, return_tensors='pt', padding='max_length',
            truncation=True, max_length=seq_len,
        ).to(device)

        # ── Baseline forward: capture last-token activation at layer+1 and last-token logits
        out = model(**tokens, output_hidden_states=True)
        raw = out.hidden_states[layer + 1][:, -1, :].float()  # [B, d]
        logits_base = out.logits[:, -1, :].float()  # [B, V]

        # ── Reconstruct
        recon_sae = sae(raw.to(next(sae.parameters()).dtype)).float()
        recon_ae = ae(raw.to(next(ae.parameters()).dtype)).float()

        # ── Swap-in forward (SAE)
        def make_hook(recon_cast):
            def hook(module, inp, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden = hidden.clone()
                hidden[:, -1, :] = recon_cast.to(hidden.dtype)
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            return hook

        h_sae = blocks[layer].register_forward_hook(make_hook(recon_sae))
        out_sae = model(**tokens)
        h_sae.remove()
        logits_sae = out_sae.logits[:, -1, :].float()

        h_ae = blocks[layer].register_forward_hook(make_hook(recon_ae))
        out_ae = model(**tokens)
        h_ae.remove()
        logits_ae = out_ae.logits[:, -1, :].float()

        raw_list.append(raw.cpu())
        sae_list.append(recon_sae.cpu())
        ae_list.append(recon_ae.cpu())
        logits_base_list.append(logits_base.cpu())
        logits_sae_list.append(logits_sae.cpu())
        logits_ae_list.append(logits_ae.cpu())

        if (i // batch_size) % 5 == 0:
            print(f"  {min(i + batch_size, len(texts))}/{len(texts)}", end='\r')

    print()
    raw = torch.cat(raw_list)
    sae_r = torch.cat(sae_list)
    ae_r = torch.cat(ae_list)
    lb = torch.cat(logits_base_list)
    ls = torch.cat(logits_sae_list)
    la = torch.cat(logits_ae_list)

    def recon_stats(recon):
        diff = recon - raw
        mse = (diff ** 2).mean().item()
        cos = F.cosine_similarity(recon, raw, dim=-1).mean().item()
        rel_l2 = (diff.norm(dim=-1) / raw.norm(dim=-1)).mean().item()
        return {'mse': round(mse, 5),
                'cos': round(cos, 4),
                'rel_l2': round(rel_l2, 4)}

    def kl_stats(logits_q):
        # KL(P || Q): P = softmax(logits_base), Q = softmax(logits_q)
        log_p = F.log_softmax(lb, dim=-1)
        log_q = F.log_softmax(logits_q, dim=-1)
        p = log_p.exp()
        kl = (p * (log_p - log_q)).sum(dim=-1)  # [B]
        return {'kl_mean': round(kl.mean().item(), 4),
                'kl_std': round(kl.std(unbiased=True).item(), 4),
                'kl_median': round(kl.median().item(), 4)}

    return {
        'n': len(raw),
        'sae': {**recon_stats(sae_r), **kl_stats(ls)},
        'ae': {**recon_stats(ae_r), **kl_stats(la)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--layer', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_pool', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', default='float16')
    parser.add_argument('--sae_weights',
                        default='results/trained_sae/l10_doubled_k32_10M/sae_weights.pt')
    parser.add_argument('--ae_weights',
                        default='results/trained_autoencoder/l10_scaling/l10_10M_drop_model.pt')
    parser.add_argument('--sources', nargs='+',
                        default=['fineweb', 'wiki_zh'],
                        choices=list(SOURCE_LOADERS.keys()))
    parser.add_argument('--outdir', default='results/recon_ood')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16,
                 'float32': torch.float32}

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype_map[args.dtype], device_map='auto',
    )
    model.eval()
    device = next(model.parameters()).device

    print(f"\nLoading SAE: {args.sae_weights}")
    sae = load_sae(args.sae_weights, device).eval()
    print(f"Loading AE:  {args.ae_weights}")
    ae = load_ae(args.ae_weights, device).eval()

    min_chars = args.seq_len * 10
    results = {
        'model': args.model, 'layer': args.layer, 'seq_len': args.seq_len,
        'n_pool': args.n_pool, 'seed': args.seed,
        'sae_weights': args.sae_weights, 'ae_weights': args.ae_weights,
        'sources': {},
    }
    outpath = os.path.join(args.outdir,
                           f'recon_ood_layer{args.layer}.json')

    for name in args.sources:
        try:
            res = run_source(name, model, tokenizer, sae, ae, device,
                             args.n_pool, args.seq_len, args.layer,
                             args.batch_size, min_chars, args.seed)
        except Exception as e:
            print(f"  FAILED {name}: {e!r}")
            results['sources'][name] = {'error': repr(e)}
            with open(outpath, 'w') as f:
                json.dump(results, f, indent=2)
            continue
        results['sources'][name] = res
        print(f"  {name}:")
        for k in ('sae', 'ae'):
            r = res[k]
            print(f"    {k:3s}  mse={r['mse']:.5f}  cos={r['cos']:.4f}  "
                  f"rel_l2={r['rel_l2']:.4f}  KL={r['kl_mean']:.4f} "
                  f"(median={r['kl_median']:.4f})")
        with open(outpath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  saved partial: {outpath}")

    print(f"\nFinal: {outpath}")


if __name__ == '__main__':
    main()
