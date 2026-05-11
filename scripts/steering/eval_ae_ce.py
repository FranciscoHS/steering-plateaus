#!/usr/bin/env python3
"""Evaluate autoencoder CE loss: how much does AE reconstruction affect model output?

Compares cross-entropy of next-token predictions with and without AE inserted
at layer 2, on both training-distribution and held-out FineWeb data.

Usage:
    python -u scripts/steering/eval_ae_ce.py \
        --ae_weights results/on_manifold_steering/qwen3-1.7b_layer2/ae_weights.pt
"""

import argparse
import os
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from src.autoencoders import Autoencoder


# ── CE evaluation ────────────────────────────────────────────────────────

def eval_ce(model, ae, token_ids_list, layer, device, batch_size=32, label=""):
    """Compute CE loss with and without AE at given layer.

    For each batch:
      1. Run model normally → CE on next-token predictions (baseline)
      2. Run model with AE hook at layer → CE on same targets (AE)

    Returns dict with mean CE for baseline and AE.
    """
    ce_baseline_all = []
    ce_ae_all = []
    cos_all = []

    n = len(token_ids_list)
    for i in range(0, n, batch_size):
        batch = torch.stack(token_ids_list[i:i + batch_size]).to(device)
        labels = batch[:, 1:]  # next-token targets

        # Baseline: no hooks
        with torch.no_grad():
            out_base = model(batch)
        logits_base = out_base.logits[:, :-1, :]  # [B, T-1, V]
        ce_base = F.cross_entropy(
            logits_base.reshape(-1, logits_base.shape[-1]),
            labels.reshape(-1), reduction='mean')
        ce_baseline_all.append(ce_base.item())

        # AE hook: reconstruct all positions at layer
        captured = {}

        def ae_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            # Only apply AE to last token (matching training distribution)
            last_tok = hidden[:, -1, :].float()  # [B, D]
            recon, _ = ae(last_tok)
            captured['cos'] = F.cosine_similarity(
                last_tok, recon, dim=-1).mean().item()
            hidden[:, -1, :] = recon.to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        layers = _get_blocks(model)
        handle = layers[layer].register_forward_hook(ae_hook)

        with torch.no_grad():
            out_ae = model(batch)
        handle.remove()

        logits_ae = out_ae.logits[:, :-1, :]
        ce_ae = F.cross_entropy(
            logits_ae.reshape(-1, logits_ae.shape[-1]),
            labels.reshape(-1), reduction='mean')
        ce_ae_all.append(ce_ae.item())
        cos_all.append(captured.get('cos', 0))

        if i % (batch_size * 10) == 0:
            print(f"  {label} {i}/{n}: base_CE={ce_base.item():.4f}, "
                  f"ae_CE={ce_ae.item():.4f}, cos={captured.get('cos', 0):.4f}")

    mean_base = sum(ce_baseline_all) / len(ce_baseline_all)
    mean_ae = sum(ce_ae_all) / len(ce_ae_all)
    mean_cos = sum(cos_all) / len(cos_all)

    return {'ce_baseline': mean_base, 'ce_ae': mean_ae, 'cos': mean_cos}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--ae_weights', required=True)
    parser.add_argument('--n_train', type=int, default=1000,
                        help='Number of training-distribution samples to eval')
    parser.add_argument('--n_heldout', type=int, default=1000,
                        help='Number of held-out samples to eval')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dtype', default='bfloat16')
    args = parser.parse_args()

    dtype_map = {'float32': None, 'bfloat16': torch.bfloat16}

    # Load model
    print(f"Loading {args.model}...")
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])

    # Load AE
    print(f"Loading AE from {args.ae_weights}...")
    checkpoint = torch.load(args.ae_weights, weights_only=True, map_location=device)
    config = checkpoint['config']
    ae = Autoencoder(config['d_in'], config['k'], config.get('hidden_dims')).to(device)
    ae.load_state_dict(checkpoint['state_dict'])
    ae.eval()
    print(f"  d_in={config['d_in']}, k={config['k']}")

    # Load data — training distribution uses seed=42 starting at index 8000
    # (matching the AE training cache from on_manifold_steering.py)
    # Held-out uses a different seed
    total_needed = args.n_train + args.n_heldout
    print(f"\nLoading FineWeb data...")

    # Training distribution: seed=42 (same as AE training cache)
    train_tokens = load_fineweb_fixed_length(
        args.n_train, tokenizer, seq_len=args.seq_len, seed=42)

    # Held-out: different seed
    heldout_tokens = load_fineweb_fixed_length(
        args.n_heldout, tokenizer, seq_len=args.seq_len, seed=999)

    # Eval
    print(f"\n{'='*60}")
    print("Training distribution")
    print(f"{'='*60}")
    train_results = eval_ce(model, ae, train_tokens, args.layer, device,
                            batch_size=args.batch_size, label="train")

    print(f"\n{'='*60}")
    print("Held-out")
    print(f"{'='*60}")
    heldout_results = eval_ce(model, ae, heldout_tokens, args.layer, device,
                              batch_size=args.batch_size, label="heldout")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Training:  baseline CE={train_results['ce_baseline']:.4f}, "
          f"AE CE={train_results['ce_ae']:.4f}, "
          f"delta={train_results['ce_ae'] - train_results['ce_baseline']:.4f}, "
          f"cos={train_results['cos']:.4f}")
    print(f"  Held-out:  baseline CE={heldout_results['ce_baseline']:.4f}, "
          f"AE CE={heldout_results['ce_ae']:.4f}, "
          f"delta={heldout_results['ce_ae'] - heldout_results['ce_baseline']:.4f}, "
          f"cos={heldout_results['cos']:.4f}")


if __name__ == '__main__':
    main()
