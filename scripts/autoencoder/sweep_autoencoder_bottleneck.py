#!/usr/bin/env python3
"""
Sweep autoencoder bottleneck dimension and compare reconstruction quality to SAE.

Trains autoencoders with bottleneck k in [5, 10, 15, 20, 25, 30] using the
67M-param architecture (hidden_dims=[4096, 4096, 2048]), then plots cos similarity
vs bottleneck dimension with SAE baseline.

Usage:
    python -u scripts/autoencoder/sweep_autoencoder_bottleneck.py \
        --model Qwen/Qwen3-1.7B --layer 2 \
        --cache_path /tmp/cached_activations.pt --reuse_cache \
        --output_dir /tmp/steering-plateaus/results/trained_autoencoder/bottleneck_sweep
"""

import argparse
import json
import os
import sys
import time
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from scripts.autoencoder.train_autoencoder import cache_activations
from src.autoencoders import DeepAutoencoder
from src.model import load_model
from src.data import load_fineweb_fixed_length


def train_one(ae, activations, n_steps, batch_size, lr, device):
    """Train autoencoder, return final metrics."""
    n_cache = activations.shape[0]
    activations_gpu = activations.to(device)

    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    t0 = time.time()
    for step in range(n_steps):
        idx = torch.randint(0, n_cache, (batch_size,), device=device)
        batch = activations_gpu[idx]

        recon, z = ae(batch)
        loss = F.mse_loss(recon, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 1000 == 0:
            with torch.no_grad():
                rel_l2 = (batch - recon).norm(dim=-1).mean() / batch.norm(dim=-1).mean()
                cos = F.cosine_similarity(batch, recon, dim=-1).mean()
            print(f"    Step {step:5d}: loss={loss.item():.6f}, cos={cos.item():.4f}")

    # Final eval on larger sample
    ae.eval()
    with torch.no_grad():
        eval_sample = activations_gpu[:10000]
        recon, z = ae(eval_sample)
        mse = F.mse_loss(recon, eval_sample).item()
        rel_l2 = ((eval_sample - recon).norm(dim=-1) / eval_sample.norm(dim=-1)).mean().item()
        cos = F.cosine_similarity(eval_sample, recon, dim=-1).mean().item()

    elapsed = time.time() - t0
    return {'mse': mse, 'rel_l2': rel_l2, 'cos_sim': cos, 'time': elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--bottlenecks', type=int, nargs='+', default=[5, 10, 15, 20, 25, 30])
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[4096, 4096, 2048])
    parser.add_argument('--n_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--cache_start', type=int, default=8000)
    parser.add_argument('--n_cache', type=int, default=2000000)
    parser.add_argument('--n_extract', type=int, default=500000)
    parser.add_argument('--reuse_cache', action='store_true')
    parser.add_argument('--cache_path', default=None)
    parser.add_argument('--output_dir', default='results/trained_autoencoder/bottleneck_sweep')
    # SAE baseline (provide manually or train)
    parser.add_argument('--sae_cos', type=float, default=None,
                        help='SAE cosine similarity baseline (if known)')
    parser.add_argument('--batchnorm', action='store_true',
                        help='Add BatchNorm1d between layers')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # Load model and activations
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    activations, cache_path = cache_activations(model, tokenizer, device, args)
    del model
    torch.cuda.empty_cache()

    # Run sweep
    results = {}
    for k in args.bottlenecks:
        print(f"\n{'='*60}")
        print(f"Training autoencoder with bottleneck k={k}")
        print(f"{'='*60}")

        torch.manual_seed(args.seed)
        ae = DeepAutoencoder(d_in=d_model, k=k, hidden_dims=args.hidden_dims,
                             use_batchnorm=args.batchnorm).to(device)
        n_params = sum(p.numel() for p in ae.parameters())
        print(f"  Architecture: {d_model} → {' → '.join(str(h) for h in args.hidden_dims)} → {k}")
        print(f"  Parameters: {n_params / 1e6:.2f}M")

        metrics = train_one(ae, activations, args.n_steps, args.batch_size, args.lr, device)
        metrics['n_params'] = n_params
        results[k] = metrics

        print(f"  Final: MSE={metrics['mse']:.6f}, rel_L2={metrics['rel_l2']:.4f}, "
              f"cos={metrics['cos_sim']:.4f} ({metrics['time']:.0f}s)")

        # Save weights
        save_path = os.path.join(args.output_dir, f"ae_k{k}.pt")
        torch.save({
            'state_dict': ae.state_dict(),
            'config': {'d_in': d_model, 'k': k, 'hidden_dims': args.hidden_dims},
            'metrics': metrics,
        }, save_path)

        del ae
        torch.cuda.empty_cache()

    # Save summary
    summary_path = os.path.join(args.output_dir, "sweep_results.json")
    # Convert to serializable format
    summary = {str(k): v for k, v in results.items()}
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # Plot
    ks = sorted(results.keys())
    cos_vals = [results[k]['cos_sim'] for k in ks]
    rel_l2_vals = [results[k]['rel_l2'] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ks, cos_vals, 'o-', color='C0', label='Autoencoder (67M)')
    if args.sae_cos is not None:
        ax1.axhline(args.sae_cos, color='C1', linestyle='--', label=f'SAE 8x k=32 (67M)')
    ax1.set_xlabel('Bottleneck dimension')
    ax1.set_ylabel('Cosine similarity')
    ax1.set_title('Reconstruction quality vs bottleneck')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, rel_l2_vals, 'o-', color='C0', label='Autoencoder (67M)')
    ax2.set_xlabel('Bottleneck dimension')
    ax2.set_ylabel('Relative L2 error')
    ax2.set_title('Reconstruction error vs bottleneck')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "bottleneck_sweep.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")

    # Print summary table
    print(f"\n{'k':>4} {'cos':>8} {'rel_L2':>8} {'MSE':>10} {'params':>10}")
    print('-' * 44)
    for k in ks:
        r = results[k]
        print(f"{k:4d} {r['cos_sim']:8.4f} {r['rel_l2']:8.4f} {r['mse']:10.6f} {r['n_params']/1e6:8.2f}M")


if __name__ == '__main__':
    main()
