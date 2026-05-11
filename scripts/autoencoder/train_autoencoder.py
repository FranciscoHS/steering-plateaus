#!/usr/bin/env python3
"""
Train a vanilla deep autoencoder (MLP encoder/decoder) on resid_post activations.

k-dimensional bottleneck, MSE reconstruction loss. No sparsity constraints.
Reuses activation caching infrastructure from train_sae.py.

Usage:
    python -u scripts/autoencoder/train_autoencoder.py \
        --model Qwen/Qwen3-1.7B --layer 2 --k 10 \
        --output_dir results/trained_autoencoder/qwen3-1.7b_layer2_k10

    # Reuse cached activations from a previous run:
    python -u scripts/autoencoder/train_autoencoder.py \
        --model Qwen/Qwen3-1.7B --layer 2 --k 10 \
        --reuse_cache --cache_path results/trained_autoencoder/qwen3-1.7b_layer2_k10/cached_activations.pt \
        --output_dir results/trained_autoencoder/qwen3-1.7b_layer2_k20
"""

import argparse
import json
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from src.model import load_model
from src.data import load_fineweb_fixed_length
from src.autoencoders import DeepAutoencoder


# ── Activation caching (reused from train_sae.py) ──────────────────────

def cache_activations(model, tokenizer, device, args):
    """Extract and cache resid_post activations."""
    cache_path = args.cache_path or os.path.join(args.output_dir, "cached_activations.pt")

    if args.reuse_cache and os.path.exists(cache_path):
        print(f"Loading cached activations from {cache_path}")
        activations = torch.load(cache_path, weights_only=True)
        print(f"  Shape: {activations.shape}")
        return activations, cache_path

    n_total_needed = args.cache_start + args.n_cache
    print(f"Loading {n_total_needed} FineWeb samples (seed={args.seed}, seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        n_total_needed, tokenizer, seq_len=args.seq_len, seed=args.seed
    )

    n_extract = args.n_extract or args.n_cache
    cache_tokens = token_ids_list[args.cache_start:args.cache_start + n_extract]
    n_cache = len(cache_tokens)
    print(f"Extracting {n_cache} activations at layer {args.layer} "
          f"(indices [{args.cache_start}, {args.cache_start + n_cache}))...")

    all_activations = []
    batch_size = 64
    t0 = time.time()
    for i in range(0, n_cache, batch_size):
        batch = torch.stack(cache_tokens[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
        hidden = outputs.hidden_states[args.layer + 1]
        all_activations.append(hidden[:, -1, :].float().cpu())
        if (i // batch_size) % 100 == 0:
            done = min(i + batch_size, n_cache)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (n_cache - done) / rate if rate > 0 else 0
            print(f"  {done}/{n_cache} ({rate:.0f} acts/s, ~{remaining:.0f}s remaining)",
                  end='\r')
    print()

    activations = torch.cat(all_activations, dim=0)
    print(f"Cached {activations.shape[0]} activations, "
          f"mean norm={activations.norm(dim=-1).mean():.2f}")

    torch.save(activations, cache_path)
    print(f"Saved to {cache_path} ({activations.nelement() * 4 / 1e9:.1f} GB)")

    return activations, cache_path


# ── Training ────────────────────────────────────────────────────────────

def train(ae, activations, args, device):
    """Train the autoencoder on cached activations."""
    n_cache = activations.shape[0]
    activations_gpu = activations.to(device)

    optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps)

    training_log = []
    t_start = time.time()

    print(f"\nTraining: {args.n_steps} steps, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  Bottleneck k={ae.k}")
    print(f"  Cache: {n_cache} activations, "
          f"~{args.n_steps * args.batch_size / n_cache:.1f} epochs")
    print()

    for step in range(args.n_steps):
        # Sample random batch
        idx = torch.randint(0, n_cache, (args.batch_size,), device=device)
        batch = activations_gpu[idx]

        # Forward
        recon, z = ae(batch)
        loss = F.mse_loss(recon, batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        if step % args.log_interval == 0:
            with torch.no_grad():
                rel_l2 = (batch - recon).norm(dim=-1).mean() / batch.norm(dim=-1).mean()
                cos_sim = F.cosine_similarity(batch, recon, dim=-1).mean()

            entry = {
                'step': step,
                'loss': loss.item(),
                'rel_l2': rel_l2.item(),
                'cos_sim': cos_sim.item(),
                'lr': scheduler.get_last_lr()[0],
            }
            training_log.append(entry)

            if step % (args.log_interval * 10) == 0:
                elapsed = time.time() - t_start
                print(f"  Step {step:6d}: loss={loss.item():.6f}, "
                      f"rel_L2={rel_l2.item():.4f}, cos={cos_sim.item():.4f}, "
                      f"lr={entry['lr']:.2e}, ({elapsed:.0f}s)")

    total_time = time.time() - t_start
    print(f"\nTraining complete: {args.n_steps} steps in {total_time:.1f}s "
          f"({total_time / args.n_steps * 1000:.1f}ms/step)")

    return training_log


# ── Plotting ────────────────────────────────────────────────────────────

def make_plot(training_log, output_path):
    """Plot training loss curve."""
    steps = [e['step'] for e in training_log]
    losses = [e['loss'] for e in training_log]
    cos = [e['cos_sim'] for e in training_log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(steps, losses)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, cos)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Reconstruction Quality')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train deep autoencoder on resid_post')
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--k', type=int, default=10, help='Bottleneck dimension')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=None,
                        help='Hidden layer dims for encoder (default: d_in//2, d_in//4)')
    parser.add_argument('--cache_start', type=int, default=8000,
                        help='First FineWeb index for training cache')
    parser.add_argument('--n_cache', type=int, default=2000000,
                        help='Number of FineWeb tokens to load (for cache matching)')
    parser.add_argument('--n_extract', type=int, default=None,
                        help='Number of activations to extract (default: n_cache)')
    parser.add_argument('--n_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--output_dir',
                        default='results/trained_autoencoder/qwen3-1.7b_layer2_k10')
    parser.add_argument('--reuse_cache', action='store_true',
                        help='Reuse cached activations if available')
    parser.add_argument('--cache_path', default=None,
                        help='Path to cached activations (overrides default)')
    parser.add_argument('--batchnorm', action='store_true',
                        help='Add BatchNorm1d between layers')
    parser.add_argument('--resume_from', default=None,
                        help='Path to ae_weights.pt to resume training from')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # Load model to get d_model
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']

    hidden_dims = args.hidden_dims if args.hidden_dims else [d_model // 2, d_model // 4]

    print("Train Deep Autoencoder")
    print(f"  model: {args.model}, layer: {args.layer}")
    print(f"  d_in: {d_model}, k: {args.k}")
    print(f"  encoder: {d_model} → {' → '.join(str(h) for h in hidden_dims)} → {args.k}")
    print(f"  decoder: {args.k} → {' → '.join(str(h) for h in reversed(hidden_dims))} → {d_model}")
    print(f"  cache: [{args.cache_start}, {args.cache_start + args.n_cache})")
    print(f"  n_steps: {args.n_steps}, batch_size: {args.batch_size}, lr: {args.lr}")
    print()

    # ── Cache activations ────────────────────────────────────────────────
    activations, cache_path = cache_activations(model, tokenizer, device, args)

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # ── Train ────────────────────────────────────────────────────────────
    ae = DeepAutoencoder(d_in=d_model, k=args.k, hidden_dims=hidden_dims,
                         use_batchnorm=args.batchnorm).to(device)
    n_params = sum(p.numel() for p in ae.parameters())
    print(f"\nAutoencoder parameters: {n_params / 1e6:.2f}M")

    if args.resume_from:
        print(f"Resuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, weights_only=True)
        ae.load_state_dict(checkpoint['state_dict'])
        print("  Loaded weights")

    training_log = train(ae, activations, args, device)

    # ── Save ─────────────────────────────────────────────────────────────
    save_data = {
        'state_dict': ae.state_dict(),
        'config': {
            'd_in': d_model, 'k': args.k,
            'hidden_dims': hidden_dims,
        },
        'training_log': training_log,
        'args': vars(args),
    }
    weights_path = os.path.join(args.output_dir, "ae_weights.pt")
    torch.save(save_data, weights_path)
    print(f"\nSaved autoencoder to {weights_path}")

    config_path = os.path.join(args.output_dir, "ae_config.json")
    with open(config_path, 'w') as f:
        json.dump(save_data['config'], f, indent=2)

    log_path = os.path.join(args.output_dir, "training_log.pkl")
    with open(log_path, 'wb') as f:
        pickle.dump(training_log, f)

    # ── Plot ─────────────────────────────────────────────────────────────
    plot_path = os.path.join(args.output_dir, "training_plot.png")
    make_plot(training_log, plot_path)

    # ── Quick eval summary ───────────────────────────────────────────────
    ae.eval()
    with torch.no_grad():
        eval_sample = activations[:10000].to(device)
        recon, z = ae(eval_sample)
        mse = F.mse_loss(recon, eval_sample).item()
        rel_l2 = ((eval_sample - recon).norm(dim=-1) / eval_sample.norm(dim=-1)).mean().item()
        cos = F.cosine_similarity(eval_sample, recon, dim=-1).mean().item()
    print(f"\nFinal eval (10k cache sample): MSE={mse:.6f}, rel_L2={rel_l2:.4f}, cos={cos:.4f}")


if __name__ == '__main__':
    main()
