#!/usr/bin/env python3
"""
Train a TopK SAE on resid_post activations.

Pre-caches activations from the model, then trains a TopK sparse autoencoder.
Designed for apples-to-apples comparison with our pruned dictionary.

Usage:
    python -u scripts/jacobian/train_sae.py \
        --model meta-llama/Llama-3.2-1B --layer 2 \
        --k 32 --d_sae 16384 --n_steps 50000 \
        --output_dir results/trained_sae/llama-3.2-1b_layer2_8x_k32

    # Also train k=20 variant:
    python -u scripts/jacobian/train_sae.py \
        --model meta-llama/Llama-3.2-1B --layer 2 \
        --k 20 --d_sae 16384 --n_steps 50000 \
        --output_dir results/trained_sae/llama-3.2-1b_layer2_8x_k20 \
        --reuse_cache --cache_path results/trained_sae/llama-3.2-1b_layer2_8x_k32/cached_activations.pt
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


# ── TopK SAE ─────────────────────────────────────────────────────────────

class TopKSAE(nn.Module):
    """TopK sparse autoencoder with unit-norm decoder columns."""

    def __init__(self, d_in, d_sae, k):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

        self.encoder = nn.Linear(d_in, d_sae)
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Initialize
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.W_dec)
        # Normalize decoder columns
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)

    def encode(self, x):
        """Encode activations to sparse codes.

        Returns:
            top_vals: (batch, k) top-k activation values
            top_idx: (batch, k) top-k feature indices
            pre_acts: (batch, d_sae) full pre-top-k activations (post-ReLU)
        """
        x_centered = x - self.b_dec
        pre_acts = F.relu(self.encoder(x_centered))
        top_vals, top_idx = pre_acts.topk(self.k, dim=-1)
        return top_vals, top_idx, pre_acts

    def decode(self, top_vals, top_idx):
        """Decode sparse codes to reconstructed activations.

        Args:
            top_vals: (batch, n) activation values
            top_idx: (batch, n) feature indices

        Returns:
            recon: (batch, d_in) reconstructed activations
        """
        # Gather decoder columns for active features: (batch, n, d_in)
        W_selected = self.W_dec[top_idx]
        # Weighted sum: (batch, d_in)
        recon = (top_vals.unsqueeze(-1) * W_selected).sum(dim=1) + self.b_dec
        return recon

    def forward(self, x):
        """Full forward pass: encode then decode using top-k."""
        top_vals, top_idx, pre_acts = self.encode(x)
        recon = self.decode(top_vals, top_idx)
        return recon, top_vals, top_idx

    def normalize_decoder(self):
        """Normalize decoder columns to unit norm."""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)


# ── Dead feature resampling ──────────────────────────────────────────────

def resample_dead_features(sae, optimizer, activations, feature_fired, n_sample=8192):
    """Reinitialize dead features toward high-loss inputs."""
    dead_mask = ~feature_fired
    n_dead = dead_mask.sum().item()
    if n_dead == 0:
        return 0

    # Sample activations and find high-loss ones
    idx = torch.randint(0, activations.shape[0], (n_sample,), device=activations.device)
    sample = activations[idx]

    with torch.no_grad():
        recon, _, _ = sae(sample)
        per_sample_loss = (sample - recon).pow(2).sum(dim=-1)  # (n_sample,)

    # Use the highest-loss inputs for reinitialization
    dead_indices = dead_mask.nonzero(as_tuple=True)[0]
    n_reinit = min(n_dead, n_sample)
    top_loss_idx = per_sample_loss.topk(n_reinit).indices
    high_loss_acts = sample[top_loss_idx]  # (n_reinit, d_in)

    with torch.no_grad():
        # Reinitialize encoder rows
        for i, dead_idx in enumerate(dead_indices[:n_reinit]):
            source = high_loss_acts[i % n_reinit]
            # Encoder row: point toward high-loss input (centered)
            direction = source - sae.b_dec
            direction = direction / (direction.norm() + 1e-8)
            sae.encoder.weight.data[dead_idx] = direction * 0.1
            sae.encoder.bias.data[dead_idx] = 0.0
            # Decoder column: normalized high-loss input
            sae.W_dec.data[dead_idx] = F.normalize(source.unsqueeze(0), dim=-1).squeeze(0)

        # Reset optimizer state for reinitialized parameters
        for param in [sae.encoder.weight, sae.encoder.bias, sae.W_dec]:
            state = optimizer.state.get(param)
            if state is None:
                continue
            for key in ['exp_avg', 'exp_avg_sq']:
                if key in state:
                    if state[key].dim() >= 1 and state[key].shape[0] == sae.d_sae:
                        state[key][dead_indices[:n_reinit]] = 0

    return n_dead


# ── Activation caching ───────────────────────────────────────────────────

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

    cache_tokens = token_ids_list[args.cache_start:args.cache_start + args.n_cache]
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


# ── Training ─────────────────────────────────────────────────────────────

def train(sae, activations, args, device):
    """Train the SAE on cached activations."""
    n_cache = activations.shape[0]
    activations_gpu = activations.to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps)

    # Dead feature tracking (from step 0)
    feature_fired = torch.zeros(sae.d_sae, dtype=torch.bool, device=device)

    training_log = []
    t_start = time.time()

    print(f"\nTraining: {args.n_steps} steps, batch_size={args.batch_size}, "
          f"lr={args.lr}, k={sae.k}")
    print(f"  Cache: {n_cache} activations, "
          f"~{args.n_steps * args.batch_size / n_cache:.0f} epochs")
    print(f"  Resample interval: {args.resample_interval} steps")
    print()

    for step in range(args.n_steps):
        # Sample random batch
        idx = torch.randint(0, n_cache, (args.batch_size,), device=device)
        batch = activations_gpu[idx]

        # Forward
        recon, top_vals, top_idx = sae(batch)
        loss = F.mse_loss(recon, batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Normalize decoder columns
        sae.normalize_decoder()

        # Track which features fired
        feature_fired[top_idx.unique()] = True

        # Dead feature resampling
        if step > 0 and step % args.resample_interval == 0:
            n_dead = resample_dead_features(
                sae, optimizer, activations_gpu, feature_fired
            )
            feature_fired.zero_()
            sae.normalize_decoder()
            print(f"  Step {step}: resampled {n_dead} dead features "
                  f"({n_dead / sae.d_sae * 100:.1f}%)")

        # Logging
        if step % args.log_interval == 0:
            with torch.no_grad():
                rel_l2 = (batch - recon).norm(dim=-1).mean() / batch.norm(dim=-1).mean()
                cos_sim = F.cosine_similarity(batch, recon, dim=-1).mean()
                n_dead_now = (~feature_fired).sum().item()

            entry = {
                'step': step,
                'loss': loss.item(),
                'rel_l2': rel_l2.item(),
                'cos_sim': cos_sim.item(),
                'n_dead': n_dead_now,
                'lr': scheduler.get_last_lr()[0],
            }
            training_log.append(entry)

            if step % (args.log_interval * 10) == 0:
                elapsed = time.time() - t_start
                print(f"  Step {step:6d}: loss={loss.item():.6f}, "
                      f"rel_L2={rel_l2.item():.4f}, cos={cos_sim.item():.4f}, "
                      f"dead={n_dead_now}, lr={entry['lr']:.2e}, "
                      f"({elapsed:.0f}s)")

    total_time = time.time() - t_start
    print(f"\nTraining complete: {args.n_steps} steps in {total_time:.1f}s "
          f"({total_time / args.n_steps * 1000:.1f}ms/step)")

    return training_log


# ── Plotting ─────────────────────────────────────────────────────────────

def make_plot(training_log, output_path):
    """Plot loss curve and dead feature count."""
    steps = [e['step'] for e in training_log]
    losses = [e['loss'] for e in training_log]
    dead = [e['n_dead'] for e in training_log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(steps, losses)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, dead)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Dead Features')
    ax2.set_title('Dead Feature Count')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train TopK SAE on resid_post')
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--d_sae', type=int, default=16384, help='SAE latent dimension')
    parser.add_argument('--k', type=int, default=32, help='TopK sparsity')
    parser.add_argument('--cache_start', type=int, default=8000,
                        help='First FineWeb index for training cache')
    parser.add_argument('--n_cache', type=int, default=2000000,
                        help='Number of activations to cache')
    parser.add_argument('--n_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--resample_interval', type=int, default=25000,
                        help='Dead feature resampling interval (steps)')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--output_dir',
                        default='results/trained_sae/llama-3.2-1b_layer2_8x')
    parser.add_argument('--reuse_cache', action='store_true',
                        help='Reuse cached activations if available')
    parser.add_argument('--cache_path', default=None,
                        help='Path to cached activations (overrides default)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    d_model = 2048  # known for Llama-3.2-1B

    print("Train TopK SAE")
    print(f"  model: {args.model}, layer: {args.layer}")
    print(f"  d_in: {d_model}, d_sae: {args.d_sae}, k: {args.k}")
    print(f"  cache: [{args.cache_start}, {args.cache_start + args.n_cache})")
    print(f"  n_steps: {args.n_steps}, batch_size: {args.batch_size}, lr: {args.lr}")
    print()

    # ── Cache activations ────────────────────────────────────────────────
    model, tokenizer, device = load_model(args.model)
    activations, cache_path = cache_activations(model, tokenizer, device, args)

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # ── Train ────────────────────────────────────────────────────────────
    sae = TopKSAE(d_in=d_model, d_sae=args.d_sae, k=args.k).to(device)
    print(f"\nSAE parameters: {sum(p.numel() for p in sae.parameters()) / 1e6:.1f}M")

    training_log = train(sae, activations, args, device)

    # ── Save ─────────────────────────────────────────────────────────────
    save_data = {
        'state_dict': sae.state_dict(),
        'config': {'d_in': d_model, 'd_sae': args.d_sae, 'k': args.k},
        'training_log': training_log,
        'args': vars(args),
    }
    weights_path = os.path.join(args.output_dir, "sae_weights.pt")
    torch.save(save_data, weights_path)
    print(f"\nSaved SAE to {weights_path}")

    # Save config separately for easy inspection
    config_path = os.path.join(args.output_dir, "sae_config.json")
    with open(config_path, 'w') as f:
        json.dump({'d_in': d_model, 'd_sae': args.d_sae, 'k': args.k}, f, indent=2)

    # Save training log
    log_path = os.path.join(args.output_dir, "training_log.pkl")
    with open(log_path, 'wb') as f:
        pickle.dump(training_log, f)

    # ── Plot ─────────────────────────────────────────────────────────────
    plot_path = os.path.join(args.output_dir, "training_plot.png")
    make_plot(training_log, plot_path)

    # ── Quick eval summary ───────────────────────────────────────────────
    sae.eval()
    with torch.no_grad():
        eval_sample = activations[:10000].to(device)
        recon, _, _ = sae(eval_sample)
        mse = F.mse_loss(recon, eval_sample).item()
        rel_l2 = ((eval_sample - recon).norm(dim=-1) / eval_sample.norm(dim=-1)).mean().item()
        cos = F.cosine_similarity(eval_sample, recon, dim=-1).mean().item()
    print(f"\nFinal eval (10k cache sample): MSE={mse:.6f}, rel_L2={rel_l2:.4f}, cos={cos:.4f}")


if __name__ == '__main__':
    main()
