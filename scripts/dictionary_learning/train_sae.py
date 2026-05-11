#!/usr/bin/env python3
"""
Train a TopK SAE on resid_post activations.

Pre-caches activations from the model, then trains a TopK sparse autoencoder.
Designed for apples-to-apples comparison with our pruned dictionary.

Usage:
    python -u scripts/dictionary_learning/train_sae.py \
        --model meta-llama/Llama-3.2-1B --layer 2 \
        --k 32 --d_sae 16384 --n_steps 50000 \
        --output_dir results/trained_sae/llama-3.2-1b_layer2_8x_k32

    # Also train k=20 variant:
    python -u scripts/dictionary_learning/train_sae.py \
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
from src.autoencoders import TopKSAE


# ── Dead feature resampling ──────────────────────────────────────────────

def resample_dead_features(sae, optimizer, activations, feature_fired, n_sample=8192):
    """Reinitialize dead features toward high-loss inputs."""
    dead_mask = ~feature_fired
    n_dead = dead_mask.sum().item()
    if n_dead == 0:
        return 0

    # Sample activations and find high-loss ones
    sae_device = sae.encoder.weight.device
    idx = torch.randint(0, activations.shape[0], (n_sample,))
    sample = activations[idx].to(sae_device).float()

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
        meta_path = cache_path + '.meta.json'
        if os.path.exists(meta_path):
            print(f"Loading cached activations (memmap) from {cache_path}")
            with open(meta_path) as f:
                meta = json.load(f)
            shape = tuple(meta['shape'])
            np_dtype = np.dtype(meta['dtype'])
            mm = np.memmap(cache_path, dtype=np_dtype, mode='r', shape=shape)
            activations = torch.from_numpy(mm)
        else:
            print(f"Loading cached activations from {cache_path}")
            activations = torch.load(cache_path, weights_only=True)
        print(f"  Shape: {activations.shape}, dtype: {activations.dtype}")
        if args.max_samples is not None and args.max_samples < activations.shape[0]:
            activations = activations[:args.max_samples]
            print(f"  Truncated to first {args.max_samples} samples -> {activations.shape}")
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

    batch_size = 256
    t0 = time.time()

    # Hook-based extraction: only run through target layer, skip the rest
    captured = {}

    # Capture in fp16 if --cache_fp16, else fp32
    save_dtype = torch.float16 if getattr(args, 'cache_fp16', False) else torch.float32
    np_dtype = np.float16 if save_dtype == torch.float16 else np.float32

    def capture_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured['act'] = hidden[:, -1, :].to(save_dtype).cpu()

    class EarlyExit(Exception):
        pass

    def exit_hook(module, input, output):
        raise EarlyExit()

    layers = model.model.layers
    handle_capture = layers[args.layer].register_forward_hook(capture_hook)
    handle_exit = None
    if args.layer + 1 < len(layers):
        handle_exit = layers[args.layer + 1].register_forward_hook(exit_hook)

    # Probe d_model by running one small batch
    probe = torch.stack(cache_tokens[:1]).to(device)
    with torch.no_grad():
        try:
            model(probe)
        except EarlyExit:
            pass
    d_model = captured['act'].shape[-1]

    # Pre-allocate a disk-backed memmap so RAM stays tiny (~1 batch).
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    memmap_path = cache_path  # write raw bytes here; sidecar stores meta
    memmap = np.memmap(memmap_path, dtype=np_dtype, mode='w+',
                       shape=(n_cache, d_model))
    print(f"Streaming to memmap at {memmap_path}: shape ({n_cache}, {d_model}), {np_dtype}")

    for i in range(0, n_cache, batch_size):
        batch = torch.stack(cache_tokens[i:i+batch_size]).to(device)
        with torch.no_grad():
            try:
                model(batch)
            except EarlyExit:
                pass
        act = captured['act']
        memmap[i:i+act.shape[0]] = act.numpy()
        if (i // batch_size) % 100 == 0:
            done = min(i + batch_size, n_cache)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (n_cache - done) / rate if rate > 0 else 0
            print(f"  {done}/{n_cache} ({rate:.0f} acts/s, ~{remaining:.0f}s remaining)",
                  end='\r')

    memmap.flush()
    del memmap
    handle_capture.remove()
    if handle_exit is not None:
        handle_exit.remove()
    print()

    # Sidecar metadata for loaders.
    meta_path = cache_path + '.meta.json'
    with open(meta_path, 'w') as f:
        json.dump({'shape': [n_cache, d_model],
                   'dtype': str(np_dtype().dtype),
                   'format': 'memmap_raw'}, f)

    nbytes = n_cache * d_model * np.dtype(np_dtype).itemsize
    print(f"Saved raw memmap to {cache_path} ({nbytes / 1e9:.1f} GB, {np_dtype})")
    print(f"  Meta: {meta_path}")

    # Return a memmap-backed tensor view (no copy) so callers keep working.
    mm = np.memmap(cache_path, dtype=np_dtype, mode='r', shape=(n_cache, d_model))
    activations = torch.from_numpy(mm)
    return activations, cache_path


# ── Training ─────────────────────────────────────────────────────────────

def train(sae, activations, args, device):
    """Train the SAE on cached activations."""
    # Val split: hold out the last val_frac of the cache.
    val_frac = getattr(args, 'val_frac', 0.1)
    n_total = activations.shape[0]
    n_val = int(n_total * val_frac)
    n_cache = n_total - n_val
    acts_train = activations[:n_cache]
    acts_val = activations[n_cache:n_cache + n_val]
    print(f"  Train split: {n_cache} | Val split: {n_val}")

    # If cache fits in GPU memory (fp32), upload once. Otherwise keep on CPU
    # (memmap) and batch to GPU each step.
    fp32_gb = n_cache * activations.shape[1] * 4 / 1e9
    if fp32_gb < 8:
        activations_gpu = acts_train.to(device).float()
        cpu_batched = False
    else:
        activations_gpu = acts_train  # stays on CPU (possibly memmap)
        cpu_batched = True
        print(f"  Cache is {fp32_gb:.1f} GB fp32 — streaming batches CPU→GPU")

    # Prebuild a single validation batch (10k rows) on GPU as fp32.
    val_sample_n = min(10000, n_val)
    val_batch = acts_val[:val_sample_n].to(device).float()

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
        if cpu_batched:
            idx = torch.randint(0, n_cache, (args.batch_size,))
            batch = activations_gpu[idx].to(device).float()
        else:
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

                # Val metrics on held-out batch
                sae.eval()
                val_recon, _, _ = sae(val_batch)
                val_rel_l2 = ((val_batch - val_recon).norm(dim=-1)
                              / val_batch.norm(dim=-1)).mean()
                val_cos = F.cosine_similarity(val_batch, val_recon, dim=-1).mean()
                sae.train()

            entry = {
                'step': step,
                'loss': loss.item(),
                'rel_l2': rel_l2.item(),
                'cos_sim': cos_sim.item(),
                'val_rel_l2': val_rel_l2.item(),
                'val_cos': val_cos.item(),
                'n_dead': n_dead_now,
                'lr': scheduler.get_last_lr()[0],
            }
            training_log.append(entry)

            if step % (args.log_interval * 10) == 0:
                elapsed = time.time() - t_start
                print(f"  Step {step:6d}: loss={loss.item():.6f}, "
                      f"rel_L2={rel_l2.item():.4f}, cos={cos_sim.item():.4f}, "
                      f"val_cos={val_cos.item():.4f}, "
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
    parser.add_argument('--cache_fp16', action='store_true',
                        help='Save cached activations in float16 to halve disk usage')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='When reusing a cache, truncate to the first N samples.')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Fraction of cache held out for validation.')
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

    # ── Quick eval summary (held-out val tail) ───────────────────────────
    val_frac = args.val_frac
    n_total = activations.shape[0]
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val
    sae.eval()
    with torch.no_grad():
        eval_sample = activations[n_train:n_train + min(10000, n_val)].to(device).float()
        recon, _, _ = sae(eval_sample)
        mse = F.mse_loss(recon, eval_sample).item()
        rel_l2 = ((eval_sample - recon).norm(dim=-1) / eval_sample.norm(dim=-1)).mean().item()
        cos = F.cosine_similarity(eval_sample, recon, dim=-1).mean().item()
    print(f"\nFinal eval (10k val sample): MSE={mse:.6f}, rel_L2={rel_l2:.4f}, cos={cos:.4f}")


if __name__ == '__main__':
    main()
