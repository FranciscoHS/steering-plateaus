#!/usr/bin/env python3
"""
Hillclimb autoencoder reconstruction quality.

Systematically tests architecture/training variations to improve cos similarity.
Fixed constraints: ~67M params, k=30 bottleneck, 10k training steps.

Usage:
    python -u scripts/autoencoder/hillclimb_autoencoder.py \
        --cache_path /path/to/cached_activations.pt \
        --experiments baseline gelu silu \
        --output_dir results/trained_autoencoder/hillclimb
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Canonical FlexAutoencoder lives in src.autoencoders so probe/steering
# scripts can share one definition. Block/SwiGLU/GEGLU/ACTIVATIONS are
# re-exported in case other modules reach in for them.
from src.autoencoders import (
    FlexAutoencoder,
    Block,
    SwiGLU,
    GEGLU,
    ACTIVATIONS,
)


class FiLMAutoencoder(nn.Module):
    """Autoencoder where decoder layers are FiLM-conditioned on bottleneck z.

    Each decoder layer receives gamma, beta = linear(z) and computes
    x = gamma * block(x) + beta, giving the decoder access to z at every layer.
    """

    def __init__(self, d_in, k, hidden_dims, activation='gelu', norm='batchnorm',
                 bottleneck_noise=0.0):
        super().__init__()
        self.d_in = d_in
        self.k = k
        self.bottleneck_noise = bottleneck_noise

        enc_dims = hidden_dims
        dec_dims = list(reversed(hidden_dims))

        # Encoder (same as FlexAutoencoder)
        encoder_blocks = []
        prev = d_in
        for h in enc_dims:
            encoder_blocks.append(Block(prev, h, activation, norm))
            prev = h
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.encoder_out = nn.Linear(prev, k)

        # Decoder with FiLM conditioning
        decoder_blocks = []
        film_layers = []
        prev = k
        for h in dec_dims:
            decoder_blocks.append(Block(prev, h, activation, norm))
            # FiLM: z -> (gamma, beta) for this layer's output dim
            film_layers.append(nn.Linear(k, h * 2))
            prev = h
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.film_layers = nn.ModuleList(film_layers)
        self.decoder_out = nn.Linear(prev, d_in)

    def encode(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        return self.encoder_out(x)

    def decode(self, z):
        x = z
        for block, film in zip(self.decoder_blocks, self.film_layers):
            x = block(x)
            # FiLM modulation
            gb = film(z)
            gamma, beta = gb.chunk(2, dim=-1)
            x = (1 + gamma) * x + beta
        return self.decoder_out(x)

    def forward(self, x):
        z = self.encode(x)
        if self.training and self.bottleneck_noise > 0:
            z = z + torch.randn_like(z) * self.bottleneck_noise
        recon = self.decode(z)
        return recon, z


# ── EMA helper ────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)  # integer buffers (e.g. BN num_batches_tracked)

    def apply(self, model):
        """Temporarily apply EMA weights. Returns original state_dict."""
        original = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return original

    def restore(self, model, original):
        """Restore original weights after EMA eval."""
        model.load_state_dict(original)


# ── Loss functions ──────────────────────────────────────────────────────

def compute_loss(recon, batch, z, loss_cfg, device):
    """Compute training loss based on config."""
    loss_type = loss_cfg['type']

    if loss_type == 'mse':
        return F.mse_loss(recon, batch)
    elif loss_type == 'cosine':
        return 1.0 - F.cosine_similarity(recon, batch, dim=-1).mean()
    elif loss_type == 'mse+cosine':
        alpha = loss_cfg.get('alpha', 0.1)
        mse = F.mse_loss(recon, batch)
        cos = 1.0 - F.cosine_similarity(recon, batch, dim=-1).mean()
        return mse + alpha * cos
    elif loss_type == 'huber':
        return F.smooth_l1_loss(recon, batch)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def decorrelation_penalty(z):
    """Penalize correlations between bottleneck dimensions."""
    z_c = z - z.mean(dim=0)
    cov = (z_c.T @ z_c) / z.shape[0]
    d = torch.sqrt(torch.diag(cov) + 1e-8)
    corr = cov / (d.unsqueeze(0) * d.unsqueeze(1))
    eye = torch.eye(corr.shape[0], device=corr.device)
    return (corr - eye).pow(2).mean()


# ── Preprocessing ───────────────────────────────────────────────────────

def preprocess(activations, mode):
    """Preprocess activations. Returns (processed, state_dict)."""
    if mode == 'none':
        return activations, {}

    mean = activations.mean(dim=0)

    if mode == 'center':
        return activations - mean, {'mean': mean}

    elif mode == 'whiten':
        centered = activations - mean
        # SVD on subsample for efficiency
        sub = centered[:50000]
        _, S, Vt = torch.linalg.svd(sub, full_matrices=False)
        # Whitening: scale by 1/std per component
        scale = S / math.sqrt(len(sub))
        whiten_mat = Vt.T @ torch.diag(1.0 / (scale + 1e-5))
        unwhiten_mat = torch.diag(scale + 1e-5) @ Vt
        whitened = centered @ whiten_mat
        return whitened, {'mean': mean, 'whiten_mat': whiten_mat,
                         'unwhiten_mat': unwhiten_mat}

    raise ValueError(f"Unknown preprocess mode: {mode}")


def inverse_preprocess(recon, state, mode):
    """Inverse preprocess reconstruction back to original space."""
    if mode == 'none':
        return recon
    elif mode == 'center':
        return recon + state['mean'].to(recon.device)
    elif mode == 'whiten':
        unwhitened = recon @ state['unwhiten_mat'].to(recon.device)
        return unwhitened + state['mean'].to(recon.device)


# ── Run one experiment ──────────────────────────────────────────────────

def run_experiment(config, activations_raw, device):
    """Train one autoencoder config, return metrics."""
    torch.manual_seed(config['seed'])
    name = config['name']

    # Preprocess
    activations, pp_state = preprocess(activations_raw, config['preprocess'])

    # Handle PCA baseline (no neural network)
    if config.get('pca_baseline', False):
        return _run_pca_baseline(config, activations_raw, device)

    # Build model
    if config.get('film_decoder', False):
        ae = FiLMAutoencoder(
            d_in=config['d_in'], k=config['k'],
            hidden_dims=config['hidden_dims'],
            activation=config['activation'],
            norm=config['norm'],
            bottleneck_noise=config.get('bottleneck_noise', 0.0),
        ).to(device)
    else:
        ae = FlexAutoencoder(
            d_in=config['d_in'], k=config['k'],
            hidden_dims=config['hidden_dims'],
            activation=config['activation'],
            norm=config['norm'],
            residual=config['residual'],
            bottleneck_noise=config.get('bottleneck_noise', 0.0),
            encoder_dims=config.get('encoder_dims'),
            decoder_dims=config.get('decoder_dims'),
            dropout=config.get('dropout', 0.0),
            spectral_norm=config.get('spectral_norm', False),
        ).to(device)

    n_params = sum(p.numel() for p in ae.parameters())
    print(f"  Params: {n_params / 1e6:.2f}M")

    # PCA initialization of bottleneck layers
    if config.get('pca_init', False):
        with torch.no_grad():
            sub = activations[:50000]
            mean = sub.mean(dim=0)
            centered = sub - mean
            _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
            pca_dirs = Vt[:config['k']].to(device)  # (k, d_in)
            mean_d = mean.to(device)
            # Init encoder_out if it maps directly from d_in
            if ae.encoder_out.in_features == config['d_in']:
                ae.encoder_out.weight.data = pca_dirs
                ae.encoder_out.bias.data.zero_()
            # Init decoder_out if it maps directly to d_in from k
            if ae.decoder_out.in_features == config['k']:
                ae.decoder_out.weight.data = pca_dirs.T
                ae.decoder_out.bias.data = mean_d
        print("  PCA initialized bottleneck layers")

    # Optimizer
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(ae.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(ae.parameters(), lr=config['lr'],
                                       weight_decay=config['weight_decay'])

    # Scheduler with optional warmup and warm restarts
    warmup = config['warmup_steps']
    sched_type = config.get('scheduler', 'cosine')

    if sched_type == 'warm_restarts':
        t0_restart = config.get('restart_period', 2500)
        t_mult = config.get('restart_mult', 2)
        if warmup > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-7 / config['lr'], total_iters=warmup)
            restart_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=t0_restart, T_mult=t_mult)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_sched, restart_sched], milestones=[warmup])
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=t0_restart, T_mult=t_mult)
    elif warmup > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-7 / config['lr'], total_iters=warmup)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['n_steps'] - warmup)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_sched, cosine_sched], milestones=[warmup])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['n_steps'])

    # EMA
    ema = None
    if config.get('ema_decay', 0) > 0:
        ema = EMA(ae, decay=config['ema_decay'])

    grad_clip = config.get('grad_clip', 0)

    # Training — keep activations on CPU, batch to GPU to save VRAM
    val_frac = config.get('val_frac', 0.1)
    patience = config.get('patience', 5000)
    val_interval = config.get('val_interval', 500)

    n_total = activations.shape[0]
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val
    activations_train = activations[:n_train]  # stays on CPU
    activations_val = activations[n_train:]    # stays on CPU

    t0 = time.time()

    best_val_cos = -1
    best_state_dict = None
    steps_since_improvement = 0

    checkpoint_interval = config.get('checkpoint_interval', 0)
    checkpoint_path = None
    if checkpoint_interval > 0 and config.get('save_model', False):
        save_dir = config.get('save_dir', '.')
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, f"{name}_model_inprogress.pt")

    for step in range(config['n_steps']):
        idx = torch.randint(0, n_train, (config['batch_size'],))
        batch = activations_train[idx].to(device).float()

        recon, z = ae(batch)
        loss = compute_loss(recon, batch, z, config['loss'], device)

        if config['decorr_weight'] > 0:
            loss = loss + config['decorr_weight'] * decorrelation_penalty(z)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=grad_clip)

        optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update(ae)

        # Validation + early stopping
        if step % val_interval == 0:
            ae.eval()
            with torch.no_grad():
                val_batch = activations_val[:10000].to(device).float()
                val_recon, _ = ae(val_batch)
                val_cos = F.cosine_similarity(val_batch, val_recon, dim=-1).mean().item()
                train_cos = F.cosine_similarity(batch, recon, dim=-1).mean().item()

            if step % 1000 == 0:
                print(f"    Step {step:5d}: loss={loss.item():.6f}, "
                      f"train_cos={train_cos:.4f}, val_cos={val_cos:.4f}")

            if val_cos > best_val_cos:
                best_val_cos = val_cos
                best_state_dict = {k: v.cpu().clone() for k, v in ae.state_dict().items()}
                steps_since_improvement = 0
            else:
                steps_since_improvement += val_interval

            if checkpoint_path is not None and step > 0 and step % checkpoint_interval == 0:
                torch.save({
                    'state_dict': best_state_dict,
                    'config': {k: v for k, v in config.items()
                               if k not in ('save_model', 'save_dir')},
                    'step': step,
                    'best_val_cos': best_val_cos,
                }, checkpoint_path)

            if steps_since_improvement >= patience and step > config.get('warmup_steps', 0):
                print(f"    Early stopping at step {step} "
                      f"(no improvement for {patience} steps, best val_cos={best_val_cos:.4f})")
                break

            ae.train()
        elif step % 1000 == 0:
            with torch.no_grad():
                cos = F.cosine_similarity(batch, recon, dim=-1).mean()
            print(f"    Step {step:5d}: loss={loss.item():.6f}, cos={cos.item():.4f}")

    elapsed = time.time() - t0

    # Restore best checkpoint
    if best_state_dict is not None:
        ae.load_state_dict(best_state_dict)
        print(f"  Restored best checkpoint (val_cos={best_val_cos:.4f})")

    # Apply EMA weights for eval if available
    original_state = None
    if ema is not None:
        original_state = ema.apply(ae)
        print("  Using EMA weights for eval")

    # Eval on raw activations
    ae.eval()
    with torch.no_grad():
        eval_raw = activations_raw[:10000].to(device).float()
        eval_input = eval_raw
        if config['preprocess'] != 'none':
            if config['preprocess'] == 'center':
                eval_input = activations_raw[:10000].to(device).float() - pp_state['mean'].to(device)
            elif config['preprocess'] == 'whiten':
                centered = activations_raw[:10000].to(device).float() - pp_state['mean'].to(device)
                eval_input = centered @ pp_state['whiten_mat'].to(device)

        recon_pp, _ = ae(eval_input)
        recon_raw = inverse_preprocess(recon_pp, pp_state, config['preprocess'])

        cos_sim = F.cosine_similarity(eval_raw, recon_raw, dim=-1).mean().item()
        mse = F.mse_loss(recon_raw, eval_raw).item()
        rel_l2 = ((eval_raw - recon_raw).norm(dim=-1) / eval_raw.norm(dim=-1)).mean().item()

    if original_state is not None:
        ema.restore(ae, original_state)

    print(f"  Final: cos={cos_sim:.4f}, mse={mse:.6f}, rel_L2={rel_l2:.4f} ({elapsed:.0f}s)")

    # Save model weights if requested
    if config.get('save_model', False):
        save_dir = config.get('save_dir', '.')
        os.makedirs(save_dir, exist_ok=True)
        save_data = {
            'state_dict': ae.state_dict(),
            'config': {k: v for k, v in config.items()
                       if k not in ('save_model', 'save_dir')},
        }
        save_path = os.path.join(save_dir, f"{name}_model.pt")
        torch.save(save_data, save_path)
        print(f"  Saved model to {save_path}")

    return {
        'name': name,
        'cos_sim': cos_sim,
        'mse': mse,
        'rel_l2': rel_l2,
        'n_params': n_params,
        'time': elapsed,
    }


def _run_pca_baseline(config, activations_raw, device):
    """Compute PCA-k reconstruction quality (no neural network)."""
    import time
    t0 = time.time()
    k = config['k']
    sub = activations_raw[:50000].to(device)
    mean = sub.mean(dim=0)
    centered = sub - mean
    _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
    pca_dirs = Vt[:k]  # (k, d_in)

    # Eval
    eval_raw = activations_raw[:10000].to(device)
    centered_eval = eval_raw - mean
    projected = centered_eval @ pca_dirs.T  # (n, k)
    recon = projected @ pca_dirs + mean  # (n, d_in)

    cos_sim = F.cosine_similarity(eval_raw, recon, dim=-1).mean().item()
    mse = F.mse_loss(recon, eval_raw).item()
    rel_l2 = ((eval_raw - recon).norm(dim=-1) / eval_raw.norm(dim=-1)).mean().item()
    elapsed = time.time() - t0

    print(f"  PCA-{k}: cos={cos_sim:.4f}, mse={mse:.6f}, rel_L2={rel_l2:.4f} ({elapsed:.0f}s)")
    return {
        'name': config['name'],
        'cos_sim': cos_sim,
        'mse': mse,
        'rel_l2': rel_l2,
        'n_params': 0,
        'time': elapsed,
    }


# ── Experiment configs ──────────────────────────────────────────────────

BASELINE = {
    'd_in': 2048,
    'k': 30,
    'hidden_dims': [4096, 4096, 2048],
    'activation': 'relu',
    'norm': 'batchnorm',
    'residual': False,
    'preprocess': 'none',
    'loss': {'type': 'mse'},
    'optimizer': 'adam',
    'weight_decay': 0,
    'lr': 1e-3,
    'warmup_steps': 0,
    'n_steps': 10000,
    'batch_size': 4096,
    'decorr_weight': 0,
    'val_frac': 0.1,
    'patience': 5000,
    'val_interval': 500,
    'seed': 42,
}

ALL_EXPERIMENTS = {
    # Baseline
    'baseline': {},

    # Activation functions
    'gelu': {'activation': 'gelu'},
    'silu': {'activation': 'silu'},

    # Data preprocessing
    'center': {'preprocess': 'center'},
    'whiten': {'preprocess': 'whiten'},

    # LR warmup
    'warmup_500': {'warmup_steps': 500},
    'warmup_1000': {'warmup_steps': 1000},

    # Loss functions
    'cosine_loss': {'loss': {'type': 'cosine'}},
    'mse+cos_0.1': {'loss': {'type': 'mse+cosine', 'alpha': 0.1}},
    'mse+cos_1.0': {'loss': {'type': 'mse+cosine', 'alpha': 1.0}},
    'huber': {'loss': {'type': 'huber'}},

    # AdamW
    'adamw_1e-2': {'optimizer': 'adamw', 'weight_decay': 1e-2},
    'adamw_1e-1': {'optimizer': 'adamw', 'weight_decay': 1e-1},

    # Residual connections
    'residual': {'residual': True},

    # Decorrelation
    'decorr_0.01': {'decorr_weight': 0.01},
    'decorr_0.1': {'decorr_weight': 0.1},

    # Layer norm
    'layernorm': {'norm': 'layernorm'},

    # ── Round 2: combinations of winners ──
    'gelu+warmup500': {'activation': 'gelu', 'warmup_steps': 500},
    'gelu+warmup1000': {'activation': 'gelu', 'warmup_steps': 1000},
    'gelu+mse+cos_1.0': {'activation': 'gelu', 'loss': {'type': 'mse+cosine', 'alpha': 1.0}},
    'gelu+adamw_1e-1': {'activation': 'gelu', 'optimizer': 'adamw', 'weight_decay': 1e-1},
    'gelu+warmup500+adamw': {'activation': 'gelu', 'warmup_steps': 500,
                              'optimizer': 'adamw', 'weight_decay': 1e-1},
    'gelu+warmup500+mse+cos': {'activation': 'gelu', 'warmup_steps': 500,
                                'loss': {'type': 'mse+cosine', 'alpha': 1.0}},
    'gelu+warmup500+adamw+mse+cos': {'activation': 'gelu', 'warmup_steps': 500,
                                      'optimizer': 'adamw', 'weight_decay': 1e-1,
                                      'loss': {'type': 'mse+cosine', 'alpha': 1.0}},
    'silu+warmup500': {'activation': 'silu', 'warmup_steps': 500},
    'silu+adamw_1e-1': {'activation': 'silu', 'optimizer': 'adamw', 'weight_decay': 1e-1},

    # ── Round 3: new ideas from critique ──

    # Bottleneck noise (denoising autoencoder)
    'best+noise_0.1': {'activation': 'gelu', 'warmup_steps': 500,
                        'loss': {'type': 'mse+cosine', 'alpha': 1.0},
                        'bottleneck_noise': 0.1},
    'best+noise_0.5': {'activation': 'gelu', 'warmup_steps': 500,
                        'loss': {'type': 'mse+cosine', 'alpha': 1.0},
                        'bottleneck_noise': 0.5},
    'best+noise_1.0': {'activation': 'gelu', 'warmup_steps': 500,
                        'loss': {'type': 'mse+cosine', 'alpha': 1.0},
                        'bottleneck_noise': 1.0},

    # Spectral normalization
    'best+specnorm': {'activation': 'gelu', 'warmup_steps': 500,
                       'loss': {'type': 'mse+cosine', 'alpha': 1.0},
                       'spectral_norm': True},

    # PCA init (shallow encoder: just d_in -> k, deep decoder)
    'pca_shallow': {'encoder_dims': [], 'decoder_dims': [2048, 4096, 4096, 2048],
                    'activation': 'gelu', 'norm': 'batchnorm',
                    'warmup_steps': 500, 'loss': {'type': 'mse+cosine', 'alpha': 1.0},
                    'pca_init': True},

    # Asymmetric: shallow encoder, deep decoder (SAE-like structure)
    # encoder: 2048 -> 2048 -> 30, decoder: 30 -> 2048 -> 4096 -> 4096 -> 2048
    'asymmetric_shallow_enc': {
        'encoder_dims': [2048], 'decoder_dims': [2048, 4096, 4096, 2048],
        'activation': 'gelu', 'norm': 'batchnorm',
        'warmup_steps': 500, 'loss': {'type': 'mse+cosine', 'alpha': 1.0}},

    # Asymmetric: deep encoder, shallow decoder
    'asymmetric_shallow_dec': {
        'encoder_dims': [4096, 4096, 2048], 'decoder_dims': [2048],
        'activation': 'gelu', 'norm': 'batchnorm',
        'warmup_steps': 500, 'loss': {'type': 'mse+cosine', 'alpha': 1.0}},

    # Cosine loss alpha sweep (finer)
    'best+cos_0.5': {'activation': 'gelu', 'warmup_steps': 500,
                      'loss': {'type': 'mse+cosine', 'alpha': 0.5}},
    'best+cos_2.0': {'activation': 'gelu', 'warmup_steps': 500,
                      'loss': {'type': 'mse+cosine', 'alpha': 2.0}},
    'best+cos_5.0': {'activation': 'gelu', 'warmup_steps': 500,
                      'loss': {'type': 'mse+cosine', 'alpha': 5.0}},

    # ── Round 4: new ideas from brainstorm+critique ──

    # PCA baseline (no neural network, just for calibration)
    'pca_30': {'pca_baseline': True},

    # EMA of weights
    'best+ema_0.999': {'activation': 'gelu', 'warmup_steps': 500,
                        'loss': {'type': 'mse+cosine', 'alpha': 5.0},
                        'ema_decay': 0.999},
    'best+ema_0.9999': {'activation': 'gelu', 'warmup_steps': 500,
                         'loss': {'type': 'mse+cosine', 'alpha': 5.0},
                         'ema_decay': 0.9999},

    # Gradient clipping
    'best+gradclip_1.0': {'activation': 'gelu', 'warmup_steps': 500,
                           'loss': {'type': 'mse+cosine', 'alpha': 5.0},
                           'grad_clip': 1.0},
    'best+gradclip_0.5': {'activation': 'gelu', 'warmup_steps': 500,
                           'loss': {'type': 'mse+cosine', 'alpha': 5.0},
                           'grad_clip': 0.5},

    # EMA + gradient clipping combo
    'best+ema+gradclip': {'activation': 'gelu', 'warmup_steps': 500,
                           'loss': {'type': 'mse+cosine', 'alpha': 5.0},
                           'ema_decay': 0.999, 'grad_clip': 1.0},

    # Warm restarts (cosine annealing with periodic LR resets)
    'best+warm_restarts': {'activation': 'gelu', 'warmup_steps': 500,
                            'loss': {'type': 'mse+cosine', 'alpha': 5.0},
                            'scheduler': 'warm_restarts',
                            'restart_period': 2500, 'restart_mult': 2},

    # SwiGLU activation (gated; hidden dims reduced to keep ~67M params)
    # With SwiGLU, Linear is d_in -> 2*d_out, so effective params double per layer.
    # Reduce hidden dims: [2816, 2816, 1408] gives ~67M
    'best+swiglu': {'activation': 'swiglu', 'warmup_steps': 500,
                     'loss': {'type': 'mse+cosine', 'alpha': 5.0},
                     'hidden_dims': [2816, 2816, 1408]},

    # GEGLU activation (similar to SwiGLU but with GELU gate)
    'best+geglu': {'activation': 'geglu', 'warmup_steps': 500,
                    'loss': {'type': 'mse+cosine', 'alpha': 5.0},
                    'hidden_dims': [2816, 2816, 1408]},

    # FiLM-conditioned decoder (bottleneck z injected at every decoder layer)
    'best+film': {'activation': 'gelu', 'warmup_steps': 500,
                   'loss': {'type': 'mse+cosine', 'alpha': 5.0},
                   'film_decoder': True},

    # ── Final comparison: k=32, 50k steps, save models ──

    'final_mse+cos': {'k': 32, 'n_steps': 50000, 'activation': 'gelu',
                       'warmup_steps': 500, 'grad_clip': 0.5,
                       'loss': {'type': 'mse+cosine', 'alpha': 5.0},
                       'save_model': True},

    'final_mse': {'k': 32, 'n_steps': 50000, 'activation': 'gelu',
                   'warmup_steps': 500, 'grad_clip': 0.5,
                   'loss': {'type': 'mse'},
                   'save_model': True},

    # L10 bottleneck sweep — clone of final_mse with k swapped
    'l10_k5':  {'k': 5,  'n_steps': 50000, 'activation': 'gelu',
                'warmup_steps': 500, 'grad_clip': 0.5,
                'loss': {'type': 'mse'}, 'save_model': True},
    'l10_k10': {'k': 10, 'n_steps': 50000, 'activation': 'gelu',
                'warmup_steps': 500, 'grad_clip': 0.5,
                'loss': {'type': 'mse'}, 'save_model': True},
    'l10_k20': {'k': 20, 'n_steps': 50000, 'activation': 'gelu',
                'warmup_steps': 500, 'grad_clip': 0.5,
                'loss': {'type': 'mse'}, 'save_model': True},
    'l10_k40': {'k': 40, 'n_steps': 50000, 'activation': 'gelu',
                'warmup_steps': 500, 'grad_clip': 0.5,
                'loss': {'type': 'mse'}, 'save_model': True},

    # ── Doubled params (~134M) ──

    'doubled_mse+cos': {'k': 32, 'n_steps': 50000, 'activation': 'gelu',
                         'warmup_steps': 500, 'grad_clip': 0.5,
                         'loss': {'type': 'mse+cosine', 'alpha': 5.0},
                         'hidden_dims': [6400, 6400, 2048],
                         'save_model': True},

    'doubled_mse': {'k': 32, 'n_steps': 50000, 'activation': 'gelu',
                     'warmup_steps': 500, 'grad_clip': 0.5,
                     'loss': {'type': 'mse'},
                     'hidden_dims': [6400, 6400, 2048],
                     'save_model': True},

    # ── Depth sweep (~134M params) ──

    'deep4_mse': {'k': 32, 'n_steps': 50000, 'activation': 'gelu',
                   'warmup_steps': 500, 'grad_clip': 0.5,
                   'loss': {'type': 'mse'},
                   'hidden_dims': [4800, 4800, 4800, 2048],
                   'save_model': True},

    'deep5_mse': {'k': 32, 'n_steps': 50000, 'activation': 'gelu',
                   'warmup_steps': 500, 'grad_clip': 0.5,
                   'loss': {'type': 'mse'},
                   'hidden_dims': [4096, 4096, 4096, 4096, 2048],
                   'save_model': True},

    'deep8_mse': {'k': 32, 'n_steps': 50000, 'activation': 'gelu',
                   'warmup_steps': 500, 'grad_clip': 0.5,
                   'loss': {'type': 'mse'},
                   'hidden_dims': [2816, 2816, 2816, 2816, 2816, 2816, 2816, 2048],
                   'save_model': True},

    # ── L10 overfitting sweep: more data + dropout ──

    'l10_nodropout': {'k': 32, 'n_steps': 50000, 'activation': 'gelu',
                      'warmup_steps': 500, 'grad_clip': 0.5,
                      'loss': {'type': 'mse'},
                      'save_model': True},

    'l10_dropout': {'k': 32, 'n_steps': 50000, 'activation': 'gelu',
                    'warmup_steps': 500, 'grad_clip': 0.5,
                    'loss': {'type': 'mse'},
                    'dropout': 0.1,
                    'save_model': True},

    # ── L10 scaling sweep: more data + dropout + 200k steps ──

    'l10_1M_drop': {'k': 32, 'n_steps': 200000, 'activation': 'gelu',
                    'warmup_steps': 500, 'grad_clip': 0.5,
                    'loss': {'type': 'mse'},
                    'dropout': 0.1,
                    'patience': 2000,
                    'save_model': True},

    'l10_2M_drop': {'k': 32, 'n_steps': 200000, 'activation': 'gelu',
                    'warmup_steps': 500, 'grad_clip': 0.5,
                    'loss': {'type': 'mse'},
                    'dropout': 0.1,
                    'patience': 2000,
                    'save_model': True},

    'l10_4M_drop': {'k': 32, 'n_steps': 200000, 'activation': 'gelu',
                    'warmup_steps': 500, 'grad_clip': 0.5,
                    'loss': {'type': 'mse'},
                    'dropout': 0.1,
                    'save_model': True},

    'l10_6M_drop': {'k': 32, 'n_steps': 200000, 'activation': 'gelu',
                    'warmup_steps': 500, 'grad_clip': 0.5,
                    'loss': {'type': 'mse'},
                    'dropout': 0.1,
                    'save_model': True},

    'l10_8M_drop': {'k': 32, 'n_steps': 200000, 'activation': 'gelu',
                    'warmup_steps': 500, 'grad_clip': 0.5,
                    'loss': {'type': 'mse'},
                    'dropout': 0.1,
                    'save_model': True},

    'l10_10M_drop': {'k': 32, 'n_steps': 200000, 'activation': 'gelu',
                     'warmup_steps': 500, 'grad_clip': 0.5,
                     'loss': {'type': 'mse'},
                     'dropout': 0.1,
                     'save_model': True},

    # ── Gemma-4-E2B-it layer 10 (d_in=1536) ──
    # Same hparams as l10_8M_drop (best Qwen run), only d_in changes to match the model.
    # Hidden dims left at [4096, 4096, 2048] per user instruction (option a).
    'gemma_l10_8M_drop': {'d_in': 1536,
                          'k': 32, 'n_steps': 200000, 'activation': 'gelu',
                          'warmup_steps': 500, 'grad_clip': 0.5,
                          'loss': {'type': 'mse'},
                          'dropout': 0.1,
                          'checkpoint_interval': 10000,
                          'save_model': True},
}


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_path', required=True)
    parser.add_argument('--output_dir', default='results/trained_autoencoder/hillclimb')
    parser.add_argument('--experiments', nargs='+', default=None,
                        help='Which experiments to run (default: all)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='If set, only use the first N rows of the cache.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading activations from {args.cache_path}...")
    meta_path = args.cache_path + '.meta.json'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        shape = tuple(meta['shape'])
        np_dtype = np.dtype(meta['dtype'])
        mm = np.memmap(args.cache_path, dtype=np_dtype, mode='r', shape=shape)
        # Keep memmap-backed; training loop will cast batches to float at GPU time.
        activations_raw = torch.from_numpy(mm)
    else:
        activations_raw = torch.load(args.cache_path, weights_only=True).float()
    print(f"  Shape: {activations_raw.shape}, dtype: {activations_raw.dtype}")

    if args.max_samples is not None and args.max_samples < activations_raw.shape[0]:
        activations_raw = activations_raw[:args.max_samples]
        print(f"  Truncated to first {args.max_samples} samples -> {activations_raw.shape}")

    # Select experiments
    if args.experiments:
        exp_names = args.experiments
    else:
        exp_names = list(ALL_EXPERIMENTS.keys())

    results = []
    for i, name in enumerate(exp_names):
        if name not in ALL_EXPERIMENTS:
            print(f"Unknown experiment: {name}, skipping")
            continue

        config = {**BASELINE, 'name': name, 'save_dir': args.output_dir, **ALL_EXPERIMENTS[name]}

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(exp_names)}] {name}")
        print(f"  Config overrides: {ALL_EXPERIMENTS[name]}")
        print(f"{'='*60}")

        result = run_experiment(config, activations_raw, device)
        results.append(result)

        # Save incrementally
        out_path = os.path.join(args.output_dir, 'hillclimb_results.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Leaderboard
    print(f"\n{'='*60}")
    print("LEADERBOARD")
    print(f"{'='*60}")
    sorted_results = sorted(results, key=lambda r: r['cos_sim'], reverse=True)
    for i, r in enumerate(sorted_results):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1}. {r['name']:20s} cos={r['cos_sim']:.4f} "
              f"mse={r['mse']:.6f} rel_L2={r['rel_l2']:.4f}{marker}")
    print(f"     {'SAE baseline':>20s} cos=0.9420")


if __name__ == '__main__':
    main()
