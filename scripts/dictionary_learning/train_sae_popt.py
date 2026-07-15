#!/usr/bin/env python3
"""
Train a TopK SAE with a differentiable superellipse-p term in the loss.

Same base loop as train_sae.py (MSE on cached resid_post), plus, every
`popt_interval` steps, a differentiable penalty that pushes the decoder atoms
toward super-additive (high-p) geometry:

    loss = MSE(x_hat, x)  +  lam * mean over sampled pairs [ R_mix / (R_d + R_c) ]

R_d, R_c, R_mix are downstream L2 responses to norm-matched perturbations along
two decoder atoms and their bisector (perturb at `layer`, measure at
`measure_layer`), computed WITH gradient so the term backprops into W_dec. The
ratio is 0.5 at no-interference (p~2) and < 0.5 when super-additive (p>2), so
minimizing it raises p. This is the gradient counterpart to the (selection-only)
cull in train_sae_pcull.py -- it *creates* high-p directions rather than hoping
to select them.

Usage:
    python -u scripts/dictionary_learning/train_sae_popt.py \
        --model Qwen/Qwen3-1.7B --layer 10 --d_sae 32768 --k 32 --n_steps 50000 \
        --reuse_cache --cache_path .../l10_cache_10M/cached_activations.pt \
        --max_samples 2000000 --lam 1.0 --popt_interval 10 --n_pairs 16 --popt_anchor 8 \
        --output_dir results/trained_sae_popt/l10_lam1
"""

import argparse
import importlib.util
import json
import math
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from src.autoencoders import TopKSAE

_p = os.path.join(os.path.dirname(__file__), 'train_sae_pcull.py')
_spec = importlib.util.spec_from_file_location('pcull', _p)
pcull = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pcull)
cache_activations = pcull.cache_activations
build_anchor_pool = pcull.build_anchor_pool
resample_dead_features = pcull.resample_dead_features
score_atoms = pcull.score_atoms
apply_tied_init = pcull.base.apply_tied_init
auxk_loss = pcull.base.auxk_loss


def pair_ratio_loss(model, D, C, anchors, cos_t, sin_t, layer, measure_layer):
    """Differentiable mean R_mix/(R_d+R_c) over pairs (D[i],C[i]) and anchors.

    D, C: [P, d] decoder rows WITH grad. Returns scalar (mean ratio).
    Mirrors the geometry of learn_superellipse_dict.pair_responses, batched.
    """
    model_dtype = next(model.parameters()).dtype
    d = D.shape[1]
    P = D.shape[0]
    total = 0.0
    count = 0
    for a, ctx, f_ref in zip(*anchors):
        a_norm = a.norm()
        a_hat = a / (a_norm + 1e-10)
        dp = D - (D @ a_hat).unsqueeze(-1) * a_hat
        dp = dp / (dp.norm(dim=-1, keepdim=True) + 1e-10)
        cp = C - (C @ a_hat).unsqueeze(-1) * a_hat
        cp = cp / (cp.norm(dim=-1, keepdim=True) + 1e-10)
        mix = dp + cp
        mix = mix / (mix.norm(dim=-1, keepdim=True) + 1e-10)
        td = torch.stack([dp, cp, mix], dim=1)                 # [P, 3, d]
        perturbed = cos_t * a.view(1, 1, d) + sin_t * a_norm * td
        points = perturbed.reshape(-1, 1, d).to(model_dtype)   # [P*3, 1, d]
        resp = forward_from_layer_to_layer(
            model, ctx, points, layer, measure_layer).float()
        resp = (resp - f_ref.unsqueeze(0)).norm(dim=-1).reshape(P, 3)
        ratio = resp[:, 2] / (resp[:, 0] + resp[:, 1] + 1e-8)
        total = total + ratio.sum()
        count += P
    return total / count


def train(sae, model, activations, anchors, args, device, measure_layer):
    val_frac = args.val_frac
    n_total = activations.shape[0]
    n_val = int(n_total * val_frac)
    n_cache = n_total - n_val
    acts_train = activations[:n_cache]
    acts_val = activations[n_cache:n_cache + n_val]
    print(f"  Train split: {n_cache} | Val split: {n_val}")

    native_gb = n_cache * activations.shape[1] * acts_train.element_size() / 1e9
    if not args.force_cpu_cache and native_gb < args.gpu_cache_gb:
        activations_gpu = acts_train.to(device)
        cpu_batched = False
        print(f"  Cache {native_gb:.1f} GB ({acts_train.dtype}) resident on GPU")
    else:
        activations_gpu = acts_train
        cpu_batched = True
        print(f"  Cache {native_gb:.1f} GB — streaming batches CPU→GPU")

    val_batch = acts_val[:min(10000, n_val)].to(device).float()
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps)
    # "Fired" = in the top-k AND value>0 (zero-valued latents occupy top-k slots)
    feature_fired = torch.zeros(sae.d_sae, dtype=torch.bool, device=device)
    last_fired = torch.zeros(sae.d_sae, dtype=torch.long, device=device)
    fired_counts = torch.zeros(sae.d_sae, dtype=torch.long, device=device)
    g = torch.Generator(device='cpu').manual_seed(args.seed + 1)

    cos_t = math.cos(math.radians(args.angle))
    sin_t = math.sin(math.radians(args.angle))
    popt_anchors = tuple(x[:args.popt_anchor] for x in anchors)
    mon_anchors = tuple(x[:args.monitor_anchor] for x in anchors)

    training_log, p_log = [], []
    t_start = time.time()
    print(f"\nTraining: {args.n_steps} steps, batch={args.batch_size}, lr={args.lr}, k={sae.k}")
    print(f"  p-opt: every {args.popt_interval} steps, lam={args.lam}, "
          f"{args.n_pairs} pairs x {args.popt_anchor} anchors @ {args.angle}deg\n")

    for step in range(args.n_steps):
        if cpu_batched:
            idx = torch.randint(0, n_cache, (args.batch_size,))
            batch = activations_gpu[idx].to(device).float()
        else:
            idx = torch.randint(0, n_cache, (args.batch_size,), device=device)
            batch = activations_gpu[idx].float()

        top_vals, top_idx, pre_acts = sae.encode(batch)
        recon = sae.decode(top_vals, top_idx)
        mse = F.mse_loss(recon, batch)

        ploss_val = float('nan')
        if args.lam > 0 and step % args.popt_interval == 0:
            W = F.normalize(sae.W_dec, dim=-1)                 # grad flows to W_dec
            pa = torch.randint(0, sae.d_sae, (args.n_pairs,), generator=g)
            pb = torch.randint(0, sae.d_sae, (args.n_pairs,), generator=g)
            same = pa == pb
            pb[same] = (pb[same] + 1) % sae.d_sae
            ploss = pair_ratio_loss(model, W[pa.to(device)], W[pb.to(device)],
                                    popt_anchors, cos_t, sin_t, args.layer, measure_layer)
            loss = mse + args.lam * ploss
            ploss_val = ploss.item()
        else:
            loss = mse

        # AuxK: reconstruct the main residual with the top-k_aux dead latents
        # (dead = no fire in the last dead_steps_threshold steps; last_fired
        # inits to 0 so nothing counts as dead before the warmup elapses).
        aux_val = float('nan')
        if args.auxk:
            dead_mask = (step - last_fired) > args.dead_steps_threshold
            if dead_mask.any():
                aux = auxk_loss(sae, batch, recon, pre_acts, dead_mask, args.k_aux)
                loss = loss + args.auxk_alpha * aux
                aux_val = aux.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        sae.normalize_decoder()
        with torch.no_grad():
            fired = top_idx[top_vals > 0]
            if fired.numel():
                feature_fired[fired] = True
                last_fired[fired] = step
                fired_counts += torch.bincount(fired, minlength=sae.d_sae)

        if args.log_p_interval > 0 and step % args.log_p_interval == 0:
            mcand = torch.randperm(sae.d_sae, generator=g)[:args.monitor_atoms].to(device)
            mcontr = torch.randperm(sae.d_sae, generator=g)[:args.monitor_contrast].to(device)
            sae.eval()
            ms = score_atoms(model, sae, mcand, mcontr, mon_anchors,
                             args.layer, measure_layer, args.angle, chunk_pairs=args.chunk_pairs)
            sae.train()
            p_log.append({'step': step, 'L_med': ms.median().item(), 'L_mean': ms.mean().item(),
                          'L_p10': ms.quantile(0.10).item(), 'L_p90': ms.quantile(0.90).item()})

        if step > 0 and step % args.resample_interval == 0:
            n_dead = resample_dead_features(sae, optimizer, activations_gpu, feature_fired)
            feature_fired.zero_()
            sae.normalize_decoder()
            print(f"  Step {step}: resampled {n_dead} dead features")

        if step % args.log_interval == 0:
            with torch.no_grad():
                rel_l2 = (batch - recon).norm(dim=-1).mean() / batch.norm(dim=-1).mean()
                # True dead count: no fire (value>0) in the last 1000 steps
                n_dead_now = ((step - last_fired) >= 1000).sum().item()
                sae.eval()
                val_recon, _, _ = sae(val_batch)
                val_cos = F.cosine_similarity(val_batch, val_recon, dim=-1).mean()
                sae.train()
            training_log.append({'step': step, 'loss': loss.item(), 'mse': mse.item(),
                                 'ploss': ploss_val, 'aux_loss': aux_val,
                                 'rel_l2': rel_l2.item(),
                                 'val_cos': val_cos.item(), 'n_dead': n_dead_now,
                                 'lr': scheduler.get_last_lr()[0]})
            if step % (args.log_interval * 10) == 0:
                lp = p_log[-1] if p_log else {'L_med': float('nan')}
                print(f"  Step {step:6d}: mse={mse.item():.5f}, ploss={ploss_val:.4f}, "
                      f"aux={aux_val:.4f}, "
                      f"rel_L2={rel_l2.item():.4f}, val_cos={val_cos.item():.4f}, "
                      f"dead={n_dead_now}, L_med={lp['L_med']:.3f}, ({time.time()-t_start:.0f}s)")

    print(f"\nTraining complete in {time.time()-t_start:.1f}s")
    # torch tensor (not numpy) so the checkpoint stays weights_only-loadable
    return training_log, p_log, fired_counts.cpu()


def make_plot(training_log, p_log, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    steps = [e['step'] for e in training_log]
    axes[0].plot(steps, [e['mse'] for e in training_log], label='mse')
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('MSE'); axes[0].set_yscale('log')
    axes[0].set_title('Reconstruction MSE'); axes[0].grid(True, alpha=0.3); axes[0].legend()
    pl = [e['ploss'] for e in training_log]
    axes[1].plot(steps, pl)
    axes[1].axhline(0.5, color='gray', ls=':', label='0.5')
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('p-term ratio'); axes[1].set_title('p-loss (train pairs)')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    if p_log:
        ps = [e['step'] for e in p_log]
        axes[2].plot(ps, [e['L_med'] for e in p_log], label='median L')
        axes[2].fill_between(ps, [e['L_p10'] for e in p_log], [e['L_p90'] for e in p_log], alpha=0.2)
        axes[2].axhline(0.5, color='gray', ls=':')
        axes[2].set_xlabel('Step'); axes[2].set_ylabel('L (monitor)')
        axes[2].set_title('monitor p-surrogate (lower=higher p)'); axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def main():
    p = argparse.ArgumentParser(description='Train TopK SAE with differentiable p-term')
    p.add_argument('--model', default='Qwen/Qwen3-1.7B')
    p.add_argument('--layer', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seq_len', type=int, default=10)
    p.add_argument('--d_sae', type=int, default=32768)
    p.add_argument('--k', type=int, default=32)
    p.add_argument('--cache_start', type=int, default=8000)
    p.add_argument('--n_cache', type=int, default=2000000)
    p.add_argument('--n_steps', type=int, default=50000)
    p.add_argument('--batch_size', type=int, default=4096)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--resample_interval', type=int, default=25000)
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--output_dir', default='results/trained_sae_popt/l10_lam1')
    p.add_argument('--reuse_cache', action='store_true')
    p.add_argument('--cache_path', default=None)
    p.add_argument('--cache_fp16', action='store_true')
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--gpu_cache_gb', type=float, default=12.0)
    p.add_argument('--force_cpu_cache', action='store_true')
    p.add_argument('--model_dtype', default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    p.add_argument('--measure_layer', type=int, default=None)
    # p-opt term
    p.add_argument('--lam', type=float, default=1.0, help='weight of the p-term (0 disables)')
    p.add_argument('--popt_interval', type=int, default=10, help='apply p-term every N steps')
    p.add_argument('--n_pairs', type=int, default=16, help='atom pairs per p-term step')
    p.add_argument('--popt_anchor', type=int, default=8)
    p.add_argument('--angle', type=float, default=30.0)
    p.add_argument('--anchor_seed', type=int, default=999)
    p.add_argument('--chunk_pairs', type=int, default=1024)
    # monitor
    p.add_argument('--log_p_interval', type=int, default=1000)
    p.add_argument('--monitor_atoms', type=int, default=64)
    p.add_argument('--monitor_contrast', type=int, default=4)
    p.add_argument('--monitor_anchor', type=int, default=4)
    # Dead-latent fixes (Gao et al. 2024); off by default so old runs reproduce
    p.add_argument('--tied_init', action='store_true')
    p.add_argument('--auxk', action='store_true')
    p.add_argument('--auxk_alpha', type=float, default=1/32)
    p.add_argument('--k_aux', type=int, default=512)
    p.add_argument('--dead_steps_threshold', type=int, default=2000)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    print("Train TopK SAE with differentiable p-term")
    print(f"  model={args.model}, layer={args.layer}, d_sae={args.d_sae}, k={args.k}\n")

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.model_dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = args.measure_layer if args.measure_layer is not None else n_layers - 2
    print(f"  n_layers={n_layers}, measure_layer={measure_layer}")

    activations, cache_path = cache_activations(model, tokenizer, device, args)
    d_model = activations.shape[1]

    print(f"Building anchor pool ({args.popt_anchor} anchors, seed={args.anchor_seed})...")
    anchors = build_anchor_pool(model, tokenizer, args.layer, args.popt_anchor,
                                args.seq_len, args.anchor_seed, device, measure_layer)

    sae = TopKSAE(d_in=d_model, d_sae=args.d_sae, k=args.k).to(device)
    if args.tied_init:
        apply_tied_init(sae, activations)
    print(f"SAE params: {sum(p.numel() for p in sae.parameters())/1e6:.1f}M")

    training_log, p_log, fired_counts = train(sae, model, activations, anchors, args, device, measure_layer)

    save_data = {'state_dict': sae.state_dict(),
                 'config': {'d_in': d_model, 'd_sae': args.d_sae, 'k': args.k},
                 'training_log': training_log, 'p_log': p_log,
                 'fired_counts': fired_counts, 'args': vars(args)}
    weights_path = os.path.join(args.output_dir, 'sae_weights.pt')
    torch.save(save_data, weights_path)
    print(f"\nSaved SAE to {weights_path}")
    with open(os.path.join(args.output_dir, 'sae_config.json'), 'w') as f:
        json.dump({'d_in': d_model, 'd_sae': args.d_sae, 'k': args.k}, f, indent=2)
    with open(os.path.join(args.output_dir, 'training_log.pkl'), 'wb') as f:
        pickle.dump({'training_log': training_log, 'p_log': p_log}, f)
    make_plot(training_log, p_log, os.path.join(args.output_dir, 'training_plot.png'))


if __name__ == '__main__':
    main()
