#!/usr/bin/env python3
"""
Train a TopK SAE with periodic *p-culling* of low-superellipse-p atoms.

Same base loop as train_sae.py (MSE on cached resid_post activations), plus a
periodic non-gradient step that:
  - measures each candidate decoder atom's superellipse-p (via the cheap
    single-angle surrogate L = R_mix / (R_d + R_c), averaged over a sample of
    contrast atoms and anchor activations);
  - deletes the worst (lowest-p == highest-L) fraction and reinitializes those
    slots toward high-reconstruction-error inputs (same mechanism as dead-feature
    resampling), with the optimizer moments reset.

The p-measurement requires partial forwards through the LLM (perturb at `layer`,
measure L2 at `measure_layer`), so the model is kept resident (NOT deleted after
caching). Measurement runs under no_grad with the non-differentiable surrogate --
we only need to *rank* atoms, not backprop through p.

NOTE: this is *selection only* (cull + reinit). MSE has no gradient toward p, and
reinit'd atoms are random (p~2), so whether the dictionary's p actually rises is
an empirical question -- that's what this run is meant to test. A long cull
interval gives reinit'd atoms a grace window to train up before they're eligible
again.

Usage:
    python -u scripts/dictionary_learning/train_sae_pcull.py \
        --model meta-llama/Llama-3.2-1B --layer 2 \
        --k 32 --d_sae 16384 --n_steps 50000 \
        --reuse_cache --cache_path results/trained_sae/llama-3.2-1b_layer2_8x_k32/cached_activations.pt \
        --output_dir results/trained_sae_pcull/llama-3.2-1b_layer2_8x_k32
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
from src.data import load_fineweb_fixed_length
from src.autoencoders import TopKSAE

# Reuse cache_activations / resample_dead_features / make_plot from the base
# trainer without copy-pasting. Importing the file is safe: its main() only runs
# under __main__.
_base_path = os.path.join(os.path.dirname(__file__), 'train_sae.py')
_spec = importlib.util.spec_from_file_location('train_sae_base', _base_path)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
cache_activations = base.cache_activations


def resample_dead_features(sae, optimizer, activations, feature_fired, n_sample=8192):
    """Device-safe clone of base.resample_dead_features (handles GPU-resident cache)."""
    dead_mask = ~feature_fired
    n_dead = dead_mask.sum().item()
    if n_dead == 0:
        return 0
    sae_device = sae.encoder.weight.device
    idx = torch.randint(0, activations.shape[0], (n_sample,), device=activations.device)
    sample = activations[idx].to(sae_device).float()
    with torch.no_grad():
        recon, _, _ = sae(sample)
        per_sample_loss = (sample - recon).pow(2).sum(dim=-1)
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        n_reinit = min(n_dead, n_sample)
        top_loss_idx = per_sample_loss.topk(n_reinit).indices
        high_loss_acts = sample[top_loss_idx]
        for i, dead_idx in enumerate(dead_indices[:n_reinit]):
            source = high_loss_acts[i % n_reinit]
            direction = source - sae.b_dec
            direction = direction / (direction.norm() + 1e-8)
            sae.encoder.weight.data[dead_idx] = direction * 0.1
            sae.encoder.bias.data[dead_idx] = 0.0
            sae.W_dec.data[dead_idx] = F.normalize(source.unsqueeze(0), dim=-1).squeeze(0)
        for param in [sae.encoder.weight, sae.encoder.bias, sae.W_dec]:
            state = optimizer.state.get(param)
            if state is None:
                continue
            for key in ['exp_avg', 'exp_avg_sq']:
                if key in state and state[key].dim() >= 1 and state[key].shape[0] == sae.d_sae:
                    state[key][dead_indices[:n_reinit]] = 0
    return n_dead


# ── Anchor activations (with context) for p-measurement ──────────────────

def extract_context_and_activation(model, token_ids_list, layer, device, batch_size=64):
    """Final-token activation + prefix context at `layer` (HF hidden states)."""
    all_contexts, all_activations = [], []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i + batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].float().cpu())
        all_activations.append(hidden[:, -1, :].float().cpu())
    return torch.cat(all_contexts, 0), torch.cat(all_activations, 0)


def build_anchor_pool(model, tokenizer, layer, n_anchor, seq_len, seed, device,
                      measure_layer):
    """Load n_anchor activations w/ context; precompute unperturbed responses.

    Returns lists (acts[d], ctxs[1,seq-1,d] in model dtype, f_refs[d]).
    """
    model_dtype = next(model.parameters()).dtype
    toks = load_fineweb_fixed_length(max(n_anchor, 8), tokenizer, seq_len=seq_len, seed=seed)
    ctx_all, act_all = extract_context_and_activation(model, toks[:n_anchor], layer, device)
    acts, ctxs, f_refs = [], [], []
    with torch.no_grad():
        for i in range(n_anchor):
            a = act_all[i].to(device=device, dtype=torch.float32)
            ctx = ctx_all[i].unsqueeze(0).to(device=device, dtype=model_dtype)
            point = a.reshape(1, 1, -1).to(model_dtype)
            f_ref = forward_from_layer_to_layer(
                model, ctx, point, layer, measure_layer).squeeze(0).float()
            acts.append(a); ctxs.append(ctx); f_refs.append(f_ref)
    return acts, ctxs, f_refs


# ── p-surrogate scoring ──────────────────────────────────────────────────

def _pair_ratios_one_anchor(model, a, a_hat, a_norm, ctx, f_ref, D, C,
                            cos_t, sin_t, layer, measure_layer, chunk_pairs):
    """Surrogate L = R_mix/(R_d+R_c) for pairs (D[i], C[i]) at one anchor.

    D, C: [m, d] unit dirs (candidate, contrast). Returns L [m] (float32).
    All ops under no_grad; geometry in float32, cast to model dtype at forward.
    """
    model_dtype = next(model.parameters()).dtype
    m = D.shape[0]
    d = D.shape[1]
    out = torch.empty(m, device=D.device, dtype=torch.float32)
    for s in range(0, m, chunk_pairs):
        e = min(m, s + chunk_pairs)
        Db = D[s:e]; Cb = C[s:e]
        dp = Db - (Db @ a_hat).unsqueeze(-1) * a_hat
        dp = dp / (dp.norm(dim=-1, keepdim=True) + 1e-10)
        cp = Cb - (Cb @ a_hat).unsqueeze(-1) * a_hat
        cp = cp / (cp.norm(dim=-1, keepdim=True) + 1e-10)
        mix = dp + cp
        mix = mix / (mix.norm(dim=-1, keepdim=True) + 1e-10)
        td = torch.stack([dp, cp, mix], dim=1)               # [b, 3, d]
        perturbed = cos_t * a.view(1, 1, d) + sin_t * a_norm * td  # [b, 3, d]
        points = perturbed.reshape(-1, 1, d).to(model_dtype)      # [b*3, 1, d]
        resp = forward_from_layer_to_layer(
            model, ctx, points, layer, measure_layer).float()    # [b*3, d]
        resp = (resp - f_ref.unsqueeze(0)).norm(dim=-1).reshape(-1, 3)  # [b, 3]
        out[s:e] = resp[:, 2] / (resp[:, 0] + resp[:, 1] + 1e-8)
    return out


@torch.no_grad()
def score_atoms(model, sae, cand_idx, contrast_idx, anchors, layer, measure_layer,
                angle_deg, chunk_pairs=2048):
    """Per-atom p-surrogate score (mean L over contrasts & anchors). Lower = higher p.

    cand_idx: [C] long. contrast_idx: [G] long. anchors: (acts, ctxs, f_refs).
    Returns score [C] float32 (mean L; self-pairs excluded).
    """
    acts, ctxs, f_refs = anchors
    device = sae.W_dec.device
    W = F.normalize(sae.W_dec.detach(), dim=-1)               # [d_sae, d]
    Cn = cand_idx.shape[0]; G = contrast_idx.shape[0]
    cos_t = math.cos(math.radians(angle_deg))
    sin_t = math.sin(math.radians(angle_deg))

    # Flatten pairs: pair p -> cand = cand_idx[p//G], contrast = contrast_idx[p%G].
    cand_rows = cand_idx.repeat_interleave(G)                 # [Cn*G]
    contr_rows = contrast_idx.repeat(Cn)                      # [Cn*G]
    D = W[cand_rows]                                          # [Cn*G, d]
    Cc = W[contr_rows]
    valid = (cand_rows != contr_rows).float()                # exclude self-pairs

    L_sum = torch.zeros(Cn * G, device=device)
    for a, ctx, f_ref in zip(*anchors):
        a_norm = a.norm()
        a_hat = a / (a_norm + 1e-10)
        L = _pair_ratios_one_anchor(
            model, a, a_hat, a_norm, ctx, f_ref, D, Cc,
            cos_t, sin_t, layer, measure_layer, chunk_pairs)
        L_sum += L
    n_anchor = len(acts)
    L_mean_pair = (L_sum / n_anchor).reshape(Cn, G)
    valid = valid.reshape(Cn, G)
    score = (L_mean_pair * valid).sum(-1) / valid.sum(-1).clamp_min(1.0)
    return score


def reinit_atoms(sae, optimizer, activations, atom_idx, n_sample=8192):
    """Reinit the given atom slots toward high-reconstruction-error inputs.

    Same mechanism as base.resample_dead_features, but on an explicit index set.
    """
    n = atom_idx.numel()
    if n == 0:
        return 0
    sae_device = sae.encoder.weight.device
    idx = torch.randint(0, activations.shape[0], (n_sample,), device=activations.device)
    sample = activations[idx].to(sae_device).float()
    with torch.no_grad():
        recon, _, _ = sae(sample)
        per_sample_loss = (sample - recon).pow(2).sum(dim=-1)
        n_reinit = min(n, n_sample)
        top_loss_idx = per_sample_loss.topk(n_reinit).indices
        high_loss_acts = sample[top_loss_idx]                 # [n_reinit, d_in]
        for i, dead_idx in enumerate(atom_idx[:n_reinit]):
            source = high_loss_acts[i % n_reinit]
            direction = source - sae.b_dec
            direction = direction / (direction.norm() + 1e-8)
            sae.encoder.weight.data[dead_idx] = direction * 0.1
            sae.encoder.bias.data[dead_idx] = 0.0
            sae.W_dec.data[dead_idx] = F.normalize(source.unsqueeze(0), dim=-1).squeeze(0)
        for param in [sae.encoder.weight, sae.encoder.bias, sae.W_dec]:
            state = optimizer.state.get(param)
            if state is None:
                continue
            for key in ['exp_avg', 'exp_avg_sq']:
                if key in state and state[key].dim() >= 1 and state[key].shape[0] == sae.d_sae:
                    state[key][atom_idx[:n_reinit]] = 0
    return int(n_reinit)


# ── Training ─────────────────────────────────────────────────────────────

def train(sae, model, activations, anchors, args, device, measure_layer):
    val_frac = args.val_frac
    n_total = activations.shape[0]
    n_val = int(n_total * val_frac)
    n_cache = n_total - n_val
    acts_train = activations[:n_cache]
    acts_val = activations[n_cache:n_cache + n_val]
    print(f"  Train split: {n_cache} | Val split: {n_val}")

    # Keep the cache resident on the GPU in its native dtype (fp16 here ->
    # ~half the fp32 size), converting each batch to fp32 on the fly. This fits
    # alongside the resident LLM on a 24GB card and avoids per-step CPU->GPU
    # streaming. Falls back to CPU streaming only if it wouldn't fit.
    native_gb = n_cache * activations.shape[1] * acts_train.element_size() / 1e9
    if not args.force_cpu_cache and native_gb < args.gpu_cache_gb:
        activations_gpu = acts_train.to(device)          # native dtype (fp16)
        cpu_batched = False
        print(f"  Cache {native_gb:.1f} GB ({acts_train.dtype}) resident on GPU")
    else:
        activations_gpu = acts_train
        cpu_batched = True
        print(f"  Cache {native_gb:.1f} GB — streaming batches CPU→GPU")

    val_batch = acts_val[:min(10000, n_val)].to(device).float()

    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps)

    feature_fired = torch.zeros(sae.d_sae, dtype=torch.bool, device=device)
    g = torch.Generator(device='cpu').manual_seed(args.seed + 1)

    training_log, p_log = [], []
    t_start = time.time()
    print(f"\nTraining: {args.n_steps} steps, batch={args.batch_size}, lr={args.lr}, k={sae.k}")
    cull_desc = (f"every {args.cull_interval} steps, delete bottom {args.delete_frac:.0%} "
                 f"of {args.n_candidates} candidates ({args.n_contrast}c x {args.n_anchor}a @ {args.angle}deg)"
                 if args.cull_interval > 0 else "DISABLED (control run)")
    print(f"  p-cull: {cull_desc}")
    print(f"  p-monitor: every {args.log_p_interval} steps "
          f"({args.monitor_atoms}atoms x {args.monitor_contrast}c x {args.monitor_anchor}a)\n")

    cull_anchors = tuple(x[:args.n_anchor] for x in anchors)
    mon_anchors = tuple(x[:args.monitor_anchor] for x in anchors)

    for step in range(args.n_steps):
        if cpu_batched:
            idx = torch.randint(0, n_cache, (args.batch_size,))
            batch = activations_gpu[idx].to(device).float()
        else:
            idx = torch.randint(0, n_cache, (args.batch_size,), device=device)
            batch = activations_gpu[idx].float()

        recon, top_vals, top_idx = sae(batch)
        loss = F.mse_loss(recon, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        sae.normalize_decoder()
        feature_fired[top_idx.unique()] = True

        # ── periodic p-cull (selection only); cull_interval<=0 disables ──
        if args.cull_interval > 0 and step > 0 and step % args.cull_interval == 0:
            t0 = time.time()
            cand = torch.randperm(sae.d_sae, generator=g)[:args.n_candidates].to(device)
            contrast = torch.randperm(sae.d_sae, generator=g)[:args.n_contrast].to(device)
            sae.eval()
            score = score_atoms(model, sae, cand, contrast, cull_anchors,
                                 args.layer, measure_layer, args.angle,
                                 chunk_pairs=args.chunk_pairs)
            n_del = max(1, int(args.delete_frac * args.n_candidates))
            worst = score.topk(n_del).indices            # highest L == lowest p
            del_atoms = cand[worst]
            n_reinit = reinit_atoms(sae, optimizer, activations_gpu, del_atoms)
            sae.normalize_decoder()
            sae.train()
            print(f"  Step {step}: p-cull deleted {n_reinit} atoms "
                  f"(cand L: med={score.median().item():.3f}, "
                  f"culled>={score[worst].min().item():.3f}) [{time.time()-t0:.0f}s]")

        # ── cheap p-monitor; log_p_interval<=0 disables ──
        if args.log_p_interval > 0 and step % args.log_p_interval == 0:
            mcand = torch.randperm(sae.d_sae, generator=g)[:args.monitor_atoms].to(device)
            mcontr = torch.randperm(sae.d_sae, generator=g)[:args.monitor_contrast].to(device)
            sae.eval()
            ms = score_atoms(model, sae, mcand, mcontr, mon_anchors,
                             args.layer, measure_layer, args.angle,
                             chunk_pairs=args.chunk_pairs)
            sae.train()
            p_log.append({'step': step, 'L_med': ms.median().item(),
                          'L_mean': ms.mean().item(),
                          'L_p10': ms.quantile(0.10).item(),
                          'L_p90': ms.quantile(0.90).item()})

        # ── dead-feature resampling (unchanged) ──
        if step > 0 and step % args.resample_interval == 0:
            n_dead = resample_dead_features(sae, optimizer, activations_gpu, feature_fired)
            feature_fired.zero_()
            sae.normalize_decoder()
            print(f"  Step {step}: resampled {n_dead} dead features")

        if step % args.log_interval == 0:
            with torch.no_grad():
                rel_l2 = (batch - recon).norm(dim=-1).mean() / batch.norm(dim=-1).mean()
                cos_sim = F.cosine_similarity(batch, recon, dim=-1).mean()
                n_dead_now = (~feature_fired).sum().item()
                sae.eval()
                val_recon, _, _ = sae(val_batch)
                val_rel_l2 = ((val_batch - val_recon).norm(dim=-1) / val_batch.norm(dim=-1)).mean()
                val_cos = F.cosine_similarity(val_batch, val_recon, dim=-1).mean()
                sae.train()
            training_log.append({
                'step': step, 'loss': loss.item(), 'rel_l2': rel_l2.item(),
                'cos_sim': cos_sim.item(), 'val_rel_l2': val_rel_l2.item(),
                'val_cos': val_cos.item(), 'n_dead': n_dead_now,
                'lr': scheduler.get_last_lr()[0],
            })
            if step % (args.log_interval * 10) == 0:
                lp = p_log[-1] if p_log else {'L_med': float('nan')}
                print(f"  Step {step:6d}: loss={loss.item():.6f}, rel_L2={rel_l2.item():.4f}, "
                      f"val_cos={val_cos.item():.4f}, dead={n_dead_now}, "
                      f"L_med={lp['L_med']:.3f}, ({time.time()-t_start:.0f}s)")

    print(f"\nTraining complete in {time.time()-t_start:.1f}s")
    return training_log, p_log


# ── Plotting ─────────────────────────────────────────────────────────────

def make_plot(training_log, p_log, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    steps = [e['step'] for e in training_log]
    axes[0].plot(steps, [e['loss'] for e in training_log])
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('MSE'); axes[0].set_yscale('log')
    axes[0].set_title('Training loss'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, [e['n_dead'] for e in training_log])
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Dead features')
    axes[1].set_title('Dead feature count'); axes[1].grid(True, alpha=0.3)

    if p_log:
        ps = [e['step'] for e in p_log]
        axes[2].plot(ps, [e['L_med'] for e in p_log], label='median L')
        axes[2].fill_between(ps, [e['L_p10'] for e in p_log],
                             [e['L_p90'] for e in p_log], alpha=0.2, label='p10-p90')
        axes[2].axhline(0.5, color='gray', ls=':', label='0.5 (no interference)')
        axes[2].set_xlabel('Step'); axes[2].set_ylabel('L = R_mix/(R1+R2)')
        axes[2].set_title('p-surrogate (lower = higher p)')
        axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Train TopK SAE with periodic p-culling')
    # base args (mirror train_sae.py)
    p.add_argument('--model', default='meta-llama/Llama-3.2-1B')
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seq_len', type=int, default=10)
    p.add_argument('--d_sae', type=int, default=16384)
    p.add_argument('--k', type=int, default=32)
    p.add_argument('--cache_start', type=int, default=8000)
    p.add_argument('--n_cache', type=int, default=2000000)
    p.add_argument('--n_steps', type=int, default=50000)
    p.add_argument('--batch_size', type=int, default=4096)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--resample_interval', type=int, default=25000)
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--output_dir', default='results/trained_sae_pcull/llama-3.2-1b_layer2_8x_k32')
    p.add_argument('--reuse_cache', action='store_true')
    p.add_argument('--cache_path', default=None)
    p.add_argument('--cache_fp16', action='store_true')
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--gpu_cache_gb', type=float, default=12.0,
                   help='Keep cache on GPU if its native-dtype size is under this (GB).')
    p.add_argument('--force_cpu_cache', action='store_true',
                   help='Force CPU streaming of the cache regardless of size.')
    p.add_argument('--model_dtype', default='bfloat16',
                   choices=['float32', 'bfloat16', 'float16'],
                   help='Model dtype for p-cull forwards (bf16 halves model VRAM).')
    # p-cull args
    p.add_argument('--measure_layer', type=int, default=None, help='default: n_layers-2')
    p.add_argument('--cull_interval', type=int, default=2000)
    p.add_argument('--delete_frac', type=float, default=0.05)
    p.add_argument('--n_candidates', type=int, default=2048)
    p.add_argument('--n_contrast', type=int, default=8)
    p.add_argument('--n_anchor', type=int, default=16)
    p.add_argument('--angle', type=float, default=30.0)
    p.add_argument('--anchor_seed', type=int, default=999)
    p.add_argument('--chunk_pairs', type=int, default=1024)
    # cheap monitor
    p.add_argument('--log_p_interval', type=int, default=100)
    p.add_argument('--monitor_atoms', type=int, default=128)
    p.add_argument('--monitor_contrast', type=int, default=4)
    p.add_argument('--monitor_anchor', type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    print("Train TopK SAE with p-culling")
    print(f"  model={args.model}, layer={args.layer}, d_sae={args.d_sae}, k={args.k}\n")

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.model_dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = args.measure_layer if args.measure_layer is not None else n_layers - 2
    print(f"  n_layers={n_layers}, measure_layer={measure_layer}")

    activations, cache_path = cache_activations(model, tokenizer, device, args)
    d_model = activations.shape[1]

    # Keep the model resident for p-measurement; build anchor pool.
    print(f"Building anchor pool ({args.n_anchor} anchors, seed={args.anchor_seed})...")
    anchors = build_anchor_pool(model, tokenizer, args.layer, args.n_anchor,
                                args.seq_len, args.anchor_seed, device, measure_layer)

    sae = TopKSAE(d_in=d_model, d_sae=args.d_sae, k=args.k).to(device)
    print(f"SAE params: {sum(p.numel() for p in sae.parameters())/1e6:.1f}M")

    training_log, p_log = train(sae, model, activations, anchors, args, device, measure_layer)

    save_data = {
        'state_dict': sae.state_dict(),
        'config': {'d_in': d_model, 'd_sae': args.d_sae, 'k': args.k},
        'training_log': training_log, 'p_log': p_log, 'args': vars(args),
    }
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
