#!/usr/bin/env python3
"""
Anisotropy ratio vs layer: for each layer, compute r_feat, r_plateau, and their ratio
using -diff-of-means (EN→ZH direction) as the feature direction.

Only does 1D sweeps along the feature and orthogonal axes (no full 2D grid),
so runs fast.

Usage:
    python scripts/run_anisotropy_per_layer.py --model qwen3-0.6b
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_allpos,
)
from src.data import load_fineweb_fixed_length
from data.bilingual_prompts import ENGLISH_PROMPTS, CHINESE_PROMPTS


def get_activations_at_layer(model, tokenizer, prompts, layer, device):
    """Extract final-token activations for a list of prompts at a given layer."""
    acts = []
    for p in prompts:
        tokens = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        acts.append(outputs.hidden_states[layer + 1][:, -1, :].squeeze(0).cpu())
    return torch.stack(acts)


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device,
                                   batch_size=64):
    """Extract layer-L context and final-token activation for fixed-length sequences."""
    all_contexts = []
    all_activations = []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].cpu())
        all_activations.append(hidden[:, -1, :].cpu())
    return torch.cat(all_contexts, 0), torch.cat(all_activations, 0)


def build_perturbed_hidden(contexts, activations, direction, angle_rad):
    """Spherical perturbation along a single direction."""
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms

    proj = (direction.unsqueeze(0) * a_unit).sum(dim=-1, keepdim=True)
    u_perp = direction.unsqueeze(0) - proj * a_unit
    u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)

    a_pert = a_norms * (cos_t * a_unit + sin_t * u_perp)
    return torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)


def sweep_direction(model, contexts, activations, ref_log_probs, direction,
                    angles_rad, layer, device):
    """Sweep a single direction, return per-activation KL at each angle. [N, n_angles]"""
    N = activations.shape[0]
    kl_divs = np.zeros((N, len(angles_rad)))
    direction_dev = direction.to(device)

    for i, theta in enumerate(angles_rad):
        with torch.no_grad():
            hidden = build_perturbed_hidden(contexts, activations, direction_dev, theta)
            logits = forward_from_layer_allpos(model, hidden, layer)
            log_probs = F.log_softmax(logits, dim=-1)
            kl = (ref_log_probs.exp() * (ref_log_probs - log_probs)).sum(dim=-1)
            kl_divs[:, i] = kl.cpu().numpy()
    return kl_divs


def find_breaking_angle(kl_vs_angle, angles_deg, threshold=1.0):
    """Find first angle where KL exceeds threshold."""
    above = np.where(kl_vs_angle > threshold)[0]
    if len(above) == 0:
        return float('nan')
    return angles_deg[above[0]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--n_act', type=int, default=50)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=30)
    parser.add_argument('--n_orth_dirs', type=int, default=5)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/anisotropy_per_layer/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Setup =====
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']

    # Determine number of layers
    from src.model import _get_blocks
    n_layers = len(_get_blocks(model))
    print(f"Device: {device}, d_model: {d_model}, n_layers: {n_layers}")

    # Load FineWeb tokens once (reused across layers)
    print(f"\nLoading {args.n_act} FineWeb sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles)
    angles_rad = np.deg2rad(angles_deg)

    # ===== Sweep each layer =====
    results_per_layer = []

    for layer in range(n_layers):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Layer {layer}/{n_layers-1}")
        print(f"{'='*60}")

        # Diff-of-means at this layer (negated: push toward Mandarin)
        en_acts = get_activations_at_layer(model, tokenizer, ENGLISH_PROMPTS, layer, device)
        zh_acts = get_activations_at_layer(model, tokenizer, CHINESE_PROMPTS, layer, device)
        D = en_acts - zh_acts
        mean_vec = D.mean(dim=0)
        feat_dir = -mean_vec / mean_vec.norm()  # negative: EN→ZH
        print(f"  Diff-of-means norm: {mean_vec.norm():.2f}")

        # FineWeb activations at this layer
        contexts, activations = extract_context_and_activation(
            model, tokenizer, token_ids_list, layer, device)
        contexts_dev = contexts.to(device)
        activations_dev = activations.to(device)

        # Reference logits
        with torch.no_grad():
            ref_hidden = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
            ref_logits = forward_from_layer_allpos(model, ref_hidden, layer)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        # Sweep along feature direction
        kl_feat = sweep_direction(
            model, contexts_dev, activations_dev, ref_log_probs,
            feat_dir, angles_rad, layer, device)
        kl_feat_median = np.median(kl_feat, axis=0)

        # Sweep along random orthogonal directions
        kl_orth_all = np.zeros((args.n_act, args.n_angles))
        for oi in range(args.n_orth_dirs):
            v = torch.randn(d_model, device=device)
            v = v - (v @ feat_dir.to(device)) * feat_dir.to(device)
            v = v / v.norm()
            kl_orth = sweep_direction(
                model, contexts_dev, activations_dev, ref_log_probs,
                v.cpu(), angles_rad, layer, device)
            kl_orth_all += kl_orth
        kl_orth_all /= args.n_orth_dirs
        kl_orth_median = np.median(kl_orth_all, axis=0)

        # Breaking angles
        r_feat = find_breaking_angle(kl_feat_median, angles_deg)
        r_plateau = find_breaking_angle(kl_orth_median, angles_deg)
        ratio = r_plateau / r_feat if r_feat > 0 and not np.isnan(r_feat) else float('nan')

        elapsed = time.time() - t0
        print(f"  r_feat={r_feat:.1f} deg, r_plateau={r_plateau:.1f} deg, ratio={ratio:.2f}")
        print(f"  Time: {elapsed:.1f}s")

        results_per_layer.append({
            'layer': layer,
            'r_feat': r_feat,
            'r_plateau': r_plateau,
            'ratio': ratio,
            'kl_feat_median': kl_feat_median,
            'kl_orth_median': kl_orth_median,
            'dom_norm': mean_vec.norm().item(),
        })

        # Checkpoint after each layer
        with open(os.path.join(args.results_dir, 'layer_results.pkl'), 'wb') as f:
            pickle.dump({'args': vars(args), 'angles_deg': angles_deg,
                         'results': results_per_layer}, f)

    # ===== Summary =====
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for r in results_per_layer:
        print(f"  Layer {r['layer']:2d}: r_feat={r['r_feat']:5.1f}, "
              f"r_plateau={r['r_plateau']:5.1f}, ratio={r['ratio']:.2f}, "
              f"dom_norm={r['dom_norm']:.2f}")

    # ===== Plot =====
    layers = [r['layer'] for r in results_per_layer]
    r_feats = [r['r_feat'] for r in results_per_layer]
    r_plateaus = [r['r_plateau'] for r in results_per_layer]
    ratios = [r['ratio'] for r in results_per_layer]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax = axes[0]
    ax.plot(layers, r_feats, 'r.-', lw=2, markersize=8, label='r_feat (feature dir)')
    ax.plot(layers, r_plateaus, 'b.-', lw=2, markersize=8, label='r_plateau (orth dir)')
    ax.set_ylabel('Breaking angle (deg)')
    ax.set_title(f'Anisotropy vs Layer: {args.model} (EN→ZH, -diff-of-means)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(layers, ratios, 'k.-', lw=2, markersize=8)
    ax.set_ylabel('Anisotropy ratio\n(r_plateau / r_feat)')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    dom_norms = [r['dom_norm'] for r in results_per_layer]
    ax.plot(layers, dom_norms, 'g.-', lw=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Diff-of-means norm')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(args.results_dir, 'anisotropy_vs_layer.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
