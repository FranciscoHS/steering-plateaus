#!/usr/bin/env python3
"""
Anisotropy ratio vs layer for multiple EN-X language pairs,
using L2 distance at the second-to-last layer as the metric
(instead of KL divergence).

Layers n-2 and n-1 are excluded since the measurement is taken at layer n-2.

Usage:
    python scripts/run_anisotropy_pairs_x_layers_l2.py --model qwen3-0.6b
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, _get_blocks, _is_rope_model, _get_position_embeddings, _run_block,
)
from src.data import load_fineweb_fixed_length
from data.multilingual_prompts import PROMPT_SETS, LANGUAGES


LANG_NAMES = {
    'zh': 'Mandarin', 'nl': 'Dutch', 'es': 'Spanish', 'fr': 'French',
    'de': 'German', 'ja': 'Japanese', 'it': 'Italian', 'ru': 'Russian',
    'pt': 'Portuguese',
}


def forward_to_layer(model, hidden_states, start_layer, end_layer):
    """Forward from start_layer to end_layer (inclusive), return final-token hidden."""
    hidden = hidden_states
    blocks = _get_blocks(model)
    rope = _is_rope_model(model)
    pos_emb = _get_position_embeddings(model, hidden) if rope else None
    for block in blocks[start_layer + 1:end_layer + 1]:
        hidden = _run_block(block, hidden, model, pos_emb)
    return hidden[:, -1, :]


def get_language_activations(model, tokenizer, device, layer, lang_code):
    """Extract final-token activations for all prompts in a given language."""
    acts = []
    for pset in PROMPT_SETS:
        prompt = pset[lang_code]
        tokens = tokenizer(prompt, return_tensors='pt').to(device)
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


def sweep_direction_l2(model, contexts, activations, ref_hidden_target,
                       direction, angles_rad, start_layer, target_layer, device):
    """Sweep a direction, return per-activation L2 distance at target layer. [N, n_angles]"""
    N = activations.shape[0]
    l2_dists = np.zeros((N, len(angles_rad)))
    direction_dev = direction.to(device)
    for i, theta in enumerate(angles_rad):
        with torch.no_grad():
            hidden = build_perturbed_hidden(contexts, activations, direction_dev, theta)
            pert_hidden = forward_to_layer(model, hidden, start_layer, target_layer)
            l2 = (pert_hidden - ref_hidden_target).norm(dim=-1)
            l2_dists[:, i] = l2.cpu().numpy()
    return l2_dists


def find_breaking_angle(values_vs_angle, angles_deg, threshold):
    """Find first angle where values exceed threshold."""
    above = np.where(values_vs_angle > threshold)[0]
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
        args.results_dir = f'results/anisotropy_pairs_x_layers_l2/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    target_layer = n_layers - 2  # second-to-last
    print(f"Device: {device}, d_model: {d_model}, n_layers: {n_layers}")
    print(f"Target layer for L2 measurement: {target_layer}")

    # Load FineWeb tokens once
    print(f"\nLoading {args.n_act} FineWeb sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles)
    angles_rad = np.deg2rad(angles_deg)

    target_langs = [l for l in LANGUAGES if l != 'en']
    results = {t: [] for t in target_langs}

    # First pass: determine L2 threshold from reference norms at target layer
    # We'll compute this once and use it for all layers
    print(f"\nComputing reference activation norms at target layer {target_layer}...")
    contexts_tgt, activations_tgt = extract_context_and_activation(
        model, tokenizer, token_ids_list, target_layer, device)
    ref_norms = activations_tgt.norm(dim=-1)
    median_norm = ref_norms.median().item()
    print(f"  Median activation norm at layer {target_layer}: {median_norm:.1f}")

    t_total = time.time()

    # Sweep layers 0 to target_layer - 1
    max_layer = target_layer  # exclusive
    for layer in range(max_layer):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Layer {layer}/{max_layer-1}")
        print(f"{'='*60}")

        # Extract language activations at this layer
        lang_acts = {}
        for lang in LANGUAGES:
            lang_acts[lang] = get_language_activations(
                model, tokenizer, device, layer, lang)

        # FineWeb activations at this layer
        contexts, activations = extract_context_and_activation(
            model, tokenizer, token_ids_list, layer, device)
        contexts_dev = contexts.to(device)
        activations_dev = activations.to(device)

        # Reference hidden state at target layer (unperturbed)
        with torch.no_grad():
            ref_hidden = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
            ref_hidden_target = forward_to_layer(model, ref_hidden, layer, target_layer)

        for target in target_langs:
            D = lang_acts['en'] - lang_acts[target]
            mean_vec = D.mean(dim=0)
            feat_dir = -mean_vec / mean_vec.norm()

            # Sweep feature direction
            l2_feat = sweep_direction_l2(
                model, contexts_dev, activations_dev, ref_hidden_target,
                feat_dir, angles_rad, layer, target_layer, device)
            l2_feat_median = np.median(l2_feat, axis=0)

            # Sweep random orthogonal directions
            l2_orth_all = np.zeros((args.n_act, args.n_angles))
            for oi in range(args.n_orth_dirs):
                v = torch.randn(d_model, device=device)
                v = v - (v @ feat_dir.to(device)) * feat_dir.to(device)
                v = v / v.norm()
                l2_orth = sweep_direction_l2(
                    model, contexts_dev, activations_dev, ref_hidden_target,
                    v.cpu(), angles_rad, layer, target_layer, device)
                l2_orth_all += l2_orth
            l2_orth_all /= args.n_orth_dirs
            l2_orth_median = np.median(l2_orth_all, axis=0)

            results[target].append({
                'layer': layer,
                'l2_feat_median': l2_feat_median,
                'l2_orth_median': l2_orth_median,
                'dom_norm': mean_vec.norm().item(),
            })

        elapsed = time.time() - t0
        # Print sample L2 values at max angle for orientation
        zh_data = results['zh'][-1]
        print(f"  EN-ZH at max angle: feat L2={zh_data['l2_feat_median'][-1]:.1f}, "
              f"orth L2={zh_data['l2_orth_median'][-1]:.1f}")
        print(f"  Time: {elapsed:.1f}s")

        # Checkpoint
        with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
            pickle.dump({'args': vars(args), 'angles_deg': angles_deg,
                         'results': results, 'target_layer': target_layer,
                         'median_norm': median_norm}, f)

    total_time = time.time() - t_total
    print(f"\nTotal time: {total_time:.0f}s")

    # ===== Determine threshold from data =====
    # Look at the range of L2 values to pick a good threshold
    all_l2_feat_max = []
    all_l2_orth_max = []
    for target in target_langs:
        for r in results[target]:
            all_l2_feat_max.append(r['l2_feat_median'][-1])
            all_l2_orth_max.append(r['l2_orth_median'][-1])
    print(f"\nL2 at max angle ({args.max_angle} deg):")
    print(f"  Feature dir: median={np.median(all_l2_feat_max):.1f}, "
          f"range=[{np.min(all_l2_feat_max):.1f}, {np.max(all_l2_feat_max):.1f}]")
    print(f"  Orth dir: median={np.median(all_l2_orth_max):.1f}, "
          f"range=[{np.min(all_l2_orth_max):.1f}, {np.max(all_l2_orth_max):.1f}]")

    # Use median_norm * 0.1 as a reasonable threshold (10% of activation norm)
    # Also try a few thresholds and pick the one that gives the cleanest separation
    thresholds_to_try = [median_norm * f for f in [0.05, 0.1, 0.2]]
    print(f"\nTrying thresholds (fractions of median norm {median_norm:.1f}):")
    for thresh in thresholds_to_try:
        # Quick check: how many breaking angles are nan vs at boundary?
        n_nan = 0
        n_boundary = 0
        for target in target_langs:
            for r in results[target]:
                ba = find_breaking_angle(r['l2_feat_median'], angles_deg, thresh)
                if np.isnan(ba):
                    n_nan += 1
                elif ba >= args.max_angle - 1:
                    n_boundary += 1
        total = len(target_langs) * len(results[target_langs[0]])
        print(f"  thresh={thresh:.1f}: {n_nan} nan, {n_boundary} at boundary, "
              f"{total - n_nan - n_boundary} good")

    # Use 10% of median norm
    threshold = median_norm * 0.1
    print(f"\nUsing threshold: {threshold:.1f} (10% of median norm)")

    # Compute breaking angles with chosen threshold
    for target in target_langs:
        for r in results[target]:
            r['r_feat'] = find_breaking_angle(r['l2_feat_median'], angles_deg, threshold)
            r['r_plateau'] = find_breaking_angle(r['l2_orth_median'], angles_deg, threshold)
            r_f = r['r_feat']
            r_p = r['r_plateau']
            r['ratio'] = r_p / r_f if r_f > 0 and not np.isnan(r_f) else float('nan')

    # Save final results
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'angles_deg': angles_deg,
                     'results': results, 'target_layer': target_layer,
                     'median_norm': median_norm, 'threshold': threshold}, f)

    # ===== Plot: anisotropy ratio vs layer =====
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(target_langs)))
    for i, target in enumerate(target_langs):
        layers = [r['layer'] for r in results[target]]
        ratios = [r['ratio'] for r in results[target]]
        label = f"EN-{target.upper()} ({LANG_NAMES[target]})"
        ax.plot(layers, ratios, '.-', color=colors[i], lw=1.5, markersize=5, label=label)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Anisotropy ratio (r_plateau / r_feat)', fontsize=12)
    ax.set_title(f'Anisotropy ratio vs layer (L2 at layer {target_layer}): {args.model}',
                 fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'ratio_vs_layer_l2.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ===== Plot: breaking angles vs layer =====
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    for i, target in enumerate(target_langs):
        layers = [r['layer'] for r in results[target]]
        r_feats = [r['r_feat'] for r in results[target]]
        ax.plot(layers, r_feats, '.-', color=colors[i], lw=1.5, markersize=5,
                label=f"EN-{target.upper()}")
    ax.set_ylabel('r_feat (deg)', fontsize=12)
    ax.set_title(f'Breaking angles vs layer (L2 at layer {target_layer}, '
                 f'threshold={threshold:.1f}): {args.model}', fontsize=13)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, target in enumerate(target_langs):
        layers = [r['layer'] for r in results[target]]
        r_plats = [r['r_plateau'] for r in results[target]]
        ax.plot(layers, r_plats, '.-', color=colors[i], lw=1.5, markersize=5,
                label=f"EN-{target.upper()}")
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('r_plateau (deg)', fontsize=12)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(args.results_dir, 'breaking_angles_vs_layer_l2.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
