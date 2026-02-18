#!/usr/bin/env python3
"""
Anisotropy ratio across different EN-X language pairs at a fixed layer.

For each target language X, computes -diff-of-means(EN, X) as the feature direction,
then measures r_feat, r_plateau, and their ratio by perturbing FineWeb activations.

Usage:
    python scripts/run_anisotropy_across_pairs.py --model qwen3-0.6b --layer 14
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

from src.model import load_model, forward_from_layer_allpos
from src.data import load_fineweb_fixed_length
from data.multilingual_prompts import PROMPT_SETS, LANGUAGES


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


def sweep_direction(model, contexts, activations, ref_log_probs, direction,
                    angles_rad, layer, device):
    """Sweep a single direction, return per-activation KL. [N, n_angles]"""
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


# Human-readable language names
LANG_NAMES = {
    'en': 'English', 'zh': 'Mandarin', 'nl': 'Dutch', 'es': 'Spanish',
    'fr': 'French', 'de': 'German', 'ja': 'Japanese', 'it': 'Italian',
    'ru': 'Russian', 'pt': 'Portuguese',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
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
        args.results_dir = f'results/anisotropy_across_pairs/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Setup =====
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    # ===== Extract all language activations at this layer =====
    print(f"\nExtracting activations for all languages at layer {args.layer}...")
    lang_acts = {}
    for lang in LANGUAGES:
        lang_acts[lang] = get_language_activations(
            model, tokenizer, device, args.layer, lang)
        print(f"  {lang}: {lang_acts[lang].shape}")

    # ===== FineWeb activations =====
    print(f"\nLoading {args.n_act} FineWeb sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)

    # Reference logits
    with torch.no_grad():
        ref_hidden = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_logits = forward_from_layer_allpos(model, ref_hidden, args.layer)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles)
    angles_rad = np.deg2rad(angles_deg)

    # ===== Sweep each EN-X pair =====
    target_langs = [l for l in LANGUAGES if l != 'en']
    results_per_pair = []

    for target in target_langs:
        t0 = time.time()
        pair_name = f"EN-{target.upper()}"
        print(f"\n{'='*60}")
        print(f"{pair_name}")
        print(f"{'='*60}")

        # Diff-of-means: -(en - target) = target - en (push toward target)
        D = lang_acts['en'] - lang_acts[target]
        mean_vec = D.mean(dim=0)
        feat_dir = -mean_vec / mean_vec.norm()
        print(f"  Diff-of-means norm: {mean_vec.norm():.2f}")

        # Cosine similarity of this feature direction with EN-ZH (for comparison)
        if target != 'zh':
            D_zh = lang_acts['en'] - lang_acts['zh']
            zh_dir = D_zh.mean(dim=0)
            zh_dir = zh_dir / zh_dir.norm()
            cos_with_zh = (feat_dir @ zh_dir).abs().item()
            print(f"  cos(this dir, EN-ZH dir): {cos_with_zh:.3f}")

        # Sweep along feature direction
        kl_feat = sweep_direction(
            model, contexts_dev, activations_dev, ref_log_probs,
            feat_dir, angles_rad, args.layer, device)
        kl_feat_median = np.median(kl_feat, axis=0)

        # Sweep along random orthogonal directions
        kl_orth_all = np.zeros((args.n_act, args.n_angles))
        for oi in range(args.n_orth_dirs):
            v = torch.randn(d_model, device=device)
            v = v - (v @ feat_dir.to(device)) * feat_dir.to(device)
            v = v / v.norm()
            kl_orth = sweep_direction(
                model, contexts_dev, activations_dev, ref_log_probs,
                v.cpu(), angles_rad, args.layer, device)
            kl_orth_all += kl_orth
        kl_orth_all /= args.n_orth_dirs
        kl_orth_median = np.median(kl_orth_all, axis=0)

        r_feat = find_breaking_angle(kl_feat_median, angles_deg)
        r_plateau = find_breaking_angle(kl_orth_median, angles_deg)
        ratio = r_plateau / r_feat if r_feat > 0 and not np.isnan(r_feat) else float('nan')

        elapsed = time.time() - t0
        print(f"  r_feat={r_feat:.1f} deg, r_plateau={r_plateau:.1f} deg, ratio={ratio:.2f}")
        print(f"  Time: {elapsed:.1f}s")

        results_per_pair.append({
            'pair': pair_name,
            'target': target,
            'r_feat': r_feat,
            'r_plateau': r_plateau,
            'ratio': ratio,
            'dom_norm': mean_vec.norm().item(),
            'kl_feat_median': kl_feat_median,
            'kl_orth_median': kl_orth_median,
        })

    # ===== Save =====
    all_results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'results': results_per_pair,
    }
    with open(os.path.join(args.results_dir, 'pair_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)

    # ===== Summary =====
    print(f"\n{'='*60}")
    print(f"SUMMARY (layer {args.layer})")
    print(f"{'='*60}")
    for r in results_per_pair:
        print(f"  {r['pair']:6s}: r_feat={r['r_feat']:5.1f}, "
              f"r_plateau={r['r_plateau']:5.1f}, ratio={r['ratio']:.2f}, "
              f"dom_norm={r['dom_norm']:.2f}")

    # ===== Plots =====
    pairs = [r['pair'] for r in results_per_pair]
    r_feats = [r['r_feat'] for r in results_per_pair]
    r_plateaus = [r['r_plateau'] for r in results_per_pair]
    ratios = [r['ratio'] for r in results_per_pair]
    dom_norms = [r['dom_norm'] for r in results_per_pair]

    # Sort by r_feat for cleaner visualization
    sort_idx = np.argsort(r_feats)
    pairs_s = [pairs[i] for i in sort_idx]
    r_feats_s = [r_feats[i] for i in sort_idx]
    r_plateaus_s = [r_plateaus[i] for i in sort_idx]
    ratios_s = [ratios[i] for i in sort_idx]
    dom_norms_s = [dom_norms[i] for i in sort_idx]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    x = np.arange(len(pairs_s))

    ax = axes[0]
    ax.bar(x - 0.15, r_feats_s, 0.3, color='red', alpha=0.7, label='r_feat')
    ax.bar(x + 0.15, r_plateaus_s, 0.3, color='blue', alpha=0.7, label='r_plateau')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs_s, rotation=45, ha='right')
    ax.set_ylabel('Breaking angle (deg)')
    ax.set_title(f'Anisotropy across language pairs: {args.model}, layer {args.layer}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x, ratios_s, color='black', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs_s, rotation=45, ha='right')
    ax.set_ylabel('Anisotropy ratio')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    ax.bar(x, dom_norms_s, color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs_s, rotation=45, ha='right')
    ax.set_ylabel('Diff-of-means norm')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(args.results_dir, 'anisotropy_across_pairs.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")

    # Also plot axis profiles overlaid
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_per_pair)))
    for i, r in enumerate(results_per_pair):
        ax.plot(angles_deg, r['kl_feat_median'], color=colors[i], lw=2,
                label=f"{r['pair']} (r={r['r_feat']:.0f}°)")
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('Median KL divergence (nats)')
    ax.set_title(f'Feature direction profiles: {args.model}, layer {args.layer}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    path = os.path.join(args.results_dir, 'feat_profiles.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
