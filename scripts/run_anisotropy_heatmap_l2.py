#!/usr/bin/env python3
"""
2D Anisotropy Heatmap using L2 distance at the second-to-last layer.

Same 2D perturbation as the KL version, but measures L2 distance of the
hidden state at layer n-2 instead of KL divergence at the output.

Usage:
    python scripts/run_anisotropy_heatmap_l2.py --model qwen3-0.6b
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
from data.bilingual_prompts import ENGLISH_PROMPTS, CHINESE_PROMPTS


def forward_to_layer(model, hidden_states, start_layer, end_layer):
    """Forward from start_layer to end_layer (inclusive), return final-token hidden."""
    hidden = hidden_states
    blocks = _get_blocks(model)
    rope = _is_rope_model(model)
    pos_emb = _get_position_embeddings(model, hidden) if rope else None
    for block in blocks[start_layer + 1:end_layer + 1]:
        hidden = _run_block(block, hidden, model, pos_emb)
    return hidden[:, -1, :]


def extract_diff_of_means(model, tokenizer, device, layer):
    """Compute EN-ZH diff-of-means direction at given layer."""
    def get_acts(prompts):
        acts = []
        for p in prompts:
            tokens = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            acts.append(outputs.hidden_states[layer + 1][:, -1, :].squeeze(0).cpu())
        return torch.stack(acts)

    print(f"Extracting bilingual activations at layer {layer}...")
    en_acts = get_acts(ENGLISH_PROMPTS)
    zh_acts = get_acts(CHINESE_PROMPTS)
    D = en_acts - zh_acts

    mean_vec = D.mean(dim=0)
    mean_dir = mean_vec / mean_vec.norm()
    print(f"  D shape: {D.shape}, mean diff norm: {mean_vec.norm():.2f}")

    return mean_dir


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


def build_perturbed_hidden_2d(contexts, activations, u_feat, u_orth,
                               theta_y_rad, theta_x_rad):
    """Construct hidden states after 2D perturbation.

    a' = ||a|| * normalize(a_hat + tan(theta_y) * u_feat_perp + tan(theta_x) * u_orth_perp)
    """
    tan_y = np.tan(theta_y_rad)
    tan_x = np.tan(theta_x_rad)

    a_norms = activations.norm(dim=-1, keepdim=True)
    a_hat = activations / a_norms

    # Project u_feat orthogonal to a_hat
    proj_f = (u_feat.unsqueeze(0) * a_hat).sum(dim=-1, keepdim=True)
    u_f_perp = u_feat.unsqueeze(0) - proj_f * a_hat
    u_f_perp = u_f_perp / (u_f_perp.norm(dim=-1, keepdim=True) + 1e-10)

    # Project u_orth orthogonal to a_hat AND u_f_perp (Gram-Schmidt)
    proj_o_a = (u_orth.unsqueeze(0) * a_hat).sum(dim=-1, keepdim=True)
    proj_o_f = (u_orth.unsqueeze(0) * u_f_perp).sum(dim=-1, keepdim=True)
    u_o_perp = u_orth.unsqueeze(0) - proj_o_a * a_hat - proj_o_f * u_f_perp
    u_o_perp = u_o_perp / (u_o_perp.norm(dim=-1, keepdim=True) + 1e-10)

    # Perturb
    v = a_hat + tan_y * u_f_perp + tan_x * u_o_perp
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-10)
    a_pert = a_norms * v

    return torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)


def find_breaking_angle(values_vs_angle, angles_deg, threshold):
    """Find first angle at which values exceed threshold."""
    above = np.where(values_vs_angle > threshold)[0]
    if len(above) == 0:
        return angles_deg[-1]
    return angles_deg[above[0]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--n_act', type=int, default=50)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_grid', type=int, default=25)
    parser.add_argument('--n_orth_dirs', type=int, default=5)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/anisotropy_heatmap_l2/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    target_layer = n_layers - 2
    print(f"Device: {device}, d_model: {d_model}, n_layers: {n_layers}")
    print(f"Perturbation layer: {args.layer}, L2 measurement layer: {target_layer}")

    # Feature direction (negated: push toward Mandarin)
    feat_dir = -extract_diff_of_means(model, tokenizer, device, args.layer)
    feat_dir_dev = feat_dir.to(device)

    # FineWeb activations at perturbation layer
    print(f"\nLoading {args.n_act} FineWeb sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.layer} activations...")
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Activation shape: {activations.shape}, "
          f"mean norm: {activations.norm(dim=-1).mean():.2f}")

    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)

    # Reference hidden state at target layer (unperturbed)
    with torch.no_grad():
        ref_hidden = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_hidden_target = forward_to_layer(model, ref_hidden, args.layer, target_layer)

    # Threshold: 10% of median activation norm at target layer
    ref_norms = ref_hidden_target.norm(dim=-1)
    median_norm = ref_norms.median().item()
    threshold = median_norm * 0.1
    print(f"\nMedian activation norm at layer {target_layer}: {median_norm:.1f}")
    print(f"L2 threshold (10%): {threshold:.1f}")

    # Random orthogonal directions
    print(f"\nGenerating {args.n_orth_dirs} random directions orthogonal to feature dir...")
    orth_dirs = []
    for i in range(args.n_orth_dirs):
        v = torch.randn(d_model, device=device)
        v = v - (v @ feat_dir_dev) * feat_dir_dev
        v = v / v.norm()
        orth_dirs.append(v)

    # 2D sweep
    angles_deg = np.linspace(0, args.max_angle, args.n_grid)
    angles_rad = np.deg2rad(angles_deg)
    n_grid = len(angles_deg)
    N = activations_dev.shape[0]

    l2_all = np.zeros((n_grid, n_grid, N))

    total_steps = n_grid * n_grid * args.n_orth_dirs
    print(f"\n{'='*60}")
    print(f"2D SWEEP: {n_grid}x{n_grid} grid, {args.n_orth_dirs} orth dirs")
    print(f"Angles: 0 to {args.max_angle} deg")
    print(f"Total forward passes: {total_steps} (batch size {N})")
    print(f"{'='*60}")

    t0 = time.time()
    step = 0

    for orth_idx, u_orth in enumerate(orth_dirs):
        l2_this_dir = np.zeros((n_grid, n_grid, N))

        for iy, theta_y in enumerate(angles_rad):
            for ix, theta_x in enumerate(angles_rad):
                with torch.no_grad():
                    if theta_y == 0 and theta_x == 0:
                        l2_this_dir[iy, ix, :] = 0.0
                    else:
                        hidden = build_perturbed_hidden_2d(
                            contexts_dev, activations_dev,
                            feat_dir_dev, u_orth,
                            theta_y, theta_x)
                        pert_hidden = forward_to_layer(
                            model, hidden, args.layer, target_layer)
                        l2 = (pert_hidden - ref_hidden_target).norm(dim=-1)
                        l2_this_dir[iy, ix, :] = l2.cpu().numpy()

                step += 1
                if step % 200 == 0:
                    elapsed = time.time() - t0
                    rate = step / elapsed
                    eta = (total_steps - step) / rate
                    print(f"  Step {step}/{total_steps} ({step/total_steps*100:.0f}%), "
                          f"ETA: {eta:.0f}s")

        l2_all += l2_this_dir
        print(f"  Orth dir {orth_idx+1}/{args.n_orth_dirs} done "
              f"({time.time()-t0:.0f}s elapsed)")

    l2_all /= args.n_orth_dirs
    elapsed = time.time() - t0
    print(f"\nSweep complete in {elapsed:.1f}s")

    # Summary statistics
    l2_median = np.median(l2_all, axis=2)
    l2_mean = np.mean(l2_all, axis=2)

    # Breaking angles along the axes
    l2_feat_axis = l2_median[:, 0]
    l2_orth_axis = l2_median[0, :]
    r_feat = find_breaking_angle(l2_feat_axis, angles_deg, threshold)
    r_plateau = find_breaking_angle(l2_orth_axis, angles_deg, threshold)
    anisotropy = r_plateau / r_feat if r_feat > 0 else float('inf')

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"r_feat (breaking angle along feature dir): {r_feat:.1f} deg")
    print(f"r_plateau (breaking angle along orth dir): {r_plateau:.1f} deg")
    print(f"Anisotropy ratio (r_plateau / r_feat): {anisotropy:.2f}")

    # Save
    results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'l2_all': l2_all,
        'l2_median': l2_median,
        'l2_mean': l2_mean,
        'feat_dir': feat_dir.cpu(),
        'orth_dirs': [d.cpu() for d in orth_dirs],
        'r_feat': r_feat,
        'r_plateau': r_plateau,
        'anisotropy': anisotropy,
        'target_layer': target_layer,
        'median_norm': median_norm,
        'threshold': threshold,
    }
    pkl_path = os.path.join(args.results_dir, 'heatmap_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {pkl_path}")

    # Plots

    # 1. Heatmap with diverging colormap centered at threshold
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(l2_median, origin='lower', aspect='equal',
                   extent=[0, args.max_angle, 0, args.max_angle],
                   cmap='RdBu_r', vmin=0, vmax=threshold * 2)
    ax.set_xlabel('Orthogonal direction angle (deg)', fontsize=12)
    ax.set_ylabel('Feature direction angle (deg)', fontsize=12)
    ax.set_ylim(0, 40)
    ax.set_title(f'Median L2 distance at layer {target_layer}: {args.model}, '
                 f'layer {args.layer}\n'
                 f'r_feat={r_feat:.1f} deg, r_plateau={r_plateau:.1f} deg, '
                 f'ratio={anisotropy:.2f}', fontsize=11)
    plt.colorbar(im, ax=ax, label=f'L2 distance (threshold={threshold:.0f})')
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'anisotropy_heatmap_l2.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # 2. Axis profiles
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(angles_deg, l2_feat_axis, 'r-', lw=2,
            label=f'Feature dir (r={r_feat:.1f} deg)')
    ax.plot(angles_deg, l2_orth_axis, 'b-', lw=2,
            label=f'Orthogonal dir (r={r_plateau:.1f} deg)')
    ax.axhline(threshold, color='gray', linestyle='--', alpha=0.5,
               label=f'threshold={threshold:.0f}')
    if r_feat < args.max_angle:
        ax.axvline(r_feat, color='r', linestyle=':', alpha=0.5)
    if r_plateau < args.max_angle:
        ax.axvline(r_plateau, color='b', linestyle=':', alpha=0.5)
    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('Median L2 distance')
    ax.set_title(f'Axis profiles (L2 at layer {target_layer}) — ratio: {anisotropy:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(args.results_dir, 'axis_profiles_l2.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
