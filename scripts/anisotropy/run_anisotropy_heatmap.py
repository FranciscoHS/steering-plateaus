#!/usr/bin/env python3
"""
2D Anisotropy Heatmap: visualize the plateau shape in the plane spanned by
a feature direction (diff-of-means) and orthogonal random directions.

At each grid point (theta_x, theta_y), perturb each activation:
    a' = ||a|| * normalize(a_hat + tan(theta_y) * u_feat_perp + tan(theta_x) * u_orth_perp)
and measure KL divergence from the unperturbed output distribution.

The feature direction is the EN-ZH diff-of-means. Orthogonal directions are random
(projected out of the feature direction). Results are averaged over multiple random
orthogonal directions and reported as median over activations.

Usage:
    python scripts/run_anisotropy_heatmap.py --model qwen3-0.6b
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

    u_feat_perp and u_orth_perp are the feature and orthogonal directions projected
    orthogonal to each activation's unit vector, then Gram-Schmidt'd to ensure
    orthogonality in the tangent plane.

    Args:
        contexts: [N, seq_len-1, d] on device
        activations: [N, d] on device
        u_feat: [d] unit vector (feature direction), on device
        u_orth: [d] unit vector (orthogonal direction), on device
        theta_y_rad: angle along feature direction
        theta_x_rad: angle along orthogonal direction

    Returns:
        hidden: [N, seq_len, d]
    """
    tan_y = np.tan(theta_y_rad)
    tan_x = np.tan(theta_x_rad)

    a_norms = activations.norm(dim=-1, keepdim=True)  # [N, 1]
    a_hat = activations / a_norms  # [N, d]

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


def find_breaking_angle(kl_vs_angle, angles_deg, threshold=1.0):
    """Find first angle at which KL exceeds threshold. Returns max angle if never."""
    above = np.where(kl_vs_angle > threshold)[0]
    if len(above) == 0:
        return angles_deg[-1]
    return angles_deg[above[0]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--n_act', type=int, default=50)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--max_angle', type=float, default=60.0,
                        help='Maximum angle in degrees for both axes')
    parser.add_argument('--n_grid', type=int, default=25,
                        help='Number of grid points per axis (including 0)')
    parser.add_argument('--n_orth_dirs', type=int, default=5,
                        help='Number of random orthogonal directions to average over')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/anisotropy_heatmap/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Phase 1: Setup =====
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    # Feature direction from bilingual diff-of-means
    feat_dir = extract_diff_of_means(model, tokenizer, device, args.layer)
    feat_dir_dev = feat_dir.to(device)

    # ===== Phase 2: FineWeb activations =====
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

    # Reference (unperturbed) logits
    with torch.no_grad():
        ref_hidden = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_logits = forward_from_layer_allpos(model, ref_hidden, args.layer)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # ===== Phase 3: Random orthogonal directions =====
    print(f"\nGenerating {args.n_orth_dirs} random directions orthogonal to feature dir...")
    orth_dirs = []
    for i in range(args.n_orth_dirs):
        v = torch.randn(d_model, device=device)
        v = v - (v @ feat_dir_dev) * feat_dir_dev
        v = v / v.norm()
        orth_dirs.append(v)

    # ===== Phase 4: 2D sweep =====
    angles_deg = np.linspace(0, args.max_angle, args.n_grid)
    angles_rad = np.deg2rad(angles_deg)
    n_grid = len(angles_deg)
    N = activations_dev.shape[0]

    # kl_all[iy, ix, n] = KL for activation n at grid point (iy, ix), averaged over orth dirs
    kl_all = np.zeros((n_grid, n_grid, N))

    total_steps = n_grid * n_grid * args.n_orth_dirs
    print(f"\n{'='*60}")
    print(f"2D SWEEP: {n_grid}x{n_grid} grid, {args.n_orth_dirs} orth dirs")
    print(f"Angles: 0 to {args.max_angle} deg")
    print(f"Total forward passes: {total_steps} (batch size {N})")
    print(f"{'='*60}")

    t0 = time.time()
    step = 0

    for orth_idx, u_orth in enumerate(orth_dirs):
        kl_this_dir = np.zeros((n_grid, n_grid, N))

        for iy, theta_y in enumerate(angles_rad):
            for ix, theta_x in enumerate(angles_rad):
                with torch.no_grad():
                    if theta_y == 0 and theta_x == 0:
                        kl_this_dir[iy, ix, :] = 0.0
                    else:
                        hidden = build_perturbed_hidden_2d(
                            contexts_dev, activations_dev,
                            feat_dir_dev, u_orth,
                            theta_y, theta_x)
                        logits = forward_from_layer_allpos(model, hidden, args.layer)
                        log_probs = F.log_softmax(logits, dim=-1)
                        kl = (ref_log_probs.exp() * (ref_log_probs - log_probs)).sum(dim=-1)
                        kl_this_dir[iy, ix, :] = kl.cpu().numpy()

                step += 1
                if step % 200 == 0:
                    elapsed = time.time() - t0
                    rate = step / elapsed
                    eta = (total_steps - step) / rate
                    print(f"  Step {step}/{total_steps} ({step/total_steps*100:.0f}%), "
                          f"ETA: {eta:.0f}s")

        kl_all += kl_this_dir
        print(f"  Orth dir {orth_idx+1}/{args.n_orth_dirs} done "
              f"({time.time()-t0:.0f}s elapsed)")

    kl_all /= args.n_orth_dirs
    elapsed = time.time() - t0
    print(f"\nSweep complete in {elapsed:.1f}s")

    # ===== Phase 5: Summary statistics =====
    kl_median = np.median(kl_all, axis=2)  # [n_grid_y, n_grid_x]
    kl_mean = np.mean(kl_all, axis=2)

    # Breaking angles along the axes
    kl_feat_axis = kl_median[:, 0]   # y-axis: perturb along feature only
    kl_orth_axis = kl_median[0, :]   # x-axis: perturb along orth only
    r_feat = find_breaking_angle(kl_feat_axis, angles_deg, threshold=1.0)
    r_plateau = find_breaking_angle(kl_orth_axis, angles_deg, threshold=1.0)
    anisotropy = r_plateau / r_feat if r_feat > 0 else float('inf')

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"r_feat (breaking angle along feature dir): {r_feat:.1f} deg")
    print(f"r_plateau (breaking angle along orth dir): {r_plateau:.1f} deg")
    print(f"Anisotropy ratio (r_plateau / r_feat): {anisotropy:.2f}")

    # ===== Phase 6: Save =====
    results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'kl_all': kl_all,
        'kl_median': kl_median,
        'kl_mean': kl_mean,
        'feat_dir': feat_dir.cpu(),
        'orth_dirs': [d.cpu() for d in orth_dirs],
        'r_feat': r_feat,
        'r_plateau': r_plateau,
        'anisotropy': anisotropy,
    }
    pkl_path = os.path.join(args.results_dir, 'heatmap_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {pkl_path}")

    # ===== Phase 7: Plots =====

    # 1. Main heatmap (two panels: linear and log scale)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(kl_median, origin='lower', aspect='equal',
                   extent=[0, args.max_angle, 0, args.max_angle],
                   cmap='viridis')
    cs = ax.contour(angles_deg, angles_deg, kl_median, levels=[1.0],
                    colors='white', linewidths=2, linestyles='--')
    ax.clabel(cs, fmt='KL=%.0f')
    ax.set_xlabel('Orthogonal direction angle (deg)')
    ax.set_ylabel('Feature direction angle (deg)')
    ax.set_title('Median KL divergence')
    plt.colorbar(im, ax=ax, label='KL (nats)')

    ax = axes[1]
    kl_log = np.log10(np.clip(kl_median, 1e-4, None))
    im = ax.imshow(kl_log, origin='lower', aspect='equal',
                   extent=[0, args.max_angle, 0, args.max_angle],
                   cmap='viridis')
    levels = [l for l in [0.1, 0.5, 1.0, 2.0, 5.0] if l <= kl_median.max()]
    if levels:
        ax.contour(angles_deg, angles_deg, kl_median, levels=levels,
                   colors='white', linewidths=1, linestyles='--')
    ax.set_xlabel('Orthogonal direction angle (deg)')
    ax.set_ylabel('Feature direction angle (deg)')
    ax.set_title('log10(Median KL)')
    plt.colorbar(im, ax=ax, label='log10(KL)')

    fig.suptitle(f'Plateau Anisotropy: {args.model}, layer {args.layer}\n'
                 f'r_feat={r_feat:.1f} deg, r_plateau={r_plateau:.1f} deg, '
                 f'ratio={anisotropy:.2f}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'anisotropy_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # 2. Axis profiles
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(angles_deg, kl_feat_axis, 'r-', lw=2, label=f'Feature dir (r={r_feat:.1f} deg)')
    ax.plot(angles_deg, kl_orth_axis, 'b-', lw=2, label=f'Orthogonal dir (r={r_plateau:.1f} deg)')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='KL = 1')
    if r_feat < args.max_angle:
        ax.axvline(r_feat, color='r', linestyle=':', alpha=0.5)
    if r_plateau < args.max_angle:
        ax.axvline(r_plateau, color='b', linestyle=':', alpha=0.5)
    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('Median KL divergence (nats)')
    ax.set_title(f'Axis profiles — anisotropy ratio: {anisotropy:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(args.results_dir, 'axis_profiles.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
