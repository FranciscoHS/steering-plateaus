#!/usr/bin/env python3
"""
Plateau width vs steering magnitude experiment.

Steers English activations toward Mandarin (zh) at increasing angles,
then measures plateau width via perturbation analysis at each magnitude.
Tests whether the plateau shrinks as we move away from the English manifold.

Perturbation directions are NOT orthogonalized — just random unit vectors
plus the zh steering direction itself.

Usage:
    python scripts/perturbation/run_plateau_vs_steering.py
"""

import sys
import os
import pickle
import time
import numpy as np
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_final, forward_from_layer_to_layer, _get_final_norm, _get_lm_head, _get_blocks

LAYER = 14
MODEL_NAME = 'qwen3-0.6b'
DIRECTIONS_PATH = 'results/allpairs_contrastive/qwen3-0.6b/directions.pkl'

# Steering magnitudes (degrees)
STEER_ANGLES_DEG = np.arange(0, 65, 5)  # 0, 5, 10, ..., 60

# Perturbation sweep
N_RANDOM_DIRS = 10
N_PERT_ANGLES = 45
PERT_MAX_DEG = 90.0

# KL thresholds for plateau width
KL_THRESHOLDS = [0.1, 0.2, 0.5]

# L2 plateau: fraction of off-manifold linear baseline
L2_RATIO_THRESHOLDS = [0.5, 0.8]

# Cosine distance thresholds for plateau width
COSDIST_THRESHOLDS = [0.01, 0.05, 0.1]

SEED = 42

# Diverse English prompts (from run_steering_mandarin_fraction.py)
PROMPTS = [
    "The weather today is",
    "I went to the store and",
    "My favorite food is",
    "Last weekend I decided to",
    "The best thing about summer is",
    "Once upon a time, there",
    "She opened the door and",
    "The old man sat quietly",
    "In the dark forest, a",
    "He picked up the letter and",
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest ocean on Earth is",
    "Photosynthesis is the process by which",
    "The first person to walk on the moon",
    "In recent years, scientists have",
    "The theory of relativity states that",
    "Machine learning algorithms are used to",
    "Climate change has led to",
    "The human brain contains approximately",
    "The meaning of life is",
    "Democracy is important because",
    "Education should focus on",
    "The internet has changed how",
    "Happiness can be found in",
    "To make a cup of coffee",
    "The first step in solving this",
    "When writing a good essay",
    "A healthy diet should include",
    "The most efficient way to",
]


def spherical_rotate(activation, direction, angle_rad):
    """Rotate activation toward direction by angle_rad on the sphere.

    a_rotated = cos(θ) * a + sin(θ) * ||a|| * v_perp
    where v_perp is direction made orthogonal to a and normalized.

    Args:
        activation: [d_model] tensor
        direction: [d_model] unit vector
        angle_rad: scalar

    Returns:
        rotated: [d_model] tensor (same norm as activation)
    """
    R = activation.norm()
    a_unit = activation / R

    # Make direction orthogonal to activation
    proj = (direction @ a_unit) * a_unit
    v_perp = direction - proj
    v_norm = v_perp.norm()
    if v_norm < 1e-10:
        return activation  # direction parallel to activation
    v_perp = v_perp / v_norm

    return R * (np.cos(angle_rad) * a_unit + np.sin(angle_rad) * v_perp)


def compute_metrics(model, context, activation, directions, pert_angles_rad,
                    layer, device, measure_layer=None):
    """Compute KL divergences, L2 distances, and cosine distances for perturbations.

    Perturbation: a_perturbed = cos(θ) * a + sin(θ) * ||a|| * v
    where v is a raw unit vector (NOT orthogonalized to a).

    Returns:
        kl_divs: [n_angles, n_dirs]
        l2_dists: [n_angles, n_dirs] — L2 on final-layer activations (before ln_f)
        cos_dists: [n_angles, n_dirs] — cosine distance at measure_layer (or final)
    """
    model_dtype = next(model.parameters()).dtype
    n_dirs = directions.shape[0]
    n_angles = len(pert_angles_rad)

    ctx = context.to(device=device, dtype=model_dtype)  # already [1, seq-1, d]
    a = activation.to(device=device, dtype=model_dtype)
    R = a.norm()
    dirs = directions.to(device=device, dtype=model_dtype)

    # Reference: final-layer activations + logits + measure-layer activations
    with torch.no_grad():
        ref_final = forward_from_layer_to_final(model, ctx, a.unsqueeze(0).unsqueeze(0), layer)
        ref_ln = _get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1)
        ref_logits = _get_lm_head(model)(ref_ln)
        ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
        if measure_layer is not None:
            ref_measure = forward_from_layer_to_layer(
                model, ctx, a.unsqueeze(0).unsqueeze(0), layer, measure_layer).float()
        else:
            ref_measure = ref_final.float()

    kl_divs = np.zeros((n_angles, n_dirs))
    l2_dists = np.zeros((n_angles, n_dirs))
    cos_dists = np.zeros((n_angles, n_dirs))

    for ang_idx, theta in enumerate(pert_angles_rad):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        perturbed = cos_t * a.unsqueeze(0) + sin_t * R * dirs  # [n_dirs, d]
        perturbed = perturbed.unsqueeze(1)  # [n_dirs, 1, d]

        with torch.no_grad():
            pert_final = forward_from_layer_to_final(model, ctx, perturbed, layer)

            # L2 on final-layer activations
            l2_dists[ang_idx] = (pert_final - ref_final).float().norm(dim=-1).cpu().numpy()

            # KL on logits
            pert_ln = _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)
            pert_logits = _get_lm_head(model)(pert_ln)
            pert_log_probs = F.log_softmax(pert_logits.float(), dim=-1)
            ref_probs = ref_log_probs.exp()
            kl = (ref_probs * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            kl_divs[ang_idx] = kl.cpu().numpy()

            # Cosine distance at measure_layer
            if measure_layer is not None:
                pert_measure = forward_from_layer_to_layer(
                    model, ctx, perturbed, layer, measure_layer).float()
            else:
                pert_measure = pert_final.float()
            cos_dists[ang_idx] = (1.0 - F.cosine_similarity(
                ref_measure, pert_measure, dim=-1)).cpu().numpy()

    return kl_divs, l2_dists, cos_dists


def compute_plateau_width(pert_angles_deg, medians, threshold):
    """Find the angle where median crosses the threshold via interpolation.

    Returns max angle if never crosses, 0 if already above at first angle.
    """
    above = np.where(medians > threshold)[0]
    if len(above) == 0:
        return pert_angles_deg[-1]
    first_above = above[0]
    if first_above == 0:
        return 0.0
    x0, x1 = pert_angles_deg[first_above - 1], pert_angles_deg[first_above]
    y0, y1 = medians[first_above - 1], medians[first_above]
    return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    results_dir = f'results/plateau_vs_steering/{MODEL_NAME}'
    os.makedirs(results_dir, exist_ok=True)

    # Perturbation angles (skip 0)
    pert_angles_deg = np.linspace(0, PERT_MAX_DEG, N_PERT_ANGLES + 1)[1:]
    pert_angles_rad = np.deg2rad(pert_angles_deg)

    print(f"Config: model={MODEL_NAME}, layer={LAYER}")
    print(f"Steering angles: {list(STEER_ANGLES_DEG)}°")
    print(f"Perturbation: {N_RANDOM_DIRS + 1} dirs, {len(pert_angles_deg)} angles "
          f"[{pert_angles_deg[0]:.1f}°–{pert_angles_deg[-1]:.1f}°]")
    print(f"KL thresholds: {KL_THRESHOLDS}")
    print(f"Prompts: {len(PROMPTS)}")

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer, device = load_model(MODEL_NAME)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2  # 2nd-to-last
    print(f"Device: {device}, d_model: {d_model}, n_layers: {n_layers}, "
          f"cosine distance measured at layer {measure_layer}")

    # Load zh direction
    print(f"Loading zh direction from {DIRECTIONS_PATH}")
    with open(DIRECTIONS_PATH, 'rb') as f:
        dir_data = pickle.load(f)
    zh_dir = dir_data['features']['zh'].to(dtype=torch.float32)
    print(f"zh direction norm: {zh_dir.norm().item():.4f}")

    # Extract activations for each prompt
    print(f"\nExtracting layer-{LAYER} activations for {len(PROMPTS)} prompts...")
    contexts = []
    activations = []

    for prompt in PROMPTS:
        tokens = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[LAYER + 1]
        contexts.append(hidden[:, :-1, :].float().cpu())
        activations.append(hidden[:, -1, :].squeeze(0).float().cpu())

    activations = torch.stack(activations)  # [n_prompts, d_model]
    print(f"Activation norms: mean={activations.norm(dim=-1).mean():.1f}, "
          f"std={activations.norm(dim=-1).std():.1f}")

    # Perturbation directions: zh + random (NOT orthogonalized)
    print("Sampling perturbation directions (random unit vectors + zh)...")
    random_dirs = F.normalize(torch.randn(N_RANDOM_DIRS, d_model), dim=-1)
    all_dirs = torch.cat([zh_dir.unsqueeze(0), random_dirs], dim=0)  # [11, d_model]
    print(f"Total perturbation directions: {all_dirs.shape[0]} (1 zh + {N_RANDOM_DIRS} random)")

    # Main experiment loop
    n_prompts = len(PROMPTS)
    n_steer = len(STEER_ANGLES_DEG)
    n_pert = len(pert_angles_deg)
    n_dirs = all_dirs.shape[0]

    all_kl = np.zeros((n_steer, n_prompts, n_pert, n_dirs))
    all_l2 = np.zeros((n_steer, n_prompts, n_pert, n_dirs))
    all_cd = np.zeros((n_steer, n_prompts, n_pert, n_dirs))

    t0 = time.time()

    for s_idx, steer_deg in enumerate(STEER_ANGLES_DEG):
        steer_rad = np.deg2rad(steer_deg)
        print(f"\n=== Steering angle: {steer_deg}° ===")

        for p_idx in range(n_prompts):
            if steer_deg == 0:
                steered = activations[p_idx]
            else:
                steered = spherical_rotate(activations[p_idx], zh_dir, steer_rad)

            kl_divs, l2_dists, cos_dists = compute_metrics(
                model, contexts[p_idx], steered, all_dirs,
                pert_angles_rad, LAYER, device, measure_layer=measure_layer
            )
            all_kl[s_idx, p_idx] = kl_divs
            all_l2[s_idx, p_idx] = l2_dists
            all_cd[s_idx, p_idx] = cos_dists

        elapsed = time.time() - t0
        print(f"  Done {n_prompts} prompts ({elapsed:.1f}s total)")

    print(f"\nTotal time: {time.time() - t0:.1f}s")

    # Off-manifold linear baseline for L2 (fit to highest steering, random dirs, 0-30°)
    l2_max_steer = all_l2[-1]  # [n_prompts, n_pert, n_dirs]
    l2_rand_max = l2_max_steer[:, :, 1:].transpose(0, 2, 1).reshape(-1, n_pert)
    median_rand_max = np.median(l2_rand_max, axis=0)
    fit_mask = pert_angles_deg <= 30
    x_fit = pert_angles_deg[fit_mask]
    y_fit = median_rand_max[fit_mask]
    slope_offmanifold = np.sum(x_fit * y_fit) / np.sum(x_fit ** 2)
    print(f"\nOff-manifold linear fit: slope={slope_offmanifold:.2f} L2/deg "
          f"(from steer {STEER_ANGLES_DEG[-1]}°, random dirs, 0–30°)")

    # KL plateau widths
    plateau_widths_kl = {}
    for thresh in KL_THRESHOLDS:
        pw_zh = np.zeros(n_steer)
        pw_rand = np.zeros(n_steer)

        print(f"\nPlateau widths (KL threshold = {thresh}):")
        for s_idx, steer_deg in enumerate(STEER_ANGLES_DEG):
            kl = all_kl[s_idx]  # [n_prompts, n_pert, n_dirs]
            median_zh = np.median(kl[:, :, 0], axis=0)
            pw_zh[s_idx] = compute_plateau_width(pert_angles_deg, median_zh, thresh)
            median_rand = np.median(kl[:, :, 1:], axis=(0, 2))
            pw_rand[s_idx] = compute_plateau_width(pert_angles_deg, median_rand, thresh)
            print(f"  {steer_deg:3.0f}°: zh={pw_zh[s_idx]:.1f}°, random={pw_rand[s_idx]:.1f}°")

        plateau_widths_kl[thresh] = {'zh': pw_zh, 'random': pw_rand}

    # L2 plateau widths (fraction of linear baseline)
    baseline_l2 = slope_offmanifold * pert_angles_deg  # expected L2 at each angle
    plateau_widths_l2 = {}
    for ratio_thresh in L2_RATIO_THRESHOLDS:
        pw_zh = np.zeros(n_steer)
        pw_rand = np.zeros(n_steer)

        print(f"\nPlateau widths (L2 ratio threshold = {ratio_thresh}):")
        for s_idx, steer_deg in enumerate(STEER_ANGLES_DEG):
            l2 = all_l2[s_idx]  # [n_prompts, n_pert, n_dirs]

            # Compute ratio: actual_L2 / baseline_L2 for each angle
            median_zh = np.median(l2[:, :, 0], axis=0)
            ratio_zh = median_zh / baseline_l2
            pw_zh[s_idx] = compute_plateau_width(pert_angles_deg, ratio_zh, ratio_thresh)

            median_rand = np.median(l2[:, :, 1:], axis=(0, 2))
            ratio_rand = median_rand / baseline_l2
            pw_rand[s_idx] = compute_plateau_width(pert_angles_deg, ratio_rand, ratio_thresh)

            print(f"  {steer_deg:3.0f}°: zh={pw_zh[s_idx]:.1f}°, random={pw_rand[s_idx]:.1f}°")

        plateau_widths_l2[ratio_thresh] = {'zh': pw_zh, 'random': pw_rand}

    # Cosine distance plateau widths
    plateau_widths_cd = {}
    for thresh in COSDIST_THRESHOLDS:
        pw_zh = np.zeros(n_steer)
        pw_rand = np.zeros(n_steer)

        print(f"\nPlateau widths (cosine distance threshold = {thresh}):")
        for s_idx, steer_deg in enumerate(STEER_ANGLES_DEG):
            cd = all_cd[s_idx]  # [n_prompts, n_pert, n_dirs]
            median_zh = np.median(cd[:, :, 0], axis=0)
            pw_zh[s_idx] = compute_plateau_width(pert_angles_deg, median_zh, thresh)
            median_rand = np.median(cd[:, :, 1:], axis=(0, 2))
            pw_rand[s_idx] = compute_plateau_width(pert_angles_deg, median_rand, thresh)
            print(f"  {steer_deg:3.0f}°: zh={pw_zh[s_idx]:.1f}°, random={pw_rand[s_idx]:.1f}°")

        plateau_widths_cd[thresh] = {'zh': pw_zh, 'random': pw_rand}

    # Save results
    results = {
        'config': {
            'model': MODEL_NAME,
            'layer': LAYER,
            'measure_layer': measure_layer,
            'steer_angles_deg': STEER_ANGLES_DEG,
            'pert_angles_deg': pert_angles_deg,
            'n_random_dirs': N_RANDOM_DIRS,
            'kl_thresholds': KL_THRESHOLDS,
            'l2_ratio_thresholds': L2_RATIO_THRESHOLDS,
            'cosdist_thresholds': COSDIST_THRESHOLDS,
            'slope_offmanifold': slope_offmanifold,
            'seed': SEED,
            'n_prompts': len(PROMPTS),
        },
        'kl_divs': all_kl,
        'l2_dists': all_l2,
        'cos_dists': all_cd,
        'plateau_widths_kl': plateau_widths_kl,
        'plateau_widths_l2': plateau_widths_l2,
        'plateau_widths_cd': plateau_widths_cd,
        'prompts': PROMPTS,
    }
    pkl_path = os.path.join(results_dir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # Plot 1a: KL plateau width vs steering magnitude
    n_thresh = len(KL_THRESHOLDS)
    fig, axes = plt.subplots(1, n_thresh, figsize=(6 * n_thresh, 5))
    if n_thresh == 1:
        axes = [axes]
    for ax, thresh in zip(axes, KL_THRESHOLDS):
        pw = plateau_widths_kl[thresh]
        ax.plot(STEER_ANGLES_DEG, pw['zh'], 's-', color='red', linewidth=2,
                markersize=5, label='zh direction')
        ax.plot(STEER_ANGLES_DEG, pw['random'], '^-', color='blue', linewidth=2,
                markersize=5, label='Random directions')
        ax.set_xlabel('Steering angle toward zh (degrees)')
        ax.set_ylabel('Plateau width (degrees)')
        ax.set_title(f'KL threshold = {thresh}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, STEER_ANGLES_DEG[-1] + 1)
        ax.set_ylim(bottom=0)

    fig.suptitle(f'KL plateau width vs steering — {MODEL_NAME}, layer {LAYER}', fontsize=13)
    plot_path = os.path.join(results_dir, 'plateau_width_vs_steering_kl.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Plot 1b: L2 plateau width vs steering magnitude
    n_l2_thresh = len(L2_RATIO_THRESHOLDS)
    fig, axes = plt.subplots(1, n_l2_thresh, figsize=(6 * n_l2_thresh, 5))
    if n_l2_thresh == 1:
        axes = [axes]
    for ax, ratio_thresh in zip(axes, L2_RATIO_THRESHOLDS):
        pw = plateau_widths_l2[ratio_thresh]
        ax.plot(STEER_ANGLES_DEG, pw['zh'], 's-', color='red', linewidth=2,
                markersize=5, label='zh direction')
        ax.plot(STEER_ANGLES_DEG, pw['random'], '^-', color='blue', linewidth=2,
                markersize=5, label='Random directions')
        ax.set_xlabel('Steering angle toward zh (degrees)')
        ax.set_ylabel('Plateau width (degrees)')
        ax.set_title(f'L2/baseline ratio = {ratio_thresh}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, STEER_ANGLES_DEG[-1] + 1)
        ax.set_ylim(bottom=0)

    fig.suptitle(f'L2 plateau width vs steering — {MODEL_NAME}, layer {LAYER}\n'
                 f'(baseline: slope={slope_offmanifold:.1f} L2/deg from steer {STEER_ANGLES_DEG[-1]}°)',
                 fontsize=13)
    plot_path = os.path.join(results_dir, 'plateau_width_vs_steering_l2.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Plot 2: KL curves at selected steering magnitudes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    steer_indices = [0, len(STEER_ANGLES_DEG) // 2, len(STEER_ANGLES_DEG) - 1]

    for ax, s_idx in zip(axes, steer_indices):
        steer_deg = STEER_ANGLES_DEG[s_idx]
        kl = all_kl[s_idx]

        # zh direction
        median_zh = np.median(kl[:, :, 0], axis=0)
        q25_zh = np.percentile(kl[:, :, 0], 25, axis=0)
        q75_zh = np.percentile(kl[:, :, 0], 75, axis=0)
        ax.plot(pert_angles_deg, median_zh, 'r-', linewidth=2, label='zh')
        ax.fill_between(pert_angles_deg, q25_zh, q75_zh, color='red', alpha=0.15)

        # Random directions — flatten prompts × dirs, keeping angle axis
        kl_rand = kl[:, :, 1:].transpose(0, 2, 1).reshape(-1, n_pert)
        median_rand = np.median(kl_rand, axis=0)
        q25_rand = np.percentile(kl_rand, 25, axis=0)
        q75_rand = np.percentile(kl_rand, 75, axis=0)
        ax.plot(pert_angles_deg, median_rand, 'b-', linewidth=2, label='Random')
        ax.fill_between(pert_angles_deg, q25_rand, q75_rand, color='blue', alpha=0.15)

        for thresh in KL_THRESHOLDS:
            ax.axhline(thresh, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Perturbation angle (deg)')
        ax.set_ylabel('KL divergence')
        ax.set_title(f'Steer {steer_deg:.0f}°')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, PERT_MAX_DEG)

    fig.suptitle(f'KL curves at different steering magnitudes — {MODEL_NAME}, layer {LAYER}', fontsize=12)
    plt.tight_layout()
    curves_path = os.path.join(results_dir, 'kl_curves_by_steering.png')
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {curves_path}")

    # Plot 3: L2 curves at selected steering magnitudes, with off-manifold linear baseline
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, s_idx in zip(axes, steer_indices):
        steer_deg = STEER_ANGLES_DEG[s_idx]
        l2 = all_l2[s_idx]

        # zh direction
        median_zh = np.median(l2[:, :, 0], axis=0)
        q25_zh = np.percentile(l2[:, :, 0], 25, axis=0)
        q75_zh = np.percentile(l2[:, :, 0], 75, axis=0)
        ax.plot(pert_angles_deg, median_zh, 'r-', linewidth=2, label='zh')
        ax.fill_between(pert_angles_deg, q25_zh, q75_zh, color='red', alpha=0.15)

        # Random directions
        l2_rand = l2[:, :, 1:].transpose(0, 2, 1).reshape(-1, n_pert)
        median_rand = np.median(l2_rand, axis=0)
        q25_rand = np.percentile(l2_rand, 25, axis=0)
        q75_rand = np.percentile(l2_rand, 75, axis=0)
        ax.plot(pert_angles_deg, median_rand, 'b-', linewidth=2, label='Random')
        ax.fill_between(pert_angles_deg, q25_rand, q75_rand, color='blue', alpha=0.15)

        # Off-manifold linear baseline
        ax.plot(pert_angles_deg, slope_offmanifold * pert_angles_deg,
                'k--', linewidth=1.5, alpha=0.5,
                label=f'Linear baseline (steer {STEER_ANGLES_DEG[-1]}°)')

        ax.set_xlabel('Perturbation angle (deg)')
        ax.set_ylabel('L2 distance (final layer)')
        ax.set_title(f'Steer {steer_deg:.0f}°')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, PERT_MAX_DEG)
        ax.set_ylim(bottom=0)

    fig.suptitle(f'L2 distance at different steering magnitudes — {MODEL_NAME}, layer {LAYER}', fontsize=12)
    plt.tight_layout()
    l2_path = os.path.join(results_dir, 'l2_curves_by_steering.png')
    plt.savefig(l2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {l2_path}")

    # Plot 4: Cosine distance plateau width vs steering magnitude
    n_cd_thresh = len(COSDIST_THRESHOLDS)
    fig, axes = plt.subplots(1, n_cd_thresh, figsize=(6 * n_cd_thresh, 5))
    if n_cd_thresh == 1:
        axes = [axes]
    for ax, thresh in zip(axes, COSDIST_THRESHOLDS):
        pw = plateau_widths_cd[thresh]
        ax.plot(STEER_ANGLES_DEG, pw['zh'], 's-', color='red', linewidth=2,
                markersize=5, label='zh direction')
        ax.plot(STEER_ANGLES_DEG, pw['random'], '^-', color='blue', linewidth=2,
                markersize=5, label='Random directions')
        ax.set_xlabel('Steering angle toward zh (degrees)')
        ax.set_ylabel('Plateau width (degrees)')
        ax.set_title(f'Cosine dist threshold = {thresh}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, STEER_ANGLES_DEG[-1] + 1)
        ax.set_ylim(bottom=0)

    fig.suptitle(f'Cosine distance plateau width vs steering — {MODEL_NAME}, '
                 f'layer {LAYER}, measured at layer {measure_layer}', fontsize=13)
    plot_path = os.path.join(results_dir, 'plateau_width_vs_steering_cosdist.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Plot 5: Cosine distance curves at selected steering magnitudes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, s_idx in zip(axes, steer_indices):
        steer_deg = STEER_ANGLES_DEG[s_idx]
        cd = all_cd[s_idx]

        # zh direction
        median_zh = np.median(cd[:, :, 0], axis=0)
        q25_zh = np.percentile(cd[:, :, 0], 25, axis=0)
        q75_zh = np.percentile(cd[:, :, 0], 75, axis=0)
        ax.plot(pert_angles_deg, median_zh, 'r-', linewidth=2, label='zh')
        ax.fill_between(pert_angles_deg, q25_zh, q75_zh, color='red', alpha=0.15)

        # Random directions
        cd_rand = cd[:, :, 1:].transpose(0, 2, 1).reshape(-1, n_pert)
        median_rand = np.median(cd_rand, axis=0)
        q25_rand = np.percentile(cd_rand, 25, axis=0)
        q75_rand = np.percentile(cd_rand, 75, axis=0)
        ax.plot(pert_angles_deg, median_rand, 'b-', linewidth=2, label='Random')
        ax.fill_between(pert_angles_deg, q25_rand, q75_rand, color='blue', alpha=0.15)

        for thresh in COSDIST_THRESHOLDS:
            ax.axhline(thresh, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Perturbation angle (deg)')
        ax.set_ylabel(f'Cosine distance (layer {measure_layer})')
        ax.set_title(f'Steer {steer_deg:.0f}°')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, PERT_MAX_DEG)

    fig.suptitle(f'Cosine distance at different steering magnitudes — {MODEL_NAME}, '
                 f'layer {LAYER}, measured at layer {measure_layer}', fontsize=12)
    plt.tight_layout()
    cd_path = os.path.join(results_dir, 'cosdist_curves_by_steering.png')
    plt.savefig(cd_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cd_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
