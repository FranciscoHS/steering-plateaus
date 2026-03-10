#!/usr/bin/env python3
"""Measure plateau preservation under GLP on-manifold projection.

For each real activation:
1. Steer by various angles in a random direction (takes it off-manifold)
2. Measure plateau along random directions (baseline)
3. Project the steered activation back on-manifold with GLP
4. Measure plateau again (hypothesis: plateau is preserved)

Tests whether GLP projection restores the plateau that steering destroys.

Usage:
    python scripts/perturbation/sweep_glp_plateau.py [--model ...] [--glp_model ...]
"""

import os, sys, pickle, time, argparse
import torch
import torch.nn as nn
# Patch for torch < 2.5 (nn.Buffer added in 2.5)
# nn.Buffer registers tensors as buffers that move with .to(device)
# We emulate this by wrapping as nn.Parameter with requires_grad=False
if not hasattr(nn, 'Buffer'):
    nn.Buffer = lambda x: nn.Parameter(x, requires_grad=False)
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.model import load_model, forward_from_layer_to_layer
from data.bilingual_prompts_spanish import PROMPT_PAIRS, ENGLISH_PROMPTS, SPANISH_PROMPTS

# Config
SEED = 42
N_SWEEP = 20        # english activations to steer and sweep
N_DIRS = 10         # random perturbation directions per condition
MAX_ANGLE = 60.0    # max perturbation angle
N_ANGLES = 120      # perturbation angle grid
STEER_ANGLES = [0, 5, 10, 15, 20, 30, 45, 60]  # steering magnitudes (degrees)
GLP_U = 0.5         # noise level for GLP projection
GLP_TIMESTEPS = 50  # diffusion steps for GLP
RESULTS_DIR = 'results/llama_plateau'


def spherical_rotate(activation, direction, angle_rad):
    """Rotate activation toward direction by angle_rad on the sphere."""
    R = activation.norm()
    a_unit = activation / R
    # Orthogonalize direction to activation
    v_perp = direction - (direction @ a_unit) * a_unit
    v_perp = v_perp / v_perp.norm()
    return R * (torch.cos(angle_rad) * a_unit + torch.sin(angle_rad) * v_perp)


def load_glp_model(glp_model_name, device):
    """Load GLP denoiser model."""
    from glp.denoiser import load_glp
    glp = load_glp(glp_model_name, device=device, checkpoint="final")
    return glp


def project_on_manifold(glp_model, activations, u=0.5, num_timesteps=50, layer_idx=None):
    """Project activations onto the learned manifold using GLP.

    Args:
        glp_model: loaded GLP model
        activations: [batch, d_model] tensor
        u: noise level (0=minimal change, 1=full regeneration)
        num_timesteps: diffusion steps
        layer_idx: layer index for multi-layer GLPs

    Returns:
        projected: [batch, d_model] tensor on the manifold
    """
    from glp import flow_matching

    device = activations.device
    orig_dtype = activations.dtype

    # GLP expects [batch, 1, d_model] in float32
    latents = activations.float().unsqueeze(1)

    # Normalize
    latents = glp_model.normalizer.normalize(latents, layer_idx=layer_idx)

    # Add noise at level u
    noise = torch.randn_like(latents)
    noisy_latents, _, timesteps, _ = flow_matching.fm_prepare(
        glp_model.scheduler,
        latents,
        noise,
        u=torch.ones(latents.shape[0]) * u,  # must be on CPU (scheduler.timesteps is CPU)
    )

    # Denoise back to manifold
    projected = flow_matching.sample_on_manifold(
        glp_model,
        noisy_latents,
        start_timestep=timesteps[0].item(),
        num_timesteps=num_timesteps,
        layer_idx=layer_idx,
    )

    # Denormalize and reshape
    projected = glp_model.normalizer.denormalize(projected, layer_idx=layer_idx)
    projected = projected.squeeze(1).to(dtype=orig_dtype)

    return projected


def measure_plateau(model, ctx, act, inject_layer, measure_layer, cos_t, sin_t, n_dirs, d_model, model_dtype, device):
    """Measure cosine distance curves for perturbations around act.

    Returns: [n_dirs, n_angles] array of cosine distances.
    """
    act = act.to(device=device, dtype=model_dtype)
    ctx = ctx.to(device=device, dtype=model_dtype)
    R = act.norm()
    a_unit = act / R
    n_angles = len(cos_t)
    curves = np.zeros((n_dirs, n_angles))

    with torch.no_grad():
        ref = forward_from_layer_to_layer(model, ctx, act[None, None], inject_layer, measure_layer).float()

    for j in range(n_dirs):
        d = torch.randn(d_model, device=device, dtype=model_dtype)
        d = d - (d @ a_unit) * a_unit
        d = F.normalize(d, dim=0)
        perturbed = R * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * d[None])
        with torch.no_grad():
            measured = forward_from_layer_to_layer(
                model, ctx, perturbed.unsqueeze(1), inject_layer, measure_layer
            ).float()
            cos_dist = 1 - F.cosine_similarity(ref, measured, dim=-1)
            curves[j] = cos_dist.cpu().numpy()

    return curves


def compute_plateau_width(angles_deg, medians, threshold):
    """Find angle where median response crosses threshold (linear interpolation)."""
    above = np.where(medians > threshold)[0]
    if len(above) == 0:
        return angles_deg[-1]
    if above[0] == 0:
        return 0.0
    x0, x1 = angles_deg[above[0]-1], angles_deg[above[0]]
    y0, y1 = medians[above[0]-1], medians[above[0]]
    return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inject_layer', type=int, default=7)
    parser.add_argument('--measure_layer', type=int, default=14)
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--glp_model', type=str, default='generative-latent-prior/glp-llama1b-d6')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'bfloat16'])
    parser.add_argument('--glp_u', type=float, default=GLP_U)
    parser.add_argument('--tag', type=str, default='glp_plateau')
    args = parser.parse_args()

    INJECT_LAYER = args.inject_layer
    MEASURE_LAYER = args.measure_layer

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load LLM
    load_dtype = torch.bfloat16 if args.dtype == 'bfloat16' else None
    print(f"Loading {args.model} (dtype={args.dtype})...")
    model, tokenizer, device = load_model(args.model, device=device, dtype=load_dtype)
    model_dtype = next(model.parameters()).dtype
    d_model = model._plateau_info['d_model']
    n_layers = len(model.model.layers)
    print(f"d_model={d_model}, n_layers={n_layers}")

    # Load GLP
    print(f"Loading GLP: {args.glp_model}...")
    glp = load_glp_model(args.glp_model, device)
    print("GLP loaded.")

    # Extract activations for both languages
    def extract_acts(texts, label):
        ctxs, acts = [], []
        for text in texts:
            tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[INJECT_LAYER + 1]
            ctxs.append(hidden[:, :-1, :].cpu())
            acts.append(hidden[:, -1, :].squeeze(0).cpu())
        print(f"  {label}: {len(acts)} activations")
        return ctxs, acts

    print(f"\nExtracting layer-{INJECT_LAYER} activations...")
    en_ctxs, en_acts = extract_acts(ENGLISH_PROMPTS, "English")
    es_ctxs, es_acts = extract_acts(SPANISH_PROMPTS, "Spanish")

    # Compute diff-of-means steering vector: Spanish - English
    en_mean = torch.stack(en_acts).mean(dim=0)
    es_mean = torch.stack(es_acts).mean(dim=0)
    steer_vec = es_mean - en_mean  # direction: English → Spanish
    steer_vec_unit = steer_vec / steer_vec.norm()
    print(f"Steering vector norm: {steer_vec.norm():.3f}")

    N_SWEEP_ACTUAL = min(N_SWEEP, len(en_acts))
    print(f"Will sweep {N_SWEEP_ACTUAL} English activations")

    # Angle grid for perturbation measurement
    angles_deg = np.linspace(0, MAX_ANGLE, N_ANGLES + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)
    cos_t = torch.tensor(np.cos(angles_rad), dtype=model_dtype, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=model_dtype, device=device)

    # Main experiment loop
    # Results: [n_steer_angles, n_activations, n_dirs, n_angles] for each condition
    n_steer = len(STEER_ANGLES)
    baseline_curves = np.zeros((n_steer, N_SWEEP_ACTUAL, N_DIRS, N_ANGLES))
    glp_curves = np.zeros((n_steer, N_SWEEP_ACTUAL, N_DIRS, N_ANGLES))

    t0 = time.time()

    for i in range(N_SWEEP_ACTUAL):
        ctx = en_ctxs[i]
        act = en_acts[i]

        for si, steer_deg in enumerate(STEER_ANGLES):
            steer_rad = torch.tensor(np.deg2rad(steer_deg))

            if steer_deg == 0:
                steered = act.clone()
            else:
                steered = spherical_rotate(act, steer_vec_unit, steer_rad)

            # Baseline: measure plateau from steered activation
            baseline_curves[si, i] = measure_plateau(
                model, ctx, steered, INJECT_LAYER, MEASURE_LAYER,
                cos_t, sin_t, N_DIRS, d_model, model_dtype, device
            )

            # GLP projection: project steered activation back on-manifold
            with torch.no_grad():
                projected = project_on_manifold(
                    glp, steered.to(device).unsqueeze(0), u=args.glp_u,
                    num_timesteps=GLP_TIMESTEPS
                ).squeeze(0).cpu()

            # Measure plateau from projected activation
            glp_curves[si, i] = measure_plateau(
                model, ctx, projected, INJECT_LAYER, MEASURE_LAYER,
                cos_t, sin_t, N_DIRS, d_model, model_dtype, device
            )

        elapsed = time.time() - t0
        print(f"  {i+1}/{N_SWEEP_ACTUAL}  ({elapsed:.0f}s elapsed)")

    total_time = time.time() - t0
    print(f"Done in {total_time:.1f}s")

    # Save
    results = {
        'config': {
            'model': args.model,
            'glp_model': args.glp_model,
            'inject_layer': INJECT_LAYER,
            'measure_layer': MEASURE_LAYER,
            'n_sweep': N_SWEEP_ACTUAL,
            'n_dirs': N_DIRS,
            'max_angle': MAX_ANGLE,
            'n_angles': N_ANGLES,
            'steer_angles': STEER_ANGLES,
            'glp_u': args.glp_u,
            'glp_timesteps': GLP_TIMESTEPS,
            'seed': SEED,
        },
        'angles_deg': angles_deg,
        'steer_angles_deg': np.array(STEER_ANGLES),
        'baseline_curves': baseline_curves,  # [n_steer, n_act, n_dirs, n_angles]
        'glp_curves': glp_curves,
    }
    pkl_path = os.path.join(RESULTS_DIR, f'{args.tag}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # Plot 1: plateau width vs steering angle
    threshold = 0.05
    baseline_widths = []
    glp_widths = []
    for si in range(n_steer):
        # Per-activation median across directions, then plateau width per activation
        bw = []
        gw = []
        for i in range(N_SWEEP_ACTUAL):
            b_med = np.median(baseline_curves[si, i], axis=0)
            g_med = np.median(glp_curves[si, i], axis=0)
            bw.append(compute_plateau_width(angles_deg, b_med, threshold))
            gw.append(compute_plateau_width(angles_deg, g_med, threshold))
        baseline_widths.append(bw)
        glp_widths.append(gw)

    baseline_widths = np.array(baseline_widths)  # [n_steer, n_act]
    glp_widths = np.array(glp_widths)

    fig, ax = plt.subplots(figsize=(10, 6))
    steer_arr = np.array(STEER_ANGLES)

    # Baseline
    ax.fill_between(steer_arr,
                    np.percentile(baseline_widths, 25, axis=1),
                    np.percentile(baseline_widths, 75, axis=1),
                    color='C0', alpha=0.15)
    ax.plot(steer_arr, np.median(baseline_widths, axis=1),
            'C0-o', lw=2, label='Without GLP')

    # GLP
    ax.fill_between(steer_arr,
                    np.percentile(glp_widths, 25, axis=1),
                    np.percentile(glp_widths, 75, axis=1),
                    color='C1', alpha=0.15)
    ax.plot(steer_arr, np.median(glp_widths, axis=1),
            'C1-o', lw=2, label='With GLP projection')

    ax.set_xlabel('Steering angle (degrees)')
    ax.set_ylabel(f'Plateau width (degrees, threshold={threshold})')
    ax.set_title(f'Plateau preservation under GLP projection\n{args.model.split("/")[-1]}, inject@L{INJECT_LAYER} measure@L{MEASURE_LAYER}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f'{args.tag}_widths.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Plot 2: full curves at a few steering angles
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    show_steer = [0, 10, 20, 30, 45, 60]
    for ax, sd in zip(axes.flat, show_steer):
        si = STEER_ANGLES.index(sd)
        # Baseline
        b_per_act = np.median(baseline_curves[si], axis=1)
        ax.fill_between(angles_deg,
                        np.percentile(b_per_act, 25, axis=0),
                        np.percentile(b_per_act, 75, axis=0),
                        color='C0', alpha=0.15)
        ax.plot(angles_deg, np.median(b_per_act, axis=0), 'C0-', lw=2, label='No GLP')
        # GLP
        g_per_act = np.median(glp_curves[si], axis=1)
        ax.fill_between(angles_deg,
                        np.percentile(g_per_act, 25, axis=0),
                        np.percentile(g_per_act, 75, axis=0),
                        color='C1', alpha=0.15)
        ax.plot(angles_deg, np.median(g_per_act, axis=0), 'C1-', lw=2, label='With GLP')
        ax.set_title(f'Steer {sd}°')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, MAX_ANGLE)
        if ax in axes[:, 0]:
            ax.set_ylabel('Cosine distance')
        if ax in axes[-1]:
            ax.set_xlabel('Perturbation angle (°)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'Plateau curves: baseline vs GLP projection\n{args.model.split("/")[-1]}', fontsize=13)
    plt.tight_layout()
    plot_path2 = os.path.join(RESULTS_DIR, f'{args.tag}_curves.png')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path2}")

    # Numerical summary
    print(f"\nPlateau widths (threshold={threshold}):")
    for si, sd in enumerate(STEER_ANGLES):
        bm = np.median(baseline_widths[si])
        gm = np.median(glp_widths[si])
        print(f"  Steer {sd:3d}°: baseline={bm:.1f}°, GLP={gm:.1f}° (diff={gm-bm:+.1f}°)")


if __name__ == '__main__':
    main()
