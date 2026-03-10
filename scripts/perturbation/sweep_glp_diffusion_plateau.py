#!/usr/bin/env python3
"""Iterative GLP diffusion: does denoising build up the plateau?

Take a random point in activation space, then iteratively apply GLP
with progressively lower noise levels. At each step, measure the plateau.
If plateaus arise from being on the learned manifold, we should see the
plateau grow longer with each denoising step.

Protocol:
1. Sample random activations (Gaussian matched to real stats)
2. For a schedule of decreasing u values (e.g. [0.9, 0.7, 0.5, 0.3, 0.1]):
   - Apply GLP with that u level to the *current* activation
   - Measure plateau width
3. Compare against real activation plateaus as reference

Usage:
    python scripts/perturbation/sweep_glp_diffusion_plateau.py [--model ...] [--glp_model ...]
"""

import os, sys, pickle, time, argparse
import torch
import torch.nn as nn
if not hasattr(nn, 'Buffer'):
    nn.Buffer = lambda x: nn.Parameter(x, requires_grad=False)
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.model import load_model, forward_from_layer_to_layer
from data.bilingual_prompts import ENGLISH_PROMPTS

# Config
SEED = 42
N_REAL = 10         # real activations (for reference plateau + stats)
N_RANDOM = 10       # random starting points to denoise
N_DIRS = 10         # perturbation directions per plateau measurement
MAX_ANGLE = 60.0
N_ANGLES = 60
U_SCHEDULE = [0.9, 0.7, 0.5, 0.3, 0.1]  # decreasing noise for iterative denoising
GLP_TIMESTEPS = 50
RESULTS_DIR = 'results/llama_plateau'


def load_glp_model(glp_model_name, device, force_cpu=False):
    """Load GLP model. Use force_cpu=True when GPU memory is tight."""
    from glp.denoiser import load_glp
    glp_device = 'cpu' if force_cpu else device
    return load_glp(glp_model_name, device=glp_device, checkpoint="final"), glp_device


def project_on_manifold(glp_model, activations, u=0.5, num_timesteps=50, layer_idx=None):
    """Project activations onto the learned manifold using GLP."""
    from glp import flow_matching

    device = activations.device
    orig_dtype = activations.dtype

    latents = activations.float().unsqueeze(1)
    latents = glp_model.normalizer.normalize(latents, layer_idx=layer_idx)

    noise = torch.randn_like(latents)
    noisy_latents, _, timesteps, _ = flow_matching.fm_prepare(
        glp_model.scheduler, latents, noise,
        u=torch.ones(latents.shape[0]) * u,
    )

    projected = flow_matching.sample_on_manifold(
        glp_model, noisy_latents,
        start_timestep=timesteps[0].item(),
        num_timesteps=num_timesteps,
        layer_idx=layer_idx,
    )

    projected = glp_model.normalizer.denormalize(projected, layer_idx=layer_idx)
    projected = projected.squeeze(1).to(dtype=orig_dtype)
    return projected


def measure_plateau(model, ctx, act, inject_layer, measure_layer,
                    cos_t, sin_t, n_dirs, d_model, model_dtype, device):
    """Measure L2 distance curves for perturbations around act.

    Returns: [n_dirs, n_angles] array of L2 distances.
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
            l2_dist = torch.norm(ref - measured, dim=-1)
            curves[j] = l2_dist.cpu().numpy()

    return curves


def compute_plateau_width(angles_deg, medians, threshold):
    """Find angle where median response crosses threshold."""
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
    parser.add_argument('--inject_layer', type=int, default=6)
    parser.add_argument('--measure_layer', type=int, default=14)
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--glp_model', type=str, default='generative-latent-prior/glp-llama8b-d6')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('--n_random', type=int, default=N_RANDOM)
    parser.add_argument('--n_real', type=int, default=N_REAL)
    parser.add_argument('--u_schedule', type=float, nargs='+', default=U_SCHEDULE)
    parser.add_argument('--glp_cpu', action='store_true', help='Load GLP on CPU (needed when GPU is too small for both LLM and GLP)')
    parser.add_argument('--start_from_real', action='store_true', help='Start from real activations instead of random (sanity check)')
    parser.add_argument('--single_pass', action='store_true', help='Apply each u independently (not iteratively)')
    parser.add_argument('--tag', type=str, default='glp_diffusion')
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
    print(f"d_model={d_model}")

    # Load GLP
    print(f"Loading GLP: {args.glp_model}...")
    glp, glp_device = load_glp_model(args.glp_model, device, force_cpu=args.glp_cpu)
    print(f"GLP loaded on {glp_device}.")

    # Extract real activations (for reference + computing stats)
    print(f"\nExtracting layer-{INJECT_LAYER} activations from {len(ENGLISH_PROMPTS)} prompts...")
    real_ctxs, real_acts = [], []
    for text in ENGLISH_PROMPTS:
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[INJECT_LAYER + 1]
        real_ctxs.append(hidden[:, :-1, :].cpu())
        real_acts.append(hidden[:, -1, :].squeeze(0).cpu())
    print(f"  Got {len(real_acts)} real activations")

    if args.start_from_real:
        # Sanity check: start from real activations
        n_start = min(args.n_random, len(real_acts))
        start_acts = [real_acts[i].clone() for i in range(n_start)]
        start_ctxs = [real_ctxs[i] for i in range(n_start)]
        args.n_random = n_start
        print(f"\nStarting from {n_start} real activations (sanity check mode)")
    else:
        # Compute activation statistics for sampling random points
        act_stack = torch.stack(real_acts)  # [N, d_model]
        act_mean = act_stack.mean(dim=0)
        act_centered = act_stack - act_mean
        act_std = act_centered.std(dim=0)

        print(f"\nSampling {args.n_random} random starting points...")
        start_acts = [act_mean + act_std * torch.randn(d_model) for _ in range(args.n_random)]
        start_ctxs = [real_ctxs[0]] * args.n_random

    # Angle grid
    angles_deg = np.linspace(0, MAX_ANGLE, N_ANGLES + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)
    cos_t = torch.tensor(np.cos(angles_rad), dtype=model_dtype, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=model_dtype, device=device)

    u_schedule = sorted(args.u_schedule, reverse=True)  # high to low
    n_steps = len(u_schedule)

    # ---- Measure real activation plateaus (reference) ----
    n_real_actual = min(args.n_real, len(real_acts))
    print(f"\nMeasuring plateaus for {n_real_actual} real activations (reference)...")
    real_curves = np.zeros((n_real_actual, N_DIRS, N_ANGLES))
    for i in range(n_real_actual):
        real_curves[i] = measure_plateau(
            model, real_ctxs[i], real_acts[i], INJECT_LAYER, MEASURE_LAYER,
            cos_t, sin_t, N_DIRS, d_model, model_dtype, device
        )
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n_real_actual}")

    # ---- Denoising experiment ----
    start_label = "real" if args.start_from_real else "random"
    mode = "single-pass" if args.single_pass else "iterative"
    print(f"\n{mode} denoising: {args.n_random} {start_label} points, u_values={u_schedule}")
    all_curves = np.zeros((n_steps + 1, args.n_random, N_DIRS, N_ANGLES))

    t0 = time.time()
    for i in range(args.n_random):
        start = start_acts[i].clone()
        ctx = start_ctxs[i]

        # Step 0: measure plateau of starting activation
        all_curves[0, i] = measure_plateau(
            model, ctx, start, INJECT_LAYER, MEASURE_LAYER,
            cos_t, sin_t, N_DIRS, d_model, model_dtype, device
        )

        # Denoising steps
        current = start.clone()
        for step, u in enumerate(u_schedule):
            # In single_pass mode, always start from the original activation
            input_act = start if args.single_pass else current
            with torch.no_grad():
                current = project_on_manifold(
                    glp, input_act.to(glp_device).unsqueeze(0), u=u,
                    num_timesteps=GLP_TIMESTEPS
                ).squeeze(0).cpu()

            all_curves[step + 1, i] = measure_plateau(
                model, ctx, current, INJECT_LAYER, MEASURE_LAYER,
                cos_t, sin_t, N_DIRS, d_model, model_dtype, device
            )

        elapsed = time.time() - t0
        print(f"  {i+1}/{args.n_random}  ({elapsed:.0f}s elapsed)")

    total_time = time.time() - t0
    print(f"Done in {total_time:.1f}s")

    # Save results
    results = {
        'config': {
            'model': args.model,
            'glp_model': args.glp_model,
            'inject_layer': INJECT_LAYER,
            'measure_layer': MEASURE_LAYER,
            'n_random': args.n_random,
            'n_real': n_real_actual,
            'n_dirs': N_DIRS,
            'max_angle': MAX_ANGLE,
            'n_angles': N_ANGLES,
            'u_schedule': u_schedule,
            'glp_timesteps': GLP_TIMESTEPS,
            'seed': SEED,
        },
        'angles_deg': angles_deg,
        'u_schedule': u_schedule,
        'all_curves': all_curves,       # [n_steps+1, n_random, n_dirs, n_angles]
        'real_curves': real_curves,      # [n_real, n_dirs, n_angles]
    }
    pkl_path = os.path.join(RESULTS_DIR, f'{args.tag}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # ---- Plot 1: Plateau width vs denoising step ----
    threshold = 0.05
    step_labels = ['Random'] + [f'u={u}' for u in u_schedule]

    # Compute plateau widths per step per random point
    widths = np.zeros((n_steps + 1, args.n_random))
    for s in range(n_steps + 1):
        for i in range(args.n_random):
            med = np.median(all_curves[s, i], axis=0)
            widths[s, i] = compute_plateau_width(angles_deg, med, threshold)

    # Real reference widths
    real_widths = []
    for i in range(n_real_actual):
        med = np.median(real_curves[i], axis=0)
        real_widths.append(compute_plateau_width(angles_deg, med, threshold))
    real_widths = np.array(real_widths)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_steps + 1)

    # Individual traces (light)
    for i in range(args.n_random):
        ax.plot(x, widths[:, i], 'C0-', alpha=0.15, lw=0.8)

    # Median + IQR
    ax.fill_between(x,
                    np.percentile(widths, 25, axis=1),
                    np.percentile(widths, 75, axis=1),
                    color='C0', alpha=0.2)
    ax.plot(x, np.median(widths, axis=1), 'C0-o', lw=2.5, ms=8,
            label='Random → denoised')

    # Real reference
    ax.axhline(np.median(real_widths), color='C2', ls='--', lw=2,
               label=f'Real activations (median)')
    ax.axhspan(np.percentile(real_widths, 25), np.percentile(real_widths, 75),
               color='C2', alpha=0.1)

    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, rotation=30, ha='right')
    ax.set_xlabel('Denoising step')
    ax.set_ylabel(f'Plateau width (degrees, threshold={threshold})')
    ax.set_title(f'Iterative GLP denoising: does the plateau grow?\n'
                 f'{args.model.split("/")[-1]}, L{INJECT_LAYER}→L{MEASURE_LAYER}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f'{args.tag}_widths.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # ---- Plot 2: Full curves at each step (median across random points) ----
    fig, axes = plt.subplots(1, n_steps + 2, figsize=(4 * (n_steps + 2), 4), sharey=True)

    # Real reference
    real_per_act = np.median(real_curves, axis=1)  # [n_real, n_angles]
    ax = axes[0]
    ax.fill_between(angles_deg,
                    np.percentile(real_per_act, 25, axis=0),
                    np.percentile(real_per_act, 75, axis=0),
                    color='C2', alpha=0.2)
    ax.plot(angles_deg, np.median(real_per_act, axis=0), 'C2-', lw=2)
    ax.set_title('Real')
    ax.set_xlabel('Perturbation angle (°)')
    ax.set_ylabel('L2 distance')
    ax.grid(True, alpha=0.3)

    # Each denoising step
    for s in range(n_steps + 1):
        ax = axes[s + 1]
        per_act = np.median(all_curves[s], axis=1)  # [n_random, n_angles]
        ax.fill_between(angles_deg,
                        np.percentile(per_act, 25, axis=0),
                        np.percentile(per_act, 75, axis=0),
                        color='C0', alpha=0.2)
        ax.plot(angles_deg, np.median(per_act, axis=0), 'C0-', lw=2)
        ax.set_title(step_labels[s])
        ax.set_xlabel('Perturbation angle (°)')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Plateau curves through iterative denoising\n'
                 f'{args.model.split("/")[-1]}, L{INJECT_LAYER}→L{MEASURE_LAYER}', fontsize=13)
    plt.tight_layout()
    plot_path2 = os.path.join(RESULTS_DIR, f'{args.tag}_curves.png')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path2}")

    # Numerical summary
    print(f"\nPlateau widths (threshold={threshold}):")
    print(f"  Real activations: median={np.median(real_widths):.1f}°, "
          f"IQR=[{np.percentile(real_widths, 25):.1f}°, {np.percentile(real_widths, 75):.1f}°]")
    for s in range(n_steps + 1):
        med = np.median(widths[s])
        q25, q75 = np.percentile(widths[s], 25), np.percentile(widths[s], 75)
        print(f"  {step_labels[s]:>10s}: median={med:.1f}°, IQR=[{q25:.1f}°, {q75:.1f}°]")


if __name__ == '__main__':
    main()
