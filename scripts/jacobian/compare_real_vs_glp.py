#!/usr/bin/env python3
"""Compare properties of real vs GLP-projected activations.

Hypothesis: GLP-projected activations (at u=0.5) lack some property of real
activations that gives rise to plateaus. Start by comparing Jacobian norms.

Usage:
    python scripts/jacobian/compare_real_vs_glp.py \
        --model meta-llama/Llama-3.1-8B --layer 6 \
        --glp_model generative-latent-prior/glp-llama8b-d6 \
        --n_act 50 --n_probes 50
"""

import os, sys, argparse, pickle, time
import torch
import torch.nn as nn
# GLP compat: older code expects nn.Buffer
if not hasattr(nn, 'Buffer'):
    nn.Buffer = lambda x: nn.Parameter(x, requires_grad=False)
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.model import load_model, forward_from_layer_to_final, _get_blocks
from data.bilingual_prompts import ENGLISH_PROMPTS


def load_glp_model(glp_model_name, device, force_cpu=False):
    """Load GLP model."""
    from glp.denoiser import load_glp
    glp_device = 'cpu' if force_cpu else device
    return load_glp(glp_model_name, device=glp_device, checkpoint="final"), glp_device


def project_on_manifold(glp_model, activations, u=0.5, num_timesteps=50, layer_idx=None):
    """Project activations onto the learned manifold using GLP."""
    from glp import flow_matching

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


def compute_jacobian_norm(model, context, point, layer, n_probes=50):
    """Estimate ||J||_F via Hutchinson VJP estimator.

    J maps layer L last-token → final layer last-token.
    """
    d_model = point.shape[-1]
    estimates = []

    for k in range(n_probes):
        point_var = point.clone().reshape(1, 1, d_model).requires_grad_(True)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            final_act = forward_from_layer_to_final(model, context, point_var, layer)

        u = torch.randn_like(final_act)
        s = (final_act * u).sum()
        g = torch.autograd.grad(s, point_var)[0]
        estimates.append((g ** 2).sum().item())

    estimates = np.array(estimates)
    frob_norm = np.sqrt(max(np.mean(estimates), 0))
    return frob_norm


def extract_activations(model, tokenizer, prompts, layer, device):
    """Extract last-token activations and contexts at a given layer."""
    model_dtype = next(model.parameters()).dtype
    contexts = []
    activations = []

    for p in prompts:
        tokens = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        contexts.append(hidden[:, :-1, :].float().cpu())
        activations.append(hidden[:, -1, :].squeeze(0).float().cpu())

    return contexts, torch.stack(activations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--glp_model', default='generative-latent-prior/glp-llama8b-d6')
    parser.add_argument('--glp_cpu', action='store_true',
                        help='Load GLP on CPU (saves GPU memory)')
    parser.add_argument('--u', type=float, default=0.5)
    parser.add_argument('--n_act', type=int, default=50)
    parser.add_argument('--n_probes', type=int, default=50)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--results_dir', default='results/real_vs_glp')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load LLM
    load_dtype = torch.bfloat16 if args.dtype == 'bfloat16' else None
    print(f"Loading {args.model} (dtype={args.dtype})...")
    model, tokenizer, device = load_model(args.model, device=device, dtype=load_dtype)
    model_dtype = next(model.parameters()).dtype
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    print(f"d_model={d_model}, n_layers={n_layers}")

    # Extract real activations
    prompts = ENGLISH_PROMPTS[:args.n_act]
    print(f"\nExtracting {len(prompts)} activations at layer {args.layer}...")
    contexts, real_acts = extract_activations(model, tokenizer, prompts, args.layer, device)
    print(f"Real activation norms: mean={real_acts.norm(dim=-1).mean():.2f}, "
          f"std={real_acts.norm(dim=-1).std():.2f}")

    # Load GLP and project
    print(f"\nLoading GLP: {args.glp_model}...")
    glp, glp_device = load_glp_model(args.glp_model, device, force_cpu=args.glp_cpu)

    print(f"Projecting with u={args.u}...")
    glp_acts = project_on_manifold(
        glp, real_acts.to(glp_device), u=args.u,
    ).cpu()
    print(f"GLP activation norms: mean={glp_acts.norm(dim=-1).mean():.2f}, "
          f"std={glp_acts.norm(dim=-1).std():.2f}")

    # Free GLP memory
    del glp
    torch.cuda.empty_cache()

    # Compute Jacobian norms
    n_act = len(real_acts)
    real_jac_norms = np.zeros(n_act)
    glp_jac_norms = np.zeros(n_act)

    print(f"\nComputing Jacobian norms ({args.n_probes} Hutchinson probes each)...")
    for i in range(n_act):
        ctx = contexts[i].to(device=device, dtype=model_dtype)

        # Real
        t0 = time.time()
        real_jac_norms[i] = compute_jacobian_norm(
            model, ctx, real_acts[i].to(device=device, dtype=model_dtype),
            args.layer, n_probes=args.n_probes)

        # GLP-projected
        glp_jac_norms[i] = compute_jacobian_norm(
            model, ctx, glp_acts[i].to(device=device, dtype=model_dtype),
            args.layer, n_probes=args.n_probes)

        elapsed = time.time() - t0
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{n_act}] real={real_jac_norms[i]:.1f}, "
                  f"glp={glp_jac_norms[i]:.1f} ({elapsed:.1f}s)")

    # Summary stats
    print(f"\n--- Jacobian Frobenius norm ---")
    print(f"Real:  mean={real_jac_norms.mean():.1f}, std={real_jac_norms.std():.1f}, "
          f"median={np.median(real_jac_norms):.1f}")
    print(f"GLP:   mean={glp_jac_norms.mean():.1f}, std={glp_jac_norms.std():.1f}, "
          f"median={np.median(glp_jac_norms):.1f}")

    # Also compute some basic stats
    real_norms = real_acts.norm(dim=-1).numpy()
    glp_norms = glp_acts.norm(dim=-1).numpy()
    cosine_sim = F.cosine_similarity(real_acts, glp_acts, dim=-1).numpy()

    print(f"\n--- Activation norms ---")
    print(f"Real:  mean={real_norms.mean():.2f}, std={real_norms.std():.2f}")
    print(f"GLP:   mean={glp_norms.mean():.2f}, std={glp_norms.std():.2f}")
    print(f"Cosine(real, glp): mean={cosine_sim.mean():.4f}, std={cosine_sim.std():.4f}")

    # Save data
    save_data = {
        'args': vars(args),
        'real_jac_norms': real_jac_norms,
        'glp_jac_norms': glp_jac_norms,
        'real_norms': real_norms,
        'glp_norms': glp_norms,
        'cosine_sim': cosine_sim,
    }
    pkl_path = os.path.join(args.results_dir, 'real_vs_glp.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved: {pkl_path}")

    # Plot: histogram of Jacobian norms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bins = np.linspace(
        min(real_jac_norms.min(), glp_jac_norms.min()),
        max(real_jac_norms.max(), glp_jac_norms.max()), 25)
    ax.hist(real_jac_norms, bins=bins, alpha=0.6, label='Real', color='C0')
    ax.hist(glp_jac_norms, bins=bins, alpha=0.6, label=f'GLP (u={args.u})', color='C1')
    ax.set_xlabel('Jacobian Frobenius norm', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title('Jacobian norm: Real vs GLP', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(real_jac_norms, glp_jac_norms, alpha=0.6, s=30)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    ax.set_xlabel('Real Jacobian norm', fontsize=13)
    ax.set_ylabel('GLP Jacobian norm', fontsize=13)
    ax.set_title('Per-activation comparison', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'jacobian_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
