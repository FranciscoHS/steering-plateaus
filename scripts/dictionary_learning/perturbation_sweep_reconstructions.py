#!/usr/bin/env python3
"""Perturbation sweep comparing 4 activation types from reconstruction experiments.

Spherically perturbs each activation at various angles, measures L2 distance
at the final layer. Produces a single plot with 4 lines:
  1. MELBO reconstruction
  2. Random-direction reconstruction
  3. Real activation
  4. Random point

Usage:
    python scripts/dictionary_learning/perturbation_sweep_reconstructions.py \
        --melbo_pkl results/reconstruct_min_jacobian/qwen3-1.7b_L2_31dirs/results.pkl \
        --random_pkl results/reconstruct_min_jacobian/qwen3-1.7b_L2_31dirs_random/results.pkl
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
    load_model, forward_from_layer_to_final,
    _get_final_norm, _get_lm_head,
)
from src.data import load_fineweb_fixed_length


def run_spherical_sweep_single(model, ctx, activation, angles_rad, layer, device, n_dirs=10):
    """Spherical perturbation sweep for a single activation.

    Returns:
        l2_dists: [n_angles, n_dirs] — L2 distance at final layer
        kl_divs: [n_angles, n_dirs] — KL divergence on logits
        cos_dists: [n_angles, n_dirs] — cosine distance at final layer
    """
    n_angles = len(angles_rad)
    l2_dists = np.zeros((n_angles, n_dirs))
    kl_divs = np.zeros((n_angles, n_dirs))
    cos_dists = np.zeros((n_angles, n_dirs))

    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a = activation.to(device=device, dtype=model_dtype)
    R = a.norm()
    a_unit = a / R
    d_model = a.shape[-1]

    # Sample random directions (not orthogonalized to activation)
    v_rand = torch.randn(n_dirs, d_model, device=device, dtype=model_dtype)
    v_rand = F.normalize(v_rand, dim=-1)

    # Reference (unperturbed)
    with torch.no_grad():
        ref_point = a.unsqueeze(0).unsqueeze(0)
        ref_final = forward_from_layer_to_final(model, ctx_d, ref_point, layer)
        ref_ln = _get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1)
        ref_logits = _get_lm_head(model)(ref_ln)
        ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
        ref_probs = ref_log_probs.exp()

    for ang_idx, theta in enumerate(angles_rad):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        perturbed = R * (cos_t * a_unit.unsqueeze(0) + sin_t * v_rand)  # [n_dirs, d]
        perturbed = perturbed.unsqueeze(1)  # [n_dirs, 1, d]

        with torch.no_grad():
            pert_final = forward_from_layer_to_final(model, ctx_d, perturbed, layer)
            l2_dists[ang_idx] = (pert_final - ref_final).float().norm(dim=-1).cpu().numpy()
            cos_sim = F.cosine_similarity(pert_final.float(), ref_final.float(), dim=-1)
            cos_dists[ang_idx] = (1 - cos_sim).cpu().numpy()
            pert_ln = _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)
            pert_logits = _get_lm_head(model)(pert_ln)
            pert_log_probs = F.log_softmax(pert_logits.float(), dim=-1)
            kl = (ref_probs * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            kl_divs[ang_idx] = kl.cpu().numpy()

    return l2_dists, kl_divs, cos_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--melbo_pkl', type=str, required=True,
                        help='Path to MELBO reconstruction results.pkl')
    parser.add_argument('--extra_pkl', type=str, nargs='+', default=[],
                        help='Extra reconstruction pkls as label:path pairs (e.g. "Cov-matched:path.pkl")')
    parser.add_argument('--n_dirs', type=int, default=10,
                        help='Random perturbation directions per activation')
    parser.add_argument('--n_angles', type=int, default=60,
                        help='Number of angle points')
    parser.add_argument('--max_angle', type=float, default=60,
                        help='Maximum perturbation angle in degrees')
    parser.add_argument('--real_act_idx', type=int, default=None,
                        help='Override which real activation to use (default: from pkl)')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load reconstruction results
    print(f"Loading MELBO reconstruction: {args.melbo_pkl}")
    with open(args.melbo_pkl, 'rb') as f:
        melbo_data = pickle.load(f)

    activations = {
        'Reconstructed': melbo_data['reconstructed_activation'],
        'Real': melbo_data['real_activation'],
        'Random': melbo_data['random_activation'],
    }

    extra_data = {}
    for spec in args.extra_pkl:
        label, path = spec.split(':', 1)
        print(f"Loading extra reconstruction '{label}': {path}")
        with open(path, 'rb') as f:
            extra_data[label] = pickle.load(f)
        activations[label] = extra_data[label]['reconstructed_activation']

    # Get model/layer from the reconstruction args
    recon_args = melbo_data['args']
    model_name = recon_args.get('model') or melbo_data['melbo_args']['model']
    layer = recon_args.get('layer')
    if layer is None:
        layer = melbo_data['melbo_args']['inject_layer']
    seq_len = recon_args.get('seq_len', 1)
    act_idx = recon_args.get('act_idx', 0)

    if args.results_dir is None:
        args.results_dir = os.path.dirname(args.melbo_pkl)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\nActivation norms:")
    for name, act in activations.items():
        print(f"  {name}: {act.norm().item():.2f}")

    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer, device = load_model(model_name)

    # Get context (same as reconstruction used)
    print(f"Loading context (seq_len={seq_len}, act_idx={act_idx})...")
    token_ids_list = load_fineweb_fixed_length(
        100, tokenizer, seq_len=seq_len, seed=recon_args.get('seed', 42))
    batch_tokens = torch.stack(token_ids_list[:64]).to(device)
    with torch.no_grad():
        outputs = model(batch_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
    ctx = hidden[act_idx, :-1, :].float().unsqueeze(0)  # [1, seq-1, d]

    if args.real_act_idx is not None:
        ri = args.real_act_idx
        print(f"Overriding real activation: act_idx={ri}")
        activations['Real'] = hidden[ri, -1, :].float().cpu()
        ctx = hidden[ri, :-1, :].float().unsqueeze(0)
        print(f"  Real norm: {activations['Real'].norm().item():.2f}")

    # Angles
    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Run sweeps
    palette = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink']
    colors = {}
    for i, name in enumerate(activations):
        colors[name] = palette[i % len(palette)]
    all_l2 = {}
    all_kl = {}
    all_cos = {}

    for name, act in activations.items():
        print(f"\nSweeping: {name} (norm={act.norm().item():.2f})")
        t0 = time.time()
        l2, kl, cos = run_spherical_sweep_single(
            model, ctx, act, angles_rad, layer, device, n_dirs=args.n_dirs)
        all_l2[name] = l2  # [n_angles, n_dirs]
        all_kl[name] = kl
        all_cos[name] = cos
        median_l2 = np.median(l2, axis=1)
        median_kl = np.median(kl, axis=1)
        print(f"  Done in {time.time()-t0:.1f}s, L2 at {args.max_angle}°: "
              f"median={median_l2[-1]:.1f}, KL: median={median_kl[-1]:.2f}")

    # Save
    results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'l2': all_l2,
        'kl': all_kl,
        'cos': all_cos,
        'activation_norms': {k: v.norm().item() for k, v in activations.items()},
        'jacobian_norms': {
            'Reconstructed': melbo_data['jacobian_norms']['reconstructed'],
            'Real': melbo_data['jacobian_norms']['real'],
            'Random': melbo_data['jacobian_norms']['random'],
            **{label: d['jacobian_norms']['reconstructed'] for label, d in extra_data.items()},
        },
    }
    pkl_path = os.path.join(args.results_dir, 'perturbation_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # Plot
    for metric_key, metric_data, metric_label, fname in [
        ('l2', all_l2, 'L2 distance (final layer)', 'perturbation_sweep.png'),
        ('kl', all_kl, 'KL divergence (nats)', 'perturbation_sweep_kl.png'),
        ('cos', all_cos, 'Cosine distance (final layer)', 'perturbation_sweep_cos.png'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for name in activations:
            d = metric_data[name]  # [n_angles, n_dirs]
            median = np.median(d, axis=1)
            q25 = np.percentile(d, 25, axis=1)
            q75 = np.percentile(d, 75, axis=1)
            ax.plot(angles_deg, median, color=colors[name], label=name, linewidth=2)
            ax.fill_between(angles_deg, q25, q75, color=colors[name], alpha=0.15)

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(metric_label)
        ax.set_title(f'Perturbation sweep: reconstructed vs real vs random activation\n'
                     f'{model_name}, layer {layer}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, args.max_angle)

        plt.tight_layout()
        fig_path = os.path.join(args.results_dir, fname)
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fig_path}")


if __name__ == '__main__':
    main()
