#!/usr/bin/env python3
"""
Spherical perturbation sweep comparing real, SAE-reconstructed, and cov-matched activations.

Tests whether SAE-reconstructed activations have plateaus as pronounced as real ones.
Uses Llama-3.2-1B, layer 2, with our trained TopK SAE.

Usage:
    python scripts/perturbation/sweep_sae_recon.py
    python scripts/perturbation/sweep_sae_recon.py --sae_path results/trained_sae/llama-3.2-1b_layer2_8x_k20/sae_weights.pt
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
    _get_final_norm, _get_lm_head, _get_blocks,
)
from src.data import load_fineweb_fixed_length, sample_random_activations_v2


class TopKSAE(torch.nn.Module):
    """Minimal TopK SAE for loading weights and running encode/decode."""

    def __init__(self, d_in, d_sae, k):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.encoder = torch.nn.Linear(d_in, d_sae)
        self.W_dec = torch.nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))

    def forward(self, x):
        x_centered = x - self.b_dec
        pre_acts = F.relu(self.encoder(x_centered))
        top_vals, top_idx = pre_acts.topk(self.k, dim=-1)
        W_selected = self.W_dec[top_idx]
        recon = (top_vals.unsqueeze(-1) * W_selected).sum(dim=1) + self.b_dec
        return recon


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device, batch_size=64):
    all_contexts = []
    all_activations = []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].float().cpu())
        all_activations.append(hidden[:, -1, :].float().cpu())
    return torch.cat(all_contexts, 0), torch.cat(all_activations, 0)


def sample_orthogonal_directions(activations, n_dirs):
    n_act, d_model = activations.shape
    v_perps = torch.zeros(n_act, n_dirs, d_model)
    for i in range(n_act):
        a = activations[i]
        a_unit = a / a.norm()
        v = torch.randn(n_dirs, d_model)
        v = v - (v @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        v = F.normalize(v, dim=-1)
        v_perps[i] = v
    return v_perps


def run_spherical_sweep(model, context, activation, v_perps, angles_rad,
                        layer, device, n_dirs, measure_layer=None):
    """Sweep spherical perturbations for one activation."""
    n_angles = len(angles_rad)
    l2_dists = np.zeros((n_angles, n_dirs))
    kl_divs = np.zeros((n_angles, n_dirs))

    model_dtype = next(model.parameters()).dtype
    ctx = context.unsqueeze(0).to(device=device, dtype=model_dtype)
    a = activation.to(device=device, dtype=model_dtype)
    R = a.norm()
    a_unit = a / R
    vp = v_perps.to(device=device, dtype=model_dtype)

    with torch.no_grad():
        ref_point = a.unsqueeze(0).unsqueeze(0)
        ref_final = forward_from_layer_to_final(model, ctx, ref_point, layer)
        ref_ln = _get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1)
        ref_logits = _get_lm_head(model)(ref_ln)
        ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
        ref_measure = ref_final

    for ang_idx, theta in enumerate(angles_rad):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        perturbed = R * (cos_t * a_unit.unsqueeze(0) + sin_t * vp)
        perturbed = perturbed.unsqueeze(1)

        with torch.no_grad():
            pert_final = forward_from_layer_to_final(model, ctx, perturbed, layer)
            l2_dists[ang_idx] = (pert_final - ref_measure).float().norm(dim=-1).cpu().numpy()

            pert_ln = _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)
            pert_logits = _get_lm_head(model)(pert_ln)
            pert_log_probs = F.log_softmax(pert_logits.float(), dim=-1)
            ref_probs = ref_log_probs.exp()
            kl = (ref_probs * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            kl_divs[ang_idx] = kl.cpu().numpy()

    return l2_dists, kl_divs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--sae_path', type=str,
                        default='results/trained_sae/llama-3.2-1b_layer2_8x_k20/sae_weights.pt')
    parser.add_argument('--sae_k', type=int, default=20)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=100)
    parser.add_argument('--n_dirs', type=int, default=10)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--max_angle', type=float, default=60)
    parser.add_argument('--n_fit', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/perturbation_sweep_sae_recon'
    os.makedirs(args.results_dir, exist_ok=True)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    # Load SAE
    print(f"Loading SAE from {args.sae_path}...")
    sae_checkpoint = torch.load(args.sae_path, map_location='cpu', weights_only=False)
    sae_sd = sae_checkpoint['state_dict']
    sae_config = sae_checkpoint['config']
    d_sae = sae_config['d_sae']
    k = sae_config['k']
    sae = TopKSAE(d_model, d_sae, k)
    sae.load_state_dict(sae_sd)
    sae.eval().to(device)
    print(f"SAE: d_sae={d_sae}, k={k}")

    # Load data
    n_load = max(args.n_act, args.n_fit)
    print(f"Loading {n_load} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(n_load, tokenizer, seq_len=args.seq_len, seed=args.seed)

    print(f"Extracting layer-{args.layer} activations...")
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device
    )
    contexts = all_contexts[:args.n_act]
    activations = all_activations[:args.n_act]
    act_norms = activations.norm(dim=-1)
    act_norm_mean = act_norms.mean().item()
    print(f"Mean activation norm: {act_norm_mean:.2f}")

    # SAE reconstructions
    print("Computing SAE reconstructions...")
    with torch.no_grad():
        sae_recon = sae(activations.to(device)).cpu()
    recon_error = (activations - sae_recon).norm(dim=-1)
    recon_cos = F.cosine_similarity(activations, sae_recon, dim=-1)
    print(f"SAE recon: L2 error mean={recon_error.mean():.2f}, cos_sim mean={recon_cos.mean():.4f}")
    print(f"SAE recon norms: mean={sae_recon.norm(dim=-1).mean():.2f}")

    # Cov-matched random
    print(f"Sampling cov-matched random activations (n_fit={all_activations.shape[0]})...")
    cov_random, cov_stats = sample_random_activations_v2(
        all_activations, args.n_act, mode='full', seed=args.seed, device=device
    )
    # Match norms to real activations
    cov_random = F.normalize(cov_random, dim=-1) * act_norm_mean
    print(f"Cov-matched random: rank={cov_stats.get('rank', '?')}")

    # Three conditions
    conditions = {
        'real': {'activations': activations, 'label': 'Real'},
        'sae_recon': {'activations': sae_recon, 'label': 'SAE recon'},
        'cov_matched': {'activations': cov_random, 'label': 'Random (cov-matched)'},
    }

    # Run sweeps
    sweep_results = {}
    for cond_name, cond in conditions.items():
        cond_acts = cond['activations']
        print(f"\n=== {cond['label']} ===")

        v_perps = sample_orthogonal_directions(cond_acts, args.n_dirs)

        all_l2 = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
        all_kl = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
        t0 = time.time()
        for i in range(args.n_act):
            all_l2[i], all_kl[i] = run_spherical_sweep(
                model, contexts[i], cond_acts[i], v_perps[i],
                angles_rad, args.layer, device, args.n_dirs,
            )
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
        print(f"Done in {time.time()-t0:.1f}s")

        sweep_results[cond_name] = {'l2': all_l2, 'kl': all_kl, 'label': cond['label']}

    # Save
    results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'angles_rad': angles_rad,
        'act_norm_mean': act_norm_mean,
        'recon_error_mean': recon_error.mean().item(),
        'recon_cos_mean': recon_cos.mean().item(),
        **{k: {'l2': v['l2'], 'kl': v['kl']} for k, v in sweep_results.items()},
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {args.results_dir}/results.pkl")

    # Plot
    chord_distances = 2 * act_norm_mean * np.sin(np.deg2rad(angles_deg) / 2)
    colors = {'real': 'blue', 'sae_recon': 'green', 'cov_matched': 'red'}

    for plot_type in ['median', 'individual']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for metric_idx, (key, name) in enumerate([('l2', 'L2 distance'), ('kl', 'KL divergence')]):
            ax = axes[metric_idx]
            for cond_name, cond_data in sweep_results.items():
                d = cond_data[key]
                per_act = np.median(d, axis=2)  # median over directions
                color = colors[cond_name]
                label = cond_data['label']

                if plot_type == 'median':
                    median = np.median(per_act, axis=0)
                    q25 = np.percentile(per_act, 25, axis=0)
                    q75 = np.percentile(per_act, 75, axis=0)
                    ax.plot(angles_deg, median, color=color, label=label, linewidth=2)
                    ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.15)
                else:
                    chosen = np.random.choice(per_act.shape[0], min(10, per_act.shape[0]), replace=False)
                    for j, idx in enumerate(chosen):
                        ax.plot(angles_deg, per_act[idx], color=color, alpha=0.4, linewidth=1.5,
                                label=label if j == 0 else None)

            ax.set_xlabel('Perturbation angle (degrees)')
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, args.max_angle)

            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            tick_angles = [a for a in [0, 5, 10, 15, 20, 30, 40, 50, 60] if a <= args.max_angle]
            tick_chords = [2 * act_norm_mean * np.sin(np.deg2rad(a) / 2) for a in tick_angles]
            ax2.set_xticks(tick_angles)
            ax2.set_xticklabels([f'{c:.0f}' for c in tick_chords])
            ax2.set_xlabel('Chord distance')

        fig.suptitle(f'Llama-3.2-1B, layer {args.layer} — real vs SAE-recon vs cov-matched ({plot_type})',
                     fontsize=13, y=1.02)
        plt.tight_layout()
        path = os.path.join(args.results_dir, f'sweep_{plot_type}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
