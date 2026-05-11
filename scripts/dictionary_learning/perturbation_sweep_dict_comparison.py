#!/usr/bin/env python3
"""
Perturbation sweep comparing reconstructions from different dictionary methods.

For each eval activation, reconstructs using:
  1. Real activation (reference)
  2. MP-pruned dictionary (matching pursuit)
  3. FP-pruned dictionary (matching pursuit)
  4. Trained SAE (top-k)
  5. PCA (top-n components)
  6. Random dictionary (matching pursuit)

Then performs a spherical perturbation sweep on each, measuring output
sensitivity (L2 distance, cosine distance, KL divergence).

Usage:
    python -u scripts/dictionary_learning/perturbation_sweep_dict_comparison.py \
        --model meta-llama/Llama-3.2-1B --layer 2 \
        --mp_dict_path results/pruned_dictionary/llama-3.2-1b_layer2/pruned_dictionary.pkl \
        --fp_dict_path results/pruned_dictionary_fp/llama-3.2-1b_layer2/pruned_dictionary.pkl \
        --sae_path results/trained_sae/llama-3.2-1b_layer2_8x_k32/sae_weights.pt \
        --output_dir results/perturbation_sweep_dict_comparison/llama-3.2-1b_layer2
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


# ── Reconstruction methods ──────────────────────────────────────────────

def matching_pursuit_reconstruct(dictionary, activations, n_nonzero_coefs):
    """Reconstruct activations via matching pursuit with n atoms."""
    n_val = activations.shape[0]
    device = activations.device
    residuals = activations.clone()
    recon = torch.zeros_like(activations)

    for step in range(n_nonzero_coefs):
        corr = residuals @ dictionary.T
        best = corr.abs().argmax(dim=1)
        coefs = corr[torch.arange(n_val, device=device), best]
        contribution = coefs.unsqueeze(1) * dictionary[best]
        recon = recon + contribution
        residuals = residuals - contribution

    return recon


def sae_reconstruct_topn(sae, activations, n, batch_size=256):
    """Reconstruct activations using our TopK SAE with top-n features."""
    n_val = activations.shape[0]
    chunks = []
    for i in range(0, n_val, batch_size):
        batch = activations[i:i+batch_size]
        _, _, pre_acts = sae.encode(batch)
        topn_vals, topn_idx = pre_acts.topk(n, dim=-1)
        chunks.append(sae.decode(topn_vals, topn_idx))
    return torch.cat(chunks, dim=0)


def pca_reconstruct(activations, fit_activations, n_components):
    """Reconstruct activations using top-n PCA components fit on fit_activations."""
    mean = fit_activations.mean(dim=0)
    centered_fit = fit_activations - mean
    # Use torch.pca_lowrank for efficiency
    U, S, V = torch.pca_lowrank(centered_fit, q=n_components)
    # V: (d_model, n_components) — principal components
    centered = activations - mean
    projections = centered @ V  # (n_act, n_components)
    recon = projections @ V.T + mean
    return recon


# ── Perturbation sweep ──────────────────────────────────────────────────

def sweep_single_activation(model, ctx, activation, angles_rad, layer, device, n_dirs=10):
    """Spherical perturbation sweep for a single activation.

    Returns:
        l2_dists: (n_angles, n_dirs)
        kl_divs: (n_angles, n_dirs)
        cos_dists: (n_angles, n_dirs)
    """
    n_angles = len(angles_rad)
    l2_dists = np.zeros((n_angles, n_dirs))
    kl_divs = np.zeros((n_angles, n_dirs))
    cos_dists = np.zeros((n_angles, n_dirs))

    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a = activation.to(device=device, dtype=model_dtype)
    R = a.norm()

    if R < 1e-8:
        return l2_dists, kl_divs, cos_dists

    a_unit = a / R
    d_model = a.shape[-1]

    # Random orthogonal directions
    v_rand = torch.randn(n_dirs, d_model, device=device, dtype=model_dtype)
    v_rand = v_rand - (v_rand @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
    v_rand = F.normalize(v_rand, dim=-1)

    # Reference
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
        perturbed = R * (cos_t * a_unit.unsqueeze(0) + sin_t * v_rand)
        perturbed = perturbed.unsqueeze(1)

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


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--mp_dict_path', required=True,
                        help='MP-pruned dictionary pkl')
    parser.add_argument('--fp_dict_path', required=True,
                        help='FP-pruned dictionary pkl')
    parser.add_argument('--sae_path', default=None,
                        help='Our trained SAE weights')
    parser.add_argument('--n_features', type=int, default=20,
                        help='Number of features/atoms for reconstruction')
    parser.add_argument('--n_act', type=int, default=50,
                        help='Number of activations to sweep')
    parser.add_argument('--n_dirs', type=int, default=10,
                        help='Random perturbation directions per activation')
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--max_angle', type=float, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_total', type=int, default=8000)
    parser.add_argument('--eval_start', type=int, default=7000)
    parser.add_argument('--eval_end', type=int, default=7050,
                        help='Default: eval_start + n_act')
    parser.add_argument('--pca_fit_start', type=int, default=2000,
                        help='Start index for PCA fitting data')
    parser.add_argument('--pca_fit_end', type=int, default=7000,
                        help='End index for PCA fitting data')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    # Auto-set eval_end from n_act if using default
    if args.eval_end == 7050:
        args.eval_end = args.eval_start + args.n_act

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)
    n = args.n_features

    print(f"Perturbation sweep: dictionary comparison")
    print(f"  model={args.model}, layer={args.layer}")
    print(f"  n_features={n}, n_act={args.n_act}, n_dirs={args.n_dirs}")
    print(f"  angles: {len(angles_deg)} points, 0-{args.max_angle} deg")
    print()

    # ── Load dictionaries ───────────────────────────────────────────────
    print("Loading MP-pruned dictionary...")
    with open(args.mp_dict_path, 'rb') as f:
        mp_data = pickle.load(f)
    mp_dict = F.normalize(mp_data['dictionary'].float(), dim=-1)
    print(f"  Shape: {mp_dict.shape}")

    print("Loading FP-pruned dictionary...")
    with open(args.fp_dict_path, 'rb') as f:
        fp_data = pickle.load(f)
    fp_dict = F.normalize(fp_data['dictionary'].float(), dim=-1)
    print(f"  Shape: {fp_dict.shape}")

    # Random dictionary (same size as MP dict)
    d_model = mp_dict.shape[1]
    n_atoms = mp_dict.shape[0]
    print(f"Generating random dictionary ({n_atoms} x {d_model})...")
    torch.manual_seed(args.seed + 999)
    rand_dict = F.normalize(torch.randn(n_atoms, d_model), dim=-1)

    # ── Load SAE ────────────────────────────────────────────────────────
    sae = None
    if args.sae_path:
        print(f"Loading SAE from {args.sae_path}...")
        from src.autoencoders import TopKSAE
        sae_data = torch.load(args.sae_path, map_location='cpu', weights_only=False)
        config = sae_data['config']
        sae = TopKSAE(d_in=config['d_in'], d_sae=config['d_sae'], k=config['k'])
        sae.load_state_dict(sae_data['state_dict'])
        sae.eval()
        print(f"  d_sae={config['d_sae']}, k={config['k']}")

    # ── Load model ──────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    # Move dictionaries + SAE to device
    mp_dict = mp_dict.to(device)
    fp_dict = fp_dict.to(device)
    rand_dict = rand_dict.to(device)
    if sae is not None:
        sae = sae.to(device)

    # ── Extract activations ─────────────────────────────────────────────
    print(f"Loading {args.n_total} FineWeb samples...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_total, tokenizer, seq_len=args.seq_len, seed=args.seed)

    n_eval = args.eval_end - args.eval_start
    eval_tokens = token_ids_list[args.eval_start:args.eval_end]

    print(f"Extracting {n_eval} eval activations at layer {args.layer}...")
    all_contexts = []
    all_activations = []
    batch_size = 64
    for i in range(0, n_eval, batch_size):
        batch = torch.stack(eval_tokens[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
        hidden = outputs.hidden_states[args.layer + 1]
        all_contexts.append(hidden[:, :-1, :].float().cpu())
        all_activations.append(hidden[:, -1, :].float().cpu())
    contexts = torch.cat(all_contexts, 0)      # (n_eval, seq-1, d_model)
    activations = torch.cat(all_activations, 0)  # (n_eval, d_model)
    print(f"  Mean activation norm: {activations.norm(dim=-1).mean():.2f}")

    # ── PCA fitting data ────────────────────────────────────────────────
    print(f"Extracting PCA fitting activations [{args.pca_fit_start}:{args.pca_fit_end}]...")
    pca_tokens = token_ids_list[args.pca_fit_start:args.pca_fit_end]
    pca_acts = []
    for i in range(0, len(pca_tokens), batch_size):
        batch = torch.stack(pca_tokens[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
        hidden = outputs.hidden_states[args.layer + 1]
        pca_acts.append(hidden[:, -1, :].float().cpu())
    pca_fit_activations = torch.cat(pca_acts, 0)
    print(f"  PCA fit set: {pca_fit_activations.shape}")

    # ── Reconstruct ─────────────────────────────────────────────────────
    act_gpu = activations.to(device)

    print(f"\nReconstructing with n={n} features...")
    mp_recon = matching_pursuit_reconstruct(mp_dict, act_gpu, n)
    print(f"  MP-pruned:  cos_sim={F.cosine_similarity(act_gpu, mp_recon, dim=-1).mean():.4f}")

    fp_recon = matching_pursuit_reconstruct(fp_dict, act_gpu, n)
    print(f"  FP-pruned:  cos_sim={F.cosine_similarity(act_gpu, fp_recon, dim=-1).mean():.4f}")

    rand_recon = matching_pursuit_reconstruct(rand_dict, act_gpu, n)
    print(f"  Random:     cos_sim={F.cosine_similarity(act_gpu, rand_recon, dim=-1).mean():.4f}")

    sae_recon = None
    if sae is not None:
        sae_recon = sae_reconstruct_topn(sae, act_gpu, n)
        print(f"  SAE:        cos_sim={F.cosine_similarity(act_gpu, sae_recon, dim=-1).mean():.4f}")

    # PCA on CPU (lowrank needs float)
    pca_recon = pca_reconstruct(activations, pca_fit_activations, n).to(device)
    print(f"  PCA:        cos_sim={F.cosine_similarity(act_gpu, pca_recon, dim=-1).mean():.4f}")

    # Build methods dict — order matters for plotting
    methods = {
        'Real': act_gpu.cpu(),
        'MP-pruned': mp_recon.cpu(),
        'FP-pruned': fp_recon.cpu(),
    }
    if sae_recon is not None:
        methods['SAE'] = sae_recon.cpu()
    methods['PCA'] = pca_recon.cpu()
    methods['Random dict'] = rand_recon.cpu()

    # Free GPU memory from dictionaries
    del mp_dict, fp_dict, rand_dict, act_gpu, mp_recon, fp_recon, rand_recon, pca_recon
    if sae is not None:
        del sae, sae_recon
    torch.cuda.empty_cache()

    # ── Perturbation sweeps ─────────────────────────────────────────────
    all_results = {}
    for method_name, recon_acts in methods.items():
        print(f"\n=== Sweeping: {method_name} (n_act={n_eval}) ===")
        l2 = np.zeros((n_eval, len(angles_deg), args.n_dirs))
        kl = np.zeros((n_eval, len(angles_deg), args.n_dirs))
        cos = np.zeros((n_eval, len(angles_deg), args.n_dirs))

        t0 = time.time()
        for i in range(n_eval):
            l2[i], kl[i], cos[i] = sweep_single_activation(
                model, contexts[i], recon_acts[i],
                angles_rad, args.layer, device, n_dirs=args.n_dirs
            )
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{n_eval}  ({(i+1)/elapsed:.1f} act/s)")

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")
        all_results[method_name] = {'l2': l2, 'kl': kl, 'cos': cos}

    # ── Save ────────────────────────────────────────────────────────────
    save_data = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'methods': list(methods.keys()),
        'results': all_results,
        'activation_norms': {k: v.detach().norm(dim=-1).numpy() for k, v in methods.items()},
    }
    pkl_path = os.path.join(args.output_dir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved: {pkl_path}")

    # ── Plot ────────────────────────────────────────────────────────────
    colors = {
        'Real': 'black',
        'MP-pruned': 'C0',
        'FP-pruned': 'C1',
        'SAE': 'C2',
        'PCA': 'C4',
        'Random dict': 'C3',
    }

    for metric_key, metric_label, fname in [
        ('l2', 'L2 distance (final layer)', 'sweep_l2.png'),
        ('kl', 'KL divergence (nats)', 'sweep_kl.png'),
        ('cos', 'Cosine distance (final layer)', 'sweep_cos.png'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for method_name in methods:
            d = all_results[method_name][metric_key]  # (n_act, n_angles, n_dirs)
            per_act = np.median(d, axis=2)  # (n_act, n_angles)
            median = np.median(per_act, axis=0)
            q25 = np.percentile(per_act, 25, axis=0)
            q75 = np.percentile(per_act, 75, axis=0)
            ax.plot(angles_deg, median, color=colors[method_name],
                    label=method_name, linewidth=2)
            ax.fill_between(angles_deg, q25, q75,
                            color=colors[method_name], alpha=0.1)

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(metric_label)
        ax.set_title(f'Perturbation sweep: reconstruction comparison\n'
                     f'{args.model}, layer {args.layer}, n={n}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, args.max_angle)
        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, fname)
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fig_path}")


if __name__ == '__main__':
    main()
