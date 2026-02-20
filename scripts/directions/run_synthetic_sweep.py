#!/usr/bin/env python3
"""
Compare perturbation sweeps from real vs synthetic (Gaussian) activations.

Fits a Gaussian (mean + full covariance) to real activations, then runs
random-direction perturbation sweeps starting from both real and synthetic
activations. Tests whether the model's response to perturbations differs
between real and off-manifold starting points.

Usage:
    python scripts/directions/run_synthetic_sweep.py [--model qwen3-0.6b]
"""

import sys
import argparse
import pickle
import os
import time
import numpy as np

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import load_model, _get_blocks, _is_rope_model, _get_position_embeddings, _run_block, _get_final_norm, _get_lm_head
from src.data import load_fineweb_fixed_length


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device,
                                   batch_size=16):
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


def forward_to_hidden_and_logits(model, hidden_states, start_layer):
    """Forward pass returning penultimate hidden state and logits (last token).

    Returns L2-measurement hidden state from the 2nd-to-last block,
    and logits from the final block + LM head.
    """
    hidden = hidden_states
    blocks = _get_blocks(model)
    rope = _is_rope_model(model)
    pos_emb = _get_position_embeddings(model, hidden) if rope else None
    remaining_blocks = blocks[start_layer + 1:]
    penultimate_hidden = None
    for i, block in enumerate(remaining_blocks):
        hidden = _run_block(block, hidden, model, pos_emb)
        if i == len(remaining_blocks) - 2:
            penultimate_hidden = hidden[:, -1, :].clone()
    # Fallback if only 1 remaining block
    if penultimate_hidden is None:
        penultimate_hidden = hidden[:, -1, :]
    normed = _get_final_norm(model)(hidden[:, -1, :])
    logits = _get_lm_head(model)(normed)
    return penultimate_hidden, logits


def sweep_random_from_activations(model, contexts, activations, angles_rad, layer, device,
                                  n_dirs=10):
    """Random-direction perturbation sweep from given activations.

    Uses the provided contexts paired with activations. For synthetic activations,
    contexts come from real data (we only replace the final-token activation).
    """
    N, d = activations.shape
    n_angles = len(angles_rad)

    # Reference forward pass
    with torch.no_grad():
        ref_hidden = torch.cat([contexts, activations.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_hidden_and_logits(model, ref_hidden, layer)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    kl_divs = np.zeros((N, n_angles, n_dirs))
    l2_dists = np.zeros((N, n_angles, n_dirs))
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms

    # Generate random perpendicular directions (match dtype of activations)
    act_dtype = activations.dtype
    v_perps = torch.zeros(N, n_dirs, d, device=device, dtype=act_dtype)
    for i in range(N):
        v = torch.randn(n_dirs, d, device=device, dtype=act_dtype)
        v = v - (v @ a_unit[i]).unsqueeze(1) * a_unit[i].unsqueeze(0)
        v_perps[i] = F.normalize(v, dim=-1)

    for ang_idx, theta in enumerate(angles_rad):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        for dir_idx in range(n_dirs):
            v = v_perps[:, dir_idx, :]
            a_pert = a_norms * (cos_t * a_unit + sin_t * v)
            with torch.no_grad():
                hidden = torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)
                pert_final, pert_logits = forward_to_hidden_and_logits(model, hidden, layer)
                pert_log_probs = F.log_softmax(pert_logits, dim=-1)
                l2 = (pert_final - ref_final).norm(dim=-1)
                kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
                l2_dists[:, ang_idx, dir_idx] = l2.float().cpu().numpy()
                kl_divs[:, ang_idx, dir_idx] = kl.float().cpu().numpy()

    return kl_divs, l2_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--max_angle', type=float, default=120)
    parser.add_argument('--n_angles', type=int, default=40)
    parser.add_argument('--n_fit', type=int, default=2000,
                        help='Number of real activations to fit Gaussian')
    parser.add_argument('--n_sweep', type=int, default=50,
                        help='Number of activations to sweep from (real and synthetic)')
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bfloat16', action='store_true',
                        help='Convert model to bfloat16 (needed for larger models on 24GB)')
    parser.add_argument('--extract_batch_size', type=int, default=16)
    args = parser.parse_args()

    results_dir = f'results/genre_contrastive/{args.model}'
    os.makedirs(results_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    load_dtype = torch.bfloat16 if args.bfloat16 else None
    model, tokenizer, device = load_model(args.model, dtype=load_dtype)
    if args.bfloat16:
        print("  Loaded in bfloat16")
    n_layers = len(_get_blocks(model))
    d_model = model._plateau_info['d_model']
    print(f"  {n_layers} layers, d_model={d_model}, device={device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load real activations for fitting
    print(f"Loading {args.n_fit} FineWeb sequences for Gaussian fit...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_fit, tokenizer, seq_len=args.seq_len, seed=args.seed)

    print(f"Extracting layer-{args.layer} activations (batch_size={args.extract_batch_size})...")
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device,
        batch_size=args.extract_batch_size)

    # Fit Gaussian to real activations (cast to float32 for numpy — bfloat16 unsupported)
    acts_np = all_activations.float().numpy()  # (n_fit, d)
    mean = acts_np.mean(axis=0)
    centered = acts_np - mean
    cov = (centered.T @ centered) / (len(centered) - 1)
    print(f"Fitted Gaussian: d={acts_np.shape[1]}, n_fit={args.n_fit}")
    print(f"  ||mean|| = {np.linalg.norm(mean):.2f}")
    print(f"  trace(cov) = {np.trace(cov):.2f}")

    # Use first n_sweep real activations for the real sweep (keep original dtype)
    n_sweep = args.n_sweep
    real_contexts = all_contexts[:n_sweep].to(device)
    real_activations = all_activations[:n_sweep].to(device)

    # Sample synthetic activations from fitted Gaussian
    # Use eigendecomposition instead of Cholesky — handles rank-deficient cov
    # (when n_fit < d_model, cov has rank at most n_fit-1)
    print(f"Sampling {n_sweep} synthetic activations from fitted Gaussian...")
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0)  # clamp numerical negatives
    z = np.random.randn(n_sweep, acts_np.shape[1])
    synth_np = mean + z @ (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
    # Match model dtype (e.g. bfloat16) so sweep forward passes don't hit dtype mismatches
    model_dtype = next(model.parameters()).dtype
    synth_activations = torch.tensor(synth_np, dtype=model_dtype).to(device)

    # For synthetic activations, we still need contexts — reuse real ones
    # (the context is everything before the final token; we only replace the activation)
    synth_contexts = real_contexts

    print(f"Real activation norms: mean={real_activations.norm(dim=-1).mean():.1f}, "
          f"std={real_activations.norm(dim=-1).std():.1f}")
    print(f"Synth activation norms: mean={synth_activations.norm(dim=-1).mean():.1f}, "
          f"std={synth_activations.norm(dim=-1).std():.1f}")

    # Angles
    angles_deg = np.linspace(3, args.max_angle, args.n_angles)
    angles_rad = np.radians(angles_deg)
    print(f"Angles: {args.n_angles} steps from 3° to {args.max_angle}°")

    # Sweep from real activations
    print("\nSweeping random directions from REAL activations...")
    t0 = time.time()
    real_kl, real_l2 = sweep_random_from_activations(
        model, real_contexts, real_activations, angles_rad, args.layer, device,
        n_dirs=args.n_random_dirs)
    print(f"  Done in {time.time()-t0:.0f}s")

    # Sweep from synthetic activations
    print("Sweeping random directions from SYNTHETIC activations...")
    t0 = time.time()
    synth_kl, synth_l2 = sweep_random_from_activations(
        model, synth_contexts, synth_activations, angles_rad, args.layer, device,
        n_dirs=args.n_random_dirs)
    print(f"  Done in {time.time()-t0:.0f}s")

    # Save
    save_data = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'real': {'kl': real_kl, 'l2': real_l2},
        'synthetic': {'kl': synth_kl, 'l2': synth_l2},
        'gaussian_fit': {'mean_norm': float(np.linalg.norm(mean)), 'trace_cov': float(np.trace(cov))},
    }
    layer_suffix = f'_L{args.layer}' if args.layer != 14 else ''
    out_path = os.path.join(results_dir, f'synthetic_sweep_results{layer_suffix}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved: {out_path}")

    # Print summary
    real_kl_mean = real_kl.mean(axis=(0, 2))
    synth_kl_mean = synth_kl.mean(axis=(0, 2))
    real_l2_mean = real_l2.mean(axis=(0, 2))
    synth_l2_mean = synth_l2.mean(axis=(0, 2))

    key_angles = [15, 30, 45, 60, 90, 120]
    print(f"\nKL divergence (random dirs):")
    print(f"  {'':>10s}", end='')
    for a in key_angles:
        if a <= args.max_angle:
            print(f"  {a:>6d}°", end='')
    print()
    for name, kl_mean in [('real', real_kl_mean), ('synthetic', synth_kl_mean)]:
        print(f"  {name:>10s}", end='')
        for a in key_angles:
            if a <= args.max_angle:
                ai = np.argmin(np.abs(angles_deg - a))
                print(f"  {kl_mean[ai]:>7.3f}", end='')
        print()

    print(f"\nL2 distance (random dirs):")
    print(f"  {'':>10s}", end='')
    for a in key_angles:
        if a <= args.max_angle:
            print(f"  {a:>6d}°", end='')
    print()
    for name, l2_mean in [('real', real_l2_mean), ('synthetic', synth_l2_mean)]:
        print(f"  {name:>10s}", end='')
        for a in key_angles:
            if a <= args.max_angle:
                ai = np.argmin(np.abs(angles_deg - a))
                print(f"  {l2_mean[ai]:>7.1f}", end='')
        print()

    print("\nDone!")


if __name__ == '__main__':
    main()
