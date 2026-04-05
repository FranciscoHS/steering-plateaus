#!/usr/bin/env python3
"""
Large-scale batch decomposition of FineWeb activations.

Decomposes activations [start_idx, end_idx) into k plateau-breaking directions
each, using response-based greedy decomposition. Designed to run on RunPod pods
in parallel, with checkpointing for resume after interruptions.

Usage:
    python -u scripts/jacobian/batch_decomp.py \
        --model meta-llama/Llama-3.2-1B --layer 2 --k 20 \
        --n_total 2000 --start_idx 0 --end_idx 100 \
        --response_self --n_restarts 3 --n_steps 100 \
        --seq_len 10 --seed 42 \
        --results_dir results/batch_decomp/llama-3.2-1b_layer2
"""

import argparse
import math
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from src.data import load_fineweb_fixed_length


# ── Decomposition core (from response_decomp.py) ────────────────────────

def spherical_perturb_batch(activations, d_hat, theta):
    """Rotate each activation toward d by angle theta on the hypersphere."""
    norms = activations.norm(dim=-1, keepdim=True)
    a_hat = activations / (norms + 1e-8)
    proj = (a_hat @ d_hat).unsqueeze(-1)
    d_perp = d_hat.unsqueeze(0) - proj * a_hat
    d_perp_hat = d_perp / (d_perp.norm(dim=-1, keepdim=True) + 1e-8)
    return norms * (math.cos(theta) * a_hat + math.sin(theta) * d_perp_hat)


def decompose_one_direction(model, stacked_contexts, stacked_activations, baselines,
                            residual, layer, end_layer, prev_directions,
                            theta, lambda_orth, n_steps, lr, device):
    """Find direction d maximizing alpha * sum_j response_j - lambda_orth * sum (d . d_prev)^2."""
    d_model = residual.shape[0]
    d_raw = torch.randn(d_model, device=device, dtype=torch.float32)
    d_raw.requires_grad_(True)
    optimizer = torch.optim.Adam([d_raw], lr=lr)

    best = {'loss': float('inf'), 'd': None, 'alpha': None, 'response': None}
    r = residual.detach()

    for step in range(n_steps):
        optimizer.zero_grad()
        d_hat = d_raw / (d_raw.norm() + 1e-8)
        alpha = d_hat @ r

        perturbed = spherical_perturb_batch(stacked_activations, d_hat, theta)
        points = perturbed.unsqueeze(1)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            f_pert = forward_from_layer_to_layer(
                model, stacked_contexts, points, layer, end_layer)

        cos_sim = torch.nn.functional.cosine_similarity(f_pert, baselines, dim=-1)
        total_response = (1 - cos_sim).sum()

        orth = torch.tensor(0.0, device=device)
        for d_prev in prev_directions:
            orth = orth + (d_hat @ d_prev).pow(2)

        loss = -(alpha * total_response) + lambda_orth * orth
        loss.backward()
        torch.nn.utils.clip_grad_norm_([d_raw], 1.0)
        optimizer.step()

        if loss.item() < best['loss']:
            best['loss'] = loss.item()
            best['d'] = d_hat.detach().clone()
            best['alpha'] = alpha.item()
            best['response'] = total_response.item()

    return best['d'], best['alpha'], best['response'], best['loss']


# ── Per-activation processing ────────────────────────────────────────────

def process_one_activation(model, token_ids, act_idx, args, device, d_model, end_layer, theta):
    """Decompose a single activation into k directions with checkpointing.

    Args:
        model: loaded model
        token_ids: [seq_len] token tensor for this activation
        act_idx: global index (for directory naming and seed)
        args: parsed arguments
        device: torch device
        d_model: model hidden dimension
        end_layer: penultimate layer index
        theta: perturbation angle in radians

    Returns:
        True if completed (or was already complete), False on error
    """
    act_dir = os.path.join(args.results_dir, f"act_{act_idx:04d}")
    os.makedirs(act_dir, exist_ok=True)

    results_path = os.path.join(act_dir, "results.pkl")
    ckpt_path = os.path.join(act_dir, "checkpoint.pkl")

    # Skip if already complete
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            existing = pickle.load(f)
        if existing.get('complete', False):
            print(f"  act_{act_idx:04d}: already complete, skipping")
            return True

    # Extract activation
    tokens = token_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    hidden = outputs.hidden_states[args.layer + 1]
    target_context = hidden[:, :-1, :].detach()      # [1, seq-1, d]
    target_activation = hidden[0, -1, :].detach().float()  # [d]
    act_norm = target_activation.norm().item()

    # Set up response activations (self mode)
    if args.response_self:
        stacked_contexts = target_context
        stacked_activations = target_activation.unsqueeze(0)
    else:
        raise NotImplementedError("Only --response_self is supported in batch mode")

    # Precompute baselines
    with torch.no_grad():
        baselines = forward_from_layer_to_layer(
            model, stacked_contexts, stacked_activations.unsqueeze(1),
            args.layer, end_layer)

    # Calibrate lambda_orth
    with torch.no_grad():
        d_rand = torch.randn(d_model, device=device)
        d_rand = d_rand / d_rand.norm()
        perturbed = spherical_perturb_batch(stacked_activations, d_rand, theta)
        f_pert = forward_from_layer_to_layer(
            model, stacked_contexts, perturbed.unsqueeze(1), args.layer, end_layer)
        cos_sim = torch.nn.functional.cosine_similarity(f_pert, baselines, dim=-1)
        random_response = (1 - cos_sim).sum().item()
    lambda_orth = act_norm * random_response * 10

    # Resume from checkpoint if available
    directions = []
    magnitudes = []
    responses = []
    residual = target_activation.clone()
    start_dir = 0

    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        directions = [d.to(device) for d in ckpt['directions']]
        magnitudes = ckpt['magnitudes']
        responses = ckpt['responses']
        start_dir = ckpt['n_directions_done']
        # Reconstruct residual
        residual = target_activation.clone()
        for m, d in zip(magnitudes, directions):
            residual = residual - m * d
        print(f"  act_{act_idx:04d}: resuming from checkpoint ({start_dir}/{args.k} done)")

    # Greedy decomposition
    for i in range(start_dir, args.k):
        residual_frac = residual.norm().item() / act_norm
        if residual_frac < args.min_residual_frac:
            print(f"  act_{act_idx:04d}: stopping at dir {i}, residual_frac={residual_frac:.4f}")
            break

        best_across = {'loss': float('inf'), 'd': None, 'alpha': None, 'response': None}

        for r in range(args.n_restarts):
            torch.manual_seed(args.opt_seed + act_idx * 100000 + i * 1000 + r)
            d, alpha, response, loss = decompose_one_direction(
                model, stacked_contexts, stacked_activations, baselines,
                residual, args.layer, end_layer, directions,
                theta, lambda_orth, args.n_steps, args.lr, device
            )
            if loss < best_across['loss']:
                best_across = {'loss': loss, 'd': d, 'alpha': alpha, 'response': response}

        d_best = best_across['d']
        alpha_best = best_across['alpha']
        response_best = best_across['response']

        directions.append(d_best)
        magnitudes.append(alpha_best)
        responses.append(response_best)
        residual = residual - alpha_best * d_best

        # Checkpoint after each direction
        ckpt_data = {
            'directions': [d.cpu() for d in directions],
            'magnitudes': magnitudes[:],
            'responses': responses[:],
            'n_directions_done': len(directions),
            'act_idx': act_idx,
        }
        with open(ckpt_path, 'wb') as f:
            pickle.dump(ckpt_data, f)

    # Save final results
    n = len(directions)
    recon = sum(m * d for m, d in zip(magnitudes, directions))
    recon_error = (target_activation - recon).norm().item() / act_norm

    results = {
        'complete': True,
        'act_idx': act_idx,
        'args': vars(args),
        'directions': [d.cpu() for d in directions],
        'magnitudes': magnitudes,
        'responses': responses,
        'recon_error': recon_error,
        'residual_norm': residual.norm().item(),
        'target_activation': target_activation.cpu(),
        'random_response': random_response,
        'lambda_orth': lambda_orth,
        'n_directions': n,
    }
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    # Clean up checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"  act_{act_idx:04d}: done, {n} dirs, recon_err={recon_error:.4f}, "
          f"residual={residual.norm():.2f}/{act_norm:.2f}")
    return True


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Batch response decomposition')
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--k', type=int, default=20, help='Max features per activation')
    parser.add_argument('--min_residual_frac', type=float, default=0.05)
    parser.add_argument('--theta_deg', type=float, default=10.0)
    parser.add_argument('--response_self', action='store_true', default=True)
    parser.add_argument('--lambda_orth', type=float, default=None,
                        help='If set, override auto-calibration (not recommended)')
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--n_restarts', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_total', type=int, default=2000,
                        help='Total FineWeb samples to load (same on every pod)')
    parser.add_argument('--start_idx', type=int, required=True,
                        help='First activation index (inclusive)')
    parser.add_argument('--end_idx', type=int, required=True,
                        help='Last activation index (exclusive)')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--opt_seed', type=int, default=None)
    parser.add_argument('--results_dir', default='results/batch_decomp/llama-3.2-1b_layer2')
    args = parser.parse_args()

    if args.opt_seed is None:
        args.opt_seed = args.seed

    os.makedirs(args.results_dir, exist_ok=True)
    theta = math.radians(args.theta_deg)

    print(f"Batch decomposition: activations [{args.start_idx}, {args.end_idx})")
    print(f"Config: model={args.model}, layer={args.layer}, k={args.k}")
    print(f"  theta={args.theta_deg}, n_steps={args.n_steps}, n_restarts={args.n_restarts}")
    print(f"  response_self={args.response_self}, seq_len={args.seq_len}")

    # Load model
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    end_layer = n_layers - 2
    print(f"  d_model={d_model}, n_layers={n_layers}, end_layer={end_layer}, device={device}")

    # Load all FineWeb samples (same on every pod via same seed + n_total)
    print(f"\nLoading {args.n_total} FineWeb samples (seed={args.seed}, seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_total, tokenizer, seq_len=args.seq_len, seed=args.seed
    )
    print(f"Loaded {len(token_ids_list)} samples")

    # Process activations in range
    n_chunk = args.end_idx - args.start_idx
    print(f"\nProcessing {n_chunk} activations [{args.start_idx}, {args.end_idx})...")
    t_start = time.time()

    for idx in range(args.start_idx, args.end_idx):
        t0 = time.time()
        process_one_activation(
            model, token_ids_list[idx], idx, args,
            device, d_model, end_layer, theta
        )
        elapsed = time.time() - t0

        # Progress
        done = idx - args.start_idx + 1
        avg = (time.time() - t_start) / done
        remaining = (n_chunk - done) * avg
        print(f"  [{done}/{n_chunk}] {elapsed:.1f}s this act, "
              f"avg {avg:.1f}s, ~{remaining/60:.0f}min remaining")

    total_time = time.time() - t_start
    print(f"\nDone! {n_chunk} activations in {total_time/60:.1f} min "
          f"({total_time/n_chunk:.1f}s avg)")


if __name__ == '__main__':
    main()
