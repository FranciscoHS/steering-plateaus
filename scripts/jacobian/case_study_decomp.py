#!/usr/bin/env python3
"""
Case study: decompose a specific prompt's activation into plateau-breaking
directions, then steer other prompts with those directions.

Combines response_decomp.py (decomposition) + steer_response_dirs.py (steering)
into a single script for targeted case studies.

By default, response is measured on the target activation itself (--response_self),
since we're decomposing a single known activation rather than studying generic
structure.

Usage:
    # Eiffel Tower case study
    python scripts/jacobian/case_study_decomp.py \
        --target_prompt "The Eiffel tower is in" \
        --steer_prompts "München liegt in" \
        --results_dir results/case_study/eiffel

    # Code case study
    python scripts/jacobian/case_study_decomp.py \
        --target_prompt "def fibonacci(n)" \
        --steer_prompts "My favorite thing to do is" \
        --results_dir results/case_study/code
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
from scripts.steering.spherical_steering_generate import generate_with_spherical_steering


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
    """Find direction d maximizing α * Σ_j response_j - λ_orth * Σ (d·d_prev)²."""
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


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Case study: decompose + steer')
    # Target
    parser.add_argument('--target_prompt', required=True,
                        help='Prompt whose activation to decompose')
    parser.add_argument('--steer_prompts', nargs='+', required=True,
                        help='Prompts to steer with found directions')
    # Model
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=7)
    # Decomposition
    parser.add_argument('--k', type=int, default=10, help='Max features to extract')
    parser.add_argument('--min_residual_frac', type=float, default=0.05)
    parser.add_argument('--theta_deg', type=float, default=10.0,
                        help='Spherical perturbation angle for response measurement')
    parser.add_argument('--n_response_acts', type=int, default=10,
                        help='Number of FineWeb activations for response (ignored if --response_self)')
    parser.add_argument('--response_self', action='store_true', default=True,
                        help='Use target activation for response (default: True)')
    parser.add_argument('--no_response_self', dest='response_self', action='store_false',
                        help='Use FineWeb activations for response instead of target')
    parser.add_argument('--lambda_orth', type=float, default=None)
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    # Steering
    parser.add_argument('--steer_angles', type=str, default='0,15,30,60',
                        help='Comma-separated steering angles in degrees')
    parser.add_argument('--n_tokens', type=int, default=30)
    # Output
    parser.add_argument('--n_dirs_steer', type=int, default=5,
                        help='Number of top directions to steer with')
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--opt_seed', type=int, default=None)
    args = parser.parse_args()

    if args.opt_seed is None:
        args.opt_seed = args.seed

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    theta = math.radians(args.theta_deg)
    steer_angles = [float(a) for a in args.steer_angles.split(',')]

    print(f"=== Case study: '{args.target_prompt}' ===")
    print(f"Config: model={args.model}, layer={args.layer}, k={args.k}")
    print(f"  response_self={args.response_self}, theta={args.theta_deg}°")
    print(f"  n_steps={args.n_steps}, n_restarts={args.n_restarts}, lr={args.lr}")
    print(f"Steer prompts: {args.steer_prompts}")
    print(f"Steer angles: {steer_angles}")

    # ── Load model ──
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    end_layer = n_layers - 2
    print(f"  d_model={d_model}, n_layers={n_layers}, end_layer={end_layer}, device={device}")

    # ── Extract target activation ──
    print(f"\nExtracting target activation for: '{args.target_prompt}'")
    target_tokens = tokenizer(args.target_prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**target_tokens, output_hidden_states=True)
    target_hidden = outputs.hidden_states[args.layer + 1]
    target_context = target_hidden[:, :-1, :].detach()   # [1, seq-1, d]
    target_activation = target_hidden[0, -1, :].detach().float()  # [d]
    act_norm = target_activation.norm().item()
    print(f"  ||target activation|| = {act_norm:.2f}")
    print(f"  Target tokens: {tokenizer.decode(target_tokens['input_ids'][0])}")

    # ── Set up response activations ──
    if args.response_self:
        print("  Using target activation for response testing (self)")
        stacked_contexts = target_context  # [1, seq-1, d]
        stacked_activations = target_activation.unsqueeze(0)  # [1, d]
    else:
        print(f"  Loading {args.n_response_acts} FineWeb activations for response...")
        token_ids_list = load_fineweb_fixed_length(
            args.n_response_acts, tokenizer, seq_len=20, seed=args.seed)
        resp_contexts = []
        resp_activations = []
        for token_ids in token_ids_list:
            tokens = token_ids.unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tokens, output_hidden_states=True)
            hidden = out.hidden_states[args.layer + 1]
            resp_contexts.append(hidden[:, :-1, :].detach())
            resp_activations.append(hidden[0, -1, :].detach().float())
        stacked_contexts = torch.cat(resp_contexts, dim=0)
        stacked_activations = torch.stack(resp_activations)

    # ── Precompute baselines ──
    print("Precomputing baselines...")
    with torch.no_grad():
        baselines = forward_from_layer_to_layer(
            model, stacked_contexts, stacked_activations.unsqueeze(1),
            args.layer, end_layer)
    print(f"  baselines shape: {baselines.shape}")

    # ── Calibrate ──
    print("\nCalibrating...")
    with torch.no_grad():
        d_rand = torch.randn(d_model, device=device)
        d_rand = d_rand / d_rand.norm()
        perturbed = spherical_perturb_batch(stacked_activations, d_rand, theta)
        f_pert = forward_from_layer_to_layer(
            model, stacked_contexts, perturbed.unsqueeze(1), args.layer, end_layer)
        cos_sim = torch.nn.functional.cosine_similarity(f_pert, baselines, dim=-1)
        random_response = (1 - cos_sim).sum().item()
    print(f"  Random direction response: {random_response:.4f}")

    if args.lambda_orth is None:
        args.lambda_orth = act_norm * random_response * 10
        print(f"  Auto lambda_orth = {args.lambda_orth:.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # DECOMPOSITION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"DECOMPOSING: k={args.k}, theta={args.theta_deg}°")
    print(f"{'='*60}")

    residual = target_activation.clone()
    directions = []
    magnitudes = []
    responses = []

    for i in range(args.k):
        residual_frac = residual.norm().item() / act_norm
        if residual_frac < args.min_residual_frac:
            print(f"\n--- Stopping: ||residual||/||a|| = {residual_frac:.4f} ---")
            break
        print(f"\n--- Direction {i+1}/{args.k} | ||residual|| = {residual.norm():.2f} "
              f"({residual_frac:.1%}) ---")

        best_across = {'loss': float('inf'), 'd': None, 'alpha': None, 'response': None}

        t0 = time.time()
        for r in range(args.n_restarts):
            torch.manual_seed(args.opt_seed + i * 1000 + r)
            d, alpha, response, loss = decompose_one_direction(
                model, stacked_contexts, stacked_activations, baselines,
                residual, args.layer, end_layer, directions,
                theta, args.lambda_orth, args.n_steps, args.lr, device
            )
            if loss < best_across['loss']:
                best_across = {'loss': loss, 'd': d, 'alpha': alpha, 'response': response}
            print(f"  restart {r+1}/{args.n_restarts}: "
                  f"α={alpha:.3f}, response={response:.4f}, loss={loss:.3f}")

        elapsed = time.time() - t0
        d_best = best_across['d']
        alpha_best = best_across['alpha']
        response_best = best_across['response']

        directions.append(d_best)
        magnitudes.append(alpha_best)
        responses.append(response_best)
        residual = residual - alpha_best * d_best

        print(f"  -> best α={alpha_best:.3f}, response={response_best:.4f} [{elapsed:.0f}s]")

    n = len(directions)
    recon = sum(m * d for m, d in zip(magnitudes, directions))
    recon_error = (target_activation - recon).norm().item() / act_norm

    print(f"\n{'='*60}")
    print("DECOMPOSITION SUMMARY")
    print(f"{'='*60}")
    for i in range(n):
        print(f"  d_{i+1}: α={magnitudes[i]:.3f}, response={responses[i]:.4f}")
    print(f"Reconstruction error (relative): {recon_error:.4f}")
    print(f"Final ||residual|| = {residual.norm():.2f} (started at {act_norm:.2f})")

    # Save decomposition
    decomp_results = {
        'args': vars(args),
        'target_prompt': args.target_prompt,
        'directions': [d.cpu() for d in directions],
        'magnitudes': magnitudes,
        'responses': responses,
        'recon_error': recon_error,
        'residual_norm': residual.norm().item(),
        'target_activation': target_activation.cpu(),
        'random_response': random_response,
    }
    decomp_path = os.path.join(args.results_dir, 'decomp_results.pkl')
    with open(decomp_path, 'wb') as f:
        pickle.dump(decomp_results, f)
    print(f"\nDecomposition saved to {decomp_path}")

    # ══════════════════════════════════════════════════════════════════════
    # STEERING
    # ══════════════════════════════════════════════════════════════════════
    n_steer = min(args.n_dirs_steer, n)
    all_prompts = [args.target_prompt] + args.steer_prompts
    prompt_labels = ['target'] + [f'steer_{i+1}' for i in range(len(args.steer_prompts))]

    print(f"\n{'='*60}")
    print(f"STEERING: top {n_steer} directions, angles={steer_angles}")
    print(f"Prompts: {all_prompts}")
    print(f"{'='*60}")

    output_lines = []
    output_lines.append(f"Case study: '{args.target_prompt}'")
    output_lines.append(f"Model: {args.model}, layer {args.layer}")
    output_lines.append(f"Directions: top {n_steer}, angles: {steer_angles}")
    output_lines.append(f"Tokens per completion: {args.n_tokens}")
    output_lines.append('=' * 70)

    for d_idx in range(n_steer):
        direction = directions[d_idx].to(device).float()
        direction = direction / direction.norm()

        output_lines.append(f"\nDIRECTION {d_idx+1} (α={magnitudes[d_idx]:.2f}, "
                          f"response={responses[d_idx]:.4f})")
        output_lines.append('-' * 50)

        for prompt, label in zip(all_prompts, prompt_labels):
            output_lines.append(f"\n  Prompt ({label}): {prompt}")
            for angle in steer_angles:
                text, ids = generate_with_spherical_steering(
                    model, tokenizer, prompt, direction, angle,
                    args.layer, device, args.n_tokens, steer_mode='rolling'
                )
                output_lines.append(f"    {angle:3.0f}°: {text}")

    output_text = '\n'.join(output_lines)
    txt_path = os.path.join(args.results_dir, 'completions.txt')
    with open(txt_path, 'w') as f:
        f.write(output_text)

    print(output_text)
    print(f"\nCompletions saved to {txt_path}")


if __name__ == '__main__':
    main()
