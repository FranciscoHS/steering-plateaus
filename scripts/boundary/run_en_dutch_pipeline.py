#!/usr/bin/env python3
"""
End-to-end boundary MELBO pipeline for English-Dutch on Qwen3-0.6B.

Steps:
  1. Constrained MELBO on En-Nl difference matrix → MELBO_1
  2. Walk English activations along -MELBO_1 → find boundary
  3. Unconstrained MELBO at boundary → bnd_MELBO_1, bnd_MELBO_2
  4. Steer with found directions (+ only), with temperature sampling

Usage:
    python scripts/boundary/run_en_dutch_pipeline.py --model qwen3-0.6b
"""

import argparse
import os
import pickle
import sys
import time
import numpy as np

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_allpos,
    _get_blocks, _is_rope_model, _get_position_embeddings, _run_block,
    _get_final_norm, _get_lm_head,
)
from data.bilingual_prompts_dutch import ENGLISH_PROMPTS, DUTCH_PROMPTS

# Reuse building blocks from existing scripts
from scripts.melbo.run_melbo_constrained import (
    optimize_direction as optimize_direction_constrained,
)
from scripts.boundary.run_boundary_melbo import (
    forward_to_hidden_and_logits,
    build_perturbed_hidden,
    compute_boundary_activations,
    sweep_single_direction,
    sweep_random_directions,
    find_boundary_angles,
    optimize_direction_at_boundary,
    extract_prompts_fixed_length,
    extract_context_and_activation,
)


# =========================================================================
# Step 1: Constrained MELBO on En-Dutch differences
# =========================================================================

def extract_bilingual_diffs_dutch(model, tokenizer, device, layer):
    """Extract difference matrix D = en_acts - nl_acts at given layer."""
    def get_acts(prompts):
        acts = []
        for p in prompts:
            tokens = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            acts.append(outputs.hidden_states[layer + 1][:, -1, :].squeeze(0).cpu())
        return torch.stack(acts)

    print(f"Extracting En-Dutch activations at layer {layer}...")
    en_acts = get_acts(ENGLISH_PROMPTS)
    nl_acts = get_acts(DUTCH_PROMPTS)
    D = en_acts - nl_acts

    mean_vec = D.mean(dim=0)
    mean_dir = mean_vec / mean_vec.norm()
    print(f"  D shape: {D.shape}, mean norm: {mean_vec.norm():.2f}")

    return D, mean_dir


def run_constrained_melbo(model, tokenizer, device, layer, D, mean_dir,
                          n_act, seq_len, n_directions, args):
    """Run constrained MELBO in span(D) using FineWeb activations."""
    # SVD
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    V = Vh.T  # [d_model, r]
    pc1 = Vh[0]
    print(f"SVD: rank r={V.shape[1]}, top SV={S[0]:.2f}, "
          f"PC1 var explained={S[0]**2 / (S**2).sum():.3f}")
    print(f"cos(PC1, mean): {(pc1 @ mean_dir).abs().item():.3f}")

    # FineWeb activations
    from src.data import load_fineweb_fixed_length
    print(f"\nLoading {n_act} FineWeb sequences (seq_len={seq_len})...")
    token_ids_list = load_fineweb_fixed_length(n_act, tokenizer, seq_len=seq_len, seed=42)
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, layer, device)
    print(f"Activation shape: {activations.shape}")

    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)
    V_dev = V.to(device)

    # Reference (unperturbed)
    with torch.no_grad():
        ref_hidden = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_hidden_and_logits(model, ref_hidden, layer)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # Optimize
    angle_rad = np.deg2rad(args.opt_angle)
    print(f"\nOptimizing {n_directions} directions in span(D), angle={args.opt_angle} deg")

    melbo_alphas = []
    melbo_directions = []
    melbo_kls = []
    melbo_histories = []

    for d_idx in range(n_directions):
        print(f"\n--- Direction {d_idx+1}/{n_directions} ---")
        alpha, direction, best_kl, history = optimize_direction_constrained(
            model, contexts_dev, activations_dev, ref_log_probs,
            V_dev, angle_rad, layer, device,
            n_steps=300, lr=0.01, n_restarts=10,
            prev_alphas=melbo_alphas if melbo_alphas else None
        )
        melbo_alphas.append(alpha)
        melbo_directions.append(direction.cpu())
        melbo_kls.append(best_kl)
        melbo_histories.append(history)

        cos_mean = (direction @ mean_dir.to(device)).abs().item()
        print(f"  Best KL: {best_kl:.4f}, cos(mean): {cos_mean:.3f}")

    # Evaluation sweep
    angles_deg = np.linspace(0, 180, 61)[1:]
    angles_rad_sweep = np.deg2rad(angles_deg)

    sweep_results = {}
    for name, d in [('mean', mean_dir), ('PC1', pc1)] + \
                   [(f'MELBO_{i+1}', melbo_directions[i]) for i in range(len(melbo_directions))]:
        print(f"Sweeping: {name}...", end=' ')
        kl, l2 = sweep_single_direction(
            model, contexts_dev, activations_dev, ref_final, ref_log_probs,
            d, angles_rad_sweep, layer, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}
        print("done")

    print(f"Sweeping: random...", end=' ')
    rand_kl, rand_l2 = sweep_random_directions(
        model, contexts_dev, activations_dev, ref_final, ref_log_probs,
        angles_rad_sweep, layer, device, n_dirs=10)
    sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}
    print("done")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'random': 'gray', 'mean': 'green', 'PC1': 'blue',
              'MELBO_1': 'red', 'MELBO_2': 'orange'}
    rand_median = np.median(np.median(sweep_results['random']['kl'], axis=2), axis=0)
    ax.plot(angles_deg, rand_median, color='gray', lw=2, ls='--', label='random')
    for name in ['mean', 'PC1', 'MELBO_1', 'MELBO_2']:
        if name in sweep_results:
            median = np.median(sweep_results[name]['kl'], axis=0)
            ax.plot(angles_deg, median, color=colors.get(name, 'black'), lw=2, label=name)
    ax.set_xlabel('Perturbation angle (deg)')
    ax.set_ylabel('KL divergence')
    ax.set_xlim(0, 70)
    ax.set_title(f'Constrained MELBO: En-Dutch, {args.model}, layer {layer}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'melbo_sweeps.png'), dpi=150)
    plt.close()

    results = {
        'melbo_directions': melbo_directions,
        'melbo_kls': melbo_kls,
        'melbo_histories': melbo_histories,
        'mean_dir': mean_dir,
        'pc1': pc1,
        'singular_values': S,
        'sweep_results': sweep_results,
        'angles_deg': angles_deg,
    }
    with open(os.path.join(args.results_dir, 'melbo_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved constrained MELBO results")

    return melbo_directions[0], mean_dir, contexts_dev, activations_dev, ref_final, ref_log_probs


# =========================================================================
# Step 2 & 3: Boundary finding + Boundary MELBO
# =========================================================================

def run_boundary_melbo(model, tokenizer, device, layer, melbo_1, mean_dir, args):
    """Walk to boundary along -MELBO_1, then run unconstrained MELBO there."""
    d_model = model._plateau_info['d_model']

    # Orient MELBO_1 English-ward
    if (melbo_1 @ mean_dir) < 0:
        melbo_1 = -melbo_1
    walk_dir = -melbo_1  # toward Dutch
    melbo_1_dev = melbo_1.to(device)
    walk_dir_dev = walk_dir.to(device)

    # Extract English activations (fixed-length)
    # Try seq_len=4 first (many Dutch-experiment prompts are short)
    print(f"\nExtracting English activations for boundary finding...")
    for try_seq_len in [5, 4, 3]:
        en_token_ids, used_prompts = extract_prompts_fixed_length(
            tokenizer, ENGLISH_PROMPTS, seq_len=try_seq_len)
        if len(en_token_ids) >= 20:
            break
    print(f"  {len(en_token_ids)} prompts at seq_len={try_seq_len}")

    contexts, activations = extract_context_and_activation(
        model, tokenizer, en_token_ids, layer, device)
    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)

    # Reference log-probs
    with torch.no_grad():
        ref_hidden = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_hidden_and_logits(model, ref_hidden, layer)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # Sweep along walk direction to find boundary
    sweep_angles_deg = np.linspace(0, 90, 60)
    sweep_angles_rad = np.deg2rad(sweep_angles_deg)

    print(f"Sweeping walk direction (-MELBO_1) to find boundary...")
    kl_sweep, _ = sweep_single_direction(
        model, contexts_dev, activations_dev, ref_final, ref_log_probs,
        walk_dir, sweep_angles_rad, layer, device)

    # Find boundary angles
    if args.boundary_angle is not None:
        # Use fixed boundary angle (override automatic detection)
        med_bnd = args.boundary_angle
        print(f"Using fixed boundary angle: {med_bnd} deg")
        bnd_angles_array = np.full(kl_sweep.shape[0], med_bnd)
    else:
        boundary_angles = find_boundary_angles(kl_sweep, sweep_angles_deg)
        valid = [a for a in boundary_angles if a is not None]
        med_bnd = np.median(valid)
        print(f"Boundary: median={med_bnd:.1f} deg, mean={np.mean(valid):.1f} deg, "
              f"n_valid={len(valid)}/{len(boundary_angles)}")
        bnd_angles_array = np.array([a if a is not None else med_bnd for a in boundary_angles])
    boundary_acts = compute_boundary_activations(
        activations_dev, walk_dir_dev, torch.tensor(np.deg2rad(bnd_angles_array),
                                                     dtype=torch.float32, device=device))

    # Reference at boundary
    with torch.no_grad():
        bnd_hidden = torch.cat([contexts_dev, boundary_acts.unsqueeze(1)], dim=1)
        bnd_final, bnd_logits = forward_to_hidden_and_logits(model, bnd_hidden, layer)
        bnd_ref_log_probs = F.log_softmax(bnd_logits, dim=-1)

    # Boundary MELBO
    print(f"\nRunning boundary MELBO (2 directions, orthogonal to MELBO_1)...")
    bnd_melbo_dirs = []
    bnd_melbo_kls = []
    bnd_melbo_histories = []
    angle_rad = np.deg2rad(args.opt_angle)

    orth_constraints = [melbo_1_dev / melbo_1_dev.norm()]

    for d_idx in range(2):
        print(f"\n--- Boundary direction {d_idx+1}/2 ---")
        direction, best_kl, history = optimize_direction_at_boundary(
            model, contexts_dev, boundary_acts, bnd_ref_log_probs,
            angle_rad, layer, device, d_model,
            n_steps=300, lr=0.01, n_restarts=5,
            prev_directions=orth_constraints
        )
        bnd_melbo_dirs.append(direction.cpu())
        bnd_melbo_kls.append(best_kl)
        bnd_melbo_histories.append(history)
        orth_constraints.append(direction)
        print(f"  Best KL: {best_kl:.1f}")

    # Evaluation sweeps from boundary
    eval_angles_deg = np.linspace(0, 90, 60)
    eval_angles_rad = np.deg2rad(eval_angles_deg)

    print(f"\nEvaluation sweeps from boundary...")
    bnd_sweep_results = {}

    for name, d in [('-MELBO_1 (walk)', walk_dir),
                    ('bnd_MELBO_1', bnd_melbo_dirs[0]),
                    ('bnd_MELBO_2', bnd_melbo_dirs[1])]:
        print(f"  Sweeping: {name}...")
        kl, l2 = sweep_single_direction(
            model, contexts_dev, boundary_acts, bnd_final, bnd_ref_log_probs,
            d, eval_angles_rad, layer, device)
        bnd_sweep_results[name] = {'kl': kl, 'l2': l2}

    print(f"  Sweeping: random...")
    rand_kl, rand_l2 = sweep_random_directions(
        model, contexts_dev, boundary_acts, bnd_final, bnd_ref_log_probs,
        eval_angles_rad, layer, device, n_dirs=10)
    bnd_sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}

    # Plots
    # 1. Boundary finding
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in range(kl_sweep.shape[0]):
        ax.plot(sweep_angles_deg, kl_sweep[s], alpha=0.3, color='gray')
    ax.plot(sweep_angles_deg, np.median(kl_sweep, axis=0), color='red', lw=2.5)
    ax.axvline(med_bnd, color='blue', ls='--', lw=1.5,
               label=f'boundary ({med_bnd:.0f} deg)')
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('KL divergence')
    ax.set_title(f'Boundary finding: En→Dutch, {args.model}, layer {layer}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'boundary_finding.png'), dpi=150)
    plt.close()

    # 2. Boundary sweeps
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'random': 'gray', '-MELBO_1 (walk)': 'green',
              'bnd_MELBO_1': 'red', 'bnd_MELBO_2': 'orange'}
    rand_median = np.median(np.median(bnd_sweep_results['random']['kl'], axis=2), axis=0)
    ax.plot(eval_angles_deg, rand_median, color='gray', lw=2, ls='--', label='random')
    for name in ['-MELBO_1 (walk)', 'bnd_MELBO_1', 'bnd_MELBO_2']:
        median = np.median(bnd_sweep_results[name]['kl'], axis=0)
        ax.plot(eval_angles_deg, median, color=colors[name], lw=2, label=name)
    ax.set_xlabel('Angle from boundary (deg)')
    ax.set_ylabel('KL divergence')
    ax.set_title(f'Boundary MELBO sweeps: En→Dutch, {args.model}, layer {layer}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'boundary_sweeps.png'), dpi=150)
    plt.close()

    # Save
    results = {
        'melbo_1': melbo_1,
        'walk_dir': walk_dir,
        'boundary_angles_deg': bnd_angles_array,
        'median_boundary_deg': med_bnd,
        'kl_boundary_sweep': kl_sweep,
        'sweep_angles_deg': sweep_angles_deg,
        'bnd_melbo_directions': bnd_melbo_dirs,
        'bnd_melbo_kls': bnd_melbo_kls,
        'bnd_melbo_histories': bnd_melbo_histories,
        'eval_angles_deg': eval_angles_deg,
        'bnd_sweep_results': bnd_sweep_results,
    }
    with open(os.path.join(args.results_dir, 'boundary_melbo_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved boundary MELBO results")

    return melbo_1, bnd_melbo_dirs, med_bnd


# =========================================================================
# Step 4: Steering with temperature
# =========================================================================

def _spherical_rotate(a, direction, cos_t, sin_t):
    """Spherical rotation: rotate a toward direction by angle."""
    R = a.norm(dim=-1, keepdim=True)
    a_unit = a / (R + 1e-10)
    proj = (a_unit * direction).sum(dim=-1, keepdim=True)
    v_perp = direction.unsqueeze(0) - proj * a_unit
    v_perp = v_perp / (v_perp.norm(dim=-1, keepdim=True) + 1e-10)
    return R * (cos_t * a_unit + sin_t * v_perp)


def generate_with_steering(model, tokenizer, prompt, steer_direction,
                           angle_deg, steer_layer, device, n_tokens=30,
                           temperature=0.7):
    """Generate tokens with spherical steering and temperature sampling."""
    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    direction = steer_direction.to(device)

    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    prompt_len = input_ids.shape[1]
    first_steer_pos = prompt_len - 1

    generated_ids = []

    for step in range(n_tokens):
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            steer_slice = hidden[0, first_steer_pos:, :]
            if steer_slice.shape[0] == 0:
                return output

            steered = _spherical_rotate(steer_slice, direction, cos_theta, sin_theta)
            hidden[0, first_steer_pos:, :] = steered

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        layers = _get_blocks(model)
        handle = layers[steer_layer].register_forward_hook(steering_hook)

        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)

        handle.remove()

        logits = outputs.logits[0, -1, :]

        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
        else:
            next_token_id = logits.argmax().item()

        generated_ids.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([
            input_ids,
            torch.tensor([[next_token_id]], device=device)
        ], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_with_compound_steering(model, tokenizer, prompt, steer_direction,
                                     angle_deg, steer_layer, device,
                                     pre_direction=None, pre_angle_deg=0,
                                     n_tokens=30, temperature=0.7):
    """Generate with compound spherical steering: first rotate pre_angle along
    pre_direction, then rotate angle_deg along steer_direction."""
    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    direction = steer_direction.to(device)

    has_pre = pre_direction is not None and pre_angle_deg > 0
    if has_pre:
        pre_dir = pre_direction.to(device)
        cos_pre = np.cos(np.radians(pre_angle_deg))
        sin_pre = np.sin(np.radians(pre_angle_deg))

    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    prompt_len = input_ids.shape[1]
    first_steer_pos = prompt_len - 1

    generated_ids = []

    for step in range(n_tokens):
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            steer_slice = hidden[0, first_steer_pos:, :]
            if steer_slice.shape[0] == 0:
                return output

            # Pre-rotation (walk to boundary)
            if has_pre:
                steer_slice = _spherical_rotate(steer_slice, pre_dir, cos_pre, sin_pre)
            # Main rotation (steer from boundary)
            steered = _spherical_rotate(steer_slice, direction, cos_theta, sin_theta)
            hidden[0, first_steer_pos:, :] = steered

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        layers = _get_blocks(model)
        handle = layers[steer_layer].register_forward_hook(steering_hook)

        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)

        handle.remove()

        logits = outputs.logits[0, -1, :]

        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
        else:
            next_token_id = logits.argmax().item()

        generated_ids.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([
            input_ids,
            torch.tensor([[next_token_id]], device=device)
        ], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_steering(model, tokenizer, device, layer, melbo_1, bnd_melbo_dirs,
                 med_bnd, args):
    """Steer with matched angle ranges per direction type.

    MELBO_1(->Dutch): 15-25 deg from English (where the perturbation response lives)
    bnd_MELBO_1/2(+): 0-10 deg from boundary (where boundary perturbation response lives)
    """
    prompts = ENGLISH_PROMPTS[:args.n_steer_prompts]
    melbo_angles = list(range(15, 26, 2))      # 15, 17, 19, 21, 23, 25
    bnd_angles = list(range(0, 11, 2))          # 0, 2, 4, 6, 8, 10

    walk_dir = -melbo_1  # toward Dutch

    print(f"\nSteering: {len(prompts)} prompts, temp={args.temperature}")
    print(f"MELBO_1(->Dutch) angles: {melbo_angles}")
    print(f"bnd_MELBO angles (from {med_bnd} deg boundary): {bnd_angles}")

    output_lines = [f"Temperature: {args.temperature}"]

    for p_idx, prompt in enumerate(prompts):
        header = f"\n{'='*70}\nPrompt [{p_idx}]: {prompt}\n{'='*70}"
        print(header)
        output_lines.append(header)

        # --- MELBO_1 (->Dutch) from English ---
        dir_header = f"\n  --- MELBO_1 (->Dutch), from English ---"
        print(dir_header)
        output_lines.append(dir_header)
        for angle in melbo_angles:
            text = generate_with_steering(
                model, tokenizer, prompt, -melbo_1, angle,
                layer, device, n_tokens=args.n_tokens,
                temperature=args.temperature)
            line = f"  {angle:5.1f} deg: {text[:150]}"
            print(line)
            output_lines.append(line)

        # --- bnd_MELBO from boundary ---
        for d_idx, bnd_dir in enumerate(bnd_melbo_dirs):
            dir_header = f"\n  --- bnd_MELBO_{d_idx+1} (+), from boundary ({med_bnd} deg) ---"
            print(dir_header)
            output_lines.append(dir_header)
            for angle in bnd_angles:
                text = generate_with_compound_steering(
                    model, tokenizer, prompt, bnd_dir, angle,
                    layer, device,
                    pre_direction=walk_dir, pre_angle_deg=med_bnd,
                    n_tokens=args.n_tokens, temperature=args.temperature)
                line = f"  {angle:5.1f} deg: {text[:150]}"
                print(line)
                output_lines.append(line)

    output_path = os.path.join(args.results_dir, 'steering_outputs.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"\nSaved steering outputs to {output_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--opt_angle', type=float, default=20.0)
    parser.add_argument('--boundary_angle', type=float, default=None,
                        help='Fixed boundary angle in deg (overrides automatic detection)')
    parser.add_argument('--n_act', type=int, default=50)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_directions', type=int, default=2)
    parser.add_argument('--n_steer_prompts', type=int, default=5)
    parser.add_argument('--n_tokens', type=int, default=30)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steer_only', action='store_true',
                        help='Skip MELBO, load saved results, and only run steering')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/boundary_melbo_dutch/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    if args.steer_only:
        # Load saved results
        print("Loading saved results (--steer_only mode)")
        melbo_path = os.path.join(args.results_dir, 'melbo_results.pkl')
        bnd_path = os.path.join(args.results_dir, 'boundary_melbo_results.pkl')
        with open(melbo_path, 'rb') as f:
            melbo_results = pickle.load(f)
        with open(bnd_path, 'rb') as f:
            bnd_results = pickle.load(f)
        melbo_1 = melbo_results['melbo_directions'][0].to(device)
        mean_dir = melbo_results['mean_dir'].to(device)
        # Orient English-ward
        if (melbo_1 @ mean_dir) < 0:
            melbo_1 = -melbo_1
        bnd_melbo_dirs = [d.to(device) for d in bnd_results['bnd_melbo_directions']]
        med_bnd = bnd_results['median_boundary_deg']
        print(f"  MELBO_1 loaded, boundary={med_bnd} deg")
        print(f"  bnd_MELBO KLs: {bnd_results['bnd_melbo_kls']}")
    else:
        # Step 1: Constrained MELBO
        print(f"\n{'='*70}")
        print("STEP 1: Constrained MELBO on En-Dutch differences")
        print(f"{'='*70}")

        D, mean_dir = extract_bilingual_diffs_dutch(model, tokenizer, device, args.layer)

        melbo_1, mean_dir, _, _, _, _ = run_constrained_melbo(
            model, tokenizer, device, args.layer, D, mean_dir,
            args.n_act, args.seq_len, args.n_directions, args)

        # Step 2 & 3: Boundary + Boundary MELBO
        print(f"\n{'='*70}")
        print("STEP 2-3: Walk to boundary + Boundary MELBO")
        print(f"{'='*70}")

        melbo_1, bnd_melbo_dirs, med_bnd = run_boundary_melbo(
            model, tokenizer, device, args.layer, melbo_1, mean_dir, args)

    # Step 4: Steering
    print(f"\n{'='*70}")
    print("STEP 4: Steering with temperature")
    print(f"{'='*70}")

    run_steering(model, tokenizer, device, args.layer, melbo_1, bnd_melbo_dirs,
                 med_bnd, args)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"MELBO_1: constrained direction in En-Dutch span")
    print(f"Boundary: median {med_bnd:.1f} deg along -MELBO_1 from English")
    for i, (d, kl) in enumerate(zip(bnd_melbo_dirs, [None, None])):
        cos = F.cosine_similarity(d.unsqueeze(0), melbo_1.unsqueeze(0)).item()
        print(f"bnd_MELBO_{i+1}: cos(MELBO_1)={cos:.4f}")
    print(f"\nAll results saved to {args.results_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
