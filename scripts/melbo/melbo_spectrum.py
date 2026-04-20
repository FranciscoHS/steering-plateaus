#!/usr/bin/env python3
"""
MELBO direction spectrum: how does plateau-breaking capacity decay across
successive orthogonal MELBO directions?

For each optimization angle θ in {1, 2, 3, 5, 10, 15}°:
  1. Find MELBO direction d_1 maximizing the objective at angle θ
  2. Find d_2, soft-orthogonal to d_1, maximizing objective at θ
  3. Continue until objective drops below threshold
  4. Record objective sequence → "MELBO spectrum" at angle θ

All angles are optimized interleaved (separate forward/backward per angle per step).

Supports two objectives:
  - kl: KL(ref || perturbed) at the logits (default)
  - l2: L2 distance of residual stream at a target layer (default: penultimate)

Operates on a single activation (--act_idx). Also computes:
  - Jacobian top singular values (logit Jacobian, via Lanczos)
  - Objective produced by perturbing along each Jacobian direction at 1°
  - Random baseline perturbation sweep

Usage:
    python scripts/melbo/melbo_spectrum.py --model qwen3-1.7b --layer 2 --act_idx 0
    python scripts/melbo/melbo_spectrum.py --model qwen3-1.7b --layer 2 --objective l2 --act_idx 0
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
    load_model, forward_from_layer, forward_from_layer_to_layer,
    _get_blocks, get_logits_at_point,
)
from src.data import load_fineweb_fixed_length
from src.analysis import compute_top_singular_values


# ── Helpers ──────────────────────────────────────────────────────────────

def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device,
                                   batch_size=64):
    """Extract layer-L context (prefix positions) and final-token activation."""
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


def perturb_last_token(activation, direction, angle_rad):
    """Spherical perturbation: rotate activation toward direction by angle_rad.

    Args:
        activation: [d_model] single activation vector
        direction: [d_model] unit direction (may carry gradients)
        angle_rad: perturbation angle in radians

    Returns: [1, 1, d_model] perturbed activation
    """
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    a_norm = activation.norm()
    a_unit = activation / a_norm
    proj = (direction * a_unit).sum()
    u_perp = direction - proj * a_unit
    u_perp = u_perp / (u_perp.norm() + 1e-10)
    a_pert = a_norm * (cos_t * a_unit + sin_t * u_perp)
    return a_pert.reshape(1, 1, -1)


def normalize_and_orthogonalize(direction, prev_directions):
    """Normalize direction and project orthogonal to prev_directions (differentiable)."""
    d = direction / direction.norm()
    if prev_directions:
        for pd in prev_directions:
            d = d - (d @ pd) * pd
        d = d / (d.norm() + 1e-10)
    return d


def make_objective_fn(model, context, activation, layer, objective, ref_data,
                      target_layer=None):
    """Create objective function closure.

    Returns fn(direction, angle_rad) -> differentiable scalar (higher = more effect).
    """
    if objective == 'kl':
        ref_log_probs = ref_data
        ref_probs = ref_log_probs.exp()

        def fn(direction, angle_rad):
            pt = perturb_last_token(activation, direction, angle_rad)
            logits = forward_from_layer(model, context, pt, layer)
            pert_lp = F.log_softmax(logits, dim=-1)
            return (ref_probs * (ref_log_probs - pert_lp)).sum(dim=-1).mean()
        return fn

    elif objective == 'l2':
        ref_target_act = ref_data

        def fn(direction, angle_rad):
            pt = perturb_last_token(activation, direction, angle_rad)
            target_act = forward_from_layer_to_layer(
                model, context, pt, layer, target_layer)
            return (target_act - ref_target_act).norm(dim=-1).mean()
        return fn

    else:
        raise ValueError(f"Unknown objective: {objective}")


# ── MELBO optimization ────────────────────────────────────────────────

def optimize_one_round_interleaved(objective_fn, active_angles, prev_directions_per_angle,
                                   device, d_model,
                                   n_steps=200, lr=0.01, n_restarts=3):
    """Find one direction per active angle, interleaving optimization steps.

    Each step does a separate forward+backward per angle (no retain_graph needed).
    All angles progress together rather than sequentially.

    Args:
        objective_fn: callable(direction, angle_rad) -> differentiable scalar
        active_angles: list of (angle_idx, angle_rad)
        prev_directions_per_angle: dict {angle_idx: [prev_dir_tensors]}

    Returns:
        dict {angle_idx: (best_value, best_direction)}
    """
    best_per_angle = {idx: (-float('inf'), None) for idx, _ in active_angles}

    for restart in range(n_restarts):
        dirs = {}
        for angle_idx, angle_rad in active_angles:
            d = torch.randn(d_model, device=device)
            prev = prev_directions_per_angle.get(angle_idx, [])
            for pd in prev:
                d = d - (d @ pd) * pd
            d = F.normalize(d, dim=0).clone().detach().requires_grad_(True)
            dirs[angle_idx] = d

        optimizers = {idx: torch.optim.Adam([dirs[idx]], lr=lr) for idx, _ in active_angles}

        for step in range(n_steps):
            for angle_idx, angle_rad in active_angles:
                optimizers[angle_idx].zero_grad()

                d_normed = normalize_and_orthogonalize(dirs[angle_idx],
                    prev_directions_per_angle.get(angle_idx, []))
                obj = objective_fn(d_normed, angle_rad)

                (-obj).backward()
                optimizers[angle_idx].step()

                prev = prev_directions_per_angle.get(angle_idx, [])
                if prev:
                    with torch.no_grad():
                        for pd in prev:
                            dirs[angle_idx].data -= (dirs[angle_idx].data @ pd) * pd

        # Evaluate final objective for this restart
        for angle_idx, angle_rad in active_angles:
            with torch.no_grad():
                prev = prev_directions_per_angle.get(angle_idx, [])
                d_final = normalize_and_orthogonalize(dirs[angle_idx].detach(), prev)
                obj = objective_fn(d_final, angle_rad).item()

            if obj > best_per_angle[angle_idx][0]:
                best_per_angle[angle_idx] = (obj, d_final.clone())

    return {idx: (val, d) for idx, (val, d) in best_per_angle.items()}


# ── Random baseline ─────────────────────────────────────────────────────

def random_direction_sweep(objective_fn, angles_rad, device, d_model, n_dirs=100):
    """Compute objective for n_dirs random directions at each angle.

    Returns: obj_array [n_angles, n_dirs]
    """
    n_angles = len(angles_rad)
    obj_array = np.zeros((n_angles, n_dirs))

    directions = torch.randn(n_dirs, d_model, device=device)
    directions = F.normalize(directions, dim=-1)

    for ang_idx, theta in enumerate(angles_rad):
        for dir_idx in range(n_dirs):
            with torch.no_grad():
                obj_array[ang_idx, dir_idx] = objective_fn(
                    directions[dir_idx], theta).item()
        print(f"  Random sweep: angle {np.degrees(theta):.1f}° done", end='\r')
    print()

    return obj_array


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--objective', choices=['kl', 'l2'], default='kl',
                        help='Optimization objective')
    parser.add_argument('--target_layer', type=int, default=None,
                        help='Target layer for L2 objective (default: penultimate)')
    parser.add_argument('--act_idx', type=int, default=0,
                        help='Index of single activation to optimize on')
    parser.add_argument('--n_act', type=int, default=50,
                        help='Number of FineWeb activations to load')
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--opt_angles', type=str, default='1,2,3,5,10,15',
                        help='Comma-separated MELBO optimization angles in degrees')
    parser.add_argument('--max_directions', type=int, default=50,
                        help='Max directions to find per angle')
    parser.add_argument('--floor_ratio', type=float, default=0.05,
                        help='Stop when objective drops below this fraction of direction 1')
    parser.add_argument('--random_floor_multiple', type=float, default=None,
                        help='If set, stop when objective drops below this multiple of '
                             'the random median (overrides floor_ratio)')
    parser.add_argument('--n_steps', type=int, default=200,
                        help='Optimization steps per direction')
    parser.add_argument('--n_restarts', type=int, default=3,
                        help='Random restarts per direction')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--k_jacobian', type=int, default=50,
                        help='Number of Jacobian singular values')
    parser.add_argument('--n_random_dirs', type=int, default=100,
                        help='Random directions for baseline')
    parser.add_argument('--n_sweep_angles', type=int, default=30,
                        help='Angle points for baseline sweep')
    parser.add_argument('--max_sweep_angle', type=float, default=60,
                        help='Max angle for baseline sweep (degrees)')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint.pkl in results_dir')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    opt_angles_deg = [float(a) for a in args.opt_angles.split(',')]

    # ===== Load model =====
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    print(f"Device: {device}, d_model: {d_model}, n_layers: {n_layers}")

    # Resolve target layer for L2
    if args.objective == 'l2':
        if args.target_layer is None:
            args.target_layer = n_layers - 2
        print(f"L2 objective: measuring at layer {args.target_layer} (of {n_layers})")

    # Results dir
    if args.results_dir is None:
        obj_tag = f'_{args.objective}' if args.objective != 'kl' else ''
        tgt_tag = f'_tgt{args.target_layer}' if args.objective == 'l2' else ''
        args.results_dir = (f'results/melbo_spectrum/'
                            f'{args.model}_layer{args.layer}{obj_tag}{tgt_tag}'
                            f'_act{args.act_idx}')
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Extract activations =====
    print(f"\nLoading {args.n_act} FineWeb sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    contexts_all, activations_all = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"All activations: {activations_all.shape}")

    assert args.act_idx < len(activations_all), \
        f"act_idx {args.act_idx} >= n_act {len(activations_all)}"
    context_dev = contexts_all[args.act_idx:args.act_idx+1].to(device)  # [1, seq-1, d]
    activation_dev = activations_all[args.act_idx].to(device)  # [d]
    print(f"Using activation {args.act_idx}, norm: {activation_dev.norm():.2f}")

    # ===== Compute reference =====
    with torch.no_grad():
        ref_point = activation_dev.reshape(1, 1, -1)
        if args.objective == 'kl':
            ref_logits = forward_from_layer(model, context_dev, ref_point, args.layer)
            ref_data = F.log_softmax(ref_logits, dim=-1)
        elif args.objective == 'l2':
            ref_data = forward_from_layer_to_layer(
                model, context_dev, ref_point, args.layer, args.target_layer)
            print(f"Reference at layer {args.target_layer}, norm: {ref_data.norm():.2f}")

    # Create objective function
    objective_fn = make_objective_fn(
        model, context_dev, activation_dev, args.layer,
        args.objective, ref_data, args.target_layer)
    obj_name = 'KL' if args.objective == 'kl' else 'L2'

    # ===== Check for resume =====
    resume_start = 0
    if args.resume:
        ckpt_path = os.path.join(args.results_dir, 'checkpoint.pkl')
        assert os.path.exists(ckpt_path), f"No checkpoint found at {ckpt_path}"
        print(f"\nResuming from {ckpt_path}")
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)

        sweep_angles_deg = ckpt['sweep_angles_deg']
        random_objs = ckpt['random_objs']
        print(f"Loaded random baseline: {random_objs.shape}")

        # Rebuild state from checkpoint
        n_angles = len(opt_angles_deg)
        angles_rad = [np.deg2rad(a) for a in opt_angles_deg]
        all_directions = {i: [] for i in range(n_angles)}
        all_objs = {i: [] for i in range(n_angles)}
        active = {i: True for i in range(n_angles)}
        thresholds = {}

        for i, ang_deg in enumerate(opt_angles_deg):
            if ang_deg in ckpt['melbo_results']:
                prev = ckpt['melbo_results'][ang_deg]
                all_objs[i] = list(prev['objs'])
                all_directions[i] = [d.to(device) for d in prev['directions']]
                resume_start = max(resume_start, len(all_objs[i]))
                print(f"  {ang_deg}°: restored {len(all_objs[i])} directions, "
                      f"last L2={all_objs[i][-1]:.1f}")

        # Precompute floors
        if args.random_floor_multiple is not None:
            for i in range(n_angles):
                ang_idx = np.argmin(np.abs(sweep_angles_deg - opt_angles_deg[i]))
                rand_med = np.median(random_objs[ang_idx])
                thresholds[i] = args.random_floor_multiple * rand_med
                print(f"  {opt_angles_deg[i]}°: floor = {args.random_floor_multiple}x × "
                      f"random median ({rand_med:.4f}) = {thresholds[i]:.4f}")
        elif resume_start > 0:
            # Reconstruct floor_ratio thresholds from first direction
            for i in range(n_angles):
                if all_objs[i]:
                    thresholds[i] = all_objs[i][0] * args.floor_ratio

        print(f"Resuming from direction {resume_start + 1}")

    # ===== Part 1: Random baseline sweep =====
    if not args.resume:
        print(f"\n{'='*60}")
        print(f"RANDOM BASELINE SWEEP: {args.n_random_dirs} dirs, {args.n_sweep_angles} angles")
        print(f"{'='*60}")

        sweep_angles_deg = np.linspace(0, args.max_sweep_angle, args.n_sweep_angles + 1)[1:]
        sweep_angles_rad = np.deg2rad(sweep_angles_deg)

        t0 = time.time()
        random_objs = random_direction_sweep(
            objective_fn, sweep_angles_rad, device, d_model, n_dirs=args.n_random_dirs)
        print(f"Random sweep done in {time.time()-t0:.0f}s")

        for ang_idx, ang_deg in enumerate(sweep_angles_deg):
            median = np.median(random_objs[ang_idx])
            q75 = np.percentile(random_objs[ang_idx], 75)
            q95 = np.percentile(random_objs[ang_idx], 95)
            if ang_deg in opt_angles_deg or ang_idx == len(sweep_angles_deg) - 1:
                print(f"  {ang_deg:5.1f}°: median={median:.6f}, 75th={q75:.6f}, 95th={q95:.6f}")

    # ===== Part 2: Iterative MELBO across all angles =====
    if not args.resume:
        print(f"\n{'='*60}")
        print(f"MELBO SPECTRUM [{obj_name}] (interleaved, {len(opt_angles_deg)} angles)")
        print(f"Angles: {opt_angles_deg}")
        print(f"{'='*60}")

        n_angles = len(opt_angles_deg)
        angles_rad = [np.deg2rad(a) for a in opt_angles_deg]

        all_directions = {i: [] for i in range(n_angles)}
        all_objs = {i: [] for i in range(n_angles)}
        active = {i: True for i in range(n_angles)}
        thresholds = {}

        # Precompute random median floors if using random_floor_multiple
        if args.random_floor_multiple is not None:
            for i in range(n_angles):
                ang_idx = np.argmin(np.abs(sweep_angles_deg - opt_angles_deg[i]))
                rand_med = np.median(random_objs[ang_idx])
                thresholds[i] = args.random_floor_multiple * rand_med
                print(f"  {opt_angles_deg[i]}°: floor = {args.random_floor_multiple}x × "
                      f"random median ({rand_med:.4f}) = {thresholds[i]:.4f}")

    for d_idx in range(resume_start, args.max_directions):
        active_angles = [(i, angles_rad[i]) for i in range(n_angles) if active[i]]
        if not active_angles:
            print(f"\nAll angles stopped at direction {d_idx}")
            break

        prev_dirs = {i: all_directions[i] for i in range(n_angles)}

        t0 = time.time()
        results = optimize_one_round_interleaved(
            objective_fn, active_angles, prev_dirs,
            device, d_model,
            n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts)
        elapsed = time.time() - t0

        status_parts = []
        for i, angle_rad in active_angles:
            obj_val, direction = results[i]
            all_directions[i].append(direction)
            all_objs[i].append(obj_val)

            # Set threshold on first direction if using floor_ratio mode
            if d_idx == 0 and args.random_floor_multiple is None:
                thresholds[i] = obj_val * args.floor_ratio

            if obj_val < thresholds.get(i, -float('inf')):
                active[i] = False
                status_parts.append(f"{opt_angles_deg[i]}°:STOP({obj_name}={obj_val:.6f})")
            else:
                ang_idx = np.argmin(np.abs(sweep_angles_deg - opt_angles_deg[i]))
                rand_med = np.median(random_objs[ang_idx])
                ratio = obj_val / rand_med if rand_med > 0 else float('inf')
                status_parts.append(f"{opt_angles_deg[i]}°:{obj_val:.5f}({ratio:.0f}x)")

        print(f"  d_{d_idx+1} [{elapsed:.1f}s] | {' | '.join(status_parts)}")

        if (d_idx + 1) % 5 == 0 or not any(active.values()):
            ckpt = {
                'args': vars(args),
                'sweep_angles_deg': sweep_angles_deg,
                'random_objs': random_objs,
                'melbo_results': {opt_angles_deg[i]: {
                    'objs': all_objs[i],
                    'directions': [d.cpu() for d in all_directions[i]],
                } for i in range(n_angles)},
            }
            with open(os.path.join(args.results_dir, 'checkpoint.pkl'), 'wb') as f:
                pickle.dump(ckpt, f)
            n_active = sum(active.values())
            print(f"  [checkpoint saved, {n_active}/{n_angles} angles still active]")

    melbo_results = {}
    for i in range(n_angles):
        melbo_results[opt_angles_deg[i]] = {
            'objs': all_objs[i],
            'directions': [d.cpu() for d in all_directions[i]],
        }

    print(f"\nMELBO summary:")
    for i in range(n_angles):
        print(f"  {opt_angles_deg[i]:5.1f}°: {len(all_objs[i])} directions, "
              f"top {obj_name}={all_objs[i][0]:.6f}, last {obj_name}={all_objs[i][-1]:.6f}")

    # ===== Part 3: Jacobian spectrum =====
    print(f"\n{'='*60}")
    print(f"JACOBIAN SPECTRUM (logit Jacobian): top-{args.k_jacobian} singular values")
    print(f"{'='*60}")

    point_jac = activation_dev.reshape(1, 1, -1)
    k_actual = min(args.k_jacobian, d_model - 2)
    t0 = time.time()
    svs, eigvecs = compute_top_singular_values(
        model, context_dev, point_jac, layer=args.layer, k=k_actual)
    print(f"  top-5 SVs: {svs[:5].round(1)} [{time.time()-t0:.0f}s]")

    # ===== Part 4: Objective of Jacobian directions at smallest opt angle =====
    jac_eval_deg = min(opt_angles_deg)
    jac_eval_rad = np.deg2rad(jac_eval_deg)
    print(f"\nComputing {obj_name} for Jacobian directions at {jac_eval_deg}°...")
    jacobian_objs_at_1deg = []

    jac_dirs = torch.tensor(eigvecs, device=device, dtype=torch.float32)
    k_eval = min(args.k_jacobian, jac_dirs.shape[0])

    for k in range(k_eval):
        d_k = jac_dirs[k] / jac_dirs[k].norm()
        with torch.no_grad():
            val = objective_fn(d_k, jac_eval_rad).item()
        jacobian_objs_at_1deg.append(val)
        if k < 10 or k % 10 == 0:
            print(f"  Jac dir {k+1}: {obj_name}={val:.6f}")

    # ===== Save final results =====
    results = {
        'args': vars(args),
        'opt_angles_deg': opt_angles_deg,
        'sweep_angles_deg': sweep_angles_deg,
        'random_objs': random_objs,
        'melbo_results': melbo_results,
        'jacobian_svs': svs,
        'jacobian_eigvecs': eigvecs,
        'jacobian_objs_at_1deg': jacobian_objs_at_1deg,
        'jac_eval_deg': jac_eval_deg,
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")

    plot_results(results, args)
    print("\nDone!")


def plot_results(results, args):
    """Generate all plots from results dict."""
    out = args.results_dir
    obj_name = 'KL' if args.objective == 'kl' else 'L2'
    opt_angles_deg = results['opt_angles_deg']
    melbo_results = results['melbo_results']
    svs = results['jacobian_svs']
    jacobian_objs = results['jacobian_objs_at_1deg']
    random_objs = results['random_objs']
    sweep_angles_deg = results['sweep_angles_deg']

    tgt_tag = f' → L{args.target_layer}' if args.objective == 'l2' else ''
    subtitle = f'{args.model}, layer {args.layer}{tgt_tag}, act {args.act_idx}, obj: {obj_name}'

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(opt_angles_deg) - 1, 1)) for i in range(len(opt_angles_deg))]

    # ── Plot 1: MELBO spectra normalized ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (opt_angle, color) in enumerate(zip(opt_angles_deg, colors)):
        objs = melbo_results[opt_angle]['objs']
        if objs[0] > 0:
            objs_norm = [o / objs[0] for o in objs]
        else:
            objs_norm = objs
        x = np.arange(1, len(objs_norm) + 1)
        ax.plot(x, objs_norm, 'o-', color=color, label=f'{opt_angle}°',
                markersize=4, linewidth=1.5)
    ax.set_xlabel('Direction index')
    ax.set_ylabel(f'{obj_name} / {obj_name}(d₁)  [normalized]')
    ax.set_title(f'MELBO spectrum (normalized)\n{subtitle}')
    ax.legend(title='Opt angle')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'melbo_spectrum_normalized.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: melbo_spectrum_normalized.png")

    # ── Plot 2: MELBO spectra raw ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (opt_angle, color) in enumerate(zip(opt_angles_deg, colors)):
        objs = melbo_results[opt_angle]['objs']
        x = np.arange(1, len(objs) + 1)
        ax.plot(x, objs, 'o-', color=color, label=f'{opt_angle}°',
                markersize=4, linewidth=1.5)
        ang_idx = np.argmin(np.abs(sweep_angles_deg - opt_angle))
        random_median = np.median(random_objs[ang_idx])
        ax.axhline(random_median, color=color, ls=':', alpha=0.4, linewidth=1)
    ax.set_xlabel('Direction index')
    ax.set_ylabel(f'{obj_name}')
    ax.set_title(f'MELBO spectrum (raw)\n{subtitle}')
    ax.legend(title='Opt angle')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'melbo_spectrum_raw.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: melbo_spectrum_raw.png")

    # ── Plot 3: Jacobian vs MELBO at eval angle ──
    jac_eval_deg = results.get('jac_eval_deg', min(opt_angles_deg))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    k = len(svs)
    ax1.plot(np.arange(1, k + 1), svs, 'k-', linewidth=1.5)
    ax1.set_xlabel('Singular value index')
    ax1.set_ylabel('Singular value')
    ax1.set_title('Jacobian SV spectrum (logit map)')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Find MELBO results at the eval angle
    melbo_objs_eval = []
    for key in [jac_eval_deg, int(jac_eval_deg)]:
        if key in melbo_results:
            melbo_objs_eval = melbo_results[key]['objs']
            break

    n_jac = len(jacobian_objs)
    n_melbo = len(melbo_objs_eval)

    if n_jac > 0:
        ax2.plot(np.arange(1, n_jac + 1), jacobian_objs, 's-',
                 color='red', label=f'Jacobian dirs @ {jac_eval_deg}°',
                 markersize=3, linewidth=1.5)
    if n_melbo > 0:
        ax2.plot(np.arange(1, n_melbo + 1), melbo_objs_eval, 'o-',
                 color='blue', label=f'MELBO dirs @ {jac_eval_deg}°',
                 markersize=4, linewidth=1.5)

    ang_idx = np.argmin(np.abs(sweep_angles_deg - jac_eval_deg))
    random_median = np.median(random_objs[ang_idx])
    ax2.axhline(random_median, color='gray', ls='--',
                label=f'Random median @ {jac_eval_deg}°')

    ax2.set_xlabel('Direction index')
    ax2.set_ylabel(f'{obj_name}')
    ax2.set_title(f'Jacobian vs MELBO directions at {jac_eval_deg}°')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if max(n_jac, n_melbo) > 0:
        ax2.set_yscale('log')

    fig.suptitle(subtitle, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'jacobian_vs_melbo.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: jacobian_vs_melbo.png")

    # ── Plot 4: Random baseline ──
    fig, ax = plt.subplots(figsize=(8, 5))
    median = np.median(random_objs, axis=1)
    q25 = np.percentile(random_objs, 25, axis=1)
    q75 = np.percentile(random_objs, 75, axis=1)
    q95 = np.percentile(random_objs, 95, axis=1)
    ax.plot(sweep_angles_deg, median, 'b-', label='Median', linewidth=2)
    ax.fill_between(sweep_angles_deg, q25, q75, alpha=0.2, color='blue', label='IQR')
    ax.plot(sweep_angles_deg, q95, 'b--', alpha=0.5, label='95th pctile')
    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel(f'{obj_name}')
    ax.set_title(f'Random direction baseline\n{subtitle}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'random_baseline.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: random_baseline.png")

    # ── Plot 5: Number of directions found per angle ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (opt_angle, color) in enumerate(zip(opt_angles_deg, colors)):
        n_dirs = len(melbo_results[opt_angle]['objs'])
        ax.bar(idx, n_dirs, color=color, edgecolor='black', linewidth=0.5)
        ax.text(idx, n_dirs + 0.3, str(n_dirs), ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(opt_angles_deg)))
    ax.set_xticklabels([f'{a}°' for a in opt_angles_deg])
    ax.set_xlabel('Optimization angle')
    ax.set_ylabel('Number of directions found')
    ax.set_title(f'Directions before {obj_name} drops below {args.floor_ratio:.0%} of d₁\n'
                 f'{subtitle}')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'n_directions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: n_directions.png")


if __name__ == '__main__':
    main()
