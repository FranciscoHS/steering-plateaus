#!/usr/bin/env python3
"""
Response-DIRECTION perturbation sweep.

Standard perturbation sweeps in this repo perturb an activation at layer L along a
direction d and measure the *magnitude* of the downstream effect (L2 distance at a
later layer). Here we instead track the *direction* of the downstream response and
ask: as we increase the perturbation, does the response direction stay fixed, or
does it rotate?

Two perturbation modes:
  - rotate (default): norm-preserving rotation of the activation toward d, by angle
        theta, in the plane spanned by a and d:
            a(theta) = ||a|| * (cos(theta) * a_hat + sin(theta) * v_perp)
        where v_perp = (d - (d.a_hat) a_hat) normalized. Swept over theta (degrees).
  - additive: a + eps * d, eps relative to ||a|| (log-spaced).

Downstream response vector at a capture layer:
    r(s) = resid_capture(perturb(a, s)) - resid_capture(a)
Baseline = response direction at the smallest perturbation (s = sweep_vals[0]):
    b = r(s_min) / ||r(s_min)||
Tracked vs s:
    cos(s)  = <r(s), b> / ||r(s)||      (direction stability)
    ||r(s)||                             (magnitude, recorded for free)

cos ~ 1 -> response keeps pointing the same way as we push harder;
a drop    -> the response rotates (nonlinearity / saturation onset).

Conditions:
  - contrastive: named diff-of-means directions (10 languages + 5 programming langs)
                 at L14, the SAME directions applied to every activation.
  - random:      n_dirs isotropic unit vectors sampled per activation.

Usage:
    python scripts/perturbation/run_response_direction_sweep.py --model qwen3-1.7b --layer 14
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
    load_model, forward_from_layer_to_final, forward_from_layer_to_layer,
)
from src.data import load_fineweb_fixed_length
from scripts.perturbation.run_perturbation_sweep import (
    extract_context_and_activation, save_checkpoint, load_checkpoint,
)


# Default L14 qwen3-1.7b contrastive direction sources (name-prefixed by family).
DEFAULT_CONTRASTIVE_PKLS = [
    ('results/allpairs_contrastive/qwen3-1.7b/directions.pkl', 'lang'),
    ('results/programming_contrastive/qwen3-1.7b/directions.pkl', 'prog'),
]


def load_contrastive_directions(pkl_specs, layer):
    """Load named diff-of-means directions from one or more pkls.

    Returns (dirs [n, d] unit-norm float, names list[str]).
    Each pkl is expected to have a 'features' dict {name: vec[d]} and a 'layer'.
    """
    vecs, names = [], []
    for path, family in pkl_specs:
        d = pickle.load(open(path, 'rb'))
        if d.get('layer') != layer:
            print(f"  WARNING: {path} is layer {d.get('layer')}, expected {layer}")
        for name, v in d['features'].items():
            v = v.float()
            v = v / v.norm()
            vecs.append(v)
            names.append(f'{family}:{name}')
    dirs = torch.stack(vecs)  # [n, d]
    return dirs, names


def build_perturbations(act, dirs, sweep_vals, mode):
    """Build perturbed activations for every sweep value, batched over directions.

    Args:
        act: [d_model] activation (on device)
        dirs: [n_dirs, d_model] unit directions (on device)
        sweep_vals: list/array of sweep values
            - mode 'rotate':   angles in radians
            - mode 'additive': absolute magnitudes (eps)
        mode: 'rotate' or 'additive'
    Returns:
        list (len n_sweep) of [n_dirs, d_model] perturbed activations
    """
    if mode == 'additive':
        return [(act.unsqueeze(0) + s * dirs) for s in sweep_vals]

    # rotate: rotate a_hat toward each direction's orthogonal component
    R = act.norm()
    a_hat = act / R.clamp_min(1e-12)                       # [d]
    proj = dirs @ a_hat                                    # [n_dirs]
    d_perp = dirs - proj.unsqueeze(1) * a_hat.unsqueeze(0)  # [n_dirs, d]
    v_perp = F.normalize(d_perp, dim=-1)                   # [n_dirs, d]
    out = []
    for theta in sweep_vals:
        a_rot = R * (np.cos(theta) * a_hat.unsqueeze(0)
                     + np.sin(theta) * v_perp)             # [n_dirs, d]
        out.append(a_rot)
    return out


def run_response_sweep(model, context, activation, directions, sweep_vals,
                       mode, layer, capture_layer, device, baseline_idxs):
    """Response-direction sweep for a single activation over a set of directions.

    Computes the cosine of the response r(s) against the response at each of several
    reference sweep points (baseline_idxs), so we can ask how much the stability curve
    depends on where we anchor the baseline.

    Returns:
        cos_sim:   [n_baselines, n_sweep, n_dirs]  cosine of r(s) vs each baseline
        resp_norm: [n_sweep, n_dirs]               ||r(s)||
    """
    n_sweep, n_dirs = len(sweep_vals), directions.shape[0]
    n_base = len(baseline_idxs)
    cos_sim = np.zeros((n_base, n_sweep, n_dirs))

    ctx = context.unsqueeze(0).to(device)   # [1, seq_len-1, d_model]
    act = activation.to(device)             # [d_model]
    dirs = directions.to(device)            # [n_dirs, d_model]

    def capture(point):
        if capture_layer is None:
            return forward_from_layer_to_final(model, ctx, point, layer)
        return forward_from_layer_to_layer(model, ctx, point, layer, capture_layer)

    with torch.no_grad():
        # Reference (unperturbed) downstream activation.
        ref_cap = capture(act.unsqueeze(0).unsqueeze(0))      # [1, d_model]

        perturbed = build_perturbations(act, dirs, sweep_vals, mode)  # list of [n_dirs, d]
        responses = [capture(p.unsqueeze(1)) - ref_cap for p in perturbed]  # list [n_dirs, d]
        R = torch.stack(responses, dim=0)                     # [n_sweep, n_dirs, d]
        rn = R.norm(dim=-1)                                   # [n_sweep, n_dirs]
        resp_norm = rn.cpu().numpy()

        for b, bi in enumerate(baseline_idxs):
            base = F.normalize(R[bi], dim=-1)                 # [n_dirs, d]
            cos = (R * base.unsqueeze(0)).sum(dim=-1) / rn.clamp_min(1e-12)  # [n_sweep, n_dirs]
            cos_sim[b] = cos.cpu().numpy()

    return cos_sim, resp_norm


def plot_baseline_overlay(xvals, xlabel, xscale, baseline_x, results, model_name,
                          layer, capture_layer, mode, save_dir):
    """Overlay cosine-stability curves anchored at different baseline reference points.

    One panel per condition (contrastive | random); one line per baseline angle.
    Each line peaks at 1.0 at its own reference point.
    """
    cap_str = 'final' if capture_layer is None else f'L{capture_layer}'
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    cmap = plt.get_cmap('viridis')
    n_base = len(baseline_x)

    for ax, cond in zip(axes, ['contrastive', 'random']):
        cos = results[cond]['cos_sim']  # [n_act, n_baselines, n_sweep, n_dirs]
        for b in range(n_base):
            flat = cos[:, b].transpose(1, 0, 2).reshape(cos.shape[2], -1)  # [n_sweep, n_act*n_dirs]
            med = np.median(flat, axis=1)
            color = cmap(b / max(n_base - 1, 1))
            ax.plot(xvals, med, color=color, linewidth=2,
                    label=f'baseline @ {baseline_x[b]:g}°')
            ax.axvline(baseline_x[b], color=color, linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        ax.set_title(cond)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('cos(response(s), response@baseline)')

    fig.suptitle(f'{model_name}, {mode} L{layer} -> {cap_str}: stability vs baseline anchor '
                 f'(median over activations × directions)', fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, 'response_baseline_overlay.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {path}")


def main():
    p = argparse.ArgumentParser(description='Response-direction perturbation sweep')
    p.add_argument('--model', type=str, default='qwen3-1.7b')
    p.add_argument('--layer', type=int, default=14, help='injection layer')
    p.add_argument('--capture_layer', type=int, default=None,
                   help='layer to read response at (default: final block)')
    p.add_argument('--perturb_mode', type=str, default='rotate',
                   choices=['rotate', 'additive'])
    p.add_argument('--seq_len', type=int, default=5)
    p.add_argument('--n_act', type=int, default=50)
    p.add_argument('--n_dirs', type=int, default=10, help='random directions per activation')
    # rotate-mode sweep (linear angle, degrees)
    p.add_argument('--max_angle', type=float, default=60.0, help='rotate: max angle (deg)')
    p.add_argument('--n_angles', type=int, default=60, help='rotate: number of angle steps')
    p.add_argument('--baseline_angles', type=str, default='1,10,15,20',
                   help='rotate: comma-separated reference angles (deg) to anchor cosine baseline')
    # additive-mode sweep (log magnitude, relative to ||a||)
    p.add_argument('--n_mags', type=int, default=30)
    p.add_argument('--mag_min', type=float, default=1e-3, help='additive: relative to ||a||')
    p.add_argument('--mag_max', type=float, default=8.0, help='additive: relative to ||a||')
    p.add_argument('--results_dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/response_direction_sweep/{args.model}_L{args.layer}_{args.perturb_mode}'
    os.makedirs(args.results_dir, exist_ok=True)

    print("Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-" * 60)

    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    # Contrastive directions (same set applied to every activation).
    print("\nLoading contrastive directions...")
    contrastive_dirs, dir_names = load_contrastive_directions(DEFAULT_CONTRASTIVE_PKLS, args.layer)
    print(f"  {len(dir_names)} contrastive directions: {dir_names}")

    # Activations from FineWeb.
    print(f"\nExtracting {args.n_act} layer-{args.layer} activations (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Contexts {tuple(contexts.shape)}, activations {tuple(activations.shape)}")

    act_norm_mean = activations.norm(dim=-1).mean().item()
    print(f"Mean activation norm: {act_norm_mean:.2f}")

    # Build the sweep axis.
    if args.perturb_mode == 'rotate':
        xvals = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]  # exclude 0 (zero response)
        sweep_vals = np.deg2rad(xvals)
        xlabel, xscale = 'Rotation angle (degrees)', 'linear'
    else:
        xvals = np.logspace(np.log10(args.mag_min), np.log10(args.mag_max), args.n_mags)
        sweep_vals = xvals * act_norm_mean  # absolute eps
        xlabel, xscale = 'Perturbation magnitude (relative to ||a||)', 'log'
    n_sweep = len(sweep_vals)

    # Resolve baseline reference points to nearest sweep indices.
    req_angles = [float(x) for x in args.baseline_angles.split(',')]
    baseline_idxs = [int(np.argmin(np.abs(xvals - a))) for a in req_angles]
    baseline_x = [xvals[i] for i in baseline_idxs]
    n_base = len(baseline_idxs)
    print(f"Baseline anchors (deg, nearest sweep point): {baseline_x}")

    # Random directions, sampled per activation (isotropic unit).
    rand_dirs = torch.randn(args.n_act, args.n_dirs, d_model)
    rand_dirs = F.normalize(rand_dirs, dim=-1)

    results = {
        'contrastive': {'cos_sim': np.zeros((args.n_act, n_base, n_sweep, len(dir_names))),
                        'resp_norm': np.zeros((args.n_act, n_sweep, len(dir_names)))},
        'random': {'cos_sim': np.zeros((args.n_act, n_base, n_sweep, args.n_dirs)),
                   'resp_norm': np.zeros((args.n_act, n_sweep, args.n_dirs))},
    }

    print(f"\n=== Sweeping {args.n_act} activations ({args.perturb_mode}, {n_sweep} steps) ===")
    t0 = time.time()
    for i in range(args.n_act):
        c, rn = run_response_sweep(model, contexts[i], activations[i], contrastive_dirs,
                                   sweep_vals, args.perturb_mode, args.layer,
                                   args.capture_layer, device, baseline_idxs)
        results['contrastive']['cos_sim'][i] = c
        results['contrastive']['resp_norm'][i] = rn

        c, rn = run_response_sweep(model, contexts[i], activations[i], rand_dirs[i],
                                   sweep_vals, args.perturb_mode, args.layer,
                                   args.capture_layer, device, baseline_idxs)
        results['random']['cos_sim'][i] = c
        results['random']['resp_norm'][i] = rn

        if (i + 1) % 5 == 0 or i == args.n_act - 1:
            el = time.time() - t0
            print(f"  {i+1}/{args.n_act}  ({el:.1f}s, {(i+1)/el:.2f} act/s)")

    out = {
        'args': vars(args), 'dir_names': dir_names,
        'xvals': xvals, 'xlabel': xlabel, 'mode': args.perturb_mode,
        'baseline_idxs': baseline_idxs, 'baseline_x': baseline_x,
        'act_norm_mean': act_norm_mean, 'results': results,
    }
    save_checkpoint(out, os.path.join(args.results_dir, 'results.pkl'))
    plot_baseline_overlay(xvals, xlabel, xscale, baseline_x, results, args.model,
                          args.layer, args.capture_layer, args.perturb_mode, args.results_dir)
    print(f"\nDone in {time.time()-t0:.1f}s. Results in {args.results_dir}")


if __name__ == '__main__':
    main()
