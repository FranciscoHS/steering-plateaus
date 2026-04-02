#!/usr/bin/env python3
"""Progressive context corruption: does corrupting context shrink the plateau?

Takes sequences of length 10. Always perturbs only the last position.
Progressively replaces context positions (from pos 8 backwards) with
cov-matched random vectors. Plots all corruption levels together to
see whether the plateau shrinks as context becomes less coherent.

Corruption levels:
  k=0: all real context (standard sweep)
  k=1: pos 8 replaced with random
  k=2: pos 7-8 replaced with random
  ...
  k=8: pos 1-8 replaced with random (only pos 0 is real)

Also includes a "random activation" baseline (real context, random last-pos
activation) for reference.

Usage:
    python scripts/perturbation/sweep_progressive_context.py --model qwen3-1.7b --layer 2
"""
import argparse, os, sys, pickle, time
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from src.model import (
    load_model, forward_from_layer_to_final, forward_from_layer_to_layer,
    _get_final_norm, _get_lm_head, _get_blocks,
)
from src.data import load_fineweb_fixed_length, sample_random_activations_v2


def extract_all_positions(model, token_ids_list, layer, device, batch_size=64):
    """Extract hidden states at all positions for each sequence.

    Returns:
        contexts: [n, seq_len-1, d_model] (positions 0..seq_len-2)
        activations: [n, d_model] (last position)
        full_hidden: [n, seq_len, d_model] (all positions, for covariance fitting)
    """
    all_hidden = []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1]
        all_hidden.append(hidden.float().cpu())
    full_hidden = torch.cat(all_hidden, 0)
    contexts = full_hidden[:, :-1, :]  # [n, seq_len-1, d_model]
    activations = full_hidden[:, -1, :]  # [n, d_model]
    return contexts, activations, full_hidden


def run_sweep_with_context(model, context, activation, v_perps, angles_rad,
                           layer, device, n_dirs, measure_layer=None):
    """Standard spherical perturbation sweep on last position.

    Args:
        context: [seq_len-1, d_model] - context hidden states
        activation: [d_model] - last-position activation to perturb
        v_perps: [n_dirs, d_model] - orthogonal perturbation directions
        angles_rad: perturbation angles
        layer: injection layer
    """
    n_angles = len(angles_rad)
    l2_dists = np.zeros((n_angles, n_dirs))
    kl_divs = np.zeros((n_angles, n_dirs))
    cos_dists = np.zeros((n_angles, n_dirs))

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
        if measure_layer is not None:
            ref_measure = forward_from_layer_to_layer(model, ctx, ref_point, layer, measure_layer)
        else:
            ref_measure = ref_final

    for ang_idx, theta in enumerate(angles_rad):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        perturbed = R * (cos_t * a_unit.unsqueeze(0) + sin_t * vp)
        perturbed = perturbed.unsqueeze(1)

        with torch.no_grad():
            pert_final = forward_from_layer_to_final(model, ctx, perturbed, layer)

            if measure_layer is not None:
                pert_measure = forward_from_layer_to_layer(model, ctx, perturbed, layer, measure_layer)
            else:
                pert_measure = pert_final
            l2_dists[ang_idx] = (pert_measure - ref_measure).float().norm(dim=-1).cpu().numpy()
            cos_sim = F.cosine_similarity(pert_measure.float(), ref_measure.float(), dim=-1)
            cos_dists[ang_idx] = (1 - cos_sim).cpu().numpy()

            pert_ln = _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)
            pert_logits = _get_lm_head(model)(pert_ln)
            pert_log_probs = F.log_softmax(pert_logits.float(), dim=-1)
            ref_probs = ref_log_probs.exp()
            kl = (ref_probs * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            kl_divs[ang_idx] = kl.cpu().numpy()

    return l2_dists, kl_divs, cos_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_act', type=int, default=100)
    parser.add_argument('--n_dirs', type=int, default=10)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--max_angle', type=float, default=60)
    parser.add_argument('--measure_layer', type=int, default=None)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--corruption_levels', type=int, nargs='+', default=None,
                        help='Which corruption levels to run (default: all 0..seq_len-2)')
    parser.add_argument('--replot', type=str, default=None,
                        help='Path to existing results.pkl — just regenerate plots, no computation')
    args = parser.parse_args()

    if args.replot:
        with open(args.replot, 'rb') as f:
            results = pickle.load(f)
        results_dir = os.path.dirname(args.replot)
        angles_deg = results['angles_deg']
        act_norm_mean = results['act_norm_mean']
        corruption_levels = results['corruption_levels']
        results_by_k = results['by_k']
        random_baseline = results['random_baseline']
        saved_args = results.get('args', {})
        model_name = saved_args.get('model', 'unknown')
        layer = saved_args.get('layer', '?')
        seq_len = saved_args.get('seq_len', '?')
        max_angle = saved_args.get('max_angle', 60)
        measure_layer = saved_args.get('measure_layer', None)

        cmap = plt.cm.viridis
        norm = Normalize(vmin=0, vmax=max(corruption_levels))

        for plot_type in ['median', 'individual']:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            for metric_idx, (key, name) in enumerate([
                ('l2', 'L2 distance'), ('cos', 'Cosine distance'), ('kl', 'KL divergence')
            ]):
                ax = axes[metric_idx]
                for k in corruption_levels:
                    d = results_by_k[k][key]
                    per_act = np.median(d, axis=2)
                    color = cmap(norm(k))
                    label = f'k={k}'
                    if plot_type == 'median':
                        median = np.median(per_act, axis=0)
                        q25 = np.percentile(per_act, 25, axis=0)
                        q75 = np.percentile(per_act, 75, axis=0)
                        ax.plot(angles_deg, median, color=color, linewidth=2, label=label)
                        ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.08)
                    else:
                        np.random.seed(42)
                        chosen = np.random.choice(per_act.shape[0], min(5, per_act.shape[0]), replace=False)
                        for j, idx in enumerate(chosen):
                            ax.plot(angles_deg, per_act[idx], color=color, alpha=0.5,
                                    linewidth=1, label=label if j == 0 else None)
                rd = random_baseline[key]
                per_act_r = np.median(rd, axis=2)
                if plot_type == 'median':
                    median_r = np.median(per_act_r, axis=0)
                    ax.plot(angles_deg, median_r, color='red', linewidth=2,
                            linestyle='--', label='Random act')
                else:
                    np.random.seed(42)
                    chosen = np.random.choice(per_act_r.shape[0], min(5, per_act_r.shape[0]), replace=False)
                    for j, idx in enumerate(chosen):
                        ax.plot(angles_deg, per_act_r[idx], color='red', alpha=0.5,
                                linewidth=1, linestyle='--', label='Random act' if j == 0 else None)
                ax.set_xlabel('Perturbation angle (degrees)')
                ax.set_ylabel(name)
                ax.set_title(name)
                ax.legend(fontsize=7, ncol=2)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, max_angle)
                ax2 = ax.twiny()
                ax2.set_xlim(ax.get_xlim())
                tick_angles = [a for a in [0, 10, 20, 30, 40, 60] if a <= max_angle]
                tick_chords = [2 * act_norm_mean * np.sin(np.deg2rad(a) / 2) for a in tick_angles]
                ax2.set_xticks(tick_angles)
                ax2.set_xticklabels([f'{c:.0f}' for c in tick_chords])
                ax2.set_xlabel('Chord distance')
            measure_label = f", L2@layer {measure_layer}" if measure_layer is not None else ""
            fig.suptitle(
                f'{model_name} L{layer} seq{seq_len}: progressive context corruption ({plot_type}{measure_label})\n'
                f'k = # context positions replaced with cov-matched random (pos 0 always real)',
                fontsize=12, y=1.05)
            plt.tight_layout()
            path = os.path.join(results_dir, f'sweep_{plot_type}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {path}")

        # Separate plot: real activation vs random activation (both with real context)
        for plot_type in ['median', 'individual']:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            conditions = [
                (results_by_k[0], 'Real activation', 'blue'),
                (random_baseline, 'Random activation (cov-matched)', 'red'),
            ]
            for metric_idx, (key, name) in enumerate([
                ('l2', 'L2 distance'), ('cos', 'Cosine distance'), ('kl', 'KL divergence')
            ]):
                ax = axes[metric_idx]
                for data, label, color in conditions:
                    d = data[key]
                    per_act = np.median(d, axis=2)
                    if plot_type == 'median':
                        median = np.median(per_act, axis=0)
                        q25 = np.percentile(per_act, 25, axis=0)
                        q75 = np.percentile(per_act, 75, axis=0)
                        ax.plot(angles_deg, median, color=color, linewidth=2, label=label)
                        ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.15)
                    else:
                        np.random.seed(42)
                        chosen = np.random.choice(per_act.shape[0], min(10, per_act.shape[0]), replace=False)
                        for j, idx in enumerate(chosen):
                            ax.plot(angles_deg, per_act[idx], color=color, alpha=0.4,
                                    linewidth=1.5, label=label if j == 0 else None)
                ax.set_xlabel('Perturbation angle (degrees)')
                ax.set_ylabel(name)
                ax.set_title(name)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, max_angle)
                ax2 = ax.twiny()
                ax2.set_xlim(ax.get_xlim())
                tick_angles = [a for a in [0, 10, 20, 30, 40, 60] if a <= max_angle]
                tick_chords = [2 * act_norm_mean * np.sin(np.deg2rad(a) / 2) for a in tick_angles]
                ax2.set_xticks(tick_angles)
                ax2.set_xticklabels([f'{c:.0f}' for c in tick_chords])
                ax2.set_xlabel('Chord distance')
            fig.suptitle(
                f'{model_name} L{layer} seq{seq_len}: real vs random last-pos activation ({plot_type})\n'
                f'Both with full real context',
                fontsize=12, y=1.05)
            plt.tight_layout()
            path = os.path.join(results_dir, f'real_vs_random_{plot_type}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {path}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/progressive_context/{args.model}_L{args.layer}_seq{args.seq_len}'
    os.makedirs(args.results_dir, exist_ok=True)

    n_context = args.seq_len - 1  # number of context positions
    if args.corruption_levels is None:
        corruption_levels = list(range(n_context))  # 0, 1, ..., seq_len-2
    else:
        corruption_levels = args.corruption_levels
    print(f"Corruption levels: {corruption_levels}")

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    print(f"Config: model={args.model}, layer={args.layer}, seq_len={args.seq_len}, "
          f"n_act={args.n_act}, n_dirs={args.n_dirs}, max_angle={args.max_angle}")

    # Load model
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    if args.measure_layer is not None and args.measure_layer < 0:
        args.measure_layer = n_layers + args.measure_layer
    print(f"d_model={d_model}, n_layers={n_layers}")

    # Load data and extract hidden states
    n_fit = 2000
    n_load = max(args.n_act, n_fit)
    token_ids_list = load_fineweb_fixed_length(n_load, tokenizer,
                                                seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.layer} activations...")
    all_contexts, all_activations, all_full = extract_all_positions(
        model, token_ids_list, args.layer, device
    )
    contexts = all_contexts[:args.n_act]
    activations = all_activations[:args.n_act]
    fit_full = all_full[:n_fit]

    act_norm_mean = activations.norm(dim=-1).mean().item()
    print(f"Mean last-pos activation norm: {act_norm_mean:.2f}")

    # Fit per-position covariance for random replacements
    print(f"\nFitting per-position covariance (n_fit={n_fit})...")
    per_pos_random = {}  # pos -> [n_act, d_model] random samples
    per_pos_norm = {}
    for p in range(n_context):
        pos_acts = fit_full[:, p, :]  # [n_fit, d_model]
        pos_norm = pos_acts.norm(dim=-1).mean().item()
        per_pos_norm[p] = pos_norm
        rand_pos, _ = sample_random_activations_v2(
            pos_acts, args.n_act, mode='full', seed=args.seed + p, device=device
        )
        per_pos_random[p] = F.normalize(rand_pos, dim=-1) * pos_norm
        print(f"  pos {p}: norm={pos_norm:.2f}")

    # Also generate random last-position activations for the baseline
    last_pos_acts = fit_full[:, -1, :]  # [n_fit, d_model]
    rand_last, _ = sample_random_activations_v2(
        last_pos_acts, args.n_act, mode='full', seed=args.seed + n_context, device=device
    )
    rand_last = F.normalize(rand_last, dim=-1) * act_norm_mean

    # Sample orthogonal directions for real activations (reuse across corruption levels)
    print("Sampling orthogonal directions...")
    real_v_perps = torch.zeros(args.n_act, args.n_dirs, d_model)
    for i in range(args.n_act):
        a = activations[i]
        a_unit = a / a.norm()
        v = torch.randn(args.n_dirs, d_model)
        v = v - (v @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        real_v_perps[i] = F.normalize(v, dim=-1)

    # Orthogonal directions for random baseline
    rand_v_perps = torch.zeros(args.n_act, args.n_dirs, d_model)
    for i in range(args.n_act):
        a = rand_last[i]
        a_unit = a / a.norm()
        v = torch.randn(args.n_dirs, d_model)
        v = v - (v @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        rand_v_perps[i] = F.normalize(v, dim=-1)

    # Run sweeps for each corruption level
    results_by_k = {}
    for k in corruption_levels:
        print(f"\n=== Corruption level k={k} (replacing {k} context positions) ===")
        k_l2 = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
        k_kl = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
        k_cos = np.zeros((args.n_act, len(angles_deg), args.n_dirs))

        t0 = time.time()
        for i in range(args.n_act):
            # Build corrupted context
            ctx = contexts[i].clone()  # [n_context, d_model]
            if k > 0:
                # Replace positions (n_context - k) through (n_context - 1) with random
                for p in range(n_context - k, n_context):
                    ctx[p] = per_pos_random[p][i]

            k_l2[i], k_kl[i], k_cos[i] = run_sweep_with_context(
                model, ctx, activations[i], real_v_perps[i],
                angles_rad, args.layer, device, args.n_dirs,
                measure_layer=args.measure_layer
            )
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
        print(f"k={k} done in {time.time()-t0:.1f}s")
        results_by_k[k] = {'l2': k_l2, 'kl': k_kl, 'cos': k_cos}

    # Random activation baseline (real context, random last-position activation)
    print(f"\n=== Random activation baseline ===")
    rand_l2 = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
    rand_kl = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
    rand_cos = np.zeros((args.n_act, len(angles_deg), args.n_dirs))

    t0 = time.time()
    for i in range(args.n_act):
        rand_l2[i], rand_kl[i], rand_cos[i] = run_sweep_with_context(
            model, contexts[i], rand_last[i], rand_v_perps[i],
            angles_rad, args.layer, device, args.n_dirs,
            measure_layer=args.measure_layer
        )
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
    print(f"Random baseline done in {time.time()-t0:.1f}s")
    random_baseline = {'l2': rand_l2, 'kl': rand_kl, 'cos': rand_cos}

    # Save results
    results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'act_norm_mean': act_norm_mean,
        'corruption_levels': corruption_levels,
        'by_k': results_by_k,
        'random_baseline': random_baseline,
    }
    pkl_path = os.path.join(args.results_dir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # Plot
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=max(corruption_levels))

    for plot_type in ['median', 'individual']:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for metric_idx, (key, name) in enumerate([
            ('l2', 'L2 distance'), ('cos', 'Cosine distance'), ('kl', 'KL divergence')
        ]):
            ax = axes[metric_idx]

            # Plot each corruption level
            for k in corruption_levels:
                d = results_by_k[k][key]
                per_act = np.median(d, axis=2)
                color = cmap(norm(k))
                label = f'k={k}'

                if plot_type == 'median':
                    median = np.median(per_act, axis=0)
                    q25 = np.percentile(per_act, 25, axis=0)
                    q75 = np.percentile(per_act, 75, axis=0)
                    ax.plot(angles_deg, median, color=color, linewidth=2, label=label)
                    ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.08)
                else:
                    np.random.seed(42)
                    chosen = np.random.choice(per_act.shape[0], min(5, per_act.shape[0]), replace=False)
                    for j, idx in enumerate(chosen):
                        ax.plot(angles_deg, per_act[idx], color=color, alpha=0.5,
                                linewidth=1, label=label if j == 0 else None)

            # Random baseline
            rd = random_baseline[key]
            per_act_r = np.median(rd, axis=2)
            if plot_type == 'median':
                median_r = np.median(per_act_r, axis=0)
                ax.plot(angles_deg, median_r, color='red', linewidth=2,
                        linestyle='--', label='Random act')
            else:
                np.random.seed(42)
                chosen = np.random.choice(per_act_r.shape[0], min(5, per_act_r.shape[0]), replace=False)
                for j, idx in enumerate(chosen):
                    ax.plot(angles_deg, per_act_r[idx], color='red', alpha=0.5,
                            linewidth=1, linestyle='--', label='Random act' if j == 0 else None)

            ax.set_xlabel('Perturbation angle (degrees)')
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, args.max_angle)

            # Twin top axis: chord distance
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            tick_angles = [a for a in [0, 10, 20, 30, 40, 60] if a <= args.max_angle]
            tick_chords = [2 * act_norm_mean * np.sin(np.deg2rad(a) / 2) for a in tick_angles]
            ax2.set_xticks(tick_angles)
            ax2.set_xticklabels([f'{c:.0f}' for c in tick_chords])
            ax2.set_xlabel('Chord distance')

        measure_label = f", L2@layer {args.measure_layer}" if args.measure_layer is not None else ""
        fig.suptitle(
            f'{args.model} L{args.layer} seq{args.seq_len}: progressive context corruption ({plot_type}{measure_label})\n'
            f'k = # context positions replaced with cov-matched random (pos 0 always real)',
            fontsize=12, y=1.05)
        plt.tight_layout()
        path = os.path.join(args.results_dir, f'sweep_{plot_type}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")


if __name__ == '__main__':
    main()
