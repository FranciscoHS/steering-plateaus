#!/usr/bin/env python3
"""Perturbation sweep where ALL positions are perturbed simultaneously.

Instead of fixing the context and perturbing only the last token,
this script treats the full hidden state [seq_len, d_model] as a single
vector in R^(seq_len * d_model) and does spherical perturbation in that space.

Usage:
    python scripts/perturbation/sweep_allpos_perturbation.py --model qwen3-1.7b --seq_len 3 --layer 2
"""
import argparse, os, sys, pickle, time
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_allpos,
    _get_final_norm, _get_lm_head, _get_blocks,
)
from src.data import load_fineweb_fixed_length


def extract_full_hidden(model, token_ids_list, layer, device, batch_size=64):
    """Extract full hidden state (all positions) at a given layer."""
    all_hidden = []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1]  # [batch, seq_len, d_model]
        all_hidden.append(hidden.float().cpu())
    return torch.cat(all_hidden, 0)


def forward_allpos_to_layer(model, hidden_states, start_layer, end_layer):
    """Forward from start_layer to end_layer, patching all positions.

    Returns last-token hidden state after end_layer.
    """
    hidden = hidden_states
    blocks = _get_blocks(model)

    from src.model import _is_rope_model, _get_position_embeddings, _is_cat_pos_emb_model, _get_cat_pos_emb, _run_block
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, hidden)
    elif _is_cat_pos_emb_model(model):
        pos_emb = _get_cat_pos_emb(model, hidden.shape[1])

    for block in blocks[start_layer + 1:end_layer + 1]:
        hidden = _run_block(block, hidden, model, pos_emb)

    return hidden[:, -1, :]


def _get_ref_measure(model, ref_hidden, layer, measure_layer):
    """Compute reference measurement at final or intermediate layer."""
    if measure_layer is not None:
        return forward_allpos_to_layer(model, ref_hidden, layer, measure_layer)
    from src.model import _run_block, _is_rope_model, _get_position_embeddings, _is_cat_pos_emb_model, _get_cat_pos_emb
    blocks = _get_blocks(model)
    h = ref_hidden.clone()
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, h)
    for block in blocks[layer + 1:]:
        h = _run_block(block, h, model, pos_emb)
    return h[:, -1, :]


def _get_pert_measure(model, perturbed, layer, measure_layer):
    """Compute perturbed measurement at final or intermediate layer."""
    if measure_layer is not None:
        return forward_allpos_to_layer(model, perturbed, layer, measure_layer)
    from src.model import _run_block, _is_rope_model, _get_position_embeddings
    blocks = _get_blocks(model)
    h = perturbed.clone()
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, h)
    for block in blocks[layer + 1:]:
        h = _run_block(block, h, model, pos_emb)
    return h[:, -1, :]


def run_allpos_sweep(model, hidden_state, layer, device, n_dirs, angles_rad,
                     measure_layer=None):
    """Sweep spherical perturbations in the full [seq_len * d_model] space.

    Args:
        hidden_state: [seq_len, d_model] - full hidden state at injection layer
        layer: injection layer
        n_dirs: number of random directions
        angles_rad: perturbation angles

    Returns:
        l2_dists, kl_divs, cos_dists: each [n_angles, n_dirs]
    """
    n_angles = len(angles_rad)
    model_dtype = next(model.parameters()).dtype
    seq_len, d_model = hidden_state.shape

    # Flatten to single vector
    h_flat = hidden_state.reshape(-1).to(device=device, dtype=torch.float32)  # [seq_len * d_model]
    R = h_flat.norm()
    h_unit = h_flat / R

    # Random orthogonal directions in flattened space
    v_rand = torch.randn(n_dirs, seq_len * d_model, device=device)
    v_rand = v_rand - (v_rand @ h_unit).unsqueeze(1) * h_unit.unsqueeze(0)
    v_perps = F.normalize(v_rand, dim=-1)  # [n_dirs, seq_len * d_model]

    # Reference
    with torch.no_grad():
        ref_hidden = hidden_state.unsqueeze(0).to(device=device, dtype=model_dtype)
        ref_logits = forward_from_layer_allpos(model, ref_hidden, layer)
        ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
        ref_measure = _get_ref_measure(model, ref_hidden, layer, measure_layer)

    l2_dists = np.zeros((n_angles, n_dirs))
    kl_divs = np.zeros((n_angles, n_dirs))
    cos_dists = np.zeros((n_angles, n_dirs))

    with torch.no_grad():
        for ang_idx, theta in enumerate(angles_rad):
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            perturbed_flat = R * (cos_t * h_unit.unsqueeze(0) + sin_t * v_perps)
            perturbed = perturbed_flat.reshape(n_dirs, seq_len, d_model).to(model_dtype)

            pert_logits = forward_from_layer_allpos(model, perturbed, layer)
            pert_measure = _get_pert_measure(model, perturbed, layer, measure_layer)

            l2_dists[ang_idx] = (pert_measure - ref_measure).float().norm(dim=-1).cpu().numpy()
            cos_sim = F.cosine_similarity(pert_measure.float(), ref_measure.float(), dim=-1)
            cos_dists[ang_idx] = (1 - cos_sim).cpu().numpy()

            pert_log_probs = F.log_softmax(pert_logits.float(), dim=-1)
            ref_probs = ref_log_probs.exp()
            kl = (ref_probs * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            kl_divs[ang_idx] = kl.cpu().numpy()

    return l2_dists, kl_divs, cos_dists


def run_allpos_perpos_sweep(model, hidden_state, layer, device, n_dirs, angles_rad,
                            measure_layer=None, skip_positions=None):
    """Perturb each position independently by the same angle on its own sphere.

    For each direction sample, each position gets its own random orthogonal
    direction in R^d_model. At angle θ, every position is rotated by θ on
    its own sphere simultaneously. Positions in skip_positions are left unchanged.

    Returns:
        l2_dists, kl_divs, cos_dists: each [n_angles, n_dirs]
    """
    n_angles = len(angles_rad)
    model_dtype = next(model.parameters()).dtype
    seq_len, d_model = hidden_state.shape
    skip_positions = set(skip_positions or [])

    h = hidden_state.to(device=device, dtype=torch.float32)  # [seq_len, d_model]
    R = h.norm(dim=-1, keepdim=True)  # [seq_len, 1]
    h_unit = h / R  # [seq_len, d_model]

    # Sample orthogonal directions per position: [n_dirs, seq_len, d_model]
    # Skipped positions get zero perturbation direction
    v_perps = torch.zeros(n_dirs, seq_len, d_model, device=device)
    for p in range(seq_len):
        if p in skip_positions:
            continue
        a_unit = h_unit[p]  # [d_model]
        v = torch.randn(n_dirs, d_model, device=device)
        v = v - (v @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        v_perps[:, p, :] = F.normalize(v, dim=-1)

    # Reference
    with torch.no_grad():
        ref_hidden = hidden_state.unsqueeze(0).to(device=device, dtype=model_dtype)
        ref_logits = forward_from_layer_allpos(model, ref_hidden, layer)
        ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
        ref_measure = _get_ref_measure(model, ref_hidden, layer, measure_layer)

    l2_dists = np.zeros((n_angles, n_dirs))
    kl_divs = np.zeros((n_angles, n_dirs))
    cos_dists = np.zeros((n_angles, n_dirs))

    with torch.no_grad():
        for ang_idx, theta in enumerate(angles_rad):
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            # Rotate each position independently: [n_dirs, seq_len, d_model]
            # For skipped positions: cos_t * h_unit + sin_t * 0 = cos_t * h_unit
            # which != h_unit. Need to keep skipped positions exactly unchanged.
            perturbed = h.unsqueeze(0).expand(n_dirs, -1, -1).clone()
            for p in range(seq_len):
                if p in skip_positions:
                    continue
                perturbed[:, p, :] = R[p] * (cos_t * h_unit[p].unsqueeze(0) + sin_t * v_perps[:, p, :])
            perturbed = perturbed.to(model_dtype)

            pert_logits = forward_from_layer_allpos(model, perturbed, layer)
            pert_measure = _get_pert_measure(model, perturbed, layer, measure_layer)

            l2_dists[ang_idx] = (pert_measure - ref_measure).float().norm(dim=-1).cpu().numpy()
            cos_sim = F.cosine_similarity(pert_measure.float(), ref_measure.float(), dim=-1)
            cos_dists[ang_idx] = (1 - cos_sim).cpu().numpy()

            pert_log_probs = F.log_softmax(pert_logits.float(), dim=-1)
            ref_probs = ref_log_probs.exp()
            kl = (ref_probs * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            kl_divs[ang_idx] = kl.cpu().numpy()

    return l2_dists, kl_divs, cos_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=3)
    parser.add_argument('--n_act', type=int, default=100)
    parser.add_argument('--n_dirs', type=int, default=10)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--max_angle', type=float, default=90)
    parser.add_argument('--measure_layer', type=int, default=None)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--perpos', action='store_true',
                        help='Perturb each position independently by the same angle on its own sphere')
    parser.add_argument('--skip_pos', type=int, nargs='+', default=None,
                        help='Positions to skip (leave unperturbed). Only used with --perpos.')
    args = parser.parse_args()

    if args.skip_pos and not args.perpos:
        parser.error('--skip_pos requires --perpos')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sweep_fn = run_allpos_perpos_sweep if args.perpos else run_allpos_sweep
    skip_tag = f'_skip{"".join(str(p) for p in args.skip_pos)}' if args.skip_pos else ''
    mode_tag = ('perpos' + skip_tag) if args.perpos else 'flat'

    if args.results_dir is None:
        args.results_dir = f'results/allpos_sweep/{args.model}_L{args.layer}_seq{args.seq_len}{"_perpos" if args.perpos else ""}{skip_tag}'
    os.makedirs(args.results_dir, exist_ok=True)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    print(f"Config: model={args.model}, layer={args.layer}, seq_len={args.seq_len}, "
          f"n_act={args.n_act}, n_dirs={args.n_dirs}, max_angle={args.max_angle}, mode={mode_tag}")

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))

    if args.measure_layer is not None and args.measure_layer < 0:
        args.measure_layer = n_layers + args.measure_layer
    flat_dim = args.seq_len * d_model
    print(f"d_model={d_model}, n_layers={n_layers}, perturbation dim={flat_dim}")

    n_fit = 2000  # samples for covariance fitting
    n_load = max(args.n_act, n_fit)
    token_ids_list = load_fineweb_fixed_length(n_load, tokenizer,
                                                seq_len=args.seq_len, seed=args.seed)
    all_hidden_full = extract_full_hidden(model, token_ids_list, args.layer, device)
    all_hidden = all_hidden_full[:args.n_act]  # for sweeping
    fit_hidden = all_hidden_full[:n_fit]        # for covariance fitting

    all_hidden_flat = all_hidden.reshape(args.n_act, -1)
    act_norm_mean = all_hidden_flat.norm(dim=-1).mean().item()
    print(f"Hidden state norms (flattened): mean={act_norm_mean:.2f}")

    # Generate covariance-matched random activations
    from src.data import sample_random_activations_v2
    if args.perpos:
        # Fit covariance per-position in R^d_model
        print(f"\nFitting per-position covariance (dim={d_model}, n_fit={n_fit})...")
        rand_hidden = torch.zeros(args.n_act, args.seq_len, d_model)
        for p in range(args.seq_len):
            pos_acts = fit_hidden[:, p, :]  # [n_fit, d_model]
            pos_norm_mean = pos_acts.norm(dim=-1).mean().item()
            rand_pos, cov_stats = sample_random_activations_v2(
                pos_acts, args.n_act, mode='full', seed=args.seed + p, device=device
            )
            rand_pos = F.normalize(rand_pos, dim=-1) * pos_norm_mean
            rand_hidden[:, p, :] = rand_pos
            print(f"  pos {p}: norm={pos_norm_mean:.2f}, rank={cov_stats.get('rank', '?')}")
    else:
        print(f"\nFitting covariance in flattened space (dim={flat_dim}, n_fit={n_fit})...")
        rand_flat, cov_stats = sample_random_activations_v2(
            all_hidden_flat, args.n_act, mode='full', seed=args.seed, device=device
        )
        rand_flat = F.normalize(rand_flat, dim=-1) * act_norm_mean
        rand_hidden = rand_flat.reshape(args.n_act, args.seq_len, d_model)
        print(f"Covariance-matched random activations: rank={cov_stats.get('rank', '?')}")

    # Sweep: real
    print(f"\n=== Real activations ===")
    all_l2 = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
    all_kl = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
    all_cos = np.zeros((args.n_act, len(angles_deg), args.n_dirs))

    t0 = time.time()
    for i in range(args.n_act):
        sweep_kwargs = dict(measure_layer=args.measure_layer)
        if args.perpos:
            sweep_kwargs['skip_positions'] = args.skip_pos
        all_l2[i], all_kl[i], all_cos[i] = sweep_fn(
            model, all_hidden[i], args.layer, device, args.n_dirs, angles_rad,
            **sweep_kwargs)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
    print(f"Real done in {time.time()-t0:.1f}s")

    # Sweep: random
    print(f"\n=== Random activations (cov-matched) ===")
    rand_l2 = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
    rand_kl = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
    rand_cos = np.zeros((args.n_act, len(angles_deg), args.n_dirs))

    t0 = time.time()
    for i in range(args.n_act):
        rand_l2[i], rand_kl[i], rand_cos[i] = sweep_fn(
            model, rand_hidden[i], args.layer, device, args.n_dirs, angles_rad,
            **sweep_kwargs)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
    print(f"Random done in {time.time()-t0:.1f}s")

    results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'act_norm_mean': act_norm_mean,
        'real': {'l2': all_l2, 'kl': all_kl, 'cos': all_cos},
        'random': {'l2': rand_l2, 'kl': rand_kl, 'cos': rand_cos},
    }
    pkl_path = os.path.join(args.results_dir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {pkl_path}")

    # Plots — same style as run_perturbation_sweep_spherical.py
    conditions = [('real', 'Real'), ('random', 'Random (cov-matched)')]
    colors = ['blue', 'red']
    chord_distances = 2 * act_norm_mean * np.sin(np.deg2rad(angles_deg) / 2)

    for plot_type in ['median', 'individual']:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for metric_idx, (key, name) in enumerate([('l2', 'L2 distance'), ('cos', 'Cosine distance'), ('kl', 'KL divergence')]):
            ax = axes[metric_idx]
            for (cond_key, label), color in zip(conditions, colors):
                d = results[cond_key][key]
                per_act = np.median(d, axis=2)

                if plot_type == 'median':
                    median = np.median(per_act, axis=0)
                    q25 = np.percentile(per_act, 25, axis=0)
                    q75 = np.percentile(per_act, 75, axis=0)
                    ax.plot(angles_deg, median, color=color, label=label, linewidth=2)
                    ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.15)
                else:
                    np.random.seed(42)
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

            # Twin top axis: chord distance
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            tick_angles = [a for a in [0, 10, 20, 30, 40, 60, 90, 120, 150, 180] if a <= args.max_angle]
            tick_chords = [2 * act_norm_mean * np.sin(np.deg2rad(a) / 2) for a in tick_angles]
            ax2.set_xticks(tick_angles)
            ax2.set_xticklabels([f'{c:.0f}' for c in tick_chords])
            ax2.set_xlabel('Chord distance')

        measure_label = f", L2@layer {args.measure_layer}" if args.measure_layer is not None else ""
        fig.suptitle(f'{args.model}, layer {args.layer}, seq_len={args.seq_len} (all-pos {mode_tag}{measure_label})', fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(args.results_dir, f'sweep_{plot_type}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")


if __name__ == '__main__':
    main()
