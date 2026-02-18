#!/usr/bin/env python3
"""
Track gate switches in SwiGLU MLPs during perturbation sweeps.

For each perturbation angle along different directions, capture the gate
pre-activation values at all MLP layers downstream of the perturbation layer,
and count how many gates switch state (active<->inactive).

Usage:
    python scripts/run_gate_switches.py --model qwen3-0.6b --layer 14
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
from matplotlib import cm

from src.model import (
    load_model,
    _get_blocks, _is_rope_model, _get_position_embeddings, _run_block,
    _get_final_norm, _get_lm_head,
)
from src.data import load_fineweb_fixed_length


# ── Reused infrastructure ──

def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device,
                                   batch_size=64):
    all_contexts, all_activations = [], []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].cpu())
        all_activations.append(hidden[:, -1, :].cpu())
    return torch.cat(all_contexts, 0), torch.cat(all_activations, 0)


def build_perturbed_hidden(contexts, activations, direction, angle_rad):
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms
    proj = (direction.unsqueeze(0) * a_unit).sum(dim=-1, keepdim=True)
    u_perp = direction.unsqueeze(0) - proj * a_unit
    u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)
    a_pert = a_norms * (cos_t * a_unit + sin_t * u_perp)
    return torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)


def project_out(v, u):
    residual = v - (v @ u) * u
    return residual / residual.norm()


# ── Gate capture ──

def forward_with_gates(model, hidden_states, start_layer, layer_indices):
    """Forward pass capturing gate pre-activations (pre-silu) at specified layers.

    Returns:
        gate_acts: dict {layer_idx: [N, intermediate_size]} (last token only)
        logits: [N, vocab_size]
    """
    gate_acts = {}
    hooks = []
    blocks = _get_blocks(model)

    for idx in layer_indices:
        def make_hook(i):
            def hook_fn(module, input, output):
                gate_acts[i] = output[:, -1, :].detach()
            return hook_fn
        h = blocks[idx].mlp.gate_proj.register_forward_hook(make_hook(idx))
        hooks.append(h)

    # Forward pass through remaining blocks
    hidden = hidden_states
    rope = _is_rope_model(model)
    pos_emb = _get_position_embeddings(model, hidden) if rope else None
    for block in blocks[start_layer + 1:]:
        hidden = _run_block(block, hidden, model, pos_emb)

    for h in hooks:
        h.remove()

    # Logits
    final_hidden = hidden[:, -1, :]
    normed = _get_final_norm(model)(final_hidden)
    logits = _get_lm_head(model)(normed)

    return gate_acts, logits


def sweep_gate_switches(model, contexts, activations, ref_lp,
                        direction, angles_rad, layer, device,
                        layer_indices, thresholds):
    """Sweep one direction, tracking gate switches and KL at each angle.

    Returns:
        switches: dict[threshold] -> dict[layer_idx] -> {
            'cumulative': [n_angles],  # fraction of gates differing from ref
            'perstep': [n_angles],     # fraction of gates differing from prev
        }
        kl: [n_angles]  # median KL vs original
    """
    N = activations.shape[0]
    n_angles = len(angles_rad)
    direction_dev = direction.to(device)

    # Reference gate activations (unperturbed)
    with torch.no_grad():
        ref_hidden = torch.cat([contexts, activations.unsqueeze(1)], dim=1)
        ref_gates, _ = forward_with_gates(model, ref_hidden, layer, layer_indices)

    # Binary reference for each threshold
    ref_binary = {}
    for tau in thresholds:
        ref_binary[tau] = {idx: (ref_gates[idx] > tau) for idx in layer_indices}

    # Initialize results
    switches = {}
    for tau in thresholds:
        switches[tau] = {}
        for idx in layer_indices:
            switches[tau][idx] = {
                'cumulative': np.zeros(n_angles),
                'perstep': np.zeros(n_angles),
            }

    kl_vals = np.zeros(n_angles)

    # Previous binary state (for per-step)
    prev_binary = {}
    for tau in thresholds:
        prev_binary[tau] = {idx: ref_binary[tau][idx].clone() for idx in layer_indices}

    for ang_idx, theta in enumerate(angles_rad):
        with torch.no_grad():
            hidden = build_perturbed_hidden(contexts, activations, direction_dev, theta)
            cur_gates, logits = forward_with_gates(model, hidden, layer, layer_indices)
            log_probs = F.log_softmax(logits, dim=-1)
            kl = (ref_lp.exp() * (ref_lp - log_probs)).sum(dim=-1)
            kl_vals[ang_idx] = kl.median().item()

            for tau in thresholds:
                for idx in layer_indices:
                    cur_bin = cur_gates[idx] > tau
                    I = cur_bin.shape[1]  # intermediate_size

                    # Cumulative: fraction differing from reference, averaged over batch
                    cum_frac = (ref_binary[tau][idx] != cur_bin).float().sum(dim=1) / I
                    switches[tau][idx]['cumulative'][ang_idx] = cum_frac.mean().item()

                    # Per-step: fraction differing from previous angle
                    step_frac = (prev_binary[tau][idx] != cur_bin).float().sum(dim=1) / I
                    switches[tau][idx]['perstep'][ang_idx] = step_frac.mean().item()

                    prev_binary[tau][idx] = cur_bin

    return switches, kl_vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--n_act', type=int, default=30)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--threshold2', type=float, default=0.1,
                        help='Second threshold for gate activity (first is always 0)')
    parser.add_argument('--directions_path', default=None)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.directions_path is None:
        args.directions_path = f'results/melbo_constrained/{args.model}/optimization_results.pkl'
    if args.results_dir is None:
        args.results_dir = f'results/gate_switches/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    thresholds = [0.0, args.threshold2]

    # ===== Load directions =====
    print("Loading directions...")
    with open(args.directions_path, 'rb') as f:
        saved = pickle.load(f)
    m_hat = saved['melbo_directions'][0]
    mean_dir = saved['mean_dir']
    m_perp = project_out(m_hat, mean_dir)
    print(f"  cos(m_hat, mean_dir) = {(m_hat @ mean_dir).item():.4f}")

    # Random directions orthogonal to m_hat
    d_model = m_hat.shape[0]
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    random_dirs = torch.randn(args.n_random_dirs, d_model, device=device_str)
    d = m_hat.to(device_str)
    random_dirs = random_dirs - (random_dirs @ d).unsqueeze(1) * d.unsqueeze(0)
    random_dirs = F.normalize(random_dirs, dim=-1)
    print(f"  Sampled {args.n_random_dirs} random dirs ⊥ m_hat")

    # ===== Load model and data =====
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    blocks = _get_blocks(model)
    n_layers = len(blocks)
    layer_indices = list(range(args.layer + 1, n_layers))
    print(f"  {n_layers} total layers, tracking gates at layers {layer_indices[0]}-{layer_indices[-1]}")

    # Check intermediate_size
    intermediate_size = blocks[layer_indices[0]].mlp.gate_proj.out_features
    print(f"  intermediate_size = {intermediate_size}")

    m_hat_dev = m_hat.to(device)
    m_perp_dev = m_perp.to(device)
    random_dirs = random_dirs.to(device)

    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"  Activations: {activations.shape}")

    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)

    # Reference log-probs
    with torch.no_grad():
        ref_hidden = torch.cat([contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        _, ref_logits = forward_with_gates(model, ref_hidden, args.layer, layer_indices)
        ref_lp = F.log_softmax(ref_logits, dim=-1)

    # ===== Sweeps =====
    angles_deg = np.linspace(0, 180, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    all_results = {}

    # MELBO_1
    print(f"\nSweeping MELBO_1...", end=' ', flush=True)
    t0 = time.time()
    sw, kl = sweep_gate_switches(
        model, contexts_dev, activations_dev, ref_lp,
        m_hat_dev, angles_rad, args.layer, device,
        layer_indices, thresholds)
    all_results['MELBO_1'] = {'switches': sw, 'kl': kl}
    print(f"{time.time()-t0:.1f}s")

    # m_perp
    print(f"Sweeping MELBO_1 ⊥ diff-of-means...", end=' ', flush=True)
    t0 = time.time()
    sw, kl = sweep_gate_switches(
        model, contexts_dev, activations_dev, ref_lp,
        m_perp_dev, angles_rad, args.layer, device,
        layer_indices, thresholds)
    all_results['m_perp'] = {'switches': sw, 'kl': kl}
    print(f"{time.time()-t0:.1f}s")

    # Random directions (sweep each, then average)
    print(f"Sweeping {args.n_random_dirs} random dirs ⊥ m_hat...", end=' ', flush=True)
    t0 = time.time()
    random_switches_all = []
    random_kl_all = []
    for d_idx in range(args.n_random_dirs):
        sw, kl = sweep_gate_switches(
            model, contexts_dev, activations_dev, ref_lp,
            random_dirs[d_idx], angles_rad, args.layer, device,
            layer_indices, thresholds)
        random_switches_all.append(sw)
        random_kl_all.append(kl)
    print(f"{time.time()-t0:.1f}s")

    # Average random switches across directions
    random_mean = {}
    for tau in thresholds:
        random_mean[tau] = {}
        for idx in layer_indices:
            cum_stack = np.stack([rs[tau][idx]['cumulative'] for rs in random_switches_all])
            step_stack = np.stack([rs[tau][idx]['perstep'] for rs in random_switches_all])
            random_mean[tau][idx] = {
                'cumulative': cum_stack.mean(axis=0),
                'cumulative_std': cum_stack.std(axis=0),
                'perstep': step_stack.mean(axis=0),
                'perstep_std': step_stack.std(axis=0),
            }
    random_kl_mean = np.stack(random_kl_all).mean(axis=0)
    all_results['random'] = {'switches': random_mean, 'kl': random_kl_mean}

    # ===== Save =====
    save_data = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'layer_indices': layer_indices,
        'thresholds': thresholds,
        'intermediate_size': intermediate_size,
        'results': all_results,
        'random_switches_all': random_switches_all,
        'random_kl_all': random_kl_all,
    }
    path = os.path.join(args.results_dir, 'results.pkl')
    with open(path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved: {path}")

    # ===== Plots =====
    print("\nPlotting...")

    max_angle = 60
    mask = angles_deg <= max_angle
    x = angles_deg[mask]

    # Color map for layers
    n_tracked = len(layer_indices)
    cmap = cm.get_cmap('viridis', n_tracked)
    layer_colors = [cmap(i / (n_tracked - 1)) for i in range(n_tracked)]

    # --- Per-direction plots: 2x2 (cumulative/perstep x tau=0/tau2) ---
    direction_labels = {
        'MELBO_1': 'MELBO₁',
        'm_perp': 'MELBO₁ ⊥ diff-of-means',
        'random': f'Random ⊥ MELBO₁ (mean of {args.n_random_dirs})',
    }

    for dir_name in ['MELBO_1', 'm_perp', 'random']:
        res = all_results[dir_name]
        sw = res['switches']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for col, tau in enumerate(thresholds):
            for row, metric in enumerate(['cumulative', 'perstep']):
                ax = axes[row, col]

                for li, idx in enumerate(layer_indices):
                    y = sw[tau][idx][metric]
                    color = layer_colors[li]
                    label = f'L{idx}' if col == 0 else None
                    ax.plot(x, y[mask], color=color, lw=1.5, label=label)

                    # For random: add shading for std
                    if dir_name == 'random' and f'{metric}_std' in sw[tau][idx]:
                        std = sw[tau][idx][f'{metric}_std']
                        ax.fill_between(x, (y - std)[mask], (y + std)[mask],
                                        color=color, alpha=0.07)

                metric_label = 'Cumulative' if metric == 'cumulative' else 'Per-step'
                ax.set_title(f'{metric_label}, τ={tau}', fontsize=12)
                ax.set_xlabel('Perturbation angle (degrees)')
                ax.set_ylabel('Fraction of gates switched')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, max_angle)

                if col == 0 and row == 0:
                    ax.legend(fontsize=7, ncol=3, loc='upper left')

        fig.suptitle(f'Gate switches: {direction_labels[dir_name]}\n'
                     f'{args.model}, layer {args.layer}', fontsize=14)
        plt.tight_layout()
        path = os.path.join(args.results_dir, f'gates_{dir_name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # --- Aggregate comparison: total switches + KL overlay ---
    for tau in thresholds:
        fig, ax1 = plt.subplots(figsize=(10, 7))

        for dir_name, color, ls in [('MELBO_1', '#d62728', '-'),
                                     ('m_perp', '#ff7f0e', '-'),
                                     ('random', '#1f77b4', '--')]:
            res = all_results[dir_name]
            sw = res['switches']

            # Sum fraction across layers
            total = np.zeros(len(angles_deg))
            for idx in layer_indices:
                total += sw[tau][idx]['cumulative']

            ax1.plot(x, total[mask], color=color, lw=2.5, ls=ls,
                     label=f'{direction_labels[dir_name]} (gates)')

        ax1.set_xlabel('Perturbation angle (degrees)', fontsize=12)
        ax1.set_ylabel('Total gate switch fraction (sum over layers)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max_angle)

        # KL on secondary axis
        ax2 = ax1.twinx()
        for dir_name, color, ls in [('MELBO_1', '#d62728', '-'),
                                     ('m_perp', '#ff7f0e', '-'),
                                     ('random', '#1f77b4', '--')]:
            kl = all_results[dir_name]['kl']
            ax2.plot(x, kl[mask], color=color, lw=1.5, ls=ls, alpha=0.4)

        ax2.set_ylabel('KL divergence (faded lines)', fontsize=12, alpha=0.5)

        ax1.legend(fontsize=10, loc='upper left')
        fig.suptitle(f'Aggregate gate switches vs KL, τ={tau}\n'
                     f'{args.model}, layer {args.layer}', fontsize=13)
        plt.tight_layout()
        path = os.path.join(args.results_dir, f'aggregate_tau{tau}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # ===== Summary =====
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)

    idx_10 = np.argmin(np.abs(angles_deg - 10))
    idx_15 = np.argmin(np.abs(angles_deg - 15))

    for tau in thresholds:
        print(f"\nτ = {tau}:")
        for label, idx in [('10 deg', idx_10), ('15 deg', idx_15)]:
            print(f"  Cumulative gate switches at {label} (total over layers):")
            for dir_name in ['MELBO_1', 'm_perp', 'random']:
                sw = all_results[dir_name]['switches']
                total = sum(sw[tau][li]['cumulative'][idx] for li in layer_indices)
                print(f"    {dir_name:30s}: {total:.4f}")

    print(f"\nKL at key angles:")
    for label, idx in [('10 deg', idx_10), ('15 deg', idx_15)]:
        print(f"  {label}:")
        for dir_name in ['MELBO_1', 'm_perp', 'random']:
            print(f"    {dir_name:30s}: {all_results[dir_name]['kl'][idx]:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
