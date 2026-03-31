#!/usr/bin/env python3
"""Perturbation sweep comparing a few specific directions on the same activation.

Plots L2 curves for: contrastive (zh-en), one MELBO direction, and a few random directions.

Usage:
    python scripts/perturbation/sweep_compare_directions.py \
        --model qwen3-1.7b --inject_layer 7 --measure_layer -2 \
        --contrastive results/contrastive_directions/qwen3-1.7b_L7_mandarin/contrastive_direction.pkl \
        --melbo results/iterative_melbo_hf/qwen3-1.7b_L7_meas-2_escalation_v6/checkpoint_round3_mid.pkl \
        --n_random 5 --outdir results/sweep_control/compare_directions
"""
import argparse, os, sys, pickle
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.melbo.run_iterative_melbo_hf import (
    extract_context_and_activation, forward_to_L,
)


def sweep_single_dir(model, ctx, act, ref_L, a_unit, a_norm,
                     direction, inject_layer, measure_layer, device, angles_deg):
    """Sweep one direction, return L2 curve."""
    angles_rad = np.deg2rad(angles_deg)
    cos_t = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)

    l2_curve = np.zeros(len(angles_deg))
    for i, (c, s) in enumerate(zip(cos_t, sin_t)):
        perturbed = a_norm * (c * a_unit + s * direction)
        with torch.no_grad():
            pert_L = forward_to_L(model, ctx, perturbed.unsqueeze(0),
                                  inject_layer, measure_layer)
            l2_curve[i] = (pert_L - ref_L).norm().item()
    return l2_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--inject_layer', type=int, required=True)
    parser.add_argument('--measure_layer', type=int, required=True)
    parser.add_argument('--contrastive', required=True)
    parser.add_argument('--melbo', required=True)
    parser.add_argument('--n_random', type=int, default=5)
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=120)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    n_layers = len(_get_blocks(model))
    measure_layer = args.measure_layer
    if measure_layer < 0:
        measure_layer = n_layers + measure_layer

    # Get activation
    token_ids_list = load_fineweb_fixed_length(100, tokenizer, seq_len=5, seed=args.seed)
    _, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.inject_layer, device)
    ctx_all, _ = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.inject_layer, device)

    ctx = ctx_all[0].unsqueeze(0).to(device)
    act = all_activations[0].to(device)
    a_norm = act.norm()
    a_unit = act / a_norm

    with torch.no_grad():
        ref_L = forward_to_L(model, ctx, act.unsqueeze(0),
                             args.inject_layer, measure_layer)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]

    def orthogonalize(d):
        d = d.to(device)
        d = d - (d @ a_unit) * a_unit
        return F.normalize(d, dim=0)

    # Load contrastive direction
    with open(args.contrastive, 'rb') as f:
        c_data = pickle.load(f)
    contrastive_dir = c_data['directions'][0].float()
    contrastive_dir = orthogonalize(contrastive_dir)

    # Load MELBO direction (first one)
    with open(args.melbo, 'rb') as f:
        m_data = pickle.load(f)
    if isinstance(m_data['directions'], list):
        melbo_dir = m_data['directions'][0].float()
    else:
        melbo_dir = m_data['directions'][0].float()
    melbo_dir = orthogonalize(melbo_dir)

    # Random directions
    d_model = act.shape[0]
    rand_dirs = []
    for i in range(args.n_random):
        r = torch.randn(d_model, device=device)
        r = orthogonalize(r)
        rand_dirs.append(r)

    # Sweep all directions
    print("Sweeping contrastive direction...")
    l2_contrastive = sweep_single_dir(model, ctx, act, ref_L, a_unit, a_norm,
                                       contrastive_dir, args.inject_layer,
                                       measure_layer, device, angles_deg)

    print("Sweeping MELBO direction...")
    l2_melbo = sweep_single_dir(model, ctx, act, ref_L, a_unit, a_norm,
                                 melbo_dir, args.inject_layer,
                                 measure_layer, device, angles_deg)

    print(f"Sweeping {args.n_random} random directions...")
    l2_randoms = []
    for i, rd in enumerate(rand_dirs):
        l2_randoms.append(sweep_single_dir(model, ctx, act, ref_L, a_unit, a_norm,
                                            rd, args.inject_layer,
                                            measure_layer, device, angles_deg))

    # Save results
    os.makedirs(args.outdir, exist_ok=True)
    save_data = {
        'angles_deg': angles_deg,
        'l2_contrastive': l2_contrastive,
        'l2_melbo': l2_melbo,
        'l2_randoms': l2_randoms,
        'args': vars(args),
    }
    pkl_path = os.path.join(args.outdir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(angles_deg, l2_contrastive, color='#EE6677', lw=2, label='Contrastive (zh-en)')
    ax.plot(angles_deg, l2_melbo, color='#4477AA', lw=2, label='MELBO dir 0')
    for i, l2r in enumerate(l2_randoms):
        label = 'Random' if i == 0 else None
        ax.plot(angles_deg, l2r, color='#BBBBBB', lw=1, alpha=0.6, label=label)
    ax.axhline(100, color='k', ls='--', lw=1, alpha=0.4, label='L2=100')
    ax.set_xlabel('Perturbation angle (°)')
    ax.set_ylabel('L2 distance at measure layer')
    ax.set_title(f'{args.model}: L2 curves for different direction types')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, 'comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


if __name__ == '__main__':
    main()
