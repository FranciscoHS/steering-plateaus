#!/usr/bin/env python3
"""
Per-position perturbation sweep for seq_len=2.

Perturbs position 0 or position 1 independently (spherically) while
holding the other fixed, then measures downstream effects.

This answers: do both positions contribute to plateaus equally?

Usage:
    python scripts/perturbation/run_perposition_sweep.py --model qwen3-1.7b --layer 2
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
    load_model, _get_final_norm, _get_lm_head, _get_blocks,
    _is_rope_model, _get_position_embeddings, _run_block,
    _is_cat_pos_emb_model, _get_cat_pos_emb,
)
from src.data import load_fineweb_fixed_length


def extract_full_hidden(model, tokenizer, token_ids_list, layer, device, batch_size=64):
    """Extract full hidden states at all positions from a given layer."""
    all_hidden = []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1]
        all_hidden.append(hidden.float().cpu())
    return torch.cat(all_hidden, 0)  # [n, seq_len, d_model]


def forward_from_layer_raw(model, hidden_states, start_layer):
    """Forward from start_layer to final block. Returns pre-ln_f hidden and logits."""
    hidden = hidden_states
    blocks = _get_blocks(model)
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, hidden)
    elif _is_cat_pos_emb_model(model):
        pos_emb = _get_cat_pos_emb(model, hidden.shape[1])

    for block in blocks[start_layer + 1:]:
        hidden = _run_block(block, hidden, model, pos_emb)

    # hidden is now [batch, seq_len, d_model] after final block
    final_act = hidden[:, -1, :]  # [batch, d_model]
    ln_out = _get_final_norm(model)(final_act.unsqueeze(1)).squeeze(1)
    logits = _get_lm_head(model)(ln_out)
    return final_act, logits


def sample_orthogonal_directions(activations, n_dirs):
    """Sample random unit vectors orthogonal to each activation."""
    n_act, d_model = activations.shape
    v_perps = torch.zeros(n_act, n_dirs, d_model)
    for i in range(n_act):
        a = activations[i]
        a_unit = a / a.norm()
        v = torch.randn(n_dirs, d_model)
        v = v - (v @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        v = F.normalize(v, dim=-1)
        v_perps[i] = v
    return v_perps


def run_sweep_perposition(model, hidden_ref, v_perps, angles_rad, layer, device,
                          n_dirs, perturb_pos):
    """Sweep spherical perturbations at a specific position.

    Args:
        hidden_ref: [seq_len, d_model] - full hidden state (both positions)
        v_perps: [n_dirs, d_model] - orthogonal directions for the perturbed position
        perturb_pos: which position to perturb (0 or 1)
    """
    n_angles = len(angles_rad)
    l2_dists = np.zeros((n_angles, n_dirs))
    kl_divs = np.zeros((n_angles, n_dirs))
    cos_dists = np.zeros((n_angles, n_dirs))

    model_dtype = next(model.parameters()).dtype
    h = hidden_ref.to(device=device, dtype=model_dtype)  # [seq_len, d_model]
    a = h[perturb_pos]  # [d_model]
    R = a.norm()
    a_unit = a / R
    vp = v_perps.to(device=device, dtype=model_dtype)  # [n_dirs, d_model]

    # Reference (unperturbed)
    with torch.no_grad():
        ref_hidden = h.unsqueeze(0)  # [1, seq_len, d_model]
        ref_final, ref_logits = forward_from_layer_raw(model, ref_hidden, layer)
        ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)

    for ang_idx, theta in enumerate(angles_rad):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        # Perturbed activation at perturb_pos
        perturbed_a = R * (cos_t * a_unit.unsqueeze(0) + sin_t * vp)  # [n_dirs, d_model]

        # Build full hidden states: [n_dirs, seq_len, d_model]
        h_batch = h.unsqueeze(0).expand(n_dirs, -1, -1).clone()
        h_batch[:, perturb_pos, :] = perturbed_a

        with torch.no_grad():
            pert_final, pert_logits = forward_from_layer_raw(model, h_batch, layer)

            l2_dists[ang_idx] = (pert_final - ref_final).float().norm(dim=-1).cpu().numpy()
            cos_sim = F.cosine_similarity(pert_final.float(), ref_final.float(), dim=-1)
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
    parser.add_argument('--n_act', type=int, default=100)
    parser.add_argument('--n_dirs', type=int, default=10)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--max_angle', type=float, default=90)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    seq_len = 2  # Fixed for this experiment

    if args.results_dir is None:
        args.results_dir = f'results/perposition_sweep/{args.model}_layer{args.layer}'
    os.makedirs(args.results_dir, exist_ok=True)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    print(f"Config: model={args.model}, layer={args.layer}, seq_len={seq_len}, "
          f"n_act={args.n_act}, n_dirs={args.n_dirs}, n_angles={len(angles_deg)}, "
          f"range=[{angles_deg[0]:.1f}, {angles_deg[-1]:.1f}] deg")

    # Load model
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    print(f"Device: {device}, d_model: {d_model}, n_layers: {n_layers}")

    # Load data and extract hidden states
    print(f"Loading {args.n_act} FineWeb sequences (seq_len={seq_len})...")
    token_ids_list = load_fineweb_fixed_length(args.n_act, tokenizer, seq_len=seq_len, seed=args.seed)
    print(f"Extracting layer-{args.layer} hidden states...")
    all_hidden = extract_full_hidden(model, tokenizer, token_ids_list, args.layer, device)
    # all_hidden: [n_act, 2, d_model]

    print(f"Pos 0 mean norm: {all_hidden[:, 0, :].norm(dim=-1).mean():.2f}")
    print(f"Pos 1 mean norm: {all_hidden[:, 1, :].norm(dim=-1).mean():.2f}")

    # Norm-matched random activations for each position
    rand_activations = {}
    for pos in [0, 1]:
        norm_mean = all_hidden[:, pos, :].norm(dim=-1).mean().item()
        r = torch.randn(args.n_act, d_model)
        r = F.normalize(r, dim=-1) * norm_mean
        rand_activations[pos] = r

    results = {}

    for perturb_pos in [0, 1]:
        pos_acts = all_hidden[:, perturb_pos, :]  # [n_act, d_model]
        v_perps = sample_orthogonal_directions(pos_acts, args.n_dirs)

        print(f"\n=== Perturb position {perturb_pos} (real) ===")
        l2 = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
        kl = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
        cos = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
        t0 = time.time()
        for i in range(args.n_act):
            l2[i], kl[i], cos[i] = run_sweep_perposition(
                model, all_hidden[i], v_perps[i], angles_rad, args.layer, device,
                args.n_dirs, perturb_pos
            )
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
        print(f"Done in {time.time()-t0:.1f}s")

        results[f'pos{perturb_pos}_real'] = {'l2': l2, 'kl': kl, 'cos': cos}

    # Random baselines: replace pos N activation with norm-matched random, perturb pos N
    for perturb_pos in [0, 1]:
        print(f"\n=== Perturb position {perturb_pos} (random, norm-matched) ===")
        rand_v_perps = sample_orthogonal_directions(rand_activations[perturb_pos], args.n_dirs)
        l2 = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
        kl = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
        cos = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
        t0 = time.time()
        for i in range(args.n_act):
            h_rand = all_hidden[i].clone()
            h_rand[perturb_pos] = rand_activations[perturb_pos][i]
            l2[i], kl[i], cos[i] = run_sweep_perposition(
                model, h_rand, rand_v_perps[i], angles_rad, args.layer, device,
                args.n_dirs, perturb_pos=perturb_pos
            )
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
        print(f"Done in {time.time()-t0:.1f}s")
        results[f'pos{perturb_pos}_random'] = {'l2': l2, 'kl': kl, 'cos': cos}

    # Save
    results['args'] = vars(args)
    results['angles_deg'] = angles_deg
    results['angles_rad'] = angles_rad
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Plot
    conditions = [
        ('pos0_real', 'Perturb pos 0 (real)', 'blue'),
        ('pos0_random', 'Perturb pos 0 (random)', 'cornflowerblue'),
        ('pos1_real', 'Perturb pos 1 (real)', 'green'),
        ('pos1_random', 'Perturb pos 1 (random)', 'red'),
    ]

    for plot_type in ['median', 'individual']:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for metric_idx, (key, name) in enumerate([('l2', 'L2 distance'), ('cos', 'Cosine distance'), ('kl', 'KL divergence')]):
            ax = axes[metric_idx]
            for cond_key, label, color in conditions:
                d = results[cond_key][key]
                per_act = np.median(d, axis=2)  # [n_act, n_angles]

                if plot_type == 'median':
                    median = np.median(per_act, axis=0)
                    q25 = np.percentile(per_act, 25, axis=0)
                    q75 = np.percentile(per_act, 75, axis=0)
                    ax.plot(angles_deg, median, color=color, label=label, linewidth=2)
                    ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.15)
                else:
                    n_individual = 10
                    np.random.seed(42)
                    chosen = np.random.choice(per_act.shape[0], n_individual, replace=False)
                    for j, idx in enumerate(chosen):
                        ax.plot(angles_deg, per_act[idx], color=color, alpha=0.4, linewidth=1.5,
                                label=label if j == 0 else None)

            ax.set_xlabel('Perturbation angle (degrees)')
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, args.max_angle)

        fig.suptitle(f'{args.model}, layer {args.layer}, seq_len=2 — per-position perturbation ({plot_type})',
                     fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(args.results_dir, f'sweep_{plot_type}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
