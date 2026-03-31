#!/usr/bin/env python3
"""Perturbation sweep for each position individually (seq_len=N).

For each position p, perturb only position p while holding all others fixed.
Produces a combined plot with one curve per position.

Usage:
    python scripts/perturbation/sweep_each_position.py --model qwen3-1.7b --layer 2 --seq_len 10
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
    _is_rope_model, _get_position_embeddings, _run_block,
    _is_cat_pos_emb_model, _get_cat_pos_emb,
)
from src.data import load_fineweb_fixed_length


def extract_full_hidden(model, token_ids_list, layer, device, batch_size=64):
    all_hidden = []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1]
        all_hidden.append(hidden.float().cpu())
    return torch.cat(all_hidden, 0)


def run_single_position_sweep(model, hidden_state, layer, device, n_dirs, angles_rad,
                               perturb_pos):
    """Perturb one position spherically, hold all others fixed."""
    n_angles = len(angles_rad)
    model_dtype = next(model.parameters()).dtype
    seq_len, d_model = hidden_state.shape

    h = hidden_state.to(device=device, dtype=torch.float32)
    a = h[perturb_pos]
    R = a.norm()
    a_unit = a / R

    # Orthogonal directions for this position
    v = torch.randn(n_dirs, d_model, device=device)
    v = v - (v @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
    v_perps = F.normalize(v, dim=-1)

    # Reference
    with torch.no_grad():
        ref_hidden = hidden_state.unsqueeze(0).to(device=device, dtype=model_dtype)
        # Get final-layer hidden (before ln_f)
        blocks = _get_blocks(model)
        hr = ref_hidden.clone()
        pos_emb = None
        if _is_rope_model(model):
            pos_emb = _get_position_embeddings(model, hr)
        for block in blocks[layer + 1:]:
            hr = _run_block(block, hr, model, pos_emb)
        ref_measure = hr[:, -1, :]

        ref_logits = forward_from_layer_allpos(model, ref_hidden, layer)
        ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)

    l2_dists = np.zeros((n_angles, n_dirs))
    kl_divs = np.zeros((n_angles, n_dirs))
    cos_dists = np.zeros((n_angles, n_dirs))

    with torch.no_grad():
        for ang_idx, theta in enumerate(angles_rad):
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            perturbed_a = R * (cos_t * a_unit.unsqueeze(0) + sin_t * v_perps)  # [n_dirs, d_model]

            h_batch = h.unsqueeze(0).expand(n_dirs, -1, -1).clone()
            h_batch[:, perturb_pos, :] = perturbed_a
            h_batch = h_batch.to(model_dtype)

            # Final-layer hidden
            hp = h_batch.clone()
            pos_emb = None
            if _is_rope_model(model):
                pos_emb = _get_position_embeddings(model, hp)
            for block in blocks[layer + 1:]:
                hp = _run_block(block, hp, model, pos_emb)
            pert_measure = hp[:, -1, :]

            l2_dists[ang_idx] = (pert_measure - ref_measure).float().norm(dim=-1).cpu().numpy()
            cos_sim = F.cosine_similarity(pert_measure.float(), ref_measure.float(), dim=-1)
            cos_dists[ang_idx] = (1 - cos_sim).cpu().numpy()

            pert_logits = forward_from_layer_allpos(model, h_batch, layer)
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
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/allpos_sweep/{args.model}_L{args.layer}_seq{args.seq_len}_perpos'
    os.makedirs(args.results_dir, exist_ok=True)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    print(f"Config: model={args.model}, layer={args.layer}, seq_len={args.seq_len}, "
          f"n_act={args.n_act}, n_dirs={args.n_dirs}, max_angle={args.max_angle}")

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)
    d_model = model._plateau_info['d_model']
    print(f"d_model={d_model}, n_layers={len(_get_blocks(model))}")

    token_ids_list = load_fineweb_fixed_length(args.n_act, tokenizer,
                                                seq_len=args.seq_len, seed=args.seed)
    all_hidden = extract_full_hidden(model, token_ids_list, args.layer, device)

    for p in range(args.seq_len):
        print(f"  Pos {p} mean norm: {all_hidden[:, p, :].norm(dim=-1).mean():.2f}")

    # Sweep each position
    results = {'args': vars(args), 'angles_deg': angles_deg}

    for p in range(args.seq_len):
        print(f"\n=== Position {p}/{args.seq_len-1} ===")
        l2 = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
        kl = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
        cos = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
        t0 = time.time()
        for i in range(args.n_act):
            l2[i], kl[i], cos[i] = run_single_position_sweep(
                model, all_hidden[i], args.layer, device, args.n_dirs, angles_rad, p)
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
        print(f"  Done in {time.time()-t0:.1f}s")
        results[f'pos{p}'] = {'l2': l2, 'kl': kl, 'cos': cos}

    with open(os.path.join(args.results_dir, 'each_position_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Plot: one curve per position, median over activations and directions
    cmap = plt.cm.viridis
    for plot_type in ['median', 'individual']:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for metric_idx, (key, name) in enumerate([('l2', 'L2 distance'), ('cos', 'Cosine distance'), ('kl', 'KL divergence')]):
            ax = axes[metric_idx]
            for p in range(args.seq_len):
                color = cmap(p / max(args.seq_len - 1, 1))
                d = results[f'pos{p}'][key]
                per_act = np.median(d, axis=2)

                if plot_type == 'median':
                    median = np.median(per_act, axis=0)
                    ax.plot(angles_deg, median, color=color, label=f'pos {p}', linewidth=2)
                else:
                    np.random.seed(42)
                    chosen = np.random.choice(per_act.shape[0], min(5, per_act.shape[0]), replace=False)
                    for j, idx in enumerate(chosen):
                        ax.plot(angles_deg, per_act[idx], color=color, alpha=0.3, linewidth=1,
                                label=f'pos {p}' if j == 0 else None)

            ax.set_xlabel('Perturbation angle (degrees)')
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, args.max_angle)

        fig.suptitle(f'{args.model}, layer {args.layer}, seq_len={args.seq_len} — each position separately ({plot_type})',
                     fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(args.results_dir, f'each_position_{plot_type}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
