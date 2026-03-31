#!/usr/bin/env python3
"""Perturbation sweep with mean-ablation of attention or MLP.

Two-pass approach:
1. Run the normal perturbation sweep, recording attention/MLP outputs at each layer and angle.
2. Compute the mean of each component's output across all angles.
3. Re-run the sweep with the component's output clamped to that mean.

Usage:
    python scripts/perturbation/sweep_mean_ablation.py --model qwen3-1.7b --seq_len 2 --layer 2 --ablate attention
    python scripts/perturbation/sweep_mean_ablation.py --model qwen3-1.7b --seq_len 2 --layer 2 --ablate mlp
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
    load_model, forward_from_layer_to_final, forward_from_layer_to_layer,
    _get_final_norm, _get_lm_head, _get_blocks,
)
from src.data import load_fineweb_fixed_length


def get_modules_to_ablate(model, ablate_type):
    """Get the list of modules to ablate (attention or MLP) from each block."""
    blocks = _get_blocks(model)
    modules = []
    for block in blocks:
        if ablate_type == 'attention':
            mod = getattr(block, 'self_attn', None) or getattr(block, 'attn', None)
        else:  # mlp
            mod = getattr(block, 'mlp', None)
        if mod is None:
            raise ValueError(f"Cannot find {ablate_type} in block: {type(block)}")
        modules.append(mod)
    return modules


def record_outputs_hook(storage, layer_idx):
    """Hook that records module outputs."""
    def hook_fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        storage[layer_idx].append(out.detach().clone())
    return hook_fn


def mean_ablate_hook(mean_per_dir, mean_grand):
    """Hook that replaces module output with the per-direction mean.

    Uses per-direction mean [n_dirs, seq, d] when batch=n_dirs (perturbed),
    and grand mean [seq, d] when batch=1 (reference).
    """
    n_dirs = mean_per_dir.shape[0]
    def hook_fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if out.shape[0] == n_dirs:
            replacement = mean_per_dir
        else:
            replacement = mean_grand.expand_as(out)
        if isinstance(output, tuple):
            return (replacement,) + output[1:]
        return replacement
    return hook_fn


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device, batch_size=64):
    """Extract context and final-token activation at a given layer."""
    all_contexts = []
    all_activations = []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].float().cpu())
        all_activations.append(hidden[:, -1, :].float().cpu())
    return torch.cat(all_contexts, 0), torch.cat(all_activations, 0)


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


def run_sweep_with_mean_ablation(model, context, activation, v_perps, angles_rad,
                                  layer, device, n_dirs, ablate_type, measure_layer=None):
    """Run perturbation sweep: first record component outputs, then re-run with mean ablation.

    Returns:
        normal: (l2, kl, cos) arrays [n_angles, n_dirs] — normal sweep
        ablated: (l2, kl, cos) arrays [n_angles, n_dirs] — mean-ablated sweep
    """
    n_angles = len(angles_rad)
    model_dtype = next(model.parameters()).dtype
    ctx = context.unsqueeze(0).to(device=device, dtype=model_dtype)
    a = activation.to(device=device, dtype=model_dtype)
    R = a.norm()
    a_unit = a / R
    vp = v_perps.to(device=device, dtype=model_dtype)

    modules = get_modules_to_ablate(model, ablate_type)
    n_layers = len(modules)
    # Only layers after the injection point are actually run
    active_layers = list(range(layer + 1, n_layers))

    # Build all perturbed points
    all_perturbed = []
    for theta in angles_rad:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        perturbed = R * (cos_t * a_unit.unsqueeze(0) + sin_t * vp)  # [n_dirs, d]
        all_perturbed.append(perturbed)

    # === Pass 1: Normal sweep, recording component outputs ===
    # Only hook layers that actually run (after injection point)
    storage = {i: [] for i in active_layers}
    hooks = []
    for i in active_layers:
        hooks.append(modules[i].register_forward_hook(record_outputs_hook(storage, i)))

    normal_l2 = np.zeros((n_angles, n_dirs))
    normal_kl = np.zeros((n_angles, n_dirs))
    normal_cos = np.zeros((n_angles, n_dirs))

    with torch.no_grad():
        # Reference
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
            perturbed = all_perturbed[ang_idx].unsqueeze(1).to(model_dtype)  # [n_dirs, 1, d]
            pert_final = forward_from_layer_to_final(model, ctx, perturbed, layer)

            if measure_layer is not None:
                pert_measure = forward_from_layer_to_layer(model, ctx, perturbed, layer, measure_layer)
            else:
                pert_measure = pert_final
            normal_l2[ang_idx] = (pert_measure - ref_measure).float().norm(dim=-1).cpu().numpy()
            cos_sim = F.cosine_similarity(pert_measure.float(), ref_measure.float(), dim=-1)
            normal_cos[ang_idx] = (1 - cos_sim).cpu().numpy()

            pert_ln = _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)
            pert_logits = _get_lm_head(model)(pert_ln)
            pert_log_probs = F.log_softmax(pert_logits.float(), dim=-1)
            ref_probs = ref_log_probs.exp()
            kl = (ref_probs * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            normal_kl[ang_idx] = kl.cpu().numpy()

    # Remove recording hooks
    for h in hooks:
        h.remove()

    # === Compute means across angles (per direction) ===
    # storage[layer_idx] has (1 + n_angles) entries: ref + one per angle
    # Each entry shape depends on batch size: ref is [1, seq, d], perturbed is [n_dirs, seq, d]
    # Mean across angles only, keeping directions separate
    mean_per_dir = {}   # [n_dirs, seq, d] — per-direction mean
    mean_grand = {}     # [seq, d] — grand mean (for reference point)
    for layer_idx in active_layers:
        angle_outputs = storage[layer_idx][1:]  # n_angles tensors of [n_dirs, seq, d]
        stacked = torch.stack(angle_outputs, dim=0)  # [n_angles, n_dirs, seq, d]
        mean_per_dir[layer_idx] = stacked.mean(dim=0)  # [n_dirs, seq, d]
        mean_grand[layer_idx] = stacked.mean(dim=(0, 1))  # [seq, d]

    del storage  # free memory

    # === Pass 2: Mean-ablated sweep ===
    hooks = []
    for i in active_layers:
        hooks.append(modules[i].register_forward_hook(
            mean_ablate_hook(mean_per_dir[i], mean_grand[i])))

    ablated_l2 = np.zeros((n_angles, n_dirs))
    ablated_kl = np.zeros((n_angles, n_dirs))
    ablated_cos = np.zeros((n_angles, n_dirs))

    with torch.no_grad():
        # Re-compute reference under ablation
        ref_final_abl = forward_from_layer_to_final(model, ctx, ref_point, layer)
        ref_ln_abl = _get_final_norm(model)(ref_final_abl.unsqueeze(1)).squeeze(1)
        ref_logits_abl = _get_lm_head(model)(ref_ln_abl)
        ref_log_probs_abl = F.log_softmax(ref_logits_abl.float(), dim=-1)
        if measure_layer is not None:
            ref_measure_abl = forward_from_layer_to_layer(model, ctx, ref_point, layer, measure_layer)
        else:
            ref_measure_abl = ref_final_abl

        for ang_idx, theta in enumerate(angles_rad):
            perturbed = all_perturbed[ang_idx].unsqueeze(1).to(model_dtype)
            pert_final = forward_from_layer_to_final(model, ctx, perturbed, layer)

            if measure_layer is not None:
                pert_measure = forward_from_layer_to_layer(model, ctx, perturbed, layer, measure_layer)
            else:
                pert_measure = pert_final
            ablated_l2[ang_idx] = (pert_measure - ref_measure_abl).float().norm(dim=-1).cpu().numpy()
            cos_sim = F.cosine_similarity(pert_measure.float(), ref_measure_abl.float(), dim=-1)
            ablated_cos[ang_idx] = (1 - cos_sim).cpu().numpy()

            pert_ln = _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)
            pert_logits = _get_lm_head(model)(pert_ln)
            pert_log_probs = F.log_softmax(pert_logits.float(), dim=-1)
            ref_probs_abl = ref_log_probs_abl.exp()
            kl = (ref_probs_abl * (ref_log_probs_abl - pert_log_probs)).sum(dim=-1)
            ablated_kl[ang_idx] = kl.cpu().numpy()

    for h in hooks:
        h.remove()

    normal = (normal_l2, normal_kl, normal_cos)
    ablated = (ablated_l2, ablated_kl, ablated_cos)
    return normal, ablated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=50)
    parser.add_argument('--n_dirs', type=int, default=10)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--max_angle', type=float, default=90)
    parser.add_argument('--ablate', type=str, required=True,
                        choices=['attention', 'mlp'],
                        help='Which component to mean-ablate')
    parser.add_argument('--measure_layer', type=int, default=None)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/mean_ablation/{args.model}_L{args.layer}_seq{args.seq_len}_{args.ablate}'
    os.makedirs(args.results_dir, exist_ok=True)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    print(f"Config: model={args.model}, layer={args.layer}, seq_len={args.seq_len}, "
          f"ablate={args.ablate}, n_act={args.n_act}, n_dirs={args.n_dirs}")

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))

    if args.measure_layer is not None and args.measure_layer < 0:
        args.measure_layer = n_layers + args.measure_layer

    print(f"d_model={d_model}, n_layers={n_layers}")

    token_ids_list = load_fineweb_fixed_length(args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    act_norms = activations.norm(dim=-1)
    print(f"Mean activation norm: {act_norms.mean():.2f}")

    v_perps = sample_orthogonal_directions(activations, args.n_dirs)

    # Run sweeps
    all_normal = {'l2': [], 'kl': [], 'cos': []}
    all_ablated = {'l2': [], 'kl': [], 'cos': []}

    t0 = time.time()
    for i in range(args.n_act):
        normal, ablated = run_sweep_with_mean_ablation(
            model, contexts[i], activations[i], v_perps[i], angles_rad,
            args.layer, device, args.n_dirs, args.ablate,
            measure_layer=args.measure_layer)

        for k, idx in [('l2', 0), ('kl', 1), ('cos', 2)]:
            all_normal[k].append(normal[idx])
            all_ablated[k].append(ablated[idx])

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.2f} act/s)")

    print(f"Done in {time.time()-t0:.1f}s")

    # Stack: [n_act, n_angles, n_dirs]
    for k in all_normal:
        all_normal[k] = np.stack(all_normal[k])
        all_ablated[k] = np.stack(all_ablated[k])

    results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'normal': all_normal,
        'ablated': all_ablated,
    }
    pkl_path = os.path.join(args.results_dir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {pkl_path}")

    # Plot
    metric_labels = {'l2': 'L2 distance', 'kl': 'KL divergence', 'cos': 'Cosine distance'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ['cos', 'kl', 'l2']):
        for data, label, color in [(all_normal, 'Normal', 'C0'),
                                    (all_ablated, f'Mean-ablated {args.ablate}', 'C1')]:
            per_act = np.median(data[metric], axis=2)  # [n_act, n_angles]
            median = np.median(per_act, axis=0)
            q25 = np.percentile(per_act, 25, axis=0)
            q75 = np.percentile(per_act, 75, axis=0)
            ax.plot(angles_deg, median, color=color, lw=2, label=label)
            ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.15)

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(metric_labels[metric])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{args.model} L{args.layer} seq_len={args.seq_len}: '
                 f'mean-ablated {args.ablate}', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'sweep_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


if __name__ == '__main__':
    main()
