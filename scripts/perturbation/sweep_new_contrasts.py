#!/usr/bin/env python3
"""Perturbation sweep for new contrastive directions (programming, sentiment).

Runs on Gemma-2-9B at layer 2. For each contrast:
  - Extracts FineWeb activations (30 samples, seq_len=5)
  - Computes DoM direction from contrastive prompts
  - Negates it (perturb AWAY from FineWeb distribution)
  - Sweeps perturbation angles 0-90°
  - Compares against improved random baseline (diff of two random FineWeb activations)

Usage:
    python scripts/perturbation/sweep_new_contrasts.py --contrast programming
    python scripts/perturbation/sweep_new_contrasts.py --contrast sentiment
"""
import argparse, os, sys, pickle, time
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_to_layer, _get_blocks,
)
from src.data import load_fineweb_fixed_length
from scripts.perturbation.sweep_random_points import extract_context_and_activation
from scripts.perturbation.sweep_qwen_sae import sweep_one


def get_contrastive_direction(model, tokenizer, device, layer, prompts_a, prompts_b,
                              label_a='A', label_b='B'):
    """Compute DoM direction between two groups of prompts.

    Returns:
        dom: [d_model] unit difference-of-means direction (A - B)
    """
    print(f"  Computing {label_a}↔{label_b} DoM at layer {layer}...")
    def get_acts(prompts):
        acts = []
        for p in prompts:
            tokens = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            acts.append(outputs.hidden_states[layer + 1][:, -1, :].squeeze(0).float().cpu())
        return torch.stack(acts)

    acts_a = get_acts(prompts_a)
    acts_b = get_acts(prompts_b)
    mean_vec = acts_a.mean(0) - acts_b.mean(0)
    dom = F.normalize(mean_vec, dim=0)
    print(f"  DoM pre-norm magnitude: {mean_vec.norm():.2f}")
    return dom


def get_improved_random_directions(all_activations, n_dirs, seed=123):
    """Improved random baseline: normalized difference of two random FineWeb activations."""
    rng = np.random.RandomState(seed)
    n_act = len(all_activations)
    dirs = []
    for _ in range(n_dirs):
        i, j = rng.choice(n_act, size=2, replace=False)
        d = all_activations[i] - all_activations[j]
        d = F.normalize(d, dim=0)
        dirs.append(d)
    return torch.stack(dirs)


def plot_curves(curves, angles_deg, ylabel, title, save_path):
    """Individual curves at alpha=0.5 with thick median on top."""
    fig, ax = plt.subplots(figsize=(10, 6))
    prefix_map = {
        'random_': ('C0', 'Random'),
        'contrastive_': ('C2', 'Contrastive'),
    }
    for prefix, (color, label) in prefix_map.items():
        keys = sorted([k for k in curves if k.startswith(prefix)])
        if not keys:
            continue
        data = np.array([curves[k] for k in keys])
        # Individual curves
        for k in keys:
            ax.plot(angles_deg, curves[k], color=color, alpha=0.5, lw=0.8)
        # Thick median
        med = np.median(data, axis=0)
        ax.plot(angles_deg, med, color=color, lw=3.0, label=f'{label} ({len(keys)})')

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, angles_deg[-1])
    ax.set_title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gemma-2-9b')
    parser.add_argument('--contrast', required=True, choices=['programming', 'sentiment', 'gender'])
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=30)
    parser.add_argument('--n_curves', type=int, default=10)
    parser.add_argument('--max_angle', type=float, default=90.0)
    parser.add_argument('--n_angles', type=int, default=90)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/lesswrong_plot/{args.model}-{args.contrast}'
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16}
    load_dtype = dtype_map[args.dtype]
    print(f"Loading model: {args.model} (dtype={args.dtype})")
    model, tokenizer, device = load_model(args.model, dtype=load_dtype)
    n_layers = len(_get_blocks(model))
    d_model = model._plateau_info['d_model']
    measure_layer = n_layers - 2
    print(f"d_model={d_model}, n_layers={n_layers}, perturb@L{args.layer}, measure@L{measure_layer}")

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Load contrastive prompts
    if args.contrast == 'programming':
        from data.programming_prompts import PYTHON_PROMPTS, HASKELL_PROMPTS
        prompts_a, prompts_b = PYTHON_PROMPTS, HASKELL_PROMPTS
        label_a, label_b = 'Python', 'Haskell'
    elif args.contrast == 'sentiment':
        from data.sentiment_prompts import POSITIVE_PROMPTS, NEGATIVE_PROMPTS
        prompts_a, prompts_b = POSITIVE_PROMPTS, NEGATIVE_PROMPTS
        label_a, label_b = 'Positive', 'Negative'
    else:  # gender
        from data.gender_prompts import MALE_PROMPTS, FEMALE_PROMPTS
        prompts_a, prompts_b = MALE_PROMPTS, FEMALE_PROMPTS
        label_a, label_b = 'Male', 'Female'

    # Load held-out prompts for perturbation (separate from DoM computation)
    if args.contrast == 'programming':
        from data.python_prompts_holdout import PYTHON_PROMPTS_HOLDOUT
        perturb_prompts = PYTHON_PROMPTS_HOLDOUT
    elif args.contrast == 'sentiment':
        from data.positive_prompts_holdout import POSITIVE_PROMPTS_HOLDOUT
        perturb_prompts = POSITIVE_PROMPTS_HOLDOUT
    else:  # gender
        from data.male_prompts_holdout import MALE_PROMPTS_HOLDOUT
        perturb_prompts = MALE_PROMPTS_HOLDOUT

    # Extract activations from held-out prompts (pad to equal length)
    print(f"\nTokenizing {len(perturb_prompts)} held-out {label_a} prompts...")
    encoded = tokenizer(perturb_prompts, return_tensors='pt', padding=True)
    token_ids_list = list(encoded['input_ids'])
    print(f"Extracting layer-{args.layer} activations...")
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    args.n_act = len(all_activations)
    print(f"Activation norms: mean={all_activations.norm(dim=-1).mean():.2f}")

    # Compute contrastive DoM direction (A - B), then negate to perturb away from A
    dom = get_contrastive_direction(
        model, tokenizer, device, args.layer, prompts_a, prompts_b, label_a, label_b)
    dom = -dom  # negate: perturb away from group A toward group B
    print(f"  Using NEGATED DoM (away from {label_a} toward {label_b})")

    # Improved random baseline: diff of two random activations from group A
    n_random_dirs = args.n_curves
    rand_dirs = get_improved_random_directions(all_activations, n_random_dirs, seed=args.seed)
    print(f"Generated {n_random_dirs} improved random directions")

    # Run sweeps
    curves = {}
    rng = np.random.RandomState(args.seed)
    act_indices = rng.randint(0, args.n_act, size=args.n_curves)

    # Random curves
    print(f"\nSweeping {args.n_curves} random curves...")
    t0 = time.time()
    for k in range(args.n_curves):
        curves[f'random_{k}'] = sweep_one(
            model, all_contexts[act_indices[k]], all_activations[act_indices[k]],
            rand_dirs[k], args.layer, measure_layer, device, angles_rad, metric='l2')
    print(f"  {time.time()-t0:.1f}s")

    # Contrastive curves
    n_cont = min(args.n_curves, args.n_act)
    print(f"Sweeping {n_cont} contrastive curves...")
    t0 = time.time()
    for k in range(n_cont):
        curves[f'contrastive_{k}'] = sweep_one(
            model, all_contexts[k], all_activations[k],
            dom, args.layer, measure_layer, device, angles_rad, metric='l2')
    print(f"  {time.time()-t0:.1f}s")

    # Save
    save_data = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'curves': curves,
        'contrast': args.contrast,
        'labels': (label_a, label_b),
    }
    pkl_path = os.path.join(args.results_dir, 'curves_l2.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved: {pkl_path}")

    # Plot
    ylabel = f'L2 distance (layer {measure_layer})'
    title = f'{args.model}: {label_a}↔{label_b} DoM, inject@L{args.layer}, measure@L{measure_layer}'
    plot_curves(curves, angles_deg, ylabel, title,
                os.path.join(args.results_dir, 'plot_l2.png'))

    print("\nDone!")


if __name__ == '__main__':
    main()
