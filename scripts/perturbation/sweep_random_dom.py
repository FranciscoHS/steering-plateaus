#!/usr/bin/env python3
"""Random difference-of-means experiment.

Split FineWeb activations into two random groups of size n, compute their
difference-of-means, then perturb a held-out set along that direction.
Vary n to see if averaging helps. Compare against true EN↔ZH DoM and random.

Usage:
    python scripts/perturbation/sweep_random_dom.py --model gemma-2-9b --layer 2
"""
import argparse, os, sys, pickle, time
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.perturbation.sweep_random_points import extract_context_and_activation
from scripts.perturbation.sweep_qwen_sae import (
    sweep_one, get_contrastive_directions,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gemma-2-9b')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_pool', type=int, default=100,
                        help='Total FineWeb activations to draw from for DoM groups')
    parser.add_argument('--n_perturb', type=int, default=30,
                        help='Number of held-out activations to perturb')
    parser.add_argument('--n_curves', type=int, default=10)
    parser.add_argument('--group_sizes', default='1,5,10,15,20,25,30',
                        help='Comma-separated group sizes for random DoM')
    parser.add_argument('--max_angle', type=float, default=90.0)
    parser.add_argument('--n_angles', type=int, default=90)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    group_sizes = [int(x) for x in args.group_sizes.split(',')]

    if args.results_dir is None:
        args.results_dir = f'results/lesswrong_plot/{args.model}-random-dom'
    os.makedirs(args.results_dir, exist_ok=True)

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

    # Load enough FineWeb activations for both pool and perturbation
    total_needed = args.n_pool + args.n_perturb
    print(f"\nLoading {total_needed} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(
        total_needed, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.layer} activations...")
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)

    # Split: first n_pool for computing DoM directions, rest for perturbation
    pool_acts = all_activations[:args.n_pool]
    perturb_contexts = all_contexts[args.n_pool:]
    perturb_acts = all_activations[args.n_pool:]
    print(f"Pool: {len(pool_acts)}, Perturb: {len(perturb_acts)}")

    # True EN↔ZH DoM
    dom_dir, _ = get_contrastive_directions(model, tokenizer, device, args.layer)
    neg_dom = -dom_dir

    # Random baseline
    rand_dirs = F.normalize(torch.randn(args.n_curves, d_model), dim=-1)

    # For each group size, compute random DoM directions
    rng = np.random.RandomState(args.seed)
    random_dom_dirs = {}
    for n in group_sizes:
        dirs = []
        for _ in range(args.n_curves):
            indices = rng.choice(len(pool_acts), size=2*n, replace=False)
            group_a = pool_acts[indices[:n]]
            group_b = pool_acts[indices[n:]]
            diff = group_a.mean(0) - group_b.mean(0)
            dirs.append(F.normalize(diff, dim=0))
        random_dom_dirs[n] = torch.stack(dirs)
        # Check similarity to true DoM
        cos_sim = (random_dom_dirs[n] @ dom_dir).abs().mean().item()
        norm = diff.norm().item()
        print(f"  n={n}: |cos(random_dom, true_dom)|={cos_sim:.4f}, last diff norm={norm:.2f}")

    # Run sweeps
    curves = {}

    # Baselines: random and -DoM
    print(f"\n  Sweeping {args.n_curves} random curves...")
    t0 = time.time()
    for k in range(args.n_curves):
        curves[f'random_{k}'] = sweep_one(
            model, perturb_contexts[k], perturb_acts[k],
            rand_dirs[k], args.layer, measure_layer, device, angles_rad, metric='l2')
    print(f"    {time.time()-t0:.1f}s")

    n_dom = min(args.n_curves, len(perturb_acts))
    print(f"  Sweeping {n_dom} -DoM curves...")
    t0 = time.time()
    for k in range(n_dom):
        curves[f'neg_dom_{k}'] = sweep_one(
            model, perturb_contexts[k], perturb_acts[k],
            neg_dom, args.layer, measure_layer, device, angles_rad, metric='l2')
    print(f"    {time.time()-t0:.1f}s")

    # Random DoM at each group size
    for n in group_sizes:
        label = f'rdom_n{n}'
        print(f"  Sweeping {args.n_curves} {label} curves...")
        t0 = time.time()
        for k in range(args.n_curves):
            curves[f'{label}_{k}'] = sweep_one(
                model, perturb_contexts[k], perturb_acts[k],
                random_dom_dirs[n][k], args.layer, measure_layer, device, angles_rad, metric='l2')
        print(f"    {time.time()-t0:.1f}s")

    # Save
    pkl_path = os.path.join(args.results_dir, 'curves_l2.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'args': vars(args),
            'angles_deg': angles_deg,
            'curves': curves,
            'group_sizes': group_sizes,
        }, f)
    print(f"Saved: {pkl_path}")

    # Plot: median for each group size + baselines
    fig, ax = plt.subplots(figsize=(12, 7))

    # Random baseline
    rkeys = sorted([k for k in curves if k.startswith('random_')])
    rdata = np.array([curves[k] for k in rkeys])
    ax.plot(angles_deg, np.median(rdata, axis=0), color='C0', lw=2.5, label='Random')
    ax.fill_between(angles_deg, np.percentile(rdata, 25, axis=0),
                     np.percentile(rdata, 75, axis=0), color='C0', alpha=0.1)

    # -DoM baseline
    dkeys = sorted([k for k in curves if k.startswith('neg_dom_')])
    ddata = np.array([curves[k] for k in dkeys])
    ax.plot(angles_deg, np.median(ddata, axis=0), color='C3', lw=2.5, label='-DoM (EN↔ZH)')
    ax.fill_between(angles_deg, np.percentile(ddata, 25, axis=0),
                     np.percentile(ddata, 75, axis=0), color='C3', alpha=0.1)

    # Random DoM at each group size
    cmap = plt.cm.viridis
    for i, n in enumerate(group_sizes):
        color = cmap(i / max(1, len(group_sizes) - 1))
        label_n = f'rdom_n{n}'
        gkeys = sorted([k for k in curves if k.startswith(f'{label_n}_')])
        gdata = np.array([curves[k] for k in gkeys])
        ax.plot(angles_deg, np.median(gdata, axis=0), color=color, lw=1.5,
                label=f'Random DoM (n={n})')

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=14)
    ax.set_ylabel('L2 distance (layer 40)', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, angles_deg[-1])
    ax.set_title(f'{args.model}: Random DoM vs true DoM', fontsize=15)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'plot_l2_random_dom.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
