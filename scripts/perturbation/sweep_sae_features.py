#!/usr/bin/env python3
"""Perturbation sweep along SAE feature directions vs random directions.

Loads a Gemma Scope SAE, encodes real activations to find active features,
then sweeps perturbations along those feature directions and compares to
random directions.

Usage:
    python scripts/perturbation/sweep_sae_features.py --model gemma-2-2b --layer 2 --measure_layer -2 --dtype bfloat16
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
from scripts.perturbation.sweep_random_points import (
    extract_context_and_activation, sweep_direction,
)


def load_gemma_scope_sae(layer, width='16k', device='cpu', model_size='2b'):
    """Load a Gemma Scope SAE for a given layer.

    Args:
        model_size: '2b' or '9b' (Gemma 2 variant)

    Returns the SAE object with .encode() and .W_dec attributes.
    """
    from sae_lens import SAE
    release = f"gemma-scope-{model_size}-pt-res-canonical"
    sae_id = f"layer_{layer}/width_{width}/canonical"
    print(f"Loading SAE: {release} / {sae_id}")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release, sae_id=sae_id, device=device
    )
    print(f"SAE loaded: {sae.W_dec.shape[0]} features, d_model={sae.W_dec.shape[1]}")
    return sae


def get_top_sae_directions(sae, activations, n_dirs, device):
    """For each activation, encode through SAE and return top-n_dirs feature directions.

    Returns:
        directions: [n_act, n_dirs, d_model] tensor of unit-norm decoder directions
        feature_ids: [n_act, n_dirs] tensor of feature indices
    """
    n_act, d_model = activations.shape
    all_directions = torch.zeros(n_act, n_dirs, d_model)
    all_feature_ids = torch.zeros(n_act, n_dirs, dtype=torch.long)

    sae_device = next(sae.parameters()).device
    W_dec = sae.W_dec.data.float()  # [n_features, d_model]

    for i in range(n_act):
        act = activations[i:i+1].to(sae_device)
        # Encode to get feature activations
        feature_acts = sae.encode(act)  # [1, n_features]
        # Get top-k active features
        topk = feature_acts[0].topk(n_dirs)
        feat_ids = topk.indices
        all_feature_ids[i] = feat_ids.cpu()
        # Get decoder directions (already unit-norm in well-trained SAEs, but normalize anyway)
        dirs = W_dec[feat_ids]  # [n_dirs, d_model]
        dirs = F.normalize(dirs, dim=-1)
        all_directions[i] = dirs.cpu()

    return all_directions, all_feature_ids


def main():
    parser = argparse.ArgumentParser(
        description='Perturbation sweep: SAE feature directions vs random directions')
    parser.add_argument('--model', default='gemma-2-2b')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--measure_layer', type=int, default=-2)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=50,
                        help='Number of real activations to sweep')
    parser.add_argument('--n_dirs', type=int, default=20,
                        help='Number of directions per activation')
    parser.add_argument('--sae_width', default='16k',
                        choices=['16k', '32k', '65k', '131k'],
                        help='SAE width (default: 16k)')
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/sae_sweep/{args.model}_L{args.layer}'
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    load_dtype = dtype_map[args.dtype]
    print(f"Loading model: {args.model} (dtype={args.dtype})")
    model, tokenizer, device = load_model(args.model, dtype=load_dtype)
    n_layers = len(_get_blocks(model))
    d_model = model._plateau_info['d_model']

    if args.measure_layer < 0:
        args.measure_layer = n_layers + args.measure_layer
    print(f"d_model={d_model}, n_layers={n_layers}, "
          f"inject@layer {args.layer}, measure@layer {args.measure_layer}")

    # Load data and extract activations
    print(f"\nLoading {args.n_act} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)

    print(f"Extracting layer-{args.layer} activations...")
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Activation norms: mean={all_activations.norm(dim=-1).mean():.2f}, "
          f"std={all_activations.norm(dim=-1).std():.2f}")

    # Load SAE and get feature directions
    print(f"\nLoading Gemma Scope SAE (width={args.sae_width})...")
    sae = load_gemma_scope_sae(args.layer, width=args.sae_width, device=device)

    print(f"Encoding activations and getting top-{args.n_dirs} feature directions...")
    sae_directions, feature_ids = get_top_sae_directions(
        sae, all_activations, args.n_dirs, device)
    print(f"Feature ID range: {feature_ids.min().item()}-{feature_ids.max().item()}")

    # Free SAE memory
    del sae
    torch.cuda.empty_cache()

    # Angle grid
    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Sweep SAE feature directions
    print(f"\n=== SAE feature directions ({args.n_act} acts x {args.n_dirs} dirs) ===")
    sae_curves = np.zeros((args.n_act, args.n_dirs, len(angles_deg)))
    t0 = time.time()
    for i in range(args.n_act):
        for j in range(args.n_dirs):
            sae_curves[i, j] = sweep_direction(
                model, all_contexts[i], all_activations[i],
                sae_directions[i, j],
                args.layer, args.measure_layer, device, angles_rad)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
    print(f"SAE directions done in {time.time()-t0:.1f}s")

    # Sweep random directions for comparison
    print(f"\n=== Random directions ({args.n_act} acts x {args.n_dirs} dirs) ===")
    rand_curves = np.zeros((args.n_act, args.n_dirs, len(angles_deg)))
    t0 = time.time()
    for i in range(args.n_act):
        for j in range(args.n_dirs):
            d = torch.randn(d_model)
            rand_curves[i, j] = sweep_direction(
                model, all_contexts[i], all_activations[i], d,
                args.layer, args.measure_layer, device, angles_rad)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
    print(f"Random directions done in {time.time()-t0:.1f}s")

    # Save results
    results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'sae_curves': sae_curves,    # [n_act, n_dirs, n_angles]
        'rand_curves': rand_curves,  # [n_act, n_dirs, n_angles]
        'feature_ids': feature_ids.numpy(),
        'act_norms': all_activations.norm(dim=-1).numpy(),
    }
    pkl_path = os.path.join(args.results_dir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # Plot: median + IQR for both groups
    # Per-activation median across directions
    sae_per_act = np.median(sae_curves, axis=1)   # [n_act, n_angles]
    rand_per_act = np.median(rand_curves, axis=1)  # [n_act, n_angles]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Random directions
    ax.fill_between(angles_deg,
                    np.percentile(rand_per_act, 25, axis=0),
                    np.percentile(rand_per_act, 75, axis=0),
                    color='C0', alpha=0.15)
    ax.plot(angles_deg, np.median(rand_per_act, axis=0),
            color='C0', lw=2.5, label=f'Random directions (n={args.n_dirs}/act)')

    # SAE feature directions
    ax.fill_between(angles_deg,
                    np.percentile(sae_per_act, 25, axis=0),
                    np.percentile(sae_per_act, 75, axis=0),
                    color='C3', alpha=0.15)
    ax.plot(angles_deg, np.median(sae_per_act, axis=0),
            color='C3', lw=2.5, label=f'SAE feature directions (top-{args.n_dirs}/act)')

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel(f'Cosine distance (at layer {args.measure_layer})')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, args.max_angle)
    ax.set_title(f'{args.model}: SAE features vs random directions, '
                 f'inject@L{args.layer} measure@L{args.measure_layer}')
    plt.tight_layout()

    plot_path = os.path.join(args.results_dir, 'sae_vs_random_directions.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Numerical summary
    for name, curves in [('Random', rand_per_act), ('SAE', sae_per_act)]:
        med = np.median(curves, axis=0)
        crossings = np.where(med > 0.1)[0]
        if len(crossings) > 0:
            print(f"{name}: median crosses 0.1 at {angles_deg[crossings[0]]:.1f}°")
        else:
            print(f"{name}: median never crosses 0.1")


if __name__ == '__main__':
    main()
