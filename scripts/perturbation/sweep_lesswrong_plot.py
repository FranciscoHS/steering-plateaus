#!/usr/bin/env python3
"""Combined perturbation sweep for LessWrong post figure.

Sweeps perturbation response along three direction types:
  1. Random directions
  2. Contrastive directions (English ↔ Mandarin difference-of-means)
  3. SAE feature directions (top active Gemma Scope features)

Plots individual curves (not mean+shading). Runs three variants:
  (a) L2 distance at layer 24, perturb at layer 2   [main plot]
  (b) Cosine distance at layer 24, perturb at layer 2
  (c) L2 distance at layer 24, perturb at layer 8

Usage:
    python scripts/perturbation/sweep_lesswrong_plot.py [--n_act 30] [--n_dirs 10]
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
from scripts.perturbation.sweep_sae_features import (
    load_gemma_scope_sae, get_top_sae_directions,
)
from data.bilingual_prompts import ENGLISH_PROMPTS, CHINESE_PROMPTS


def sweep_direction_l2(model, ctx, act, direction, layer, measure_layer, device,
                       angles_rad, batch_size=64):
    """Sweep a single direction, return L2 distance array at measure_layer."""
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=model_dtype)
    R = a_d.float().norm()
    a_unit = a_d.float() / R
    d_d = direction.to(device=device, dtype=torch.float32)
    d_orth = d_d - (d_d @ a_unit) * a_unit
    d_orth = F.normalize(d_orth, dim=0)

    thetas = torch.tensor(angles_rad, dtype=torch.float32, device=device)
    cos_t = thetas.cos().unsqueeze(1)
    sin_t = thetas.sin().unsqueeze(1)
    all_points = R * (cos_t * a_unit.unsqueeze(0) + sin_t * d_orth.unsqueeze(0))
    all_points = all_points.to(model_dtype)

    l2_arr = np.zeros(len(angles_rad))
    with torch.no_grad():
        ref = forward_from_layer_to_layer(
            model, ctx_d, a_d[None, None], layer, measure_layer).float()
        for start in range(0, len(angles_rad), batch_size):
            end = min(start + batch_size, len(angles_rad))
            batch_points = all_points[start:end].unsqueeze(1)
            pf = forward_from_layer_to_layer(
                model, ctx_d, batch_points, layer, measure_layer).float()
            l2_arr[start:end] = (pf - ref).norm(dim=-1).cpu().numpy()
    return l2_arr


def sweep_direction_cosine(model, ctx, act, direction, layer, measure_layer, device,
                           angles_rad, batch_size=64):
    """Sweep a single direction, return cosine distance array at measure_layer."""
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=model_dtype)
    R = a_d.float().norm()
    a_unit = a_d.float() / R
    d_d = direction.to(device=device, dtype=torch.float32)
    d_orth = d_d - (d_d @ a_unit) * a_unit
    d_orth = F.normalize(d_orth, dim=0)

    thetas = torch.tensor(angles_rad, dtype=torch.float32, device=device)
    cos_t = thetas.cos().unsqueeze(1)
    sin_t = thetas.sin().unsqueeze(1)
    all_points = R * (cos_t * a_unit.unsqueeze(0) + sin_t * d_orth.unsqueeze(0))
    all_points = all_points.to(model_dtype)

    cos_arr = np.zeros(len(angles_rad))
    with torch.no_grad():
        ref = forward_from_layer_to_layer(
            model, ctx_d, a_d[None, None], layer, measure_layer).float()
        for start in range(0, len(angles_rad), batch_size):
            end = min(start + batch_size, len(angles_rad))
            batch_points = all_points[start:end].unsqueeze(1)
            pf = forward_from_layer_to_layer(
                model, ctx_d, batch_points, layer, measure_layer).float()
            cos_arr[start:end] = (1.0 - F.cosine_similarity(ref, pf, dim=-1)).cpu().numpy()
    return cos_arr


def get_contrastive_direction(model, tokenizer, device, layer):
    """Compute English-Mandarin difference-of-means direction at given layer."""
    print(f"  Computing English-Mandarin DoM direction at layer {layer}...")
    en_acts = []
    zh_acts = []
    for p in ENGLISH_PROMPTS:
        tokens = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        en_acts.append(outputs.hidden_states[layer + 1][:, -1, :].squeeze(0).float().cpu())
    for p in CHINESE_PROMPTS:
        tokens = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        zh_acts.append(outputs.hidden_states[layer + 1][:, -1, :].squeeze(0).float().cpu())
    en_mean = torch.stack(en_acts).mean(dim=0)
    zh_mean = torch.stack(zh_acts).mean(dim=0)
    dom = en_mean - zh_mean
    dom = F.normalize(dom, dim=0)
    print(f"  DoM direction computed (pre-norm magnitude: {(en_mean - zh_mean).norm():.2f})")
    return dom


def run_sweep(model, all_contexts, all_activations, directions_dict,
              layer, measure_layer, device, angles_rad, metric='l2'):
    """Run perturbation sweep for all directions.

    Args:
        directions_dict: {name: direction_tensor} or {name: [dir1, dir2, ...]}
            If value is a list of directions, each gets its own curve.
        metric: 'l2' or 'cosine'

    Returns:
        curves: {name: array} where array is [n_angles] for single-dir or
                {name_i: array} for multi-dir
    """
    sweep_fn = sweep_direction_l2 if metric == 'l2' else sweep_direction_cosine
    n_act = len(all_activations)
    curves = {}

    for name, dirs in directions_dict.items():
        if isinstance(dirs, list):
            # Multiple directions (e.g., per-activation SAE/random)
            for j, d in enumerate(dirs):
                label = f"{name}_{j}"
                # Average across activations for this direction index
                vals = np.zeros(len(angles_rad))
                for i in range(n_act):
                    vals += sweep_fn(model, all_contexts[i], all_activations[i],
                                     d[i] if d.dim() > 1 else d,
                                     layer, measure_layer, device, angles_rad)
                curves[label] = vals / n_act
        else:
            # Single fixed direction (e.g., contrastive)
            vals = np.zeros(len(angles_rad))
            for i in range(n_act):
                vals += sweep_fn(model, all_contexts[i], all_activations[i],
                                 dirs, layer, measure_layer, device, angles_rad)
            curves[name] = vals / n_act
        print(f"  Done: {name}")

    return curves


def run_individual_sweep(model, all_contexts, all_activations,
                         sae_dirs, rand_dirs, contrastive_dir,
                         layer, measure_layer, device, angles_rad,
                         metric='l2', n_curves_per_type=10):
    """Run sweep with individual curves per direction (not averaged).

    For SAE and random: pick n_curves_per_type random (activation, direction) pairs.
    For contrastive: one curve per activation (same direction).

    Returns dict of {label: [n_angles] array}.
    """
    sweep_fn = sweep_direction_l2 if metric == 'l2' else sweep_direction_cosine
    n_act = len(all_activations)
    n_dirs = sae_dirs.shape[1]
    curves = {}

    # SAE individual curves: pick random (act, dir) pairs
    rng = np.random.RandomState(42)
    act_indices = rng.randint(0, n_act, size=n_curves_per_type)
    dir_indices = rng.randint(0, n_dirs, size=n_curves_per_type)
    print(f"  Sweeping {n_curves_per_type} SAE curves...")
    t0 = time.time()
    for k in range(n_curves_per_type):
        i, j = act_indices[k], dir_indices[k]
        curves[f'sae_{k}'] = sweep_fn(
            model, all_contexts[i], all_activations[i], sae_dirs[i, j],
            layer, measure_layer, device, angles_rad)
    print(f"    {time.time()-t0:.1f}s")

    # Random individual curves: same (act, dir) index pattern for fairness
    print(f"  Sweeping {n_curves_per_type} random curves...")
    t0 = time.time()
    for k in range(n_curves_per_type):
        i, j = act_indices[k], dir_indices[k]
        curves[f'random_{k}'] = sweep_fn(
            model, all_contexts[i], all_activations[i], rand_dirs[i, j],
            layer, measure_layer, device, angles_rad)
    print(f"    {time.time()-t0:.1f}s")

    # Contrastive individual curves: one per activation
    n_contrastive = min(n_curves_per_type, n_act)
    print(f"  Sweeping {n_contrastive} contrastive curves...")
    t0 = time.time()
    for k in range(n_contrastive):
        curves[f'contrastive_{k}'] = sweep_fn(
            model, all_contexts[k], all_activations[k], contrastive_dir,
            layer, measure_layer, device, angles_rad)
    print(f"    {time.time()-t0:.1f}s")

    return curves


def plot_individual_curves(curves, angles_deg, metric_name, title, save_path):
    """Plot individual perturbation curves grouped by direction type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by type
    sae_keys = sorted([k for k in curves if k.startswith('sae_')])
    rand_keys = sorted([k for k in curves if k.startswith('random_')])
    cont_keys = sorted([k for k in curves if k.startswith('contrastive_')])

    for k in rand_keys:
        ax.plot(angles_deg, curves[k], color='C0', alpha=0.3, lw=0.8)
    for k in cont_keys:
        ax.plot(angles_deg, curves[k], color='C2', alpha=0.4, lw=0.8)
    for k in sae_keys:
        ax.plot(angles_deg, curves[k], color='C3', alpha=0.3, lw=0.8)

    # Legend entries (invisible lines for clean legend)
    ax.plot([], [], color='C0', lw=2, label=f'Random ({len(rand_keys)} curves)')
    ax.plot([], [], color='C2', lw=2, label=f'Contrastive EN↔ZH ({len(cont_keys)} curves)')
    ax.plot([], [], color='C3', lw=2, label=f'SAE features ({len(sae_keys)} curves)')

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, angles_deg[-1])
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Combined perturbation sweep for LessWrong post')
    parser.add_argument('--model', default='gemma-2-2b')
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=30,
                        help='Number of real activations to sample')
    parser.add_argument('--n_dirs', type=int, default=10,
                        help='Number of SAE/random directions per activation')
    parser.add_argument('--n_curves', type=int, default=15,
                        help='Number of individual curves to plot per type')
    parser.add_argument('--sae_width', default='16k')
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default=None)
    # Which variants to run
    parser.add_argument('--skip_main', action='store_true',
                        help='Skip main plot (L2, layer 2→24)')
    parser.add_argument('--skip_cosine', action='store_true',
                        help='Skip cosine variant')
    parser.add_argument('--skip_layer8', action='store_true',
                        help='Skip layer-8 variant')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/lesswrong_plot/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    load_dtype = dtype_map[args.dtype]
    print(f"Loading model: {args.model} (dtype={args.dtype})")
    model, tokenizer, device = load_model(args.model, dtype=load_dtype)
    n_layers = len(_get_blocks(model))
    d_model = model._plateau_info['d_model']
    # Extract model size for SAE loading (e.g., 'gemma-2-2b' -> '2b', 'gemma-2-9b' -> '9b')
    sae_model_size = args.model.split('-')[-1] if 'gemma' in args.model else '2b'
    print(f"d_model={d_model}, n_layers={n_layers}, sae_model_size={sae_model_size}")

    # Angle grid
    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # === Configurations to run ===
    configs = []
    if not args.skip_main:
        configs.append({
            'name': 'main',
            'layer': 2,
            'measure_layer': n_layers - 2,  # 24 for 26-layer model
            'metric': 'l2',
            'title_suffix': f'inject@L2, measure@L{n_layers-2}',
            'ylabel': f'L2 distance (layer {n_layers-2})',
        })
    if not args.skip_cosine:
        configs.append({
            'name': 'cosine',
            'layer': 2,
            'measure_layer': n_layers - 2,
            'metric': 'cosine',
            'title_suffix': f'inject@L2, measure@L{n_layers-2}',
            'ylabel': f'Cosine distance (layer {n_layers-2})',
        })
    if not args.skip_layer8:
        configs.append({
            'name': 'layer8',
            'layer': 8,
            'measure_layer': n_layers - 2,
            'metric': 'l2',
            'title_suffix': f'inject@L8, measure@L{n_layers-2}',
            'ylabel': f'L2 distance (layer {n_layers-2})',
        })

    for cfg in configs:
        layer = cfg['layer']
        measure_layer = cfg['measure_layer']
        metric = cfg['metric']
        print(f"\n{'='*60}")
        print(f"Config: {cfg['name']} — {cfg['title_suffix']}, metric={metric}")
        print(f"{'='*60}")

        # Extract activations at this layer
        print(f"\nLoading {args.n_act} FineWeb sequences...")
        token_ids_list = load_fineweb_fixed_length(
            args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
        print(f"Extracting layer-{layer} activations...")
        all_contexts, all_activations = extract_context_and_activation(
            model, tokenizer, token_ids_list, layer, device)
        print(f"Activation norms: mean={all_activations.norm(dim=-1).mean():.2f}")

        # Contrastive direction (English ↔ Mandarin)
        contrastive_dir = get_contrastive_direction(model, tokenizer, device, layer)

        # SAE directions
        print(f"\nLoading SAE (width={args.sae_width}) for layer {layer}...")
        sae = load_gemma_scope_sae(layer, width=args.sae_width, device=device,
                                   model_size=sae_model_size)
        sae_dirs, feature_ids = get_top_sae_directions(
            sae, all_activations, args.n_dirs, device)
        print(f"Top feature IDs range: {feature_ids.min().item()}-{feature_ids.max().item()}")
        del sae
        torch.cuda.empty_cache()

        # Random directions (pre-generate, matched shape to SAE)
        rand_dirs = torch.randn(args.n_act, args.n_dirs, d_model)
        rand_dirs = F.normalize(rand_dirs, dim=-1)

        # Run individual sweeps
        print(f"\nRunning sweeps ({metric})...")
        curves = run_individual_sweep(
            model, all_contexts, all_activations,
            sae_dirs, rand_dirs, contrastive_dir,
            layer, measure_layer, device, angles_rad,
            metric=metric, n_curves_per_type=args.n_curves)

        # Save data
        save_data = {
            'args': vars(args),
            'config': cfg,
            'angles_deg': angles_deg,
            'curves': curves,
            'feature_ids': feature_ids.numpy(),
        }
        pkl_path = os.path.join(args.results_dir, f'curves_{cfg["name"]}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"  Saved: {pkl_path}")

        # Plot
        title = f'{args.model}: {cfg["title_suffix"]}'
        plot_path = os.path.join(args.results_dir, f'plot_{cfg["name"]}.png')
        plot_individual_curves(curves, angles_deg, cfg['ylabel'], title, plot_path)

    print("\nAll done!")


if __name__ == '__main__':
    main()
