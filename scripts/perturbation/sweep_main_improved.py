#!/usr/bin/env python3
"""Main perturbation sweep with improved random baseline.

Experiment 1: Spherical perturbation with SAE, -DoM, and improved random baseline.
Experiment 5: Additive perturbation (same directions, not norm-preserving).

Improved random baseline: direction = normalized difference of two random activations,
rather than a Gaussian random vector.

Usage:
    python scripts/perturbation/sweep_main_improved.py --model gemma-2-9b --layer 2
"""
import argparse, os, sys, pickle, time
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_to_layer, forward_from_layer, _get_blocks,
)
from src.data import load_fineweb_fixed_length
from scripts.perturbation.sweep_random_points import extract_context_and_activation
from scripts.perturbation.sweep_sae_features import load_gemma_scope_sae
from scripts.perturbation.sweep_qwen_sae import (
    load_qwen_sae, get_top_sae_directions, get_contrastive_directions,
)


def sweep_one_spherical(model, ctx, act, direction, layer, measure_layer, device,
                        angles_rad, metric='l2', batch_size=64):
    """Spherical perturbation sweep (norm-preserving rotation on the sphere)."""
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
    all_points = (R * (cos_t * a_unit.unsqueeze(0) + sin_t * d_orth.unsqueeze(0))).to(model_dtype)

    return _measure_response(model, ctx_d, a_d, all_points, layer, measure_layer,
                             device, metric, batch_size, angles_rad)


def sweep_one_additive(model, ctx, act, direction, layer, measure_layer, device,
                       magnitudes, metric='l2', batch_size=64):
    """Additive perturbation sweep (add magnitude * direction, changes norm)."""
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=model_dtype)
    d_d = direction.to(device=device, dtype=model_dtype)

    # a + mag * d for each magnitude
    mags = torch.tensor(magnitudes, dtype=torch.float32, device=device)
    all_points = (a_d.unsqueeze(0) + mags.unsqueeze(1) * d_d.unsqueeze(0)).to(model_dtype)

    return _measure_response(model, ctx_d, a_d, all_points, layer, measure_layer,
                             device, metric, batch_size, magnitudes)


def _measure_response(model, ctx_d, a_d, all_points, layer, measure_layer,
                      device, metric, batch_size, x_values):
    """Measure downstream response for a batch of perturbed activations."""
    use_logits = metric in ('kl', 'logit_l2', 'logit_cosine')
    fwd_fn = forward_from_layer if use_logits else forward_from_layer_to_layer
    model_dtype = next(model.parameters()).dtype

    result = np.zeros(len(x_values))
    with torch.no_grad():
        if use_logits:
            ref = fwd_fn(model, ctx_d, a_d[None, None], layer).float()
        else:
            ref = fwd_fn(model, ctx_d, a_d[None, None], layer, measure_layer).float()
        if metric == 'kl':
            ref_logp = F.log_softmax(ref, dim=-1)

        for start in range(0, len(x_values), batch_size):
            end = min(start + batch_size, len(x_values))
            batch_points = all_points[start:end].unsqueeze(1)
            if use_logits:
                pf = fwd_fn(model, ctx_d, batch_points, layer).float()
            else:
                pf = fwd_fn(model, ctx_d, batch_points, layer, measure_layer).float()
            if metric == 'l2':
                result[start:end] = (pf - ref).norm(dim=-1).cpu().numpy()
            elif metric == 'cosine':
                result[start:end] = (1.0 - F.cosine_similarity(ref, pf, dim=-1)).cpu().numpy()
            elif metric == 'kl':
                pf_logp = F.log_softmax(pf, dim=-1)
                kl = F.kl_div(pf_logp, ref_logp.expand_as(pf_logp),
                              log_target=True, reduction='none').sum(-1)
                result[start:end] = kl.cpu().numpy()
    return result


def get_improved_random_dirs(all_activations, n_dirs, seed=42):
    """Improved random baseline: normalized difference of random activation pairs."""
    rng = np.random.RandomState(seed)
    n = len(all_activations)
    dirs = []
    for _ in range(n_dirs):
        i, j = rng.choice(n, size=2, replace=False)
        diff = all_activations[i] - all_activations[j]
        dirs.append(F.normalize(diff, dim=0))
    return torch.stack(dirs)


def plot_results(curves, x_values, ylabel, xlabel, title, save_path, direction_labels):
    """Individual curves at alpha=0.5 + thick median."""
    colors = {'random': 'C0', 'neg_dom': 'C2', 'sae': 'C3'}
    fig, ax = plt.subplots(figsize=(10, 6))
    for prefix, label in direction_labels.items():
        color = colors.get(prefix, 'C4')
        keys = sorted([k for k in curves if k.rsplit('_', 1)[0] == prefix])
        if not keys:
            continue
        data = np.array([curves[k] for k in keys])
        for k in keys:
            ax.plot(x_values, curves[k], color=color, alpha=0.5, lw=0.8)
        med = np.median(data, axis=0)
        ax.plot(x_values, med, color=color, lw=2.5, label=label)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_values[0], x_values[-1])
    ax.set_title(title, fontsize=15)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gemma-2-9b')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--sae_width', default='16k')
    parser.add_argument('--sae_type', default='auto', choices=['auto', 'qwen', 'gemma'])
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=30)
    parser.add_argument('--n_dirs', type=int, default=10)
    parser.add_argument('--n_curves', type=int, default=10)
    parser.add_argument('--max_angle', type=float, default=90.0)
    parser.add_argument('--n_angles', type=int, default=90)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--experiments', default='spherical,additive',
                        help='Comma-separated: spherical, additive')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/lesswrong_plot/{args.model}-improved'
    os.makedirs(args.results_dir, exist_ok=True)

    experiments = args.experiments.split(',')

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16}
    load_dtype = dtype_map[args.dtype]
    print(f"Loading model: {args.model} (dtype={args.dtype})")
    model, tokenizer, device = load_model(args.model, dtype=load_dtype)
    n_layers = len(_get_blocks(model))
    d_model = model._plateau_info['d_model']
    measure_layer = n_layers - 2
    print(f"d_model={d_model}, n_layers={n_layers}, perturb@L{args.layer}, measure@L{measure_layer}")

    # We need extra activations for the improved random baseline
    # n_act for perturbation targets, plus extra for computing random diff directions
    n_extra = max(args.n_act, args.n_curves * 2)
    n_total = args.n_act + n_extra
    print(f"\nLoading {n_total} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(
        n_total, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.layer} activations...")
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)

    # Split: first n_act for perturbation, rest for random baseline directions
    perturb_contexts = all_contexts[:args.n_act]
    perturb_acts = all_activations[:args.n_act]
    pool_acts = all_activations[args.n_act:]
    print(f"Perturb: {len(perturb_acts)}, Pool for random dirs: {len(pool_acts)}")
    print(f"Activation norms: mean={perturb_acts.norm(dim=-1).mean():.2f}")

    # Directions
    # 1. Improved random baseline
    print("\nComputing improved random baseline directions...")
    rand_dirs = get_improved_random_dirs(pool_acts, n_dirs=args.n_curves, seed=args.seed)

    # 2. -DoM
    dom_dir, _ = get_contrastive_directions(model, tokenizer, device, args.layer)
    neg_dom = -dom_dir
    print(f"  Using -DoM (toward Chinese)")

    # 3. SAE top-active features
    sae_type = args.sae_type
    if sae_type == 'auto':
        sae_type = 'gemma' if 'gemma' in args.model else 'qwen'
    print(f"\nLoading {sae_type} SAE (width={args.sae_width}) for layer {args.layer}...")
    if sae_type == 'gemma':
        model_size = '9b' if '9b' in args.model else '2b'
        sae = load_gemma_scope_sae(args.layer, width=args.sae_width,
                                    device=device, model_size=model_size)
    else:
        sae = load_qwen_sae(args.layer, width=args.sae_width, device=device,
                             dtype=torch.float32)
    sae_dirs, feature_ids = get_top_sae_directions(
        sae, perturb_acts, args.n_dirs, device)
    print(f"Top feature IDs range: {feature_ids.min().item()}-{feature_ids.max().item()}")
    del sae
    torch.cuda.empty_cache()

    # Select SAE directions (one per curve)
    rng = np.random.RandomState(args.seed)
    act_indices = rng.randint(0, args.n_act, size=args.n_curves)
    dir_indices = rng.randint(0, args.n_dirs, size=args.n_curves)

    direction_labels = {
        'random': 'Random',
        'neg_dom': 'Contrastive',
        'sae': 'SAE',
    }

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Experiment 1: Spherical perturbation
    if 'spherical' in experiments:
        print(f"\n{'='*60}")
        print("Experiment 1: Spherical perturbation (improved random baseline)")
        print(f"{'='*60}")

        curves = {}

        print(f"  Sweeping {args.n_curves} random curves...")
        t0 = time.time()
        for k in range(args.n_curves):
            curves[f'random_{k}'] = sweep_one_spherical(
                model, perturb_contexts[act_indices[k]], perturb_acts[act_indices[k]],
                rand_dirs[k], args.layer, measure_layer, device, angles_rad)
        print(f"    {time.time()-t0:.1f}s")

        n_dom = min(args.n_curves, len(perturb_acts))
        print(f"  Sweeping {n_dom} -DoM curves...")
        t0 = time.time()
        for k in range(n_dom):
            curves[f'neg_dom_{k}'] = sweep_one_spherical(
                model, perturb_contexts[k], perturb_acts[k],
                neg_dom, args.layer, measure_layer, device, angles_rad)
        print(f"    {time.time()-t0:.1f}s")

        print(f"  Sweeping {args.n_curves} SAE curves...")
        t0 = time.time()
        for k in range(args.n_curves):
            curves[f'sae_{k}'] = sweep_one_spherical(
                model, perturb_contexts[act_indices[k]], perturb_acts[act_indices[k]],
                sae_dirs[act_indices[k], dir_indices[k]],
                args.layer, measure_layer, device, angles_rad)
        print(f"    {time.time()-t0:.1f}s")

        pkl_path = os.path.join(args.results_dir, 'curves_spherical_l2.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump({'args': vars(args), 'angles_deg': angles_deg,
                         'curves': curves, 'feature_ids': feature_ids.numpy()}, f)
        print(f"  Saved: {pkl_path}")

        plot_results(curves, angles_deg,
                     f'L2 distance (layer {measure_layer})',
                     'Perturbation angle (degrees)',
                     f'{args.model}: inject@L{args.layer}, measure@L{measure_layer} (spherical)',
                     os.path.join(args.results_dir, 'plot_spherical_l2.png'),
                     direction_labels)

    # Experiment 5: Additive perturbation
    if 'additive' in experiments:
        print(f"\n{'='*60}")
        print("Experiment 5: Additive perturbation (improved random baseline)")
        print(f"{'='*60}")

        # For additive, we need magnitudes instead of angles.
        # Use the mean activation norm to set a meaningful scale.
        mean_norm = perturb_acts.norm(dim=-1).mean().item()
        # Cap equivalent angle at 80° to avoid tan(90°) blowup
        eff_angle = min(args.max_angle, 80.0)
        max_mag = mean_norm * np.tan(np.radians(eff_angle))
        magnitudes = np.linspace(0, max_mag, args.n_angles + 1)[1:]
        # Also compute equivalent angles for x-axis labeling
        equiv_angles = np.degrees(np.arctan(magnitudes / mean_norm))

        curves_add = {}

        print(f"  Mean activation norm: {mean_norm:.2f}")
        print(f"  Magnitude range: 0 to {max_mag:.1f}")

        print(f"  Sweeping {args.n_curves} random curves...")
        t0 = time.time()
        for k in range(args.n_curves):
            curves_add[f'random_{k}'] = sweep_one_additive(
                model, perturb_contexts[act_indices[k]], perturb_acts[act_indices[k]],
                rand_dirs[k], args.layer, measure_layer, device, magnitudes)
        print(f"    {time.time()-t0:.1f}s")

        n_dom = min(args.n_curves, len(perturb_acts))
        print(f"  Sweeping {n_dom} -DoM curves...")
        t0 = time.time()
        for k in range(n_dom):
            curves_add[f'neg_dom_{k}'] = sweep_one_additive(
                model, perturb_contexts[k], perturb_acts[k],
                neg_dom, args.layer, measure_layer, device, magnitudes)
        print(f"    {time.time()-t0:.1f}s")

        print(f"  Sweeping {args.n_curves} SAE curves...")
        t0 = time.time()
        for k in range(args.n_curves):
            curves_add[f'sae_{k}'] = sweep_one_additive(
                model, perturb_contexts[act_indices[k]], perturb_acts[act_indices[k]],
                sae_dirs[act_indices[k], dir_indices[k]],
                args.layer, measure_layer, device, magnitudes)
        print(f"    {time.time()-t0:.1f}s")

        pkl_path = os.path.join(args.results_dir, 'curves_additive_l2.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump({'args': vars(args), 'magnitudes': magnitudes,
                         'equiv_angles': equiv_angles, 'mean_norm': mean_norm,
                         'curves': curves_add, 'feature_ids': feature_ids.numpy()}, f)
        print(f"  Saved: {pkl_path}")

        # Plot with magnitude on x-axis
        plot_results(curves_add, magnitudes,
                     f'L2 distance (layer {measure_layer})',
                     'Perturbation magnitude',
                     f'{args.model}: inject@L{args.layer}, measure@L{measure_layer} (additive)',
                     os.path.join(args.results_dir, 'plot_additive_l2_magnitude.png'),
                     direction_labels)

        # Plot with equivalent angle on x-axis
        plot_results(curves_add, equiv_angles,
                     f'L2 distance (layer {measure_layer})',
                     'Equivalent angle (degrees)',
                     f'{args.model}: inject@L{args.layer}, measure@L{measure_layer} (additive)',
                     os.path.join(args.results_dir, 'plot_additive_l2_angle.png'),
                     direction_labels)

    print("\nDone!")


if __name__ == '__main__':
    main()
