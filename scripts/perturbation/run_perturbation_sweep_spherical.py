#!/usr/bin/env python3
"""
Spherical perturbation sweep: rotate activations on the constant-norm sphere.

Perturbation: a_perturbed = cos(theta) * a + sin(theta) * ||a|| * v_perp
where v_perp is a random unit vector orthogonal to a.

X-axis: angle theta in degrees (bottom), chord distance (top).
Y-axis: L2 distance on final-layer activations / KL divergence on logits.

Usage:
    python scripts/run_perturbation_sweep_spherical.py --model gpt2-large
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
    load_model, forward_from_layer_to_final, forward_from_layer_to_layer,
    _get_final_norm, _get_lm_head, _get_blocks, ablate_attention, ablate_mlp,
)
from src.data import load_fineweb_fixed_length, generate_random_token_sequences, load_python_code_fixed_length, generate_chess_games


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device, batch_size=64):
    all_contexts = []
    all_activations = []
    is_circuitgpt = model._plateau_info['model_type'] == 'circuitgpt'

    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            if is_circuitgpt:
                # CSP doesn't support output_hidden_states; run embedding + blocks manually
                gpt = model.circuit_model
                tok_emb = gpt.transformer.wte(batch_tokens)
                x = gpt.transformer.drop(tok_emb)
                seq_len = batch_tokens.shape[1]
                has_cat_pos = getattr(gpt.config, 'd_pos_emb', None) is not None
                if has_cat_pos:
                    pos_emb = gpt.transformer.wpe.weight[:seq_len].unsqueeze(0)
                else:
                    # Dense CSP: add position embeddings to residual stream (GPT-2 style)
                    positions = torch.arange(seq_len, device=device).unsqueeze(0)
                    x = x + gpt.transformer.wpe(positions)
                for block in gpt.transformer.h[:layer + 1]:
                    x = block(x, pos_emb) if has_cat_pos else block(x)
                hidden = x
            else:
                outputs = model(batch_tokens, output_hidden_states=True)
                hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].float().cpu())
        all_activations.append(hidden[:, -1, :].float().cpu())
    return torch.cat(all_contexts, 0), torch.cat(all_activations, 0)


def sample_orthogonal_directions(activations, n_dirs):
    """Sample random unit vectors orthogonal to each activation.

    Args:
        activations: [n_act, d_model]
        n_dirs: directions per activation

    Returns:
        v_perps: [n_act, n_dirs, d_model] unit vectors orthogonal to activations
    """
    n_act, d_model = activations.shape
    v_perps = torch.zeros(n_act, n_dirs, d_model)

    for i in range(n_act):
        a = activations[i]  # [d_model]
        a_unit = a / a.norm()
        # Random vectors, then Gram-Schmidt to remove component along a
        v = torch.randn(n_dirs, d_model)
        v = v - (v @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        v = F.normalize(v, dim=-1)
        v_perps[i] = v

    return v_perps


def run_spherical_sweep(model, context, activation, v_perps, angles_rad,
                        layer, device, n_dirs, measure_layer=None):
    """Sweep spherical perturbations for one activation.

    a_perturbed = cos(theta) * a + sin(theta) * ||a|| * v_perp

    Returns:
        l2_dists: [n_angles, n_dirs]
        kl_divs: [n_angles, n_dirs]
        cos_dists: [n_angles, n_dirs]
    """
    n_angles = len(angles_rad)
    l2_dists = np.zeros((n_angles, n_dirs))
    kl_divs = np.zeros((n_angles, n_dirs))
    cos_dists = np.zeros((n_angles, n_dirs))

    model_dtype = next(model.parameters()).dtype
    ctx = context.unsqueeze(0).to(device=device, dtype=model_dtype)
    a = activation.to(device=device, dtype=model_dtype)       # [d_model]
    R = a.norm()
    a_unit = a / R
    vp = v_perps.to(device=device, dtype=model_dtype)          # [n_dirs, d_model]

    # Reference (unperturbed)
    with torch.no_grad():
        ref_point = a.unsqueeze(0).unsqueeze(0)  # [1, 1, d]
        ref_final = forward_from_layer_to_final(model, ctx, ref_point, layer)
        ref_ln = _get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1)
        ref_logits = _get_lm_head(model)(ref_ln)
        ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
        if measure_layer is not None:
            ref_measure = forward_from_layer_to_layer(model, ctx, ref_point, layer, measure_layer)
        else:
            ref_measure = ref_final

    for ang_idx, theta in enumerate(angles_rad):
        # a_perturbed = cos(theta) * a + sin(theta) * R * v_perp
        # = R * (cos(theta) * a_unit + sin(theta) * v_perp)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        perturbed = R * (cos_t * a_unit.unsqueeze(0) + sin_t * vp)  # [n_dirs, d]
        perturbed = perturbed.unsqueeze(1)  # [n_dirs, 1, d]

        with torch.no_grad():
            pert_final = forward_from_layer_to_final(model, ctx, perturbed, layer)

            # L2 at measure_layer (or final if not set)
            if measure_layer is not None:
                pert_measure = forward_from_layer_to_layer(model, ctx, perturbed, layer, measure_layer)
            else:
                pert_measure = pert_final
            l2_dists[ang_idx] = (pert_measure - ref_measure).float().norm(dim=-1).cpu().numpy()
            cos_sim = F.cosine_similarity(pert_measure.float(), ref_measure.float(), dim=-1)
            cos_dists[ang_idx] = (1 - cos_sim).cpu().numpy()

            # KL always from final logits
            pert_ln = _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)
            pert_logits = _get_lm_head(model)(pert_ln)
            pert_log_probs = F.log_softmax(pert_logits.float(), dim=-1)
            ref_probs = ref_log_probs.exp()
            kl = (ref_probs * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            kl_divs[ang_idx] = kl.cpu().numpy()

    return l2_dists, kl_divs, cos_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=100)
    parser.add_argument('--n_dirs', type=int, default=10)
    parser.add_argument('--n_angles', type=int, default=60,
                        help='Number of angle points from 0 to 180 degrees')
    parser.add_argument('--cov_matched', action='store_true',
                        help='Use covariance-matched random activations instead of norm-matched')
    parser.add_argument('--n_fit', type=int, default=2000,
                        help='Number of samples for covariance fitting (must be > d_model)')
    parser.add_argument('--delta_path', type=str, default=None,
                        help='Path to delta optimization results .pkl. Compares real vs real+delta instead of real vs random.')
    parser.add_argument('--measure_layer', type=int, default=None,
                        help='Layer at which to measure L2 distance (default: final layer)')
    parser.add_argument('--data_source', type=str, default='auto',
                        choices=['auto', 'fineweb', 'random', 'python_code', 'chess',
                                 'english_holdout', 'male_holdout', 'coding_agnostic'],
                        help='Data source: fineweb, random tokens, python_code, chess, '
                             'english_holdout, male_holdout, coding_agnostic, or auto')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'],
                        help='Model dtype (default: model.py default)')
    parser.add_argument('--max_angle', type=float, default=180,
                        help='Maximum perturbation angle in degrees (default: 180)')
    parser.add_argument('--ablate_attention', action='store_true',
                        help='Zero-ablate all attention heads (MLP-only forward pass)')
    parser.add_argument('--ablate_mlp', action='store_true',
                        help='Zero-ablate all MLP outputs (attention-only forward pass)')
    parser.add_argument('--no_special_tokens', action='store_true',
                        help='Disable special tokens (e.g. BOS) in tokenization')
    parser.add_argument('--direction_pkl', type=str, default=None,
                        help='Path to pkl with a contrastive direction. Switches to '
                             'contrastive vs random vs combined sweep mode.')
    parser.add_argument('--direction_key', type=str, default=None,
                        help='Key in the direction pkl (e.g. d_french)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.delta_path:
        comparison_tag = 'shifted'
    elif args.cov_matched:
        comparison_tag = 'cov_matched'
    else:
        comparison_tag = 'norm_matched'
    if args.results_dir is None:
        args.results_dir = f'results/perturbation_sweep_spherical/{args.model}_{comparison_tag}'
    os.makedirs(args.results_dir, exist_ok=True)

    # Angles: linearly spaced from just above 0 to max_angle degrees
    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]  # skip 0
    angles_rad = np.deg2rad(angles_deg)

    measure_str = f", measure_layer={args.measure_layer}" if args.measure_layer is not None else ""
    print(f"Config: model={args.model}, layer={args.layer}, n_act={args.n_act}, "
          f"n_dirs={args.n_dirs}, n_angles={len(angles_deg)}, range=[{angles_deg[0]:.1f}, {angles_deg[-1]:.1f}] deg, "
          f"comparison={comparison_tag}" + measure_str
          + (f", n_fit={args.n_fit}" if args.cov_matched else "")
          + (f", delta={args.delta_path}" if args.delta_path else ""))

    # Load model
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    print(f"Loading model: {args.model}" + (f" (dtype={args.dtype})" if args.dtype else ""))
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    # Resolve negative measure_layer (e.g. -2 → 2nd-to-last)
    if args.measure_layer is not None and args.measure_layer < 0:
        args.measure_layer = n_layers + args.measure_layer
    print(f"Device: {device}, d_model: {d_model}, n_layers: {n_layers}"
          + (f", measure_layer: {args.measure_layer}" if args.measure_layer is not None else ""))

    if args.ablate_attention and args.ablate_mlp:
        raise ValueError("Cannot ablate both attention and MLP simultaneously")
    if args.ablate_attention:
        hooks = ablate_attention(model)
        print(f"Ablated attention in all {len(hooks)} layers")
    if args.ablate_mlp:
        hooks = ablate_mlp(model)
        print(f"Ablated MLP in all {len(hooks)} layers")

    # Resolve data source
    data_source = args.data_source
    if data_source == 'auto':
        if args.model.startswith('chess'):
            data_source = 'chess'
        elif model._plateau_info['model_type'] == 'circuitgpt':
            data_source = 'python_code'
        else:
            data_source = 'fineweb'

    # Extract activations
    n_load = max(args.n_act, args.n_fit) if args.cov_matched else args.n_act

    # Holdout prompt sources (variable-length, use extract_activations_holdout)
    _holdout_sources = {
        'english_holdout': ('data.english_prompts_holdout', 'ENGLISH_PROMPTS_HOLDOUT'),
        'male_holdout': ('data.male_prompts_holdout', 'MALE_PROMPTS_HOLDOUT'),
        'coding_agnostic': ('data.coding_prompts_agnostic', 'CODING_PROMPTS'),
    }
    if data_source in _holdout_sources:
        mod_name, var_name = _holdout_sources[data_source]
        import importlib
        mod = importlib.import_module(mod_name)
        holdout_prompts = getattr(mod, var_name)[:n_load]
        print(f"Using {len(holdout_prompts)} {data_source} prompts...")
        from scripts.perturbation.generate_lesswrong_figures import extract_activations_holdout
        contexts_list, activations_stacked = extract_activations_holdout(
            model, tokenizer, device, args.layer, holdout_prompts)
        # contexts_list is list of [1, seq-1, d] (variable seq lengths)
        # activations_stacked is [n, d]
        # Squeeze batch dim so contexts[i] is [seq-1, d], matching FineWeb path
        contexts = [c.squeeze(0) for c in contexts_list]
        activations = activations_stacked
    elif data_source == 'random':
        vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else model.circuit_model.config.vocab_size
        print(f"Generating {n_load} random token sequences (vocab_size={vocab_size})...")
        token_ids_list = generate_random_token_sequences(n_load, vocab_size, args.seq_len, seed=args.seed)
    elif data_source == 'python_code':
        from transformers import AutoTokenizer
        print(f"Loading CSP tokenizer...")
        csp_tokenizer = AutoTokenizer.from_pretrained("openai/circuit-sparsity", trust_remote_code=True)
        print(f"Loading {n_load} Python code sequences...")
        token_ids_list = load_python_code_fixed_length(n_load, csp_tokenizer, seq_len=args.seq_len, seed=args.seed)
    elif data_source == 'chess':
        print(f"Generating {n_load} chess game sequences...")
        token_ids_list = generate_chess_games(n_load, args.seq_len, seed=args.seed)
    else:
        print(f"Loading {n_load} FineWeb sequences...")
        add_special = not args.no_special_tokens
        token_ids_list = load_fineweb_fixed_length(n_load, tokenizer, seq_len=args.seq_len, seed=args.seed,
                                                   add_special_tokens=add_special)
    if data_source not in _holdout_sources:
        print(f"Extracting layer-{args.layer} activations...")
        all_contexts, all_activations = extract_context_and_activation(
            model, tokenizer, token_ids_list, args.layer, device
        )
        # Use first n_act for the sweep
        contexts = all_contexts[:args.n_act]
        activations = all_activations[:args.n_act]
    act_norms = activations.norm(dim=-1)
    act_norm_mean = act_norms.mean().item()
    print(f"Mean activation norm: {act_norm_mean:.2f}")

    # --- Contrastive direction mode ---
    if args.direction_pkl:
        import pickle as _pkl
        with open(args.direction_pkl, 'rb') as f:
            dir_data = _pkl.load(f)
        d_contrastive = dir_data[args.direction_key]  # [d_model]
        print(f"\nLoaded contrastive direction '{args.direction_key}' "
              f"from {args.direction_pkl}")

        # Improved random directions: diff of two activations (shared across all acts)
        rng = np.random.RandomState(args.seed)
        n_pool = len(activations)
        rand_dirs_global = []
        for _ in range(args.n_dirs):
            i_r, j_r = rng.choice(n_pool, size=2, replace=False)
            diff = activations[i_r] - activations[j_r]
            rand_dirs_global.append(F.normalize(diff, dim=0))
        rand_dirs_global = torch.stack(rand_dirs_global)  # [n_dirs, d_model]
        print(f"  Generated {args.n_dirs} improved-random directions "
              f"(diff of two activations)")

        def make_direction_sets(act, d_contr, rand_dirs):
            """For one activation, build v_perps for contrastive, random, and combined."""
            a_unit = act / act.norm()
            # Orthogonalize contrastive against activation
            dc = d_contr.float() - (d_contr.float() @ a_unit) * a_unit
            dc = F.normalize(dc, dim=0)
            # Orthogonalize improved-random directions against activation
            v_rand = rand_dirs.float() - (rand_dirs.float() @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
            v_rand = F.normalize(v_rand, dim=-1)
            # Combined: normalize(contrastive_orth + random_orth) for each random dir
            v_comb = dc.unsqueeze(0) + v_rand  # [n_dirs, d_model]
            v_comb = v_comb - (v_comb @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
            v_comb = F.normalize(v_comb, dim=-1)
            return dc.unsqueeze(0), v_rand, v_comb  # [1, d], [n_dirs, d], [n_dirs, d]

        conditions = ['contrastive', 'random', 'combined']
        sweep_results = {c: {'l2': [], 'kl': [], 'cos': []} for c in conditions}

        print(f"\n=== Contrastive direction sweep ===")
        t0 = time.time()
        for i in range(args.n_act):
            dc_vp, rand_vp, comb_vp = make_direction_sets(
                activations[i], d_contrastive, rand_dirs_global)
            for cond, vp in [('contrastive', dc_vp),
                             ('random', rand_vp),
                             ('combined', comb_vp)]:
                n_d = vp.shape[0]
                l2, kl, cos = run_spherical_sweep(
                    model, contexts[i], activations[i], vp,
                    angles_rad, args.layer, device, n_d,
                    measure_layer=args.measure_layer)
                sweep_results[cond]['l2'].append(l2)
                sweep_results[cond]['kl'].append(kl)
                sweep_results[cond]['cos'].append(cos)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
        print(f"Done in {time.time()-t0:.1f}s")

        # Stack: contrastive is [n_act, n_angles, 1], others [n_act, n_angles, n_dirs]
        for cond in conditions:
            for key in ['l2', 'kl', 'cos']:
                sweep_results[cond][key] = np.stack(sweep_results[cond][key])

        results = {
            'args': vars(args),
            'angles_deg': angles_deg,
            'angles_rad': angles_rad,
            'act_norm_mean': act_norm_mean,
            'direction_key': args.direction_key,
        }
        for cond in conditions:
            results[cond] = sweep_results[cond]

        with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
            _pkl.dump(results, f)

        # Plot
        cond_colors = {'contrastive': '#228833', 'random': '#4477AA', 'combined': '#EE6677'}
        cond_labels = {
            'contrastive': args.direction_key.replace('d_', '').capitalize(),
            'random': 'Random',
            'combined': 'Combined',
        }

        for plot_type in ['median', 'individual']:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            for metric_idx, (key, name) in enumerate(
                    [('l2', 'L2 distance'), ('cos', 'Cosine distance'), ('kl', 'KL divergence')]):
                ax = axes[metric_idx]
                for cond in conditions:
                    d = results[cond][key]
                    per_act = np.median(d, axis=2)  # [n_act, n_angles]
                    color = cond_colors[cond]
                    label = cond_labels[cond]

                    if plot_type == 'median':
                        median = np.median(per_act, axis=0)
                        q25 = np.percentile(per_act, 25, axis=0)
                        q75 = np.percentile(per_act, 75, axis=0)
                        ax.plot(angles_deg, median, color=color, label=label, linewidth=2)
                        ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.15)
                    else:
                        np.random.seed(42)
                        n_individual = min(10, per_act.shape[0])
                        chosen = np.random.choice(per_act.shape[0], n_individual, replace=False)
                        for j, idx in enumerate(chosen):
                            ax.plot(angles_deg, per_act[idx], color=color, alpha=0.4,
                                    linewidth=1.5, label=label if j == 0 else None)

                ax.set_xlabel('Perturbation angle (degrees)')
                ax.set_ylabel(name)
                ax.set_title(name)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, args.max_angle)

                ax2 = ax.twiny()
                ax2.set_xlim(ax.get_xlim())
                tick_angles = [a for a in [0, 10, 20, 30, 40, 60, 90, 120, 150, 180]
                               if a <= args.max_angle]
                tick_chords = [2 * act_norm_mean * np.sin(np.deg2rad(a) / 2)
                               for a in tick_angles]
                ax2.set_xticks(tick_angles)
                ax2.set_xticklabels([f'{c:.0f}' for c in tick_chords])
                ax2.set_xlabel('Chord distance')

            measure_label = (f", L2@layer {args.measure_layer}"
                             if args.measure_layer is not None else "")
            fig.suptitle(f'{args.model}, layer {args.layer} '
                         f'(spherical, {args.direction_key}, {plot_type}{measure_label})',
                         fontsize=14, y=1.02)
            plt.tight_layout()
            path = os.path.join(args.results_dir, f'sweep_{plot_type}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {path}")

        print("\nDone!")
        return

    # Comparison activations
    if args.delta_path:
        print(f"\nLoading delta from {args.delta_path}...")
        with open(args.delta_path, 'rb') as f:
            delta_results = pickle.load(f)
        delta = delta_results['delta']  # [d_model]
        if isinstance(delta, torch.Tensor):
            delta = delta.cpu()
        else:
            delta = torch.tensor(delta, dtype=torch.float32)
        print(f"Delta norm: {delta.norm().item():.2f}")
        # Shifted activations: real + delta (keep actual resulting norm, no renormalization)
        comparison_activations = activations + delta.unsqueeze(0)
        shifted_norms = comparison_activations.norm(dim=-1)
        print(f"Shifted activation norms: mean={shifted_norms.mean().item():.2f}, "
              f"std={shifted_norms.std().item():.2f}")
        comparison_label = 'Shifted (real+delta)'
    elif args.cov_matched:
        print(f"\nFitting covariance from {all_activations.shape[0]} activations (d_model={d_model})...")
        from src.data import sample_random_activations_v2
        cov_random, cov_stats = sample_random_activations_v2(
            all_activations, args.n_act, mode='full', seed=args.seed, device=device
        )
        comparison_activations = F.normalize(cov_random, dim=-1) * act_norm_mean
        print(f"Covariance-matched random activations: rank={cov_stats.get('rank', '?')}")
        comparison_label = 'Random (cov-matched)'
    else:
        comparison_activations = torch.randn(args.n_act, d_model)
        comparison_activations = F.normalize(comparison_activations, dim=-1) * act_norm_mean
        comparison_label = 'Random (norm-matched)'

    # Orthogonal directions
    print("Sampling orthogonal directions...")
    real_v_perps = sample_orthogonal_directions(activations, args.n_dirs)
    rand_v_perps = sample_orthogonal_directions(comparison_activations, args.n_dirs)

    # Sweep: real
    print(f"\n=== Real activations ===")
    real_l2 = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
    real_kl = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
    real_cos = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
    t0 = time.time()
    for i in range(args.n_act):
        real_l2[i], real_kl[i], real_cos[i] = run_spherical_sweep(
            model, contexts[i], activations[i], real_v_perps[i],
            angles_rad, args.layer, device, args.n_dirs,
            measure_layer=args.measure_layer
        )
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
    print(f"Real done in {time.time()-t0:.1f}s")

    # Sweep: random
    print(f"\n=== Comparison activations ({comparison_tag}) ===")
    rand_l2 = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
    rand_kl = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
    rand_cos = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
    t0 = time.time()
    for i in range(args.n_act):
        rand_l2[i], rand_kl[i], rand_cos[i] = run_spherical_sweep(
            model, contexts[i], comparison_activations[i], rand_v_perps[i],
            angles_rad, args.layer, device, args.n_dirs,
            measure_layer=args.measure_layer
        )
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
    print(f"Comparison done in {time.time()-t0:.1f}s")

    # Save
    results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'angles_rad': angles_rad,
        'act_norm_mean': act_norm_mean,
        'comparison_type': comparison_tag,
        'comparison_label': comparison_label,
        'real': {'l2': real_l2, 'kl': real_kl, 'cos': real_cos},
        'random': {'l2': rand_l2, 'kl': rand_kl, 'cos': rand_cos},
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Chord distances for top axis: 2R * sin(theta/2)
    chord_distances = 2 * act_norm_mean * np.sin(np.deg2rad(angles_deg) / 2)

    # Plots
    conditions = [('real', 'Real'), ('random', comparison_label)]
    colors = ['blue', 'red']
    np.random.seed(42)
    n_individual = 10

    for plot_type in ['median', 'individual']:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for metric_idx, (key, name) in enumerate([('l2', 'L2 distance'), ('cos', 'Cosine distance'), ('kl', 'KL divergence')]):
            ax = axes[metric_idx]
            for (cond_key, label), color in zip(conditions, colors):
                d = results[cond_key][key]
                per_act = np.median(d, axis=2)

                if plot_type == 'median':
                    median = np.median(per_act, axis=0)
                    q25 = np.percentile(per_act, 25, axis=0)
                    q75 = np.percentile(per_act, 75, axis=0)
                    ax.plot(angles_deg, median, color=color, label=label, linewidth=2)
                    ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.15)
                else:
                    chosen = np.random.choice(per_act.shape[0], n_individual, replace=False)
                    for j, idx in enumerate(chosen):
                        ax.plot(angles_deg, per_act[idx], color=color, alpha=0.4, linewidth=1.5,
                                label=label if j == 0 else None)

            ax.set_xlabel('Perturbation angle (degrees)')
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, args.max_angle)

            # Twin top axis: chord distance
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            # Place ticks at nice angle values, label with chord distance
            tick_angles = [a for a in [0, 10, 20, 30, 40, 60, 90, 120, 150, 180] if a <= args.max_angle]
            tick_chords = [2 * act_norm_mean * np.sin(np.deg2rad(a) / 2) for a in tick_angles]
            ax2.set_xticks(tick_angles)
            ax2.set_xticklabels([f'{c:.0f}' for c in tick_chords])
            ax2.set_xlabel('Chord distance')

        measure_label = f", L2@layer {args.measure_layer}" if args.measure_layer is not None else ""
        fig.suptitle(f'{args.model}, layer {args.layer} (spherical, {comparison_tag}, {plot_type}{measure_label})', fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(args.results_dir, f'sweep_{plot_type}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
