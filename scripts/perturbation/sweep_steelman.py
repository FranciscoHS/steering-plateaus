#!/usr/bin/env python3
"""Steelman perturbation sweeps: additional direction types for LessWrong post.

Three new direction types compared against random and DoM baselines:
1. diff_random: normalized difference of random activation pairs
2. sae_lang: most language-selective SAE feature (highest |mean_en - mean_zh| activation)
3. dom_recon: DoM computed on SAE reconstructions

Usage:
    python scripts/perturbation/sweep_steelman.py --model gemma-2-9b --layer 2
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
    load_qwen_sae, get_top_sae_directions, sweep_one,
    get_contrastive_directions,
)
from data.bilingual_prompts import ENGLISH_PROMPTS, CHINESE_PROMPTS


def get_bilingual_activations(model, tokenizer, device, layer):
    """Get layer activations for EN and ZH bilingual prompts."""
    def get_acts(prompts):
        acts = []
        for p in prompts:
            tokens = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            acts.append(outputs.hidden_states[layer + 1][:, -1, :].squeeze(0).float().cpu())
        return torch.stack(acts)

    en_acts = get_acts(ENGLISH_PROMPTS)
    zh_acts = get_acts(CHINESE_PROMPTS)
    return en_acts, zh_acts


def get_diff_random_directions(all_activations, n_dirs):
    """Compute normalized differences between random pairs of activations.

    Returns [n_dirs, d_model] tensor of unit directions.
    """
    n = len(all_activations)
    rng = np.random.RandomState(99)
    dirs = []
    for _ in range(n_dirs):
        i, j = rng.choice(n, size=2, replace=False)
        diff = all_activations[i] - all_activations[j]
        dirs.append(F.normalize(diff, dim=0))
    return torch.stack(dirs)


def get_sae_language_directions(sae, en_acts, zh_acts, device, top_k=5):
    """Find SAE features most differentially active between EN and ZH.

    Returns:
        directions: [top_k, d_model] decoder directions for the most language-selective features
        feature_ids: [top_k] feature indices
        en_mean_acts: mean SAE activations for English
        zh_mean_acts: mean SAE activations for Chinese
    """
    # Encode all EN and ZH activations
    en_feat = []
    for i in range(len(en_acts)):
        feat = sae.encode(en_acts[i:i+1].to(device))
        en_feat.append(feat.cpu())
    en_feat = torch.cat(en_feat, dim=0).float()  # [n_en, n_features]

    zh_feat = []
    for i in range(len(zh_acts)):
        feat = sae.encode(zh_acts[i:i+1].to(device))
        zh_feat.append(feat.cpu())
    zh_feat = torch.cat(zh_feat, dim=0).float()  # [n_zh, n_features]

    en_mean = en_feat.mean(0)  # [n_features]
    zh_mean = zh_feat.mean(0)

    # Most language-selective: largest |mean_en - mean_zh|
    diff = (en_mean - zh_mean).abs()
    topk = diff.topk(top_k)
    feat_ids = topk.indices
    print(f"  Top {top_k} language-selective SAE features:")
    for rank, fid in enumerate(feat_ids):
        fid = fid.item()
        print(f"    #{rank}: feature {fid}, en_mean={en_mean[fid]:.3f}, "
              f"zh_mean={zh_mean[fid]:.3f}, |diff|={diff[fid]:.3f}")

    W_dec = sae.W_dec.float()
    if hasattr(W_dec, 'data'):
        W_dec = W_dec.data
    directions = W_dec[feat_ids.cpu()]
    directions = F.normalize(directions, dim=-1)

    return directions, feat_ids, en_mean, zh_mean


def get_dom_reconstruction(sae, en_acts, zh_acts, device):
    """Compute DoM on SAE reconstructions.

    Encode -> decode each activation, then take mean(recon_en) - mean(recon_zh).
    Returns unit direction.
    """
    def reconstruct(acts):
        recons = []
        W_dec = sae.W_dec.float()
        if hasattr(W_dec, 'data'):
            W_dec = W_dec.data
        b_dec = sae.b_dec.float() if hasattr(sae, 'b_dec') else torch.zeros(W_dec.shape[1])
        if hasattr(b_dec, 'data'):
            b_dec = b_dec.data
        for i in range(len(acts)):
            feat = sae.encode(acts[i:i+1].to(device)).float()  # [1, n_features]
            recon = feat @ W_dec + b_dec.to(feat.device)  # [1, d_model]
            recons.append(recon.cpu())
        return torch.cat(recons, dim=0)

    en_recon = reconstruct(en_acts)
    zh_recon = reconstruct(zh_acts)
    dom_recon = F.normalize((en_recon.mean(0) - zh_recon.mean(0)), dim=0)

    # Also compute cosine similarity with raw DoM for comparison
    raw_dom = F.normalize((en_acts.mean(0) - zh_acts.mean(0)), dim=0)
    cos_sim = (dom_recon @ raw_dom).item()
    print(f"  cos(DoM_recon, DoM_raw) = {cos_sim:.4f}")

    return dom_recon


def run_sweeps(model, all_contexts, all_activations, directions_dict,
               layer, measure_layer, device, angles_rad, metric, n_curves):
    """Run sweeps for all direction types.

    directions_dict: {label: direction_or_directions}
      - If direction is 1D [d_model]: same direction for all activations (like DoM)
      - If direction is 2D [n, d_model]: one direction per curve
    """
    n_act = len(all_activations)
    curves = {}
    rng = np.random.RandomState(42)

    for label, dirs in directions_dict.items():
        if dirs.dim() == 1:
            # Fixed direction, sweep over different activations
            n = min(n_curves, n_act)
            print(f"  Sweeping {n} {label} curves (fixed direction)...")
            t0 = time.time()
            for k in range(n):
                curves[f'{label}_{k}'] = sweep_one(
                    model, all_contexts[k], all_activations[k],
                    dirs, layer, measure_layer, device, angles_rad, metric=metric)
            print(f"    {time.time()-t0:.1f}s")
        else:
            # Multiple directions
            n = min(n_curves, len(dirs))
            act_indices = rng.randint(0, n_act, size=n)
            print(f"  Sweeping {n} {label} curves...")
            t0 = time.time()
            for k in range(n):
                curves[f'{label}_{k}'] = sweep_one(
                    model, all_contexts[act_indices[k]], all_activations[act_indices[k]],
                    dirs[k % len(dirs)], layer, measure_layer, device, angles_rad, metric=metric)
            print(f"    {time.time()-t0:.1f}s")

    return curves


def plot_results(curves, angles_deg, ylabel, title, save_path, direction_labels):
    """Plot individual curves + median for each direction type."""
    colors = ['C0', 'C2', 'C3', 'C1', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (prefix, label) in enumerate(direction_labels.items()):
        color = colors[i % len(colors)]
        keys = sorted([k for k in curves if k.startswith(prefix + '_')])
        if not keys:
            continue
        data = np.array([curves[k] for k in keys])
        for k in keys:
            ax.plot(angles_deg, curves[k], color=color, alpha=0.5, lw=0.8)
        med = np.median(data, axis=0)
        ax.plot(angles_deg, med, color=color, lw=2.5, label=label)

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, angles_deg[-1])
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
    parser.add_argument('--n_sae_lang', type=int, default=5,
                        help='Number of top language-selective SAE features')
    parser.add_argument('--n_curves', type=int, default=10)
    parser.add_argument('--max_angle', type=float, default=90.0)
    parser.add_argument('--n_angles', type=int, default=90)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--metrics', default='l2', help='Comma-separated metrics')
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/lesswrong_plot/{args.model}-steelman'
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

    # FineWeb activations (the ones we perturb)
    print(f"\nLoading {args.n_act} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.layer} activations...")
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Activation norms: mean={all_activations.norm(dim=-1).mean():.2f}")

    # Bilingual activations (for computing directions)
    print("\nExtracting bilingual activations...")
    en_acts, zh_acts = get_bilingual_activations(model, tokenizer, device, args.layer)

    # 1. Baselines: random directions and -DoM
    print("\n--- Computing baseline directions ---")
    dom_dir, pc1_dir = get_contrastive_directions(model, tokenizer, device, args.layer)
    neg_dom = -dom_dir
    rand_dirs = F.normalize(torch.randn(args.n_curves, d_model), dim=-1)

    # 2. Difference of random activations
    print("\n--- Computing diff-of-random-activations directions ---")
    diff_random_dirs = get_diff_random_directions(all_activations, n_dirs=args.n_curves)
    print(f"  Generated {len(diff_random_dirs)} diff-of-random directions")
    # Check how similar these are to DoM
    cos_with_dom = (diff_random_dirs @ dom_dir).abs()
    print(f"  |cos(diff_random, DoM)|: mean={cos_with_dom.mean():.3f}, max={cos_with_dom.max():.3f}")

    # 3. Language-selective SAE features
    print("\n--- Finding language-selective SAE features ---")
    sae_type = args.sae_type
    if sae_type == 'auto':
        sae_type = 'gemma' if 'gemma' in args.model else 'qwen'
    if sae_type == 'gemma':
        model_size = '9b' if '9b' in args.model else '2b'
        sae = load_gemma_scope_sae(args.layer, width=args.sae_width,
                                    device=device, model_size=model_size)
    else:
        sae = load_qwen_sae(args.layer, width=args.sae_width, device=device,
                             dtype=torch.float32)

    sae_lang_dirs, sae_lang_ids, en_mean_feat, zh_mean_feat = get_sae_language_directions(
        sae, en_acts, zh_acts, device, top_k=args.n_sae_lang)

    # Also get top-active SAE features (existing approach, for comparison)
    sae_active_dirs, _ = get_top_sae_directions(sae, all_activations, 10, device)

    # 4. DoM on SAE reconstructions
    print("\n--- Computing DoM on SAE reconstructions ---")
    dom_recon = get_dom_reconstruction(sae, en_acts, zh_acts, device)
    neg_dom_recon = -dom_recon

    del sae
    torch.cuda.empty_cache()

    # Build directions dict
    # For SAE active, pick one per activation like the original
    rng_sae = np.random.RandomState(42)
    sae_act_indices = rng_sae.randint(0, args.n_act, size=args.n_curves)
    sae_dir_indices = rng_sae.randint(0, 10, size=args.n_curves)
    sae_active_selected = torch.stack([
        sae_active_dirs[sae_act_indices[k], sae_dir_indices[k]]
        for k in range(args.n_curves)
    ])

    directions_dict = {
        'random': rand_dirs,                    # random directions (baseline)
        'neg_dom': neg_dom,                     # -DoM (baseline, toward Chinese)
        'diff_random': diff_random_dirs,        # difference of random activation pairs
        'sae_active': sae_active_selected,      # top-active SAE features (existing)
        'sae_lang': sae_lang_dirs,              # language-selective SAE features
        'neg_dom_recon': neg_dom_recon,          # -DoM on SAE reconstructions
    }

    direction_labels = {
        'random': 'Random',
        'neg_dom': '-DoM',
        'diff_random': 'Diff of random acts',
        'sae_active': 'SAE (top active)',
        'sae_lang': 'SAE (language-selective)',
        'neg_dom_recon': '-DoM (SAE recon)',
    }

    # Run sweeps
    for metric in args.metrics.split(','):
        ylabels = {
            'l2': f'L2 distance (layer {measure_layer})',
            'kl': 'KL divergence (logits)',
            'cosine': f'Cosine distance (layer {measure_layer})',
        }
        ylabel = ylabels[metric]
        print(f"\n{'='*60}")
        print(f"Metric: {metric}")
        print(f"{'='*60}")

        curves = run_sweeps(model, all_contexts, all_activations,
                            directions_dict, args.layer, measure_layer,
                            device, angles_rad, metric, args.n_curves)

        # Save
        pkl_path = os.path.join(args.results_dir, f'curves_{metric}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'args': vars(args),
                'angles_deg': angles_deg,
                'curves': curves,
                'sae_lang_feature_ids': sae_lang_ids.cpu().numpy(),
                'direction_labels': direction_labels,
            }, f)
        print(f"  Saved: {pkl_path}")

        title = f'{args.model}: inject@L{args.layer}, measure@L{measure_layer}'
        plot_results(curves, angles_deg, ylabel, title,
                     os.path.join(args.results_dir, f'plot_{metric}.png'),
                     direction_labels)

    print("\nDone!")


if __name__ == '__main__':
    main()
