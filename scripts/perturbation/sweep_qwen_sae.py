#!/usr/bin/env python3
"""Perturbation sweep for Qwen3-1.7B with adamkarvonen batch-top-k SAEs.

Sweeps: random, EN↔ZH DoM contrastive, SAE feature directions.
Metrics: L2 and cosine distance.

Usage:
    python scripts/perturbation/sweep_qwen_sae.py [--n_act 30] [--n_curves 15]
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
    _is_rope_model, _get_position_embeddings, _run_block,
)
from src.data import load_fineweb_fixed_length
from scripts.perturbation.sweep_random_points import extract_context_and_activation
from scripts.perturbation.sweep_sae_features import load_gemma_scope_sae
from data.bilingual_prompts import ENGLISH_PROMPTS, CHINESE_PROMPTS


def load_qwen_sae(layer, width='65k', device='cpu', dtype=torch.float32):
    """Load adamkarvonen batch-top-k SAE for Qwen3-1.7B.

    Args:
        layer: 7, 14, or 21
        width: '16k' or '65k'
        device: torch device
    """
    # trainer mapping: 0=16k/L0=80, 1=16k/L0=160, 2=65k/L0=80, 3=65k/L0=160
    trainer = {'16k': 0, '65k': 2}[width]

    try:
        import interp_tools.saes.batch_topk_sae as batch_topk_sae
        repo_id = "adamkarvonen/qwen3-1.7b-saes"
        filename = f"saes_Qwen_Qwen3-1.7B_batch_top_k/resid_post_layer_{layer}/trainer_{trainer}/ae.pt"
        sae = batch_topk_sae.load_dictionary_learning_batch_topk_sae(
            repo_id=repo_id, filename=filename,
            model_name="Qwen/Qwen3-1.7B", device=device, dtype=dtype
        )
        print(f"SAE loaded via interp_tools: W_dec shape {sae.W_dec.shape}")
        return sae
    except ImportError:
        # Fallback: manual loading
        from huggingface_hub import hf_hub_download
        import json

        repo_id = "adamkarvonen/qwen3-1.7b-saes"
        base = f"saes_Qwen_Qwen3-1.7B_batch_top_k/resid_post_layer_{layer}/trainer_{trainer}"

        cfg_path = hf_hub_download(repo_id, f"{base}/config.json")
        with open(cfg_path) as f:
            config = json.load(f)
        print(f"SAE config: {config}")

        ae_path = hf_hub_download(repo_id, f"{base}/ae.pt")
        state = torch.load(ae_path, map_location=device, weights_only=True)

        # Remap legacy keys if needed
        key_map = {
            'encoder.weight': 'W_enc', 'encoder.bias': 'b_enc',
            'decoder.weight': 'W_dec', 'decoder.bias': 'b_dec',
        }
        remapped = {}
        for k, v in state.items():
            new_k = key_map.get(k, k)
            remapped[new_k] = v

        # W_dec: [n_features, d_model], W_enc: [d_model, n_features] or transposed
        W_dec = remapped['W_dec'].to(dtype=dtype)
        if W_dec.shape[0] < W_dec.shape[1]:
            W_dec = W_dec.T  # ensure [n_features, d_model]

        class SimpleSAE:
            def __init__(self, W_enc, b_enc, W_dec, b_dec):
                self.W_enc = W_enc.to(dtype=dtype)
                self.b_enc = b_enc.to(dtype=dtype)
                self.W_dec = W_dec.to(dtype=dtype)
                self.b_dec = b_dec.to(dtype=dtype)

            def encode(self, x):
                """x: [batch, d_model] -> [batch, n_features]"""
                x_centered = x - self.b_dec
                return F.relu(x_centered @ self.W_enc + self.b_enc)

        W_enc = remapped['W_enc'].to(dtype=dtype)
        if W_enc.shape[1] < W_enc.shape[0]:
            W_enc = W_enc.T  # ensure [d_model, n_features]
        b_enc = remapped['b_enc'].to(dtype=dtype)
        b_dec = remapped['b_dec'].to(dtype=dtype)

        sae = SimpleSAE(W_enc, b_enc, W_dec, b_dec)
        print(f"SAE loaded manually: W_dec shape {W_dec.shape}")
        return sae


def get_top_sae_directions(sae, activations, n_dirs, device):
    """Encode activations through SAE, return top-n_dirs decoder directions per activation."""
    n_act, d_model = activations.shape
    all_directions = torch.zeros(n_act, n_dirs, d_model)
    all_feature_ids = torch.zeros(n_act, n_dirs, dtype=torch.long)

    W_dec = sae.W_dec.float()  # [n_features, d_model]

    for i in range(n_act):
        act = activations[i:i+1].to(device)
        feature_acts = sae.encode(act)  # [1, n_features]
        topk = feature_acts[0].topk(n_dirs)
        feat_ids = topk.indices
        all_feature_ids[i] = feat_ids.cpu()
        dirs = W_dec[feat_ids.cpu()]
        dirs = F.normalize(dirs, dim=-1)
        all_directions[i] = dirs.cpu()

    return all_directions, all_feature_ids


def get_contrastive_directions(model, tokenizer, device, layer):
    """Compute English-Mandarin DoM and PC1 directions.

    Returns:
        dom: [d_model] unit difference-of-means direction
        pc1: [d_model] unit PC1 direction (first right singular vector of D = en - zh)
    """
    print(f"  Computing EN↔ZH DoM + PC1 at layer {layer}...")
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
    D = en_acts - zh_acts  # [n_pairs, d_model]
    mean_vec = D.mean(0)
    dom = F.normalize(mean_vec, dim=0)
    print(f"  DoM pre-norm magnitude: {mean_vec.norm():.2f}")

    U, S, Vh = torch.linalg.svd(D.float(), full_matrices=False)
    pc1 = Vh[0]
    if (pc1 @ dom) < 0:
        pc1 = -pc1
    var_explained = S[0]**2 / (S**2).sum()
    print(f"  PC1 variance explained: {var_explained:.3f}, cos(PC1, DoM): {(pc1 @ dom).item():.3f}")
    return dom, pc1


def sweep_one(model, ctx, act, direction, layer, measure_layer, device, angles_rad,
              metric='l2', batch_size=64):
    """Sweep a single (activation, direction) pair. Returns array of distances."""
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

    result = np.zeros(len(angles_rad))
    use_logits = metric in ('kl', 'logit_l2', 'logit_cosine')
    fwd_fn = forward_from_layer if use_logits else forward_from_layer_to_layer

    with torch.no_grad():
        if use_logits:
            ref = fwd_fn(model, ctx_d, a_d[None, None], layer).float()
        else:
            ref = fwd_fn(model, ctx_d, a_d[None, None], layer, measure_layer).float()
        if metric == 'kl':
            ref_logp = F.log_softmax(ref, dim=-1)

        for start in range(0, len(angles_rad), batch_size):
            end = min(start + batch_size, len(angles_rad))
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
                kl = F.kl_div(pf_logp, ref_logp.expand_as(pf_logp), log_target=True, reduction='none').sum(-1)
                result[start:end] = kl.cpu().numpy()
            elif metric == 'logit_l2':
                result[start:end] = (pf - ref).norm(dim=-1).cpu().numpy()
            elif metric == 'logit_cosine':
                result[start:end] = (1.0 - F.cosine_similarity(ref, pf, dim=-1)).cpu().numpy()
    return result


def run_sweep(model, all_contexts, all_activations, sae_dirs, rand_dirs,
              contrastive_dir, pc1_dir, layer, measure_layer, device, angles_rad,
              metric='l2', n_curves=15):
    """Run individual sweeps, return dict of curves."""
    n_act = len(all_activations)
    n_dirs = sae_dirs.shape[1]
    curves = {}

    rng = np.random.RandomState(42)
    act_indices = rng.randint(0, n_act, size=n_curves)
    dir_indices = rng.randint(0, n_dirs, size=n_curves)

    for label, get_dir in [
        ('sae', lambda k: sae_dirs[act_indices[k], dir_indices[k]]),
        ('random', lambda k: rand_dirs[act_indices[k], dir_indices[k]]),
    ]:
        print(f"  Sweeping {n_curves} {label} curves...")
        t0 = time.time()
        for k in range(n_curves):
            curves[f'{label}_{k}'] = sweep_one(
                model, all_contexts[act_indices[k]], all_activations[act_indices[k]],
                get_dir(k), layer, measure_layer, device, angles_rad, metric=metric)
        print(f"    {time.time()-t0:.1f}s")

    n_cont = min(n_curves, n_act)
    for label, direction in [('contrastive', contrastive_dir), ('pc1', pc1_dir)]:
        print(f"  Sweeping {n_cont} {label} curves...")
        t0 = time.time()
        for k in range(n_cont):
            curves[f'{label}_{k}'] = sweep_one(
                model, all_contexts[k], all_activations[k],
                direction, layer, measure_layer, device, angles_rad, metric=metric)
        print(f"    {time.time()-t0:.1f}s")

    return curves


def plot_individual_curves(curves, angles_deg, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    prefix_map = {
        'random_': ('C0', 'Random'), 'contrastive_': ('C2', 'Contrastive DoM'),
        'neg_dom_': ('C2', '-DoM (toward Chinese)'), 'pc1_': ('C1', 'PC1'),
        'sae_': ('C3', 'SAE features'),
    }
    for prefix, (color, label) in prefix_map.items():
        keys = sorted([k for k in curves if k.startswith(prefix)])
        if not keys:
            continue
        for k in keys:
            ax.plot(angles_deg, curves[k], color=color, alpha=0.6, lw=1.0)
        ax.plot([], [], color=color, lw=2, label=f'{label} ({len(keys)})')
    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, angles_deg[-1])
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_median(curves, angles_deg, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for prefix, color, label in [
        ('random_', 'C0', 'Random'), ('contrastive_', 'C2', 'Contrastive DoM'),
        ('neg_dom_', 'C2', '-DoM (toward Chinese)'), ('pc1_', 'C1', 'PC1'),
        ('sae_', 'C3', 'SAE features'),
    ]:
        keys = [k for k in curves if k.startswith(prefix)]
        if not keys:
            continue
        data = np.array([curves[k] for k in keys])
        med = np.median(data, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.15)
        ax.plot(angles_deg, med, color=color, lw=2.5, label=label)
    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, angles_deg[-1])
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=7)
    parser.add_argument('--sae_width', default='65k', choices=['16k', '32k', '65k', '131k'])
    parser.add_argument('--sae_type', default='auto', choices=['auto', 'qwen', 'gemma'],
                        help='SAE type (auto-detected from model name)')
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=30)
    parser.add_argument('--n_dirs', type=int, default=10)
    parser.add_argument('--n_curves', type=int, default=15)
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--negate_dom', action='store_true',
                        help='Negate the DoM direction (perturb toward Chinese)')
    parser.add_argument('--directions', default='all',
                        choices=['all', 'dom_random', 'dom_random_sae'],
                        help='Which directions to sweep')
    parser.add_argument('--prompts', default='fineweb',
                        choices=['fineweb', 'zh', 'en'],
                        help='Source of activations to perturb')
    parser.add_argument('--metrics', default='l2,cosine,kl',
                        help='Comma-separated metrics to run')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/lesswrong_plot/{args.model}'
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

    # Extract activations
    if args.prompts == 'fineweb':
        print(f"\nLoading {args.n_act} FineWeb sequences...")
        token_ids_list = load_fineweb_fixed_length(
            args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    else:
        prompts = CHINESE_PROMPTS if args.prompts == 'zh' else ENGLISH_PROMPTS
        print(f"\nTokenizing {len(prompts)} {args.prompts.upper()} prompts...")
        encoded = tokenizer(prompts[:args.n_act], return_tensors='pt', padding=True)
        token_ids_list = list(encoded['input_ids'])
    print(f"Extracting layer-{args.layer} activations...")
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Activation norms: mean={all_activations.norm(dim=-1).mean():.2f}")

    # Contrastive directions (DoM + PC1)
    contrastive_dir, pc1_dir = get_contrastive_directions(model, tokenizer, device, args.layer)
    if args.negate_dom:
        contrastive_dir = -contrastive_dir
        print("  Using NEGATED DoM direction (toward Chinese)")

    feature_ids = None
    sae_dirs = None
    if args.directions in ('all', 'dom_random_sae'):
        # SAE directions
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
            sae, all_activations, args.n_dirs, device)
        print(f"Top feature IDs range: {feature_ids.min().item()}-{feature_ids.max().item()}")
        del sae
        torch.cuda.empty_cache()

    # Random directions
    rand_dirs = F.normalize(torch.randn(args.n_act, args.n_dirs, d_model), dim=-1)

    # Run sweeps
    ylabels = {
        'l2': f'L2 distance (layer {measure_layer})',
        'cosine': f'Cosine distance (layer {measure_layer})',
        'kl': 'KL divergence (logits)',
        'logit_l2': 'L2 distance (logits)',
        'logit_cosine': 'Cosine distance (logits)',
    }
    for metric in args.metrics.split(','):
        ylabel = ylabels[metric]
        print(f"\n{'='*60}")
        print(f"Metric: {metric}")
        print(f"{'='*60}")

        if args.directions in ('dom_random', 'dom_random_sae'):
            curves = {}
            n_act = len(all_activations)
            rng = np.random.RandomState(42)
            act_indices = rng.randint(0, n_act, size=args.n_curves)
            dir_indices = rng.randint(0, args.n_dirs, size=args.n_curves)

            # Random
            print(f"  Sweeping {args.n_curves} random curves...")
            t0 = time.time()
            for k in range(args.n_curves):
                curves[f'random_{k}'] = sweep_one(
                    model, all_contexts[act_indices[k]], all_activations[act_indices[k]],
                    rand_dirs[act_indices[k], dir_indices[k]],
                    args.layer, measure_layer, device, angles_rad, metric=metric)
            print(f"    {time.time()-t0:.1f}s")

            # DoM
            n_cont = min(args.n_curves, n_act)
            dom_label = 'neg_dom' if args.negate_dom else 'contrastive'
            print(f"  Sweeping {n_cont} {dom_label} curves...")
            t0 = time.time()
            for k in range(n_cont):
                curves[f'{dom_label}_{k}'] = sweep_one(
                    model, all_contexts[k], all_activations[k],
                    contrastive_dir, args.layer, measure_layer, device, angles_rad, metric=metric)
            print(f"    {time.time()-t0:.1f}s")

            # SAE (if requested)
            if args.directions == 'dom_random_sae':
                print(f"  Sweeping {args.n_curves} SAE curves...")
                t0 = time.time()
                for k in range(args.n_curves):
                    curves[f'sae_{k}'] = sweep_one(
                        model, all_contexts[act_indices[k]], all_activations[act_indices[k]],
                        sae_dirs[act_indices[k], dir_indices[k]],
                        args.layer, measure_layer, device, angles_rad, metric=metric)
                print(f"    {time.time()-t0:.1f}s")
        else:
            curves = run_sweep(model, all_contexts, all_activations,
                               sae_dirs, rand_dirs, contrastive_dir, pc1_dir,
                               args.layer, measure_layer, device, angles_rad,
                               metric=metric, n_curves=args.n_curves)

        # Save
        save_data = {'args': vars(args), 'angles_deg': angles_deg, 'curves': curves}
        if feature_ids is not None:
            save_data['feature_ids'] = feature_ids.numpy()
        pkl_path = os.path.join(args.results_dir, f'curves_{metric}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"  Saved: {pkl_path}")

        title = f'{args.model}: inject@L{args.layer}, measure@L{measure_layer}'
        plot_individual_curves(curves, angles_deg, ylabel, title,
                               os.path.join(args.results_dir, f'plot_{metric}_3curves.png'))
        plot_median(curves, angles_deg, ylabel, title,
                    os.path.join(args.results_dir, f'plot_{metric}_median.png'))

    print("\nDone!")


if __name__ == '__main__':
    main()
