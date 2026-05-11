#!/usr/bin/env python3
"""
Perturbation sweep comparing method-specific directions to random + contrastive.

For each method (PCA, SAE, our dict), produces a plot showing the perturbation
response curve when perturbing real activations along:
  - The first 5 directions selected by that method
  - 5 random directions
  - The EN→ZH contrastive direction

"First 5" means: PCA → PC1-PC5, SAE → top-5 features by pre-act magnitude,
dict → first 5 atoms selected by matching pursuit. Since SAE/dict selections
are per-activation, results are pooled over activations.

Usage:
    python -u scripts/dictionary_learning/sweep_directions_comparison.py \
        --model meta-llama/Llama-3.2-1B --layer 2 \
        --mp_dict_path results/pruned_dictionary/llama-3.2-1b_layer2/pruned_dictionary.pkl \
        --sae_path results/trained_sae/llama-3.2-1b_layer2_8x_k32/sae_weights.pt \
        --output_dir results/sweep_directions_comparison/llama-3.2-1b_layer2
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
    load_model, forward_from_layer_to_final,
    _get_final_norm, _get_lm_head,
)
from src.data import load_fineweb_fixed_length


def sweep_single(model, ctx, activation, directions, angles_rad, layer, device):
    """Spherical perturbation sweep for one activation along multiple directions.

    Args:
        directions: (n_dirs, d_model) — specific directions to perturb along

    Returns:
        cos_dists: (n_angles, n_dirs) cosine distance at final layer
    """
    n_angles = len(angles_rad)
    n_dirs = directions.shape[0]
    cos_dists = np.zeros((n_angles, n_dirs))

    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.unsqueeze(0).to(device=device, dtype=model_dtype)
    a = activation.to(device=device, dtype=model_dtype)
    R = a.norm()
    if R < 1e-8:
        return cos_dists
    a_unit = a / R

    dirs_d = directions.to(device=device, dtype=model_dtype)
    # Orthogonalize directions against activation
    proj = (dirs_d @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
    d_perp = dirs_d - proj
    d_perp_norm = d_perp.norm(dim=-1, keepdim=True)
    d_perp = d_perp / (d_perp_norm + 1e-8)

    with torch.no_grad():
        ref_point = a.unsqueeze(0).unsqueeze(0)
        ref_final = forward_from_layer_to_final(model, ctx_d, ref_point, layer)

    for ang_idx, theta in enumerate(angles_rad):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        perturbed = R * (cos_t * a_unit.unsqueeze(0) + sin_t * d_perp)
        perturbed = perturbed.unsqueeze(1)

        with torch.no_grad():
            pert_final = forward_from_layer_to_final(model, ctx_d, perturbed, layer)
            cos_sim = F.cosine_similarity(pert_final.float(), ref_final.float(), dim=-1)
            cos_dists[ang_idx] = (1 - cos_sim).cpu().numpy()

    return cos_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--mp_dict_path', required=True)
    parser.add_argument('--sae_path', required=True)
    parser.add_argument('--n_dirs', type=int, default=5,
                        help='Number of directions per method')
    parser.add_argument('--n_act', type=int, default=30,
                        help='Number of activations to sweep')
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--max_angle', type=float, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_total', type=int, default=8000)
    parser.add_argument('--eval_start', type=int, default=7000)
    parser.add_argument('--pca_fit_start', type=int, default=2000)
    parser.add_argument('--pca_fit_end', type=int, default=7000)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    eval_end = args.eval_start + args.n_act
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)
    nd = args.n_dirs

    print(f"Direction sweep comparison")
    print(f"  model={args.model}, layer={args.layer}")
    print(f"  {nd} directions per method, {args.n_act} activations, {args.n_angles} angles")
    print()

    # ── Load dictionary ─────────────────────────────────────────────────
    print("Loading dictionary...")
    with open(args.mp_dict_path, 'rb') as f:
        mp_dict = F.normalize(pickle.load(f)['dictionary'].float(), dim=-1)
    print(f"  {mp_dict.shape}")

    # ── Load SAE ────────────────────────────────────────────────────────
    print("Loading SAE...")
    from src.autoencoders import TopKSAE
    sae_data = torch.load(args.sae_path, map_location='cpu', weights_only=False)
    config = sae_data['config']
    sae = TopKSAE(d_in=config['d_in'], d_sae=config['d_sae'], k=config['k'])
    sae.load_state_dict(sae_data['state_dict'])
    sae.eval()
    sae_decoder = F.normalize(sae.W_dec.data.float(), dim=-1)
    print(f"  d_sae={config['d_sae']}, k={config['k']}")

    # ── Load model ──────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    mp_dict = mp_dict.to(device)
    sae = sae.to(device)

    # ── Contrastive direction ───────────────────────────────────────────
    print("Computing contrastive direction...")
    from data.bilingual_prompts_hq import PROMPT_PAIRS
    en_prompts = [p[0] for p in PROMPT_PAIRS]
    zh_prompts = [p[1] for p in PROMPT_PAIRS]
    en_acts, zh_acts = [], []
    for p in en_prompts:
        toks = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**toks, output_hidden_states=True)
        en_acts.append(out.hidden_states[args.layer + 1][:, -1, :].float().squeeze(0))
    for p in zh_prompts:
        toks = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**toks, output_hidden_states=True)
        zh_acts.append(out.hidden_states[args.layer + 1][:, -1, :].float().squeeze(0))
    en_acts = torch.stack(en_acts)
    zh_acts = torch.stack(zh_acts)
    contrastive_dir = F.normalize((zh_acts - en_acts).mean(dim=0), dim=0).cpu()
    print(f"  ||mean diff|| = {(zh_acts - en_acts).mean(dim=0).norm().item():.2f}")

    # ── Extract activations ─────────────────────────────────────────────
    print(f"Loading {args.n_total} FineWeb samples...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_total, tokenizer, seq_len=args.seq_len, seed=args.seed)

    eval_tokens = token_ids_list[args.eval_start:eval_end]
    all_contexts, all_activations = [], []
    batch_size = 64
    for i in range(0, len(eval_tokens), batch_size):
        batch = torch.stack(eval_tokens[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
        hidden = outputs.hidden_states[args.layer + 1]
        all_contexts.append(hidden[:, :-1, :].float().cpu())
        all_activations.append(hidden[:, -1, :].float().cpu())
    contexts = torch.cat(all_contexts, 0)
    activations = torch.cat(all_activations, 0)
    print(f"  {activations.shape}, mean norm: {activations.norm(dim=-1).mean():.2f}")

    # PCA fitting
    print(f"Extracting PCA fitting activations...")
    pca_tokens = token_ids_list[args.pca_fit_start:args.pca_fit_end]
    pca_acts = []
    for i in range(0, len(pca_tokens), batch_size):
        batch = torch.stack(pca_tokens[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
        hidden = outputs.hidden_states[args.layer + 1]
        pca_acts.append(hidden[:, -1, :].float().cpu())
    pca_fit = torch.cat(pca_acts, 0)

    # PCA components
    mean = pca_fit.mean(dim=0)
    U, S, V = torch.pca_lowrank(pca_fit - mean, q=nd)
    pca_dirs = V.T  # (nd, d_model) — first nd principal components

    # ── Sweep ───────────────────────────────────────────────────────────
    n_act = args.n_act

    # For each activation, collect directions for each method
    # Then sweep all directions at once, store results separately

    # Results: dict of method -> list of (n_angles,) arrays (one per activation×direction)
    results = {name: [] for name in ['PCA', 'SAE', 'Dict', 'Random', 'Contrastive']}

    print(f"\nSweeping {n_act} activations...")
    t0 = time.time()
    for i in range(n_act):
        act = activations[i]
        ctx = contexts[i]
        act_gpu = act.to(device)

        # Dict: first nd atoms from matching pursuit
        residual = act_gpu.clone()
        dict_dirs = []
        for step in range(nd):
            corr = residual @ mp_dict.T
            best = corr.abs().argmax()
            dict_dirs.append(mp_dict[best].cpu())
            coef = corr[best]
            residual = residual - coef * mp_dict[best]
        dict_dirs = torch.stack(dict_dirs)  # (nd, d_model)

        # SAE: top-nd features by pre-act magnitude
        _, _, pre_acts = sae.encode(act_gpu.unsqueeze(0))
        topn_vals, topn_idx = pre_acts.topk(nd, dim=-1)
        sae_dirs = sae_decoder[topn_idx.squeeze(0).cpu()]  # (nd, d_model)

        # Random: nd random unit directions
        rand_dirs = F.normalize(torch.randn(nd, activations.shape[1]), dim=-1)

        # All directions for this activation
        all_dirs = torch.cat([
            pca_dirs,                           # nd
            sae_dirs,                           # nd
            dict_dirs,                          # nd
            rand_dirs,                          # nd
            contrastive_dir.unsqueeze(0),        # 1
        ], dim=0)  # (4*nd + 1, d_model)

        cos_dists = sweep_single(model, ctx, act, all_dirs, angles_rad, args.layer, device)
        # cos_dists: (n_angles, 4*nd + 1)

        # Split back
        offset = 0
        for name, count in [('PCA', nd), ('SAE', nd), ('Dict', nd),
                            ('Random', nd), ('Contrastive', 1)]:
            for j in range(count):
                results[name].append(cos_dists[:, offset + j])
            offset += count

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_act}  ({(i+1)/(time.time()-t0):.1f} act/s)")

    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Save ────────────────────────────────────────────────────────────
    save_data = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'results': {k: np.array(v) for k, v in results.items()},
    }
    pkl_path = os.path.join(args.output_dir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved: {pkl_path}")

    # ── Plot ────────────────────────────────────────────────────────────
    method_plots = ['PCA', 'SAE', 'Dict']
    for method in method_plots:
        fig, ax = plt.subplots(figsize=(10, 6))

        for name, color, ls in [
            (method, 'C0', '-'),
            ('Random', 'C3', '--'),
            ('Contrastive', 'red', '-'),
        ]:
            data = results[name]  # list of (n_angles,) arrays
            data = np.array(data)  # (n_samples, n_angles)
            median = np.median(data, axis=0)
            q25 = np.percentile(data, 25, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            ax.plot(angles_deg, median, color=color, linestyle=ls,
                    label=name, linewidth=2)
            ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.12)

        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel('Cosine distance (final layer)')
        ax.set_title(f'{method} directions vs random vs contrastive\n'
                     f'{args.model}, layer {args.layer}, {nd} dirs × {n_act} acts')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, args.max_angle)
        ax.axhline(y=0.05, color='gray', linestyle=':', alpha=0.5, label='threshold')

        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, f'sweep_{method.lower()}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fig_path}")


if __name__ == '__main__':
    main()
