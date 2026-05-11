#!/usr/bin/env python3
"""
Pareto plot with relative L2 on y-axis (instead of KL-div).

Reuses plateau angle cache from the KL version. Only recomputes the
reconstruction metric (relative L2 = ||orig - recon|| / ||orig||).

Usage:
    python -u scripts/dictionary_learning/pareto_plateau_vs_recon_l2.py \
        --model meta-llama/Llama-3.2-1B --layer 2 \
        --mp_dict_path results/pruned_dictionary/llama-3.2-1b_layer2/pruned_dictionary.pkl \
        --fp_dict_path results/pruned_dictionary_fp/llama-3.2-1b_layer2/pruned_dictionary.pkl \
        --sae_path results/trained_sae/llama-3.2-1b_layer2_8x_k32/sae_weights.pt \
        --pareto_pkl results/pareto_plateau_recon/llama-3.2-1b_layer2/pareto_results.pkl \
        --output_dir results/pareto_plateau_recon/llama-3.2-1b_layer2
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model
from src.data import load_fineweb_fixed_length


def matching_pursuit_with_indices(dictionary, activations, n_nonzero_coefs):
    n_val = activations.shape[0]
    device = activations.device
    residuals = activations.clone()
    recon = torch.zeros_like(activations)
    selected = torch.zeros(n_val, n_nonzero_coefs, dtype=torch.long, device=device)
    for step in range(n_nonzero_coefs):
        corr = residuals @ dictionary.T
        best = corr.abs().argmax(dim=1)
        selected[:, step] = best
        coefs = corr[torch.arange(n_val, device=device), best]
        contribution = coefs.unsqueeze(1) * dictionary[best]
        recon = recon + contribution
        residuals = residuals - contribution
    return recon, selected


def sae_reconstruct_with_indices(sae, activations, n, batch_size=256):
    n_val = activations.shape[0]
    recon_chunks, idx_chunks = [], []
    for i in range(0, n_val, batch_size):
        batch = activations[i:i+batch_size]
        _, _, pre_acts = sae.encode(batch)
        topn_vals, topn_idx = pre_acts.topk(n, dim=-1)
        recon_chunks.append(sae.decode(topn_vals, topn_idx))
        idx_chunks.append(topn_idx)
    return torch.cat(recon_chunks, 0), torch.cat(idx_chunks, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--mp_dict_path', required=True)
    parser.add_argument('--fp_dict_path', required=True)
    parser.add_argument('--sae_path', default=None)
    parser.add_argument('--pareto_pkl', required=True,
                        help='Path to pareto_results.pkl from KL version (for plateau angles)')
    parser.add_argument('--n_values', default='5,10,20,50')
    parser.add_argument('--n_eval', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_total', type=int, default=8000)
    parser.add_argument('--eval_start', type=int, default=7000)
    parser.add_argument('--pca_fit_start', type=int, default=2000)
    parser.add_argument('--pca_fit_end', type=int, default=7000)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    n_values = [int(x) for x in args.n_values.split(',')]
    eval_end = args.eval_start + args.n_eval
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load cached plateau angles
    print("Loading plateau angle cache...")
    with open(args.pareto_pkl, 'rb') as f:
        pareto_data = pickle.load(f)
    plateau_angle_lookup = pareto_data['plateau_angle_lookup']
    contrastive_angle = pareto_data['contrastive_angle']
    print(f"  {len(plateau_angle_lookup)} cached angles")

    # Load dictionaries
    print("Loading dictionaries...")
    with open(args.mp_dict_path, 'rb') as f:
        mp_dict = F.normalize(pickle.load(f)['dictionary'].float(), dim=-1)
    with open(args.fp_dict_path, 'rb') as f:
        fp_dict = F.normalize(pickle.load(f)['dictionary'].float(), dim=-1)
    d_model = mp_dict.shape[1]
    torch.manual_seed(args.seed + 999)
    rand_dict = F.normalize(torch.randn(mp_dict.shape[0], d_model), dim=-1)

    # Load SAE
    sae = None
    if args.sae_path:
        from src.autoencoders import TopKSAE
        sae_data = torch.load(args.sae_path, map_location='cpu', weights_only=False)
        config = sae_data['config']
        sae = TopKSAE(d_in=config['d_in'], d_sae=config['d_sae'], k=config['k'])
        sae.load_state_dict(sae_data['state_dict'])
        sae.eval()

    # Load model + activations
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    mp_dict = mp_dict.to(device)
    fp_dict = fp_dict.to(device)
    rand_dict = rand_dict.to(device)
    if sae:
        sae = sae.to(device)

    print(f"Loading FineWeb samples...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_total, tokenizer, seq_len=args.seq_len, seed=args.seed)

    eval_tokens = token_ids_list[args.eval_start:eval_end]
    all_activations = []
    batch_size = 64
    for i in range(0, len(eval_tokens), batch_size):
        batch = torch.stack(eval_tokens[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
        hidden = outputs.hidden_states[args.layer + 1]
        all_activations.append(hidden[:, -1, :].float().cpu())
    eval_activations = torch.cat(all_activations, 0)
    eval_act_gpu = eval_activations.to(device)
    norms = eval_act_gpu.norm(dim=-1)

    # PCA
    pca_tokens = token_ids_list[args.pca_fit_start:args.pca_fit_end]
    pca_acts = []
    for i in range(0, len(pca_tokens), batch_size):
        batch = torch.stack(pca_tokens[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
        hidden = outputs.hidden_states[args.layer + 1]
        pca_acts.append(hidden[:, -1, :].float().cpu())
    pca_fit = torch.cat(pca_acts, 0)

    # Contrastive direction
    from data.bilingual_prompts_hq import PROMPT_PAIRS
    en_acts, zh_acts = [], []
    for p in [pair[0] for pair in PROMPT_PAIRS]:
        toks = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**toks, output_hidden_states=True)
        en_acts.append(out.hidden_states[args.layer + 1][:, -1, :].float().squeeze(0))
    for p in [pair[1] for pair in PROMPT_PAIRS]:
        toks = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**toks, output_hidden_states=True)
        zh_acts.append(out.hidden_states[args.layer + 1][:, -1, :].float().squeeze(0))
    contrastive_dir = F.normalize((torch.stack(zh_acts) - torch.stack(en_acts)).mean(dim=0), dim=0)

    # Helper
    def avg_plateau_angle(selected_indices, prefix):
        total = 0.0
        count = 0
        for i in range(selected_indices.shape[0]):
            for j in range(selected_indices.shape[1]):
                idx = selected_indices[i, j].item()
                total += plateau_angle_lookup[f"{prefix}:{idx}"]
                count += 1
        return total / count

    def rel_l2(recon):
        return ((eval_act_gpu - recon).norm(dim=-1) / norms).mean().item()

    # Run reconstructions
    max_n = max(n_values)
    print(f"\nComputing Pareto points (rel L2)...")

    _, mp_sel = matching_pursuit_with_indices(mp_dict, eval_act_gpu, max_n)
    _, fp_sel = matching_pursuit_with_indices(fp_dict, eval_act_gpu, max_n)
    _, rand_sel = matching_pursuit_with_indices(rand_dict, eval_act_gpu, max_n)
    sae_sel = None
    if sae:
        _, sae_sel = sae_reconstruct_with_indices(sae, eval_act_gpu, max_n)

    pareto_points = {}

    for method_name, dict_gpu, selected, prefix in [
        ('MP-pruned', mp_dict, mp_sel, 'mp'),
        ('FP-pruned', fp_dict, fp_sel, 'fp'),
        ('Random dict', rand_dict, rand_sel, 'rand'),
    ]:
        pareto_points[method_name] = []
        print(f"\n  {method_name}:")
        for n in n_values:
            sel_n = selected[:, :n].cpu()
            avg_angle = avg_plateau_angle(sel_n, prefix)
            # Reconstruct with first n MP steps
            recon = torch.zeros_like(eval_act_gpu)
            residuals = eval_act_gpu.clone()
            for step in range(n):
                best = selected[:, step]
                coefs = (residuals * dict_gpu[best]).sum(dim=-1)
                contribution = coefs.unsqueeze(1) * dict_gpu[best]
                recon = recon + contribution
                residuals = residuals - contribution
            rl2 = rel_l2(recon)
            pareto_points[method_name].append((avg_angle, rl2, n))
            print(f"    n={n:3d}: avg_angle={avg_angle:.1f} deg, rel_L2={rl2:.4f}")

    if sae and sae_sel is not None:
        pareto_points['SAE'] = []
        print(f"\n  SAE:")
        for n in n_values:
            sel_n = sae_sel[:, :n].cpu()
            avg_angle = avg_plateau_angle(sel_n, 'sae')
            recon, _ = sae_reconstruct_with_indices(sae, eval_act_gpu, n)
            rl2 = rel_l2(recon)
            pareto_points['SAE'].append((avg_angle, rl2, n))
            print(f"    n={n:3d}: avg_angle={avg_angle:.1f} deg, rel_L2={rl2:.4f}")

    # PCA: full SVD + adaptive per-activation selection
    pca_mean = pca_fit.mean(dim=0)
    _, _, Vh = torch.linalg.svd(pca_fit - pca_mean, full_matrices=False)
    pca_all_components = Vh  # (d_model, d_model)

    pareto_points['PCA'] = []
    print(f"\n  PCA (adaptive):")
    for n in n_values:
        centered = eval_activations - pca_mean
        all_proj = centered @ pca_all_components.T  # (n_eval, d_model)
        topn_vals, topn_idx = all_proj.abs().topk(n, dim=-1)
        actual_coeffs = all_proj.gather(1, topn_idx)
        selected_dirs = pca_all_components[topn_idx]  # (n_eval, n, d_model)
        recon = ((actual_coeffs.unsqueeze(-1) * selected_dirs).sum(dim=1) + pca_mean).to(device)

        avg_angle = avg_plateau_angle(topn_idx.cpu(), 'pca')
        rl2 = rel_l2(recon)
        pareto_points['PCA'].append((avg_angle, rl2, n))
        print(f"    n={n:3d}: avg_angle={avg_angle:.1f} deg, rel_L2={rl2:.4f}")

    # Contrastive
    contrastive_gpu = contrastive_dir.to(device).unsqueeze(0)
    coefs = (eval_act_gpu * contrastive_gpu).sum(dim=-1, keepdim=True)
    contrastive_recon = coefs * contrastive_gpu
    contrastive_rl2 = rel_l2(contrastive_recon)
    pareto_points['Contrastive (EN→ZH)'] = [(contrastive_angle, contrastive_rl2, 1)]
    print(f"\n  Contrastive: angle={contrastive_angle:.1f} deg, rel_L2={contrastive_rl2:.4f}")

    # Save
    save_data = {
        'args': vars(args),
        'pareto_points': pareto_points,
        'contrastive_angle': contrastive_angle,
        'n_values': n_values,
    }
    pkl_path = os.path.join(args.output_dir, 'pareto_results_l2.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved: {pkl_path}")

    # Plot
    colors = {'MP-pruned':'C0','FP-pruned':'C1','SAE':'C2','PCA':'C4',
              'Random dict':'C3','Contrastive (EN→ZH)':'red'}
    markers = {'MP-pruned':'o','FP-pruned':'s','SAE':'^','PCA':'D',
               'Random dict':'x','Contrastive (EN→ZH)':'*'}

    fig, ax = plt.subplots(figsize=(10, 7))
    for method, points in pareto_points.items():
        angles = [p[0] for p in points]
        rl2s = [p[1] for p in points]
        ns = [p[2] for p in points]
        ax.plot(angles, rl2s, marker=markers.get(method,'o'), color=colors.get(method,'gray'),
                label=method, linewidth=2, markersize=8)
        for a, r, n in points:
            ax.annotate(f'n={n}', (a, r), textcoords='offset points',
                        xytext=(5, 5), fontsize=7, color=colors.get(method,'gray'))

    ax.set_xlabel('Avg plateau-breaking angle (degrees)\n'
                  '← more plateau-breaking | less plateau-breaking →')
    ax.set_ylabel('Relative L2 error\n← better reconstruction')
    ax.set_title(f'Plateau-breaking vs reconstruction quality (L2)\n'
                 f'{args.model}, layer {args.layer}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, 'pareto_plot_l2.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == '__main__':
    main()
