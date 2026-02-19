#!/usr/bin/env python3
"""
Perturbation sweeps for programming language directions on code-adjacent prompts.

Uses prompts like "The following code", "Write a function that", etc.
Saves sweep data for plotting.

Usage:
    python scripts/directions/sweep_programming_code_prompts.py [--model qwen3-0.6b] [--layer 14]
"""

import sys
import argparse
import os
import pickle
import numpy as np

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import (
    load_model,
    _get_blocks, _is_rope_model, _get_position_embeddings, _run_block,
    _get_final_norm, _get_lm_head,
)

CODE_PROMPTS = [
    "The following code",
    "The code below",
    "Here is some code",
    "Consider the following program",
    "The following function",
    "Below is an implementation",
]


def extract_context_and_activation_from_prompts(model, tokenizer, prompts, layer, device):
    """Extract layer-L context and final-token activation for text prompts."""
    all_contexts = []
    all_activations = []
    for p in prompts:
        tokens = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].cpu())
        all_activations.append(hidden[:, -1, :].cpu())
    # Pad contexts to same length for batching
    max_len = max(c.shape[1] for c in all_contexts)
    padded_contexts = []
    for c in all_contexts:
        if c.shape[1] < max_len:
            pad = torch.zeros(1, max_len - c.shape[1], c.shape[2])
            padded_contexts.append(torch.cat([pad, c], dim=1))
        else:
            padded_contexts.append(c)
    return torch.cat(padded_contexts, dim=0), torch.cat(all_activations, dim=0)


def forward_to_logits(model, hidden_states, start_layer):
    """Forward pass returning logits for last token."""
    hidden = hidden_states
    blocks = _get_blocks(model)
    rope = _is_rope_model(model)
    pos_emb = _get_position_embeddings(model, hidden) if rope else None
    for block in blocks[start_layer + 1:]:
        hidden = _run_block(block, hidden, model, pos_emb)
    final_hidden = hidden[:, -1, :]
    normed = _get_final_norm(model)(final_hidden)
    logits = _get_lm_head(model)(normed)
    return final_hidden, logits


def sweep_direction(model, contexts, activations, ref_final, ref_log_probs,
                    direction, angles_rad, layer, device):
    """Perturbation sweep for one direction."""
    N = activations.shape[0]
    n_angles = len(angles_rad)
    kl_divs = np.zeros((N, n_angles))
    l2_dists = np.zeros((N, n_angles))
    direction_dev = direction.to(device)
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms

    for ang_idx, theta in enumerate(angles_rad):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        proj = (direction_dev.unsqueeze(0) * a_unit).sum(dim=-1, keepdim=True)
        u_perp = direction_dev.unsqueeze(0) - proj * a_unit
        u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)
        a_pert = a_norms * (cos_t * a_unit + sin_t * u_perp)
        with torch.no_grad():
            hidden = torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)
            pert_final, pert_logits = forward_to_logits(model, hidden, layer)
            pert_log_probs = F.log_softmax(pert_logits, dim=-1)
            l2 = (pert_final - ref_final).norm(dim=-1)
            kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            l2_dists[:, ang_idx] = l2.cpu().numpy()
            kl_divs[:, ang_idx] = kl.cpu().numpy()
    return kl_divs, l2_dists


def sweep_random(model, contexts, activations, ref_final, ref_log_probs,
                 angles_rad, layer, device, n_dirs=10):
    """Perturbation sweep with random directions."""
    N, d = activations.shape
    n_angles = len(angles_rad)
    kl_divs = np.zeros((N, n_angles, n_dirs))
    l2_dists = np.zeros((N, n_angles, n_dirs))
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms

    v_perps = torch.zeros(N, n_dirs, d, device=device)
    for i in range(N):
        v = torch.randn(n_dirs, d, device=device)
        v = v - (v @ a_unit[i]).unsqueeze(1) * a_unit[i].unsqueeze(0)
        v_perps[i] = F.normalize(v, dim=-1)

    for ang_idx, theta in enumerate(angles_rad):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        for dir_idx in range(n_dirs):
            v = v_perps[:, dir_idx, :]
            a_pert = a_norms * (cos_t * a_unit + sin_t * v)
            with torch.no_grad():
                hidden = torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)
                pert_final, pert_logits = forward_to_logits(model, hidden, layer)
                pert_log_probs = F.log_softmax(pert_logits, dim=-1)
                l2 = (pert_final - ref_final).norm(dim=-1)
                kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
                l2_dists[:, ang_idx, dir_idx] = l2.cpu().numpy()
                kl_divs[:, ang_idx, dir_idx] = kl.cpu().numpy()
    return kl_divs, l2_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=40)
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_dir = f'results/programming_contrastive/{args.model}'

    # Load feature directions
    contrastive_path = os.path.join(results_dir, 'contrastive_results.pkl')
    print(f"Loading directions from {contrastive_path}")
    with open(contrastive_path, 'rb') as f:
        contrastive = pickle.load(f)
    languages = contrastive['languages']
    f_hat = contrastive['f_hat']
    print(f"Languages: {languages}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    # Extract activations for code prompts
    print(f"\nExtracting activations for {len(CODE_PROMPTS)} code-adjacent prompts...")
    contexts, activations = extract_context_and_activation_from_prompts(
        model, tokenizer, CODE_PROMPTS, args.layer, device)
    contexts = contexts.to(device)
    activations = activations.to(device)
    print(f"  contexts: {contexts.shape}, activations: {activations.shape}")

    # Reference logits
    with torch.no_grad():
        ref_hidden = torch.cat([contexts, activations.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_logits(model, ref_hidden, args.layer)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # Sweeps
    angles_deg = np.linspace(1, args.max_angle, args.n_angles)
    angles_rad = np.radians(angles_deg)

    sweep_results = {}
    for lang in languages:
        name = f'f_hat_{lang}'
        print(f"  Sweeping {name}...")
        kl, l2 = sweep_direction(
            model, contexts, activations, ref_final, ref_log_probs,
            f_hat[lang], angles_rad, args.layer, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}

    print("  Sweeping random directions...")
    rand_kl, rand_l2 = sweep_random(
        model, contexts, activations, ref_final, ref_log_probs,
        angles_rad, args.layer, device, n_dirs=args.n_random_dirs)
    sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}

    # Save
    save_data = {
        'angles_deg': angles_deg,
        'sweep_results': sweep_results,
        'prompts': CODE_PROMPTS,
        'languages': languages,
    }
    out_path = os.path.join(results_dir, 'sweep_code_prompts.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved to {out_path}")
    print("Done!")


if __name__ == '__main__':
    main()
