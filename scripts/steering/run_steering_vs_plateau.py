#!/usr/bin/env python3
"""
Steering vs plateau experiment: compare random spherical perturbations
with steering along a meaningful feature direction.

Steering vector extracted at layer 14 (mid-network), applied at layer 2
(where existing spherical sweep was run).

Usage:
    python scripts/run_steering_vs_plateau.py --model qwen3-0.6b
    python scripts/run_steering_vs_plateau.py --model qwen3-0.6b --language dutch
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
LANGUAGE_CONFIGS = {
    'chinese': {
        'module': 'data.bilingual_prompts',
        'en_attr': 'ENGLISH_PROMPTS',
        'other_attr': 'CHINESE_PROMPTS',
        'label': 'en↔zh',
    },
    'dutch': {
        'module': 'data.bilingual_prompts_dutch',
        'en_attr': 'ENGLISH_PROMPTS',
        'other_attr': 'DUTCH_PROMPTS',
        'label': 'en↔nl',
    },
}


def extract_steering_vector(model, tokenizer, device, extract_layer,
                            en_prompts, other_prompts, label='en↔other'):
    """Compute language steering vector at extract_layer via difference-of-means.

    Returns unit vector: mean(english) - mean(other_language), normalized.
    """
    def get_mean_activation(prompts):
        acts = []
        for p in prompts:
            tokens = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[extract_layer + 1]
            acts.append(hidden[:, -1, :].squeeze(0).cpu())
        return torch.stack(acts).mean(dim=0)

    print(f"Computing steering vector ({label}) from {len(en_prompts)} prompt pairs at layer {extract_layer}...")
    en_mean = get_mean_activation(en_prompts)
    other_mean = get_mean_activation(other_prompts)
    steering_vec = en_mean - other_mean
    steering_norm = steering_vec.norm().item()
    steering_unit = steering_vec / steering_vec.norm()
    print(f"Steering vector norm: {steering_norm:.2f}")
    return steering_unit, steering_norm


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device, batch_size=64):
    """Extract layer-L context and final-token activation for fixed-length sequences."""
    all_contexts = []
    all_activations = []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].cpu())
        all_activations.append(hidden[:, -1, :].cpu())
    return torch.cat(all_contexts, 0), torch.cat(all_activations, 0)


def run_steering_sweep(model, context, activation, steering_dir, angles_rad,
                       layer, device):
    """Sweep steering perturbations for one activation.

    Projects steering_dir orthogonal to activation, then applies spherically:
    a_perturbed = cos(theta) * a + sin(theta) * ||a|| * v_steer_perp

    Returns:
        l2_dists: [n_angles]
        kl_divs: [n_angles]
    """
    n_angles = len(angles_rad)
    l2_dists = np.zeros(n_angles)
    kl_divs = np.zeros(n_angles)

    ctx = context.unsqueeze(0).to(device)
    a = activation.to(device)
    R = a.norm()
    a_unit = a / R
    s = steering_dir.to(device)

    # Project steering direction orthogonal to activation
    s_perp = s - (s @ a_unit) * a_unit
    s_perp_norm = s_perp.norm()
    if s_perp_norm < 1e-8:
        print("Warning: steering direction nearly parallel to activation")
        return l2_dists, kl_divs
    s_perp = s_perp / s_perp_norm  # unit vector orthogonal to a, in steering plane

    # Reference (unperturbed)
    with torch.no_grad():
        ref_point = a.unsqueeze(0).unsqueeze(0)
        ref_final = forward_from_layer_to_final(model, ctx, ref_point, layer)
        ref_ln = _get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1)
        ref_logits = _get_lm_head(model)(ref_ln)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    for ang_idx, theta in enumerate(angles_rad):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        perturbed = R * (cos_t * a_unit + sin_t * s_perp)
        perturbed = perturbed.unsqueeze(0).unsqueeze(0)  # [1, 1, d]

        with torch.no_grad():
            pert_final = forward_from_layer_to_final(model, ctx, perturbed, layer)
            l2_dists[ang_idx] = (pert_final - ref_final).norm(dim=-1).cpu().item()

            pert_ln = _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)
            pert_logits = _get_lm_head(model)(pert_ln)
            pert_log_probs = F.log_softmax(pert_logits, dim=-1)
            ref_probs = ref_log_probs.exp()
            kl = (ref_probs * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            kl_divs[ang_idx] = kl.cpu().item()

    return l2_dists, kl_divs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-0.6b')
    parser.add_argument('--language', type=str, default='chinese',
                        choices=list(LANGUAGE_CONFIGS.keys()),
                        help='Target language for steering vector')
    parser.add_argument('--apply_layer', type=int, default=2,
                        help='Layer at which to apply perturbation/steering')
    parser.add_argument('--extract_layer', type=int, default=14,
                        help='Layer at which to extract steering vector')
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=100)
    parser.add_argument('--n_angles', type=int, default=60)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load language-specific prompts
    import importlib
    lang_cfg = LANGUAGE_CONFIGS[args.language]
    mod = importlib.import_module(lang_cfg['module'])
    en_prompts = getattr(mod, lang_cfg['en_attr'])
    other_prompts = getattr(mod, lang_cfg['other_attr'])
    lang_label = lang_cfg['label']

    if args.results_dir is None:
        args.results_dir = f'results/steering_vs_plateau/{args.model}_{args.language}'
    os.makedirs(args.results_dir, exist_ok=True)

    angles_deg = np.linspace(0, 180, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    print(f"Config: model={args.model}, language={args.language}, apply_layer={args.apply_layer}, "
          f"extract_layer={args.extract_layer}, n_act={args.n_act}, n_angles={len(angles_deg)}")

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    # Extract steering vector at extract_layer
    steering_unit, steering_norm = extract_steering_vector(
        model, tokenizer, device, args.extract_layer,
        en_prompts, other_prompts, lang_label
    )

    # Extract activations at apply_layer
    print(f"\nLoading {args.n_act} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(args.n_act, tokenizer,
                                                seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.apply_layer} activations...")
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.apply_layer, device
    )
    act_norm_mean = activations.norm(dim=-1).mean().item()
    print(f"Mean activation norm: {act_norm_mean:.2f}")

    # Check alignment of steering vector with activations
    cosines = F.cosine_similarity(activations, steering_unit.unsqueeze(0), dim=-1)
    print(f"Cosine(steering, activations): mean={cosines.mean().item():.4f}, "
          f"std={cosines.std().item():.4f}, max={cosines.abs().max().item():.4f}")

    # Steering sweep
    print(f"\n=== Steering sweep ({lang_label} direction) ===")
    steer_l2 = np.zeros((args.n_act, len(angles_deg)))
    steer_kl = np.zeros((args.n_act, len(angles_deg)))
    t0 = time.time()
    for i in range(args.n_act):
        steer_l2[i], steer_kl[i] = run_steering_sweep(
            model, contexts[i], activations[i], steering_unit,
            angles_rad, args.apply_layer, device
        )
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
    print(f"Steering done in {time.time()-t0:.1f}s")

    # Save
    results = {
        'args': vars(args),
        'language': args.language,
        'lang_label': lang_label,
        'angles_deg': angles_deg,
        'angles_rad': angles_rad,
        'act_norm_mean': act_norm_mean,
        'steering_norm': steering_norm,
        'steering_cosines': cosines.numpy(),
        'steering': {'l2': steer_l2, 'kl': steer_kl},
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {args.results_dir}/results.pkl")

    print("\nDone!")


if __name__ == '__main__':
    main()
