#!/usr/bin/env python3
"""
Diagnose the MELBO animal discrepancy: optimization KL=0.87 but sweep shows ~0.

Three diagnostics:
(i)   Perturb in -MELBO direction
(ii)  Perturb the cat/bird/fish/snake activations instead of FineWeb
(iii) Check cosine overlap of MELBO direction with activations at layer 14
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

from src.model import (
    load_model, forward_from_layer_allpos,
    _get_blocks, _is_rope_model, _get_position_embeddings, _run_block,
    _get_final_norm, _get_lm_head,
)
from src.data import load_fineweb_fixed_length
from data.animal_prompts import ALL_CATEGORIES


def extract_animal_activations(model, tokenizer, device, layer):
    """Extract final-token activations + contexts for all animal prompts."""
    acts_by_cat = {}
    contexts_by_cat = {}
    for cat_name, prompts in ALL_CATEGORIES.items():
        acts = []
        ctxs = []
        for p in prompts:
            tokens = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1]
            acts.append(hidden[:, -1, :].squeeze(0).cpu())
            ctxs.append(hidden[:, :-1, :].cpu())
        acts_by_cat[cat_name] = torch.stack(acts)
        contexts_by_cat[cat_name] = ctxs
    return acts_by_cat, contexts_by_cat


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device,
                                   batch_size=64):
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


def build_perturbed_hidden(contexts, activations, direction, angle_rad):
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms
    proj = (direction.unsqueeze(0) * a_unit).sum(dim=-1, keepdim=True)
    u_perp = direction.unsqueeze(0) - proj * a_unit
    u_perp_norm = u_perp.norm(dim=-1, keepdim=True)
    u_perp = u_perp / (u_perp_norm + 1e-10)
    a_pert = a_norms * (cos_t * a_unit + sin_t * u_perp)
    return torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)


def compute_kl_at_angle(model, contexts, activations, ref_log_probs, direction, angle_rad, layer):
    """Compute per-sample KL at a given angle."""
    with torch.no_grad():
        hidden = build_perturbed_hidden(contexts, activations, direction, angle_rad)
        logits = forward_from_layer_allpos(model, hidden, layer)
        pert_log_probs = F.log_softmax(logits, dim=-1)
        kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
    return kl.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--results_dir', default='results/melbo_animal/qwen3-0.6b_r20_s500')
    parser.add_argument('--output_dir', default='results/melbo_animal/diagnosis')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    # Load saved MELBO results
    results_path = os.path.join(args.results_dir, 'results.pkl')
    print(f"Loading results from {results_path}")
    results = pickle.load(open(results_path, 'rb'))
    melbo_dirs = results['melbo_directions']
    melbo_kls = results['melbo_kls']
    mean_dir = results['mean_dir']

    print(f"MELBO directions found: {len(melbo_dirs)}")
    for i, kl in enumerate(melbo_kls):
        print(f"  MELBO_{i+1}: optimization KL = {kl:.6f}")

    # ===== Extract activations =====
    print(f"\nExtracting animal activations at layer {args.layer}...")
    acts_by_cat, contexts_by_cat = extract_animal_activations(
        model, tokenizer, device, args.layer)

    print(f"\nLoading 50 FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(
        50, tokenizer, seq_len=5, seed=args.seed)
    fw_contexts, fw_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)

    # ===== Diagnostic (iii): Cosine overlap =====
    print(f"\n{'='*60}")
    print("DIAGNOSTIC (iii): Cosine overlap of MELBO dirs with activations")
    print(f"{'='*60}")

    for d_idx, direction in enumerate(melbo_dirs):
        d_unit = direction / direction.norm()
        print(f"\nMELBO_{d_idx+1}:")

        # Overlap with each category
        for cat_name, acts in acts_by_cat.items():
            acts_unit = acts / acts.norm(dim=-1, keepdim=True)
            cosines = (acts_unit @ d_unit).numpy()
            print(f"  {cat_name}: mean cos = {cosines.mean():.4f}, "
                  f"std = {cosines.std():.4f}, "
                  f"min = {cosines.min():.4f}, max = {cosines.max():.4f}")

        # Overlap with FineWeb
        fw_unit = fw_activations / fw_activations.norm(dim=-1, keepdim=True)
        fw_cosines = (fw_unit @ d_unit).numpy()
        print(f"  FineWeb: mean cos = {fw_cosines.mean():.4f}, "
              f"std = {fw_cosines.std():.4f}, "
              f"min = {fw_cosines.min():.4f}, max = {fw_cosines.max():.4f}")

        # Also check: how much of the direction is orthogonal to each activation?
        # This is what actually gets used in the perturbation
        for cat_name, acts in acts_by_cat.items():
            acts_unit = acts / acts.norm(dim=-1, keepdim=True)
            proj = (acts_unit @ d_unit).unsqueeze(-1) * acts_unit
            perp = d_unit.unsqueeze(0) - proj
            perp_norms = perp.norm(dim=-1).numpy()
            print(f"  ||perp({cat_name})|| : mean = {perp_norms.mean():.4f}, "
                  f"std = {perp_norms.std():.4f}")

        fw_unit_2d = fw_activations / fw_activations.norm(dim=-1, keepdim=True)
        proj = (fw_unit_2d @ d_unit).unsqueeze(-1) * fw_unit_2d
        perp = d_unit.unsqueeze(0) - proj
        perp_norms = perp.norm(dim=-1).numpy()
        print(f"  ||perp(FineWeb)||   : mean = {perp_norms.mean():.4f}, "
              f"std = {perp_norms.std():.4f}")

    # ===== Diagnostic: mean vs median KL on FineWeb =====
    print(f"\n{'='*60}")
    print("DIAGNOSTIC: Mean vs Median KL on FineWeb at 15 deg")
    print(f"{'='*60}")

    angle_15 = np.deg2rad(15.0)
    fw_ctx_dev = fw_contexts.to(device)
    fw_act_dev = fw_activations.to(device)

    # Reference logits
    with torch.no_grad():
        ref_hidden = torch.cat([fw_ctx_dev, fw_act_dev.unsqueeze(1)], dim=1)
        ref_logits = forward_from_layer_allpos(model, ref_hidden, args.layer)
        fw_ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    for d_idx, direction in enumerate(melbo_dirs):
        d_dev = direction.to(device)
        kl_pos = compute_kl_at_angle(model, fw_ctx_dev, fw_act_dev, fw_ref_log_probs,
                                     d_dev, angle_15, args.layer)
        kl_neg = compute_kl_at_angle(model, fw_ctx_dev, fw_act_dev, fw_ref_log_probs,
                                     -d_dev, angle_15, args.layer)
        print(f"\nMELBO_{d_idx+1} on FineWeb at 15 deg:")
        print(f"  +dir: mean={kl_pos.mean():.4f}, median={np.median(kl_pos):.4f}, "
              f"min={kl_pos.min():.4f}, max={kl_pos.max():.4f}")
        print(f"  -dir: mean={kl_neg.mean():.4f}, median={np.median(kl_neg):.4f}, "
              f"min={kl_neg.min():.4f}, max={kl_neg.max():.4f}")

    # ===== Diagnostic (i) & (ii): Perturb -MELBO and perturb animal activations =====
    print(f"\n{'='*60}")
    print("DIAGNOSTIC (i) & (ii): Perturb animal activations")
    print(f"{'='*60}")

    # For each non-dog category, compute ref logits and then KL for +/- MELBO
    for cat_name in ['cats', 'birds', 'fish', 'snakes']:
        cat_acts = acts_by_cat[cat_name].to(device)
        cat_ctxs = contexts_by_cat[cat_name]

        # Pad contexts to same length for batching (they may vary)
        # Actually, each prompt has different length, so process individually
        print(f"\n{cat_name} (20 prompts):")
        for d_idx, direction in enumerate(melbo_dirs[:1]):  # Just MELBO_1 for now
            d_dev = direction.to(device)
            kl_pos_list = []
            kl_neg_list = []
            for i in range(cat_acts.shape[0]):
                ctx = cat_ctxs[i].to(device)  # [1, seq_len-1, d]
                act = cat_acts[i:i+1]  # [1, d]

                with torch.no_grad():
                    ref_h = torch.cat([ctx, act.unsqueeze(1)], dim=1)
                    ref_logits = forward_from_layer_allpos(model, ref_h, args.layer)
                    ref_lp = F.log_softmax(ref_logits, dim=-1)

                    # +MELBO
                    pert_h = build_perturbed_hidden(ctx, act, d_dev, angle_15)
                    pert_logits = forward_from_layer_allpos(model, pert_h, args.layer)
                    pert_lp = F.log_softmax(pert_logits, dim=-1)
                    kl_p = (ref_lp.exp() * (ref_lp - pert_lp)).sum(dim=-1).item()
                    kl_pos_list.append(kl_p)

                    # -MELBO
                    pert_h = build_perturbed_hidden(ctx, act, -d_dev, angle_15)
                    pert_logits = forward_from_layer_allpos(model, pert_h, args.layer)
                    pert_lp = F.log_softmax(pert_logits, dim=-1)
                    kl_n = (ref_lp.exp() * (ref_lp - pert_lp)).sum(dim=-1).item()
                    kl_neg_list.append(kl_n)

            kl_pos_arr = np.array(kl_pos_list)
            kl_neg_arr = np.array(kl_neg_list)
            print(f"  MELBO_1 +dir: mean={kl_pos_arr.mean():.4f}, "
                  f"median={np.median(kl_pos_arr):.4f}, "
                  f"max={kl_pos_arr.max():.4f}")
            print(f"  MELBO_1 -dir: mean={kl_neg_arr.mean():.4f}, "
                  f"median={np.median(kl_neg_arr):.4f}, "
                  f"max={kl_neg_arr.max():.4f}")

    # Also do dogs
    print(f"\ndogs (20 prompts):")
    cat_acts = acts_by_cat['dogs'].to(device)
    cat_ctxs = contexts_by_cat['dogs']
    for d_idx, direction in enumerate(melbo_dirs[:1]):
        d_dev = direction.to(device)
        kl_pos_list = []
        kl_neg_list = []
        for i in range(cat_acts.shape[0]):
            ctx = cat_ctxs[i].to(device)
            act = cat_acts[i:i+1]
            with torch.no_grad():
                ref_h = torch.cat([ctx, act.unsqueeze(1)], dim=1)
                ref_logits = forward_from_layer_allpos(model, ref_h, args.layer)
                ref_lp = F.log_softmax(ref_logits, dim=-1)
                pert_h = build_perturbed_hidden(ctx, act, d_dev, angle_15)
                pert_logits = forward_from_layer_allpos(model, pert_h, args.layer)
                pert_lp = F.log_softmax(pert_logits, dim=-1)
                kl_p = (ref_lp.exp() * (ref_lp - pert_lp)).sum(dim=-1).item()
                kl_pos_list.append(kl_p)
                pert_h = build_perturbed_hidden(ctx, act, -d_dev, angle_15)
                pert_logits = forward_from_layer_allpos(model, pert_h, args.layer)
                pert_lp = F.log_softmax(pert_logits, dim=-1)
                kl_n = (ref_lp.exp() * (ref_lp - pert_lp)).sum(dim=-1).item()
                kl_neg_list.append(kl_n)
        kl_pos_arr = np.array(kl_pos_list)
        kl_neg_arr = np.array(kl_neg_list)
        print(f"  MELBO_1 +dir: mean={kl_pos_arr.mean():.4f}, "
              f"median={np.median(kl_pos_arr):.4f}, "
              f"max={kl_pos_arr.max():.4f}")
        print(f"  MELBO_1 -dir: mean={kl_neg_arr.mean():.4f}, "
              f"median={np.median(kl_neg_arr):.4f}, "
              f"max={kl_neg_arr.max():.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
