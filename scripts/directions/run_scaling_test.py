#!/usr/bin/env python3
"""
Multi-contrastive scaling verification experiment.

Tests whether the estimation error of multi-contrastive feature estimators
scales as 1/sqrt(k) by computing English estimators at k=2..10 using
all C(9,k-1) subsets, then measuring self-consistency against k=10 reference.

Theoretical prediction for orthogonal unit-norm features:
    cos angle(f_hat^(k), f_hat^(10)) = sqrt(10*(k-1) / (9*k))

Usage:
    python scripts/run_scaling_test.py [--model qwen3-0.6b] [--layer 14]
"""

import sys
import argparse
import os
import pickle
import numpy as np
from itertools import combinations

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import load_model
from data.multilingual_prompts import PROMPT_SETS, LANGUAGES

# Reuse get_acts from run_multi_contrastive
from scripts.run_multi_contrastive import get_acts


def theoretical_cosine(k, k_ref):
    """Predicted cosine between estimators at k and k_ref for orthogonal features."""
    return np.sqrt(k_ref * (k - 1) / (k * (k_ref - 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--max_token_spread', type=int, default=3,
                        help='Max token length difference across languages to keep a prompt set')
    args = parser.parse_args()

    MODEL = args.model
    LAYER = args.layer
    MAX_SPREAD = args.max_token_spread

    results_dir = f'results/scaling_test/{MODEL}'
    os.makedirs(results_dir, exist_ok=True)

    print(f"Model: {MODEL}, Layer: {LAYER}")
    print(f"Languages: {LANGUAGES}")
    print(f"Prompt sets: {len(PROMPT_SETS)}")

    # ================================================================
    # Phase A: Token length filtering
    # ================================================================
    print(f"\n{'='*60}")
    print("Phase A: Token length filtering")
    print(f"{'='*60}")

    print("Loading tokenizer...")
    model, tokenizer, device = load_model(MODEL)

    kept_indices = []
    dropped = []
    for i, pset in enumerate(PROMPT_SETS):
        token_counts = {}
        for lang in LANGUAGES:
            tokens = tokenizer(pset[lang], return_tensors='pt')
            token_counts[lang] = tokens['input_ids'].shape[1]
        min_t = min(token_counts.values())
        max_t = max(token_counts.values())
        spread = max_t - min_t
        if spread <= MAX_SPREAD:
            kept_indices.append(i)
        else:
            worst_langs = [f"{l}={c}" for l, c in sorted(token_counts.items(), key=lambda x: x[1])]
            dropped.append((i, spread, worst_langs))

    print(f"\nKept: {len(kept_indices)} / {len(PROMPT_SETS)} prompt sets")
    if dropped:
        print(f"Dropped {len(dropped)} sets (token spread > {MAX_SPREAD}):")
        for idx, spread, langs in dropped:
            en_text = PROMPT_SETS[idx]['en']
            print(f"  [{idx}] \"{en_text}\" — spread={spread}, counts: {', '.join(langs)}")

    assert len(kept_indices) >= 30, \
        f"Only {len(kept_indices)} prompt sets survived filtering (need >= 30). " \
        f"Consider increasing --max_token_spread or adding more prompts."

    # Use first 30 that survived
    N_PROMPTS = 30
    use_indices = kept_indices[:N_PROMPTS]
    prompt_sets = [PROMPT_SETS[i] for i in use_indices]
    print(f"\nUsing {N_PROMPTS} prompt sets for experiment.")

    # ================================================================
    # Phase B: Language verification
    # ================================================================
    print(f"\n{'='*60}")
    print("Phase B: Language verification (sample generations)")
    print(f"{'='*60}")

    for lang in LANGUAGES:
        print(f"\n--- {lang.upper()} ---")
        for j in range(3):
            prompt = prompt_sets[j][lang]
            tokens = tokenizer(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model.generate(
                    **tokens, max_new_tokens=20, temperature=0.7,
                    do_sample=True, top_p=0.9)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"  [{j}] {text}")

    # ================================================================
    # Phase C: Activation extraction
    # ================================================================
    print(f"\n{'='*60}")
    print("Phase C: Activation extraction")
    print(f"{'='*60}")

    acts = {}
    for lang in LANGUAGES:
        prompts = [pset[lang] for pset in prompt_sets]
        acts[lang] = get_acts(model, tokenizer, device, prompts, LAYER)
        print(f"  {lang}: {acts[lang].shape}")

    # ================================================================
    # Phase D: Pairwise diff-of-means directions
    # ================================================================
    print(f"\n{'='*60}")
    print("Phase D: Pairwise diff-of-means directions (EN vs each)")
    print(f"{'='*60}")

    non_en = [l for l in LANGUAGES if l != 'en']
    pairwise_dirs = {}
    for lang in non_en:
        diff = acts['en'].mean(0) - acts[lang].mean(0)
        pairwise_dirs[lang] = F.normalize(diff.unsqueeze(0), dim=-1).squeeze(0)
        cos_with_mean = F.cosine_similarity(
            acts['en'].mean(0).unsqueeze(0), acts[lang].mean(0).unsqueeze(0)).item()
        print(f"  d_E_{lang}: ||diff|| = {diff.norm():.2f}, "
              f"cos(mean_en, mean_{lang}) = {cos_with_mean:.4f}")

    # Print pairwise cosines between d_E_* directions
    print("\nPairwise cosines between d_E_* directions:")
    for i, l1 in enumerate(non_en):
        for l2 in non_en[i+1:]:
            cos = F.cosine_similarity(
                pairwise_dirs[l1].unsqueeze(0),
                pairwise_dirs[l2].unsqueeze(0)).item()
            print(f"  cos(d_E_{l1}, d_E_{l2}) = {cos:.4f}")

    # ================================================================
    # Phase E: Multi-contrastive estimation at each k
    # ================================================================
    print(f"\n{'='*60}")
    print("Phase E: Multi-contrastive estimation at k=2..10")
    print(f"{'='*60}")

    # For each k, enumerate all C(9, k-1) subsets of non-English languages
    # Estimator: f_hat_E = normalize(sum(d_E_j for j in subset) / (k-1))
    k_ref = len(LANGUAGES)  # k=10

    all_estimators = {}  # k -> list of (subset, direction) tuples
    for k in range(2, k_ref + 1):
        n_contrast = k - 1  # number of contrast languages
        subsets = list(combinations(non_en, n_contrast))
        estimators = []
        for subset in subsets:
            raw = sum(pairwise_dirs[lang] for lang in subset) / n_contrast
            direction = F.normalize(raw.unsqueeze(0), dim=-1).squeeze(0)
            estimators.append((subset, direction))
        all_estimators[k] = estimators
        print(f"  k={k}: {len(subsets)} subsets (C(9,{n_contrast}))")

    # The reference: k=10 (single estimator using all 9 contrasts)
    assert len(all_estimators[k_ref]) == 1
    ref_direction = all_estimators[k_ref][0][1]

    # ================================================================
    # Phase F: Self-consistency analysis
    # ================================================================
    print(f"\n{'='*60}")
    print("Phase F: Self-consistency analysis")
    print(f"{'='*60}")

    consistency_results = {}  # k -> dict with cosines, predicted, etc.

    print(f"\n{'k':>3} | {'n_sub':>5} | {'cos_pred':>8} | {'cos_mean':>8} | "
          f"{'cos_std':>7} | {'ang_pred':>8} | {'ang_mean':>8} | {'ang_std':>7}")
    print("-" * 75)

    for k in range(2, k_ref):
        cosines = []
        for subset, direction in all_estimators[k]:
            cos = F.cosine_similarity(
                direction.unsqueeze(0), ref_direction.unsqueeze(0)).item()
            cosines.append(cos)

        cosines = np.array(cosines)
        cos_pred = theoretical_cosine(k, k_ref)
        ang_pred = np.degrees(np.arccos(cos_pred))
        ang_mean = np.degrees(np.arccos(np.clip(cosines, -1, 1))).mean()
        ang_std = np.degrees(np.arccos(np.clip(cosines, -1, 1))).std()

        consistency_results[k] = {
            'cosines': cosines,
            'angles_deg': np.degrees(np.arccos(np.clip(cosines, -1, 1))),
            'cos_predicted': cos_pred,
            'angle_predicted': ang_pred,
        }

        print(f"{k:>3} | {len(cosines):>5} | {cos_pred:>8.4f} | {cosines.mean():>8.4f} | "
              f"{cosines.std():>7.4f} | {ang_pred:>7.1f}° | {ang_mean:>7.1f}° | {ang_std:>6.1f}°")

    # k=10 vs itself (should be 0 degrees)
    print(f"{k_ref:>3} | {'1':>5} | {'1.0000':>8} | {'1.0000':>8} | "
          f"{'0.0000':>7} | {'0.0':>7}° | {'0.0':>7}° | {'0.0':>6}°")

    # ================================================================
    # Phase G: SVD-based estimator comparison
    # ================================================================
    print(f"\n{'='*60}")
    print("Phase G: SVD-based estimator comparison")
    print(f"{'='*60}")

    svd_results = {}

    print(f"\n{'k':>3} | {'n_sub':>5} | {'cos_mean(svd→ref)':>18} | "
          f"{'ang_mean':>8} | {'cos(svd,mean)':>13}")
    print("-" * 65)

    for k in range(2, k_ref + 1):
        n_contrast = k - 1
        subsets = list(combinations(non_en, n_contrast))
        svd_cosines_to_ref = []
        svd_cosines_to_mean = []

        for subset in subsets:
            # Stack difference matrices for this subset
            diff_matrices = []
            for lang in subset:
                diffs = acts['en'] - acts[lang]  # [30, d_model]
                diff_matrices.append(diffs)
            D = torch.cat(diff_matrices, dim=0)  # [30*n_contrast, d_model]

            # SVD -> leading right singular vector
            _, S, Vh = torch.linalg.svd(D, full_matrices=False)
            u1 = Vh[0]  # leading right singular vector

            # Ensure consistent sign (positive cosine with mean of pairwise dirs)
            mean_dir = sum(pairwise_dirs[lang] for lang in subset)
            if F.cosine_similarity(u1.unsqueeze(0), mean_dir.unsqueeze(0)).item() < 0:
                u1 = -u1

            cos_to_ref = F.cosine_similarity(
                u1.unsqueeze(0), ref_direction.unsqueeze(0)).item()
            svd_cosines_to_ref.append(cos_to_ref)

            # Also compare SVD estimator to mean-based estimator for same subset
            mean_est = F.normalize(mean_dir.unsqueeze(0), dim=-1).squeeze(0)
            cos_svd_mean = F.cosine_similarity(
                u1.unsqueeze(0), mean_est.unsqueeze(0)).item()
            svd_cosines_to_mean.append(cos_svd_mean)

        svd_results[k] = {
            'cosines_to_ref': np.array(svd_cosines_to_ref),
            'cosines_to_mean': np.array(svd_cosines_to_mean),
        }

        ang_mean = np.degrees(np.arccos(np.clip(svd_cosines_to_ref, -1, 1))).mean()
        cos_sm = np.mean(svd_cosines_to_mean)

        print(f"{k:>3} | {len(subsets):>5} | {np.mean(svd_cosines_to_ref):>18.4f} | "
              f"{ang_mean:>7.1f}° | {cos_sm:>13.4f}")

    # ================================================================
    # Phase H: Save results
    # ================================================================
    print(f"\n{'='*60}")
    print("Phase H: Saving results")
    print(f"{'='*60}")

    results = {
        'model': MODEL,
        'layer': LAYER,
        'languages': LANGUAGES,
        'non_en': non_en,
        'n_prompts': N_PROMPTS,
        'kept_indices': kept_indices,
        'use_indices': use_indices,
        'pairwise_dirs': {lang: d.cpu() for lang, d in pairwise_dirs.items()},
        'ref_direction': ref_direction.cpu(),
        'all_estimators': {
            k: [(subset, d.cpu()) for subset, d in ests]
            for k, ests in all_estimators.items()
        },
        'consistency_results': consistency_results,
        'svd_results': svd_results,
        'theoretical_cosine_fn': 'sqrt(k_ref*(k-1)/(k*(k_ref-1)))',
        'k_ref': k_ref,
    }

    save_path = os.path.join(results_dir, 'scaling_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {save_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
