#!/usr/bin/env python3
"""
Validation tests for multi-contrastive scaling experiment.

Test 1: Cosine similarities between new k=10 estimators and old directions
Test 2: Perturbation sweeps comparing new vs old directions
Test 3: Steering from Mandarin prompts with new vs old directions

Usage:
    python scripts/run_scaling_validation.py
"""

import sys
import os
import pickle
import numpy as np

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import load_model
from src.data import load_fineweb_fixed_length
from data.multilingual_prompts import PROMPT_SETS, LANGUAGES

from scripts.run_multi_contrastive import (
    get_acts, extract_context_and_activation,
    forward_to_hidden_and_logits, build_perturbed_hidden,
    sweep_single_direction,
)

MODEL = 'qwen3-0.6b'
LAYER = 14
N_ACT = 50
SEQ_LEN = 5
SEED = 42
N_ANGLES = 60

results_dir = 'results/scaling_test/qwen3-0.6b'


def main():
    print("Loading model...")
    model, tokenizer, device = load_model(MODEL)

    # Load scaling results
    with open(os.path.join(results_dir, 'scaling_results.pkl'), 'rb') as f:
        scaling = pickle.load(f)

    # Load old multi-contrastive results
    mc_path = 'results/multi_contrastive/qwen3-0.6b/optimization_results.pkl'
    with open(mc_path, 'rb') as f:
        old_mc = pickle.load(f)

    # Load old MELBO constrained results
    melbo_path = 'results/melbo_constrained/qwen3-0.6b/optimization_results.pkl'
    with open(melbo_path, 'rb') as f:
        old_melbo = pickle.load(f)

    # === Reconstruct SVD k=10 direction ===
    print("\nReconstructing SVD-based k=10 direction...")
    non_en = scaling['non_en']
    use_indices = scaling['use_indices']
    prompt_sets = [PROMPT_SETS[i] for i in use_indices]

    # Extract activations (same as scaling test)
    acts = {}
    for lang in LANGUAGES:
        prompts = [pset[lang] for pset in prompt_sets]
        acts[lang] = get_acts(model, tokenizer, device, prompts, LAYER)

    # Stack all EN-X difference matrices
    diff_matrices = []
    for lang in non_en:
        diffs = acts['en'] - acts[lang]  # [30, d_model]
        diff_matrices.append(diffs)
    D_all = torch.cat(diff_matrices, dim=0)  # [270, d_model]
    _, S, Vh = torch.linalg.svd(D_all, full_matrices=False)
    svd_k10 = Vh[0].to('cpu')

    # Ensure consistent sign
    ref_mean = scaling['ref_direction']
    if F.cosine_similarity(svd_k10.unsqueeze(0), ref_mean.unsqueeze(0)).item() < 0:
        svd_k10 = -svd_k10

    print(f"  cos(svd_k10, mean_k10) = {F.cosine_similarity(svd_k10.unsqueeze(0), ref_mean.unsqueeze(0)).item():.4f}")
    print(f"  Top 5 singular values: {S[:5].tolist()}")

    # ================================================================
    # Test 1: Cosine similarities
    # ================================================================
    print(f"\n{'='*60}")
    print("Test 1: Cosine similarities with old directions")
    print(f"{'='*60}")

    old_dirs = {
        'd_EM': old_mc['directions']['d_EM'],
        'f_E (k=3)': old_mc['directions']['f_E'],
        'MELBO_EM': old_mc['directions']['MELBO_EM'],
        'MELBO_joint_1': old_mc['directions']['MELBO_joint_1'],
    }

    new_dirs = {
        'k=10 mean': ref_mean,
        'k=10 SVD': svd_k10,
        'd_E_zh (new)': scaling['pairwise_dirs']['zh'],
    }

    print(f"\n{'':>15}", end='')
    for old_name in old_dirs:
        print(f" | {old_name:>14}", end='')
    print()
    print("-" * (16 + 17 * len(old_dirs)))

    for new_name, new_d in new_dirs.items():
        print(f"{new_name:>15}", end='')
        for old_name, old_d in old_dirs.items():
            cos = F.cosine_similarity(new_d.unsqueeze(0), old_d.unsqueeze(0)).item()
            print(f" | {cos:>14.4f}", end='')
        print()

    # ================================================================
    # Test 2: Perturbation sweeps
    # ================================================================
    print(f"\n{'='*60}")
    print("Test 2: Perturbation sweeps")
    print(f"{'='*60}")

    # Load FineWeb activations (same seed as original experiments)
    print(f"\nLoading {N_ACT} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(N_ACT, tokenizer, seq_len=SEQ_LEN, seed=SEED)
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, LAYER, device)
    contexts = contexts.to(device)
    activations = activations.to(device)

    # Reference logits
    with torch.no_grad():
        ref_hidden = torch.cat([contexts, activations.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_hidden_and_logits(model, ref_hidden, LAYER)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    angles_deg = np.linspace(3, 180, N_ANGLES)
    angles_rad = np.radians(angles_deg)

    # Directions to sweep
    sweep_dirs = {
        'k=10 mean': ref_mean,
        'k=10 SVD': svd_k10,
        'd_EM (old)': old_mc['directions']['d_EM'],
        'f_E (old k=3)': old_mc['directions']['f_E'],
        'MELBO_EM': old_mc['directions']['MELBO_EM'],
    }

    sweep_results = {}
    for name, direction in sweep_dirs.items():
        print(f"  Sweeping {name}...")
        kl, l2 = sweep_single_direction(
            model, contexts, activations, ref_final, ref_log_probs,
            direction, angles_rad, LAYER, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}

    # Also sweep negatives for steering comparison
    for name_base in ['k=10 mean', 'd_EM (old)', 'MELBO_EM']:
        neg_name = f'-{name_base}'
        print(f"  Sweeping {neg_name}...")
        kl, l2 = sweep_single_direction(
            model, contexts, activations, ref_final, ref_log_probs,
            -sweep_dirs[name_base], angles_rad, LAYER, device)
        sweep_results[neg_name] = {'kl': kl, 'l2': l2}

    # ================================================================
    # Test 3: Steering from Mandarin prompts
    # ================================================================
    print(f"\n{'='*60}")
    print("Test 3: Steering from Mandarin prompts")
    print(f"{'='*60}")

    from data.bilingual_prompts import CHINESE_PROMPTS

    steer_prompts = CHINESE_PROMPTS[:20]
    steer_angles = [0, 5, 10, 15, 20, 25, 30, 45]

    # Directions to test (negative = toward English, away from Mandarin...
    # or rather: the English feature direction should steer toward English)
    steer_dirs = {
        'k=10 mean': ref_mean,
        'k=10 SVD': svd_k10,
        'd_EM (old)': old_mc['directions']['d_EM'],
        'f_E (old k=3)': old_mc['directions']['f_E'],
        '-MELBO_EM': -old_mc['directions']['MELBO_EM'],
    }

    steer_results = []

    for dir_name, direction in steer_dirs.items():
        direction = direction.to(device)
        for angle in steer_angles:
            angle_rad = np.radians(angle)
            for prompt in steer_prompts:
                tokens = tokenizer(prompt, return_tensors='pt').to(device)

                if angle == 0:
                    with torch.no_grad():
                        out = model.generate(
                            **tokens, max_new_tokens=50, temperature=0.7,
                            do_sample=True, top_p=0.9)
                else:
                    # Get activations at layer
                    with torch.no_grad():
                        outputs = model(**tokens, output_hidden_states=True)
                    hidden = outputs.hidden_states[LAYER + 1]  # [1, seq_len, d_model]
                    act = hidden[:, -1, :]  # [1, d_model]

                    # Perturb
                    act_norm = act.norm(dim=-1, keepdim=True)
                    act_unit = act / act_norm
                    # Remove component along act_unit from direction
                    d_proj = direction - (direction @ act_unit.squeeze()) * act_unit.squeeze()
                    d_proj = d_proj / d_proj.norm()
                    perturbed = act_norm * (
                        np.cos(angle_rad) * act_unit.squeeze() +
                        np.sin(angle_rad) * d_proj)

                    # Build steering hook
                    steer_vec = perturbed - act.squeeze()

                    def make_hook(sv):
                        def hook_fn(module, input, output):
                            # output is tuple for some models
                            if isinstance(output, tuple):
                                h = output[0]
                                h[:, -1, :] += sv
                                return (h,) + output[1:]
                            else:
                                output[:, -1, :] += sv
                                return output
                        return hook_fn

                    from src.model import _get_blocks
                    blocks = _get_blocks(model)
                    hook = blocks[LAYER].register_forward_hook(make_hook(steer_vec))

                    with torch.no_grad():
                        out = model.generate(
                            **tokens, max_new_tokens=50, temperature=0.7,
                            do_sample=True, top_p=0.9)
                    hook.remove()

                text = tokenizer.decode(out[0][tokens['input_ids'].shape[1]:],
                                       skip_special_tokens=True)

                # Compute CJK fraction
                cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or
                               '\u3400' <= c <= '\u4dbf' or '\uf900' <= c <= '\ufaff')
                total_chars = max(len(text.strip()), 1)
                cjk_frac = cjk_chars / total_chars

                # Simple English fraction (ASCII letters / total)
                ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
                eng_frac = ascii_letters / total_chars

                steer_results.append({
                    'direction': dir_name,
                    'angle': angle,
                    'prompt': prompt,
                    'text': text,
                    'cjk_fraction': cjk_frac,
                    'english_fraction': eng_frac,
                })

        print(f"  Done: {dir_name}")

    # Print summary
    print(f"\n{'='*60}")
    print("Steering summary: CJK fraction / English fraction by angle")
    print(f"{'='*60}")

    print(f"\n{'Direction':>20} | ", end='')
    for a in steer_angles:
        print(f"{'%d°' % a:>8}", end='')
    print()
    print("-" * (22 + 9 * len(steer_angles)))

    for dir_name in steer_dirs:
        # CJK
        print(f"{'CJK ' + dir_name:>20} | ", end='')
        for a in steer_angles:
            vals = [r['cjk_fraction'] for r in steer_results
                    if r['direction'] == dir_name and r['angle'] == a]
            print(f"{np.mean(vals):>7.1%}", end=' ')
        print()
        # English
        print(f"{'ENG ' + dir_name:>20} | ", end='')
        for a in steer_angles:
            vals = [r['english_fraction'] for r in steer_results
                    if r['direction'] == dir_name and r['angle'] == a]
            print(f"{np.mean(vals):>7.1%}", end=' ')
        print()

    # ================================================================
    # Save all results
    # ================================================================
    print(f"\n{'='*60}")
    print("Saving results")
    print(f"{'='*60}")

    validation = {
        'svd_k10': svd_k10,
        'sweep_results': sweep_results,
        'sweep_angles_deg': angles_deg,
        'steer_results': steer_results,
        'steer_angles': steer_angles,
    }

    save_path = os.path.join(results_dir, 'validation_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(validation, f)
    print(f"Saved: {save_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
