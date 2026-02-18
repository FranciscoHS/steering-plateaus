#!/usr/bin/env python3
"""
Run missing negative-direction sweeps and compute k=3 SVD direction,
then save a single plot of all plateau-breaking directions.
"""

import sys, os, pickle, numpy as np
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

results_dir = 'results/scaling_test/qwen3-0.6b'

print("Loading model...")
model, tokenizer, device = load_model(MODEL)

# Load existing results
with open(os.path.join(results_dir, 'scaling_results.pkl'), 'rb') as f:
    scaling = pickle.load(f)
with open(os.path.join(results_dir, 'validation_results.pkl'), 'rb') as f:
    val = pickle.load(f)
with open('results/multi_contrastive/qwen3-0.6b/optimization_results.pkl', 'rb') as f:
    old_mc = pickle.load(f)

ref_mean = scaling['ref_direction']
svd_k10 = val['svd_k10']

# === Compute k=3 SVD direction (EN vs ZH+NL, matching old experiment) ===
print("\nComputing k=3 SVD direction...")
use_indices = scaling['use_indices']
prompt_sets = [PROMPT_SETS[i] for i in use_indices]

acts = {}
for lang in ['en', 'zh', 'nl']:
    prompts = [pset[lang] for pset in prompt_sets]
    acts[lang] = get_acts(model, tokenizer, device, prompts, LAYER)

D_k3 = torch.cat([acts['en'] - acts['zh'], acts['en'] - acts['nl']], dim=0)  # [60, d_model]
_, S, Vh = torch.linalg.svd(D_k3, full_matrices=False)
svd_k3 = Vh[0].cpu()
# Consistent sign: positive cosine with mean-based f_E(k=3) direction
d_ez = F.normalize((acts['en'].mean(0) - acts['zh'].mean(0)).unsqueeze(0), dim=-1).squeeze(0)
d_en = F.normalize((acts['en'].mean(0) - acts['nl'].mean(0)).unsqueeze(0), dim=-1).squeeze(0)
f_e_k3_new = F.normalize((d_ez + d_en).unsqueeze(0), dim=-1).squeeze(0)
if F.cosine_similarity(svd_k3.unsqueeze(0), f_e_k3_new.unsqueeze(0)).item() < 0:
    svd_k3 = -svd_k3
print(f"  cos(svd_k3, mean_k3_new) = {F.cosine_similarity(svd_k3.unsqueeze(0), f_e_k3_new.unsqueeze(0)).item():.4f}")
print(f"  cos(svd_k3, svd_k10) = {F.cosine_similarity(svd_k3.unsqueeze(0), svd_k10.unsqueeze(0)).item():.4f}")

# === Load FineWeb for sweeps ===
print("\nLoading FineWeb activations...")
token_ids_list = load_fineweb_fixed_length(50, tokenizer, seq_len=5, seed=42)
contexts, activations = extract_context_and_activation(
    model, tokenizer, token_ids_list, LAYER, device)
contexts = contexts.to(device)
activations = activations.to(device)

with torch.no_grad():
    ref_hidden = torch.cat([contexts, activations.unsqueeze(1)], dim=1)
    ref_final, ref_logits = forward_to_hidden_and_logits(model, ref_hidden, LAYER)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

angles_deg = np.linspace(3, 180, 60)
angles_rad = np.radians(angles_deg)

# === Run ALL negative sweeps (clean set) ===
print("\nRunning negative-direction sweeps...")

neg_dirs = {
    '-k=10 mean': -ref_mean,
    '-k=10 SVD': -svd_k10,
    '-d_EM (old)': -old_mc['directions']['d_EM'],
    '-f_E mean (old k=3)': -old_mc['directions']['f_E'],
    '-f_E SVD (k=3)': -svd_k3,
    '-MELBO_EM': -old_mc['directions']['MELBO_EM'],
}

neg_sweep_results = {}
for name, direction in neg_dirs.items():
    print(f"  Sweeping {name}...")
    kl, l2 = sweep_single_direction(
        model, contexts, activations, ref_final, ref_log_probs,
        direction, angles_rad, LAYER, device)
    neg_sweep_results[name] = {'kl': kl, 'l2': l2}
    peak_kl = kl.mean(axis=0).max()
    peak_angle = angles_deg[kl.mean(axis=0).argmax()]
    print(f"    peak KL = {peak_kl:.3f} at {peak_angle:.0f}°")

# === Save ===
save_data = {
    'neg_sweep_results': neg_sweep_results,
    'angles_deg': angles_deg,
    'svd_k3': svd_k3,
}
with open(os.path.join(results_dir, 'plateau_sweeps.pkl'), 'wb') as f:
    pickle.dump(save_data, f)
print(f"\nSaved: {results_dir}/plateau_sweeps.pkl")
print("Done!")
