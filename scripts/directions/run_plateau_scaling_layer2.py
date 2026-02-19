#!/usr/bin/env python3
"""
Same as run_plateau_scaling.py but inject perturbation at layer 2 instead of 14.
More layers of processing → potentially sharper plateau / stronger error correction.
Feature directions are still the ones estimated at layer 14.
"""

import sys, os, pickle, itertools, random, numpy as np
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import (
    load_model, _get_blocks, _get_final_norm, _get_lm_head,
    _is_rope_model, _get_position_embeddings, _run_block,
)
from src.data import load_fineweb_fixed_length

MODEL = 'qwen3-0.6b'
LAYER = 2           # inject perturbation here
MEASURE_LAYER = 26  # measure L2 here
N_ACT = 50
SEQ_LEN = 5
SEED = 42
KL_THRESHOLD = 0.5
MAX_COMBOS_PER_N = 45

results_dir = 'results/allpairs_contrastive/qwen3-0.6b'


# --- Helper functions ---

def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device,
                                   batch_size=64):
    all_contexts, all_activations = [], []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].cpu())
        all_activations.append(hidden[:, -1, :].cpu())
    return torch.cat(all_contexts, 0), torch.cat(all_activations, 0)


def forward_with_measure(model, hidden_states, start_layer, measure_layer):
    hidden = hidden_states
    blocks = _get_blocks(model)
    rope = _is_rope_model(model)
    pos_emb = _get_position_embeddings(model, hidden) if rope else None

    measure_hidden = None
    for i, block in enumerate(blocks[start_layer + 1:], start=start_layer + 1):
        hidden = _run_block(block, hidden, model, pos_emb)
        if i == measure_layer:
            measure_hidden = hidden[:, -1, :].clone()

    final_hidden = hidden[:, -1, :]
    normed = _get_final_norm(model)(final_hidden)
    logits = _get_lm_head(model)(normed)
    return measure_hidden, logits


def build_perturbed_hidden(contexts, activations, direction, angle_rad):
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms
    proj = (direction.unsqueeze(0) * a_unit).sum(dim=-1, keepdim=True)
    u_perp = direction.unsqueeze(0) - proj * a_unit
    u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)
    a_pert = a_norms * (cos_t * a_unit + sin_t * u_perp)
    return torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)


def sweep_single_direction(model, contexts, activations, ref_measure, ref_log_probs,
                           direction, angles_rad, layer, measure_layer, device):
    N = activations.shape[0]
    n_angles = len(angles_rad)
    kl_divs = np.zeros((N, n_angles))
    l2_dists = np.zeros((N, n_angles))
    direction_dev = direction.to(device)
    for ang_idx, theta in enumerate(angles_rad):
        with torch.no_grad():
            hidden = build_perturbed_hidden(contexts, activations, direction_dev, theta)
            pert_measure, pert_logits = forward_with_measure(model, hidden, layer, measure_layer)
            pert_log_probs = F.log_softmax(pert_logits, dim=-1)
            l2 = (pert_measure - ref_measure).norm(dim=-1)
            kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            l2_dists[:, ang_idx] = l2.cpu().numpy()
            kl_divs[:, ang_idx] = kl.cpu().numpy()
    return kl_divs, l2_dists


def sweep_random_directions(model, contexts, activations, ref_measure, ref_log_probs,
                            angles_rad, layer, measure_layer, device, n_dirs=10):
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
                pert_measure, pert_logits = forward_with_measure(
                    model, hidden, layer, measure_layer)
                pert_log_probs = F.log_softmax(pert_logits, dim=-1)
                l2 = (pert_measure - ref_measure).norm(dim=-1)
                kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
                l2_dists[:, ang_idx, dir_idx] = l2.cpu().numpy()
                kl_divs[:, ang_idx, dir_idx] = kl.cpu().numpy()
    return kl_divs, l2_dists


# ============================================================
# MAIN
# ============================================================

print(f"Injection layer: {LAYER}, Measure layer: {MEASURE_LAYER}")

# Load feature directions (estimated at layer 14, used as-is)
print("Loading feature directions (from layer 14 estimation)...")
with open(os.path.join(results_dir, 'directions.pkl'), 'rb') as f:
    dir_data = pickle.load(f)
features = dir_data['features']
languages = dir_data['languages']
k = len(languages)

print("\nLoading model...")
model, tokenizer, device = load_model(MODEL)

print(f"Loading FineWeb activations at layer {LAYER}...")
token_ids_list = load_fineweb_fixed_length(N_ACT, tokenizer, seq_len=SEQ_LEN, seed=SEED)
contexts, activations = extract_context_and_activation(
    model, tokenizer, token_ids_list, LAYER, device)
contexts = contexts.to(device)
activations = activations.to(device)

# Reference (unperturbed)
with torch.no_grad():
    ref_hidden = torch.cat([contexts, activations.unsqueeze(1)], dim=1)
    ref_measure, ref_logits = forward_with_measure(model, ref_hidden, LAYER, MEASURE_LAYER)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

# Random baseline (need to recompute for layer 2)
print("\nSweeping random directions...")
angles_deg = np.linspace(3, 180, 60)
angles_rad = np.radians(angles_deg)
rand_kl, rand_l2 = sweep_random_directions(
    model, contexts, activations, ref_measure, ref_log_probs,
    angles_rad, LAYER, MEASURE_LAYER, device, n_dirs=10)
print(f"  random L2 at 30°: {rand_l2[:, np.argmin(np.abs(angles_deg-30)), :].mean():.1f}")

# === Composite direction sweeps ===
print("\n=== Sweeping composite directions ===")
feature_vecs = [features[lang] for lang in languages]
rng = random.Random(0)

all_results = {}
for n in range(1, k + 1):
    all_combos = list(itertools.combinations(range(k), n))
    if len(all_combos) <= MAX_COMBOS_PER_N:
        combos = all_combos
    else:
        combos = rng.sample(all_combos, MAX_COMBOS_PER_N)

    print(f"\nn={n}: {len(combos)} combinations")
    results_n = []
    for ci, combo in enumerate(combos):
        d = sum(feature_vecs[i] for i in combo)
        d = F.normalize(d.unsqueeze(0), dim=-1).squeeze(0)

        kl, l2 = sweep_single_direction(
            model, contexts, activations, ref_measure, ref_log_probs,
            d, angles_rad, LAYER, MEASURE_LAYER, device)

        results_n.append({'combo': combo, 'kl': kl, 'l2': l2})

        if ci % 10 == 0 or ci == len(combos) - 1:
            l2_30 = l2[:, np.argmin(np.abs(angles_deg - 30))].mean()
            print(f"  [{ci+1}/{len(combos)}] L2@30°={l2_30:.1f}")

    all_results[n] = results_n
    mean_l2_30 = np.mean([r['l2'][:, np.argmin(np.abs(angles_deg-30))].mean()
                          for r in results_n])
    print(f"  n={n} mean L2@30° = {mean_l2_30:.1f}")

# === Save ===
save_data = {
    'all_results': all_results,
    'random_kl': rand_kl,
    'random_l2': rand_l2,
    'angles_deg': angles_deg,
    'layer': LAYER,
    'measure_layer': MEASURE_LAYER,
    'languages': languages,
}
with open(os.path.join(results_dir, 'plateau_scaling_layer2.pkl'), 'wb') as f:
    pickle.dump(save_data, f)
print(f"\nSaved: {results_dir}/plateau_scaling_layer2.pkl")

# === Summary ===
print("\n=== Summary: mean L2 at 30° by n ===")
idx_30 = np.argmin(np.abs(angles_deg - 30))
for n in range(1, k + 1):
    vals = [r['l2'][:, idx_30].mean() for r in all_results[n]]
    print(f"  n={n}: L2={np.mean(vals):.1f} ± {np.std(vals):.1f}")
rand_30 = rand_l2[:, idx_30, :].mean()
print(f"  random: L2={rand_30:.1f}")

print("\nDone!")
