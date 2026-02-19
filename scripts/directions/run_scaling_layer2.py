#!/usr/bin/env python3
"""
Full pipeline at layer 2: estimate language feature directions + composite
perturbation sweeps (n=1..10). Feature directions estimated at layer 2,
perturbations injected at layer 2, L2 measured at layer 26.
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
from data.multilingual_prompts import PROMPT_SETS, LANGUAGES

MODEL = 'qwen3-0.6b'
LAYER = 2
MEASURE_LAYER = 26
N_ACT = 50
SEQ_LEN = 5
SEED = 42
MAX_COMBOS = 20  # sample per n value

results_dir = 'results/allpairs_contrastive/qwen3-0.6b/layer2'


# --- Helper functions ---

def get_acts(model, tokenizer, device, prompts, layer):
    acts = []
    for p in prompts:
        toks = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**toks, output_hidden_states=True)
        acts.append(out.hidden_states[layer + 1][:, -1, :].squeeze(0))
    return torch.stack(acts).cpu()


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

print(f"=== Full pipeline at layer {LAYER} ===")
print(f"Languages: {LANGUAGES}")

print("\nLoading model...")
model, tokenizer, device = load_model(MODEL)

# === Phase 1: Extract multilingual activations at layer 2 ===
print(f"\n=== Phase 1: Extracting activations at layer {LAYER} ===")
acts = {}
for lang in LANGUAGES:
    prompts = [ps[lang] for ps in PROMPT_SETS]
    acts[lang] = get_acts(model, tokenizer, device, prompts, LAYER)
    print(f"  {lang}: {acts[lang].shape}")

# === Phase 2: Pairwise differences ===
print(f"\n=== Phase 2: Pairwise differences ===")
k = len(LANGUAGES)
pairwise_means = {}
for i, li in enumerate(LANGUAGES):
    for j, lj in enumerate(LANGUAGES):
        if i < j:
            diff = (acts[li] - acts[lj]).mean(dim=0)
            pairwise_means[(li, lj)] = diff

# === Phase 3: Least-squares feature estimation ===
print(f"\n=== Phase 3: Feature estimation ===")
features_raw = {}
features = {}
for lang in LANGUAGES:
    f_hat = torch.zeros_like(acts[lang][0])
    for other in LANGUAGES:
        if other == lang:
            continue
        if (lang, other) in pairwise_means:
            f_hat += pairwise_means[(lang, other)]
        else:
            f_hat -= pairwise_means[(other, lang)]
    f_hat = f_hat / k
    features_raw[lang] = f_hat
    features[lang] = F.normalize(f_hat.unsqueeze(0), dim=-1).squeeze(0)
    print(f"  {lang}: ||f̂|| = {f_hat.norm().item():.4f}")

total = sum(features_raw.values())
print(f"  Σ f̂_i norm (should be ~0): {total.norm().item():.6f}")

# Cosine similarity matrix
print(f"\nCosine similarity matrix:")
cos_matrix = np.zeros((k, k))
header = "        " + "  ".join(f"{l:>5}" for l in LANGUAGES)
print(header)
for i, li in enumerate(LANGUAGES):
    for j, lj in enumerate(LANGUAGES):
        cos_matrix[i, j] = F.cosine_similarity(
            features[li].unsqueeze(0), features[lj].unsqueeze(0)).item()
    row = f"  {li:>4}  " + "  ".join(f"{cos_matrix[i,j]:5.2f}" for j in range(k))
    print(row)

# === Phase 4: FineWeb activations + reference ===
print(f"\n=== Phase 4: FineWeb activations at layer {LAYER} ===")
token_ids_list = load_fineweb_fixed_length(N_ACT, tokenizer, seq_len=SEQ_LEN, seed=SEED)
contexts, activations = extract_context_and_activation(
    model, tokenizer, token_ids_list, LAYER, device)
contexts = contexts.to(device)
activations = activations.to(device)

with torch.no_grad():
    ref_hidden = torch.cat([contexts, activations.unsqueeze(1)], dim=1)
    ref_measure, ref_logits = forward_with_measure(model, ref_hidden, LAYER, MEASURE_LAYER)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

angles_deg = np.linspace(3, 180, 60)
angles_rad = np.radians(angles_deg)

# === Phase 5: Random baseline ===
print(f"\n=== Phase 5: Random baseline ===")
rand_kl, rand_l2 = sweep_random_directions(
    model, contexts, activations, ref_measure, ref_log_probs,
    angles_rad, LAYER, MEASURE_LAYER, device, n_dirs=10)
print(f"  random KL shape: {rand_kl.shape}")

# === Phase 6: Composite direction sweeps ===
print(f"\n=== Phase 6: Composite direction sweeps (n=1..{k}) ===")
feature_vecs = [features[lang] for lang in LANGUAGES]
rng = random.Random(0)

all_results = {}
for n in range(1, k + 1):
    all_combos = list(itertools.combinations(range(k), n))
    if len(all_combos) <= MAX_COMBOS:
        combos = all_combos
    else:
        combos = rng.sample(all_combos, MAX_COMBOS)

    print(f"\nn={n}: {len(combos)} combinations (of {len(all_combos)} total)")
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
            kl_30 = kl[:, np.argmin(np.abs(angles_deg - 30))].mean()
            print(f"  [{ci+1}/{len(combos)}] KL@30°={kl_30:.3f}, L2@30°={l2_30:.1f}")

    all_results[n] = results_n
    idx_30 = np.argmin(np.abs(angles_deg - 30))
    mean_l2 = np.mean([r['l2'][:, idx_30].mean() for r in results_n])
    mean_kl = np.mean([r['kl'][:, idx_30].mean() for r in results_n])
    print(f"  n={n} mean: KL@30°={mean_kl:.3f}, L2@30°={mean_l2:.1f}")

# === Save ===
print(f"\n=== Saving ===")
os.makedirs(results_dir, exist_ok=True)

save_data = {
    'features': {lang: features[lang].cpu() for lang in LANGUAGES},
    'features_raw': {lang: features_raw[lang].cpu() for lang in LANGUAGES},
    'cos_matrix': cos_matrix,
    'all_results': all_results,
    'random_kl': rand_kl,
    'random_l2': rand_l2,
    'angles_deg': angles_deg,
    'layer': LAYER,
    'measure_layer': MEASURE_LAYER,
    'languages': LANGUAGES,
}
with open(os.path.join(results_dir, 'full_results.pkl'), 'wb') as f:
    pickle.dump(save_data, f)
print(f"  Saved: {results_dir}/full_results.pkl")

# === Summary ===
print(f"\n=== Summary: L2 and KL at 30° by n ===")
idx_30 = np.argmin(np.abs(angles_deg - 30))
for n in range(1, k + 1):
    l2_vals = [r['l2'][:, idx_30].mean() for r in all_results[n]]
    kl_vals = [r['kl'][:, idx_30].mean() for r in all_results[n]]
    print(f"  n={n}: L2={np.mean(l2_vals):.1f}±{np.std(l2_vals):.1f}, KL={np.mean(kl_vals):.3f}±{np.std(kl_vals):.3f}")
rand_l2_30 = rand_l2[:, idx_30, :].mean()
rand_kl_30 = rand_kl[:, idx_30, :].mean()
print(f"  random: L2={rand_l2_30:.1f}, KL={rand_kl_30:.3f}")

print("\nDone!")
