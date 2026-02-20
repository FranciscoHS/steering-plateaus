#!/usr/bin/env python3
"""
All-pairwise multi-contrastive feature estimation for 10 languages.

Uses the Section 8 least-squares estimator from multi_contrastive_projection.pdf:
  f̂_i = (1/k) Σ_{j≠i} v̄_{ij}

where v̄_{ij} is the mean difference vector for pair (i,j) and k is the number
of languages. This recovers each language's feature direction with O(εC) error,
improving over the O(C/√k) rate of the star topology.

Languages: zh, nl, es, fr, de, ja, it, ru, pt, ar (no English)
"""

import sys, os, pickle, itertools, numpy as np
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import (
    load_model, _get_blocks, _get_final_norm, _get_lm_head,
    _is_rope_model, _get_position_embeddings, _run_block,
)
from src.data import load_fineweb_fixed_length
from data.multilingual_prompts import PROMPT_SETS, LANGUAGES

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='qwen3-0.6b')
parser.add_argument('--layer', type=int, default=14)
parser.add_argument('--measure-layer', type=int, default=None,
                    help='Layer for L2 measurement (default: n_layers - 2)')
parser.add_argument('--n_act', type=int, default=50)
parser.add_argument('--seq_len', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--bfloat16', action='store_true',
                    help='Load model in bfloat16 (needed for larger models on 24GB)')
args = parser.parse_args()

MODEL = args.model
LAYER = args.layer
N_PROMPTS = 36
N_ACT = args.n_act
SEQ_LEN = args.seq_len
SEED = args.seed


# --- Helper functions (reused from rerun_plateau_sweeps.py) ---

def get_acts(model, tokenizer, device, prompts, layer):
    """Extract final-token activations at given layer for a list of prompts."""
    acts = []
    for p in prompts:
        toks = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**toks, output_hidden_states=True)
        acts.append(out.hidden_states[layer + 1][:, -1, :].squeeze(0))
    return torch.stack(acts).cpu()


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device,
                                   batch_size=64):
    """Extract layer-L context and final-token activation for fixed-length sequences."""
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
    """Forward pass capturing hidden state at measure_layer AND producing logits."""
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
    """Construct hidden states after spherical perturbation along a fixed direction."""
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
    """Perturbation sweep for one fixed direction. KL from logits, L2 at measure_layer."""
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
            l2_dists[:, ang_idx] = l2.float().cpu().numpy()
            kl_divs[:, ang_idx] = kl.float().cpu().numpy()
    return kl_divs, l2_dists


def sweep_random_directions(model, contexts, activations, ref_measure, ref_log_probs,
                            angles_rad, layer, measure_layer, device, n_dirs=10):
    """Perturbation sweep with per-activation random orthogonal directions."""
    N, d = activations.shape
    n_angles = len(angles_rad)
    kl_divs = np.zeros((N, n_angles, n_dirs))
    l2_dists = np.zeros((N, n_angles, n_dirs))
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms

    act_dtype = activations.dtype
    v_perps = torch.zeros(N, n_dirs, d, device=device, dtype=act_dtype)
    for i in range(N):
        v = torch.randn(n_dirs, d, device=device, dtype=act_dtype)
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
                l2_dists[:, ang_idx, dir_idx] = l2.float().cpu().numpy()
                kl_divs[:, ang_idx, dir_idx] = kl.float().cpu().numpy()
    return kl_divs, l2_dists


# ============================================================
# MAIN
# ============================================================

print(f"Model: {MODEL}, Layer: {LAYER}")
print(f"Languages: {LANGUAGES}")
print(f"Number of prompt sets: {N_PROMPTS}")
print(f"Number of pairwise contrasts: {len(LANGUAGES) * (len(LANGUAGES) - 1) // 2}")

print("\nLoading model...")
load_dtype = torch.bfloat16 if args.bfloat16 else None
model, tokenizer, device = load_model(MODEL, dtype=load_dtype)
if args.bfloat16:
    print("  Loaded in bfloat16")
n_layers = len(_get_blocks(model))
MEASURE_LAYER = args.measure_layer if args.measure_layer is not None else n_layers - 2
results_dir = f'results/allpairs_contrastive/{MODEL}'
print(f"  {n_layers} layers, measure_layer={MEASURE_LAYER}")

# === Phase 1: Extract activations for all languages ===
print("\n=== Phase 1: Extracting activations ===")
acts = {}
for lang in LANGUAGES:
    prompts = [ps[lang] for ps in PROMPT_SETS]
    acts[lang] = get_acts(model, tokenizer, device, prompts, LAYER)
    print(f"  {lang}: {acts[lang].shape}")

# === Phase 2: Compute all 45 pairwise mean difference vectors ===
print("\n=== Phase 2: Computing pairwise difference vectors ===")
k = len(LANGUAGES)
pairwise_means = {}  # (lang_i, lang_j) → mean diff vector
for i, li in enumerate(LANGUAGES):
    for j, lj in enumerate(LANGUAGES):
        if i < j:
            diff = (acts[li] - acts[lj]).mean(dim=0)  # [d_model]
            pairwise_means[(li, lj)] = diff
            norm = diff.norm().item()
            print(f"  {li}-{lj}: ||v̄|| = {norm:.2f}")

# === Phase 3: Apply Section 8 least-squares estimator ===
print("\n=== Phase 3: Least-squares feature estimation ===")
features_raw = {}  # unnormalized
features = {}      # normalized to unit length
for lang in LANGUAGES:
    # f̂_i = (1/k) Σ_{j≠i} v̄_{ij}
    f_hat = torch.zeros_like(acts[lang][0])
    for other in LANGUAGES:
        if other == lang:
            continue
        if (lang, other) in pairwise_means:
            f_hat += pairwise_means[(lang, other)]
        else:
            f_hat -= pairwise_means[(other, lang)]  # v̄_{ij} = -v̄_{ji}
    f_hat = f_hat / k
    features_raw[lang] = f_hat
    features[lang] = F.normalize(f_hat.unsqueeze(0), dim=-1).squeeze(0)
    print(f"  {lang}: ||f̂|| = {f_hat.norm().item():.4f}")

# Sanity check: unnormalized features should sum to ~0
total = sum(features_raw.values())
print(f"\n  Σ f̂_i norm (should be ~0): {total.norm().item():.6f}")

# === Phase 4: Diagnostics ===
print("\n=== Phase 4: Diagnostics ===")

# 10x10 cosine similarity matrix
print("\nCosine similarity matrix between recovered features:")
cos_matrix = np.zeros((k, k))
header = "        " + "  ".join(f"{l:>5}" for l in LANGUAGES)
print(header)
for i, li in enumerate(LANGUAGES):
    for j, lj in enumerate(LANGUAGES):
        cos_matrix[i, j] = F.cosine_similarity(
            features[li].unsqueeze(0), features[lj].unsqueeze(0)).item()
    row = f"  {li:>4}  " + "  ".join(f"{cos_matrix[i,j]:5.2f}" for j in range(k))
    print(row)

# Consistency check: cos(v̄_{ij}, f̂_i - f̂_j) for all pairs
print("\nConsistency: cos(v̄_{ij}, f̂_i - f̂_j) for all pairs:")
consistencies = []
for (li, lj), v_ij in pairwise_means.items():
    diff_feat = features_raw[li] - features_raw[lj]
    cos = F.cosine_similarity(v_ij.unsqueeze(0), diff_feat.unsqueeze(0)).item()
    consistencies.append(cos)
    print(f"  {li}-{lj}: {cos:.4f}")
print(f"  Mean consistency: {np.mean(consistencies):.4f}")

# === Phase 5: Perturbation sweeps ===
print("\n=== Phase 5: Perturbation sweeps ===")
print("Loading FineWeb activations...")
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

angles_deg = np.linspace(3, 180, 60)
angles_rad = np.radians(angles_deg)

# Sweep positive directions (toward each language)
print(f"\nSweeping positive directions (L2 at layer {MEASURE_LAYER})...")
sweep_results = {}
for lang in LANGUAGES:
    name = f"+{lang}"
    print(f"  Sweeping {name}...")
    kl, l2 = sweep_single_direction(
        model, contexts, activations, ref_measure, ref_log_probs,
        features[lang], angles_rad, LAYER, MEASURE_LAYER, device)
    sweep_results[name] = {'kl': kl, 'l2': l2}
    peak_kl = kl.mean(axis=0).max()
    peak_angle = angles_deg[kl.mean(axis=0).argmax()]
    print(f"    peak KL = {peak_kl:.3f} at {peak_angle:.0f}°")

# Sweep negative directions (away from each language)
print(f"\nSweeping negative directions...")
for lang in LANGUAGES:
    name = f"-{lang}"
    print(f"  Sweeping {name}...")
    kl, l2 = sweep_single_direction(
        model, contexts, activations, ref_measure, ref_log_probs,
        -features[lang], angles_rad, LAYER, MEASURE_LAYER, device)
    sweep_results[name] = {'kl': kl, 'l2': l2}
    peak_kl = kl.mean(axis=0).max()
    peak_angle = angles_deg[kl.mean(axis=0).argmax()]
    print(f"    peak KL = {peak_kl:.3f} at {peak_angle:.0f}°")

# Sweep random directions
print("\nSweeping random directions (10 dirs per activation)...")
rand_kl, rand_l2 = sweep_random_directions(
    model, contexts, activations, ref_measure, ref_log_probs,
    angles_rad, LAYER, MEASURE_LAYER, device, n_dirs=10)
sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}
print(f"  random KL shape: {rand_kl.shape}")

# === Phase 6: Save ===
print("\n=== Saving results ===")
os.makedirs(results_dir, exist_ok=True)

directions_data = {
    'features': {lang: features[lang].cpu() for lang in LANGUAGES},
    'features_raw': {lang: features_raw[lang].cpu() for lang in LANGUAGES},
    'pairwise_means': {k: v.cpu() for k, v in pairwise_means.items()},
    'cosine_matrix': cos_matrix,
    'languages': LANGUAGES,
    'layer': LAYER,
    'model': MODEL,
}
with open(os.path.join(results_dir, 'directions.pkl'), 'wb') as f:
    pickle.dump(directions_data, f)
print(f"  Saved: {results_dir}/directions.pkl")

sweep_data = {
    'sweep_results': sweep_results,
    'angles_deg': angles_deg,
    'layer': LAYER,
    'measure_layer': MEASURE_LAYER,
    'languages': LANGUAGES,
}
with open(os.path.join(results_dir, 'sweep_results.pkl'), 'wb') as f:
    pickle.dump(sweep_data, f)
print(f"  Saved: {results_dir}/sweep_results.pkl")

# === Phase 7: Plots ===
print("\n=== Generating plots ===")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Use a colormap with 10 distinct colors
cmap = plt.cm.tab10
mask = angles_deg <= 60

# --- Plot 1: Positive directions ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for i, lang in enumerate(LANGUAGES):
    name = f"+{lang}"
    kl = sweep_results[name]['kl'][:, mask]
    l2 = sweep_results[name]['l2'][:, mask]
    ax1.plot(angles_deg[mask], kl.mean(axis=0), color=cmap(i), label=lang, linewidth=1.5)
    ax2.plot(angles_deg[mask], l2.mean(axis=0), color=cmap(i), label=lang, linewidth=1.5)

# Random baseline
rand_kl_avg = rand_kl[:, mask, :].mean(axis=2)
rand_l2_avg = rand_l2[:, mask, :].mean(axis=2)
ax1.plot(angles_deg[mask], rand_kl_avg.mean(axis=0), '-', color='gray',
         label='random', linewidth=1.5)
ax1.fill_between(angles_deg[mask],
                 rand_kl_avg.mean(axis=0) - rand_kl_avg.std(axis=0),
                 rand_kl_avg.mean(axis=0) + rand_kl_avg.std(axis=0),
                 color='gray', alpha=0.15)
ax2.plot(angles_deg[mask], rand_l2_avg.mean(axis=0), '-', color='gray',
         label='random', linewidth=1.5)
ax2.fill_between(angles_deg[mask],
                 rand_l2_avg.mean(axis=0) - rand_l2_avg.std(axis=0),
                 rand_l2_avg.mean(axis=0) + rand_l2_avg.std(axis=0),
                 color='gray', alpha=0.15)

ax1.set_xlabel('Perturbation angle (°)')
ax1.set_ylabel('Mean KL divergence')
ax1.set_title('KL divergence (from final logits)')
ax1.legend(fontsize=7, ncol=2)
ax1.grid(True, alpha=0.3)
ax2.set_xlabel('Perturbation angle (°)')
ax2.set_ylabel('Mean L2 distance')
ax2.set_title(f'L2 distance (layer {MEASURE_LAYER})')
ax2.legend(fontsize=7, ncol=2)
ax2.grid(True, alpha=0.3)
fig.suptitle('Positive feature directions (+f̂_i: toward language)\n'
             f'{MODEL}, layer {LAYER}', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(results_dir, 'sweeps_positive.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {results_dir}/sweeps_positive.png")

# --- Plot 2: Negative directions ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for i, lang in enumerate(LANGUAGES):
    name = f"-{lang}"
    kl = sweep_results[name]['kl'][:, mask]
    l2 = sweep_results[name]['l2'][:, mask]
    ax1.plot(angles_deg[mask], kl.mean(axis=0), color=cmap(i), label=lang, linewidth=1.5)
    ax2.plot(angles_deg[mask], l2.mean(axis=0), color=cmap(i), label=lang, linewidth=1.5)

ax1.plot(angles_deg[mask], rand_kl_avg.mean(axis=0), '-', color='gray',
         label='random', linewidth=1.5)
ax1.fill_between(angles_deg[mask],
                 rand_kl_avg.mean(axis=0) - rand_kl_avg.std(axis=0),
                 rand_kl_avg.mean(axis=0) + rand_kl_avg.std(axis=0),
                 color='gray', alpha=0.15)
ax2.plot(angles_deg[mask], rand_l2_avg.mean(axis=0), '-', color='gray',
         label='random', linewidth=1.5)
ax2.fill_between(angles_deg[mask],
                 rand_l2_avg.mean(axis=0) - rand_l2_avg.std(axis=0),
                 rand_l2_avg.mean(axis=0) + rand_l2_avg.std(axis=0),
                 color='gray', alpha=0.15)

ax1.set_xlabel('Perturbation angle (°)')
ax1.set_ylabel('Mean KL divergence')
ax1.set_title('KL divergence (from final logits)')
ax1.legend(fontsize=7, ncol=2)
ax1.grid(True, alpha=0.3)
ax2.set_xlabel('Perturbation angle (°)')
ax2.set_ylabel('Mean L2 distance')
ax2.set_title(f'L2 distance (layer {MEASURE_LAYER})')
ax2.legend(fontsize=7, ncol=2)
ax2.grid(True, alpha=0.3)
fig.suptitle('Negative feature directions (-f̂_i: away from language)\n'
             f'{MODEL}, layer {LAYER}', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(results_dir, 'sweeps_negative.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {results_dir}/sweeps_negative.png")

# --- Plot 3: Cosine similarity heatmap ---
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cos_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(k))
ax.set_xticklabels(LANGUAGES, fontsize=9)
ax.set_yticks(range(k))
ax.set_yticklabels(LANGUAGES, fontsize=9)
for i in range(k):
    for j in range(k):
        ax.text(j, i, f'{cos_matrix[i,j]:.2f}', ha='center', va='center', fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title(f'Cosine similarity between recovered features\n{MODEL}, layer {LAYER}')
fig.tight_layout()
fig.savefig(os.path.join(results_dir, 'cosine_heatmap.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {results_dir}/cosine_heatmap.png")

print("\nDone!")
