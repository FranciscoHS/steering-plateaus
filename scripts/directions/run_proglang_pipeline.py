#!/usr/bin/env python3
"""
Run contrastive direction estimation + plateau scaling for programming languages.
Combines run_programming_contrastive.py and run_plateau_scaling.py logic.

Usage:
    python scripts/directions/run_proglang_pipeline.py --model qwen3-1.7b --languages java c rust go lisp
"""

import sys, os, pickle, itertools, random, argparse
import numpy as np
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import (
    load_model, _get_blocks, _get_final_norm, _get_lm_head,
    _is_rope_model, _get_position_embeddings, _run_block,
)
from src.data import load_fineweb_fixed_length
from data.programming_prompts import PROMPT_SETS, LANGUAGES

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='qwen3-1.7b')
parser.add_argument('--layer', type=int, default=14)
parser.add_argument('--languages', nargs='+', default=['java', 'c', 'rust', 'go', 'lisp'])
parser.add_argument('--n_act', type=int, default=50)
parser.add_argument('--seq_len', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_combos', type=int, default=45)
parser.add_argument('--bfloat16', action='store_true')
parser.add_argument('--eval_source', default='fineweb', choices=['fineweb', 'coding'],
                    help='Source of evaluation prompts: fineweb (generic text) or coding (language-agnostic code prompts)')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

langs = args.languages
k = len(langs)
print(f"Languages ({k}): {langs}")

results_dir = f'results/programming_contrastive/{args.model}'
os.makedirs(results_dir, exist_ok=True)


# === Helper functions ===

def get_acts(model, tokenizer, device, prompts, layer):
    acts = []
    for p in prompts:
        tokens = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        acts.append(outputs.hidden_states[layer + 1][:, -1, :].squeeze(0).cpu())
    return torch.stack(acts)


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
            l2_dists[:, ang_idx] = l2.float().cpu().numpy()
            kl_divs[:, ang_idx] = kl.float().cpu().numpy()
    return kl_divs, l2_dists


def sweep_random_directions(model, contexts, activations, ref_measure, ref_log_probs,
                            angles_rad, layer, measure_layer, device, n_dirs=10):
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
                pert_measure, pert_logits = forward_with_measure(model, hidden, layer, measure_layer)
                pert_log_probs = F.log_softmax(pert_logits, dim=-1)
                l2 = (pert_measure - ref_measure).norm(dim=-1)
                kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
                l2_dists[:, ang_idx, dir_idx] = l2.float().cpu().numpy()
                kl_divs[:, ang_idx, dir_idx] = kl.float().cpu().numpy()
    return kl_divs, l2_dists


def measure_plateau_size(mean_kl, angles_deg, threshold):
    for i in range(len(mean_kl) - 1):
        if mean_kl[i] < threshold <= mean_kl[i + 1]:
            frac = (threshold - mean_kl[i]) / (mean_kl[i + 1] - mean_kl[i])
            return angles_deg[i] + frac * (angles_deg[i + 1] - angles_deg[i])
    if mean_kl[-1] < threshold:
        return angles_deg[-1]
    return angles_deg[0]


# ============================================================
# PHASE 1: Load model & extract contrastive features
# ============================================================

print(f"\nLoading model: {args.model}")
load_dtype = torch.bfloat16 if args.bfloat16 else None
model, tokenizer, device = load_model(args.model, dtype=load_dtype)
n_layers = len(_get_blocks(model))
measure_layer = n_layers - 2
print(f"  {n_layers} layers, measure_layer={measure_layer}")

# Extract activations per language
prompts_per_lang = {lang: [] for lang in langs}
for pset in PROMPT_SETS:
    for lang in langs:
        prompts_per_lang[lang].append(pset[lang])
n_prompts = len(PROMPT_SETS)
print(f"\n{n_prompts} matched prompt sets across {k} languages")

print(f"\nExtracting activations (layer {args.layer})...")
acts = {}
for lang in langs:
    acts[lang] = get_acts(model, tokenizer, device, prompts_per_lang[lang], args.layer)
    print(f"  {lang}: {acts[lang].shape}")

# Compute pairwise mean differences
from itertools import combinations
pairs = list(combinations(langs, 2))
v_bar = {}
for lang_i, lang_j in pairs:
    diff = acts[lang_i] - acts[lang_j]
    v_bar[(lang_i, lang_j)] = diff.mean(dim=0)
    v_bar[(lang_j, lang_i)] = -v_bar[(lang_i, lang_j)]
    print(f"  v_bar({lang_i}, {lang_j}): norm = {v_bar[(lang_i, lang_j)].norm():.2f}")

# Recover per-language features
features = {}
features_raw = {}
for lang_i in langs:
    total = torch.zeros_like(acts[langs[0]][0])
    for lang_j in langs:
        if lang_j != lang_i:
            total += v_bar[(lang_i, lang_j)]
    f_raw = total / k
    features_raw[lang_i] = f_raw
    features[lang_i] = f_raw / f_raw.norm()
    print(f"  f_hat_{lang_i}: norm(raw) = {f_raw.norm():.4f}")

# Cosine similarity matrix
print(f"\nCosine similarity matrix:")
header = f"{'':>8s}" + "".join(f"{l[:6]:>8s}" for l in langs)
print(header)
for li in langs:
    row = f"{li[:6]:>8s}"
    for lj in langs:
        cos = (features[li] @ features[lj]).item()
        row += f"{cos:>8.3f}"
    print(row)

# Save contrastive results
contrastive_save = {
    'features': {l: features[l].cpu() for l in langs},
    'features_raw': {l: features_raw[l].cpu() for l in langs},
    'languages': langs,
    'activations': {l: acts[l].cpu() for l in langs},
    'layer': args.layer,
    'model': args.model,
}
with open(os.path.join(results_dir, 'directions.pkl'), 'wb') as f:
    pickle.dump(contrastive_save, f)
print(f"\nSaved directions to {results_dir}/directions.pkl")


# ============================================================
# PHASE 2: Single-direction sweeps (for random baseline)
# ============================================================

print(f"\n{'='*60}")
print("PERTURBATION SWEEPS")
print(f"{'='*60}")

if args.eval_source == 'coding':
    from data.coding_prompts_agnostic import CODING_PROMPTS
    # Tokenize and filter to fixed length
    all_ids = [tokenizer.encode(p) for p in CODING_PROMPTS]
    # Find most common length for batching, or use seq_len as target
    target_len = args.seq_len
    token_ids_list = [torch.tensor(ids[:target_len]) for ids in all_ids if len(ids) >= target_len]
    if len(token_ids_list) > args.n_act:
        token_ids_list = token_ids_list[:args.n_act]
    print(f"\nLoaded {len(token_ids_list)} coding prompts (truncated to {target_len} tokens)")
else:
    print(f"\nLoading {args.n_act} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)

contexts_list, activations_list = [], []
for i in range(0, len(token_ids_list), 64):
    batch = torch.stack(token_ids_list[i:i+64]).to(device)
    with torch.no_grad():
        outputs = model(batch, output_hidden_states=True)
    hidden = outputs.hidden_states[args.layer + 1]
    contexts_list.append(hidden[:, :-1, :].cpu())
    activations_list.append(hidden[:, -1, :].cpu())
contexts = torch.cat(contexts_list, 0).to(device)
activations = torch.cat(activations_list, 0).to(device)

# Reference
with torch.no_grad():
    ref_hidden = torch.cat([contexts, activations.unsqueeze(1)], dim=1)
    ref_measure, ref_logits = forward_with_measure(model, ref_hidden, args.layer, measure_layer)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

angles_deg = np.linspace(3, 180, 60)
angles_rad = np.radians(angles_deg)

# Sweep individual features
sweep_results = {}
for lang in langs:
    print(f"  Sweeping {lang}...")
    kl, l2 = sweep_single_direction(
        model, contexts, activations, ref_measure, ref_log_probs,
        features[lang], angles_rad, args.layer, measure_layer, device)
    sweep_results[lang] = {'kl': kl, 'l2': l2}

# Random baseline
print("  Sweeping random...")
rand_kl, rand_l2 = sweep_random_directions(
    model, contexts, activations, ref_measure, ref_log_probs,
    angles_rad, args.layer, measure_layer, device, n_dirs=10)
sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}

# Save sweep results
sweep_save = {
    'angles_deg': angles_deg,
    'sweep_results': sweep_results,
}
with open(os.path.join(results_dir, 'sweep_results.pkl'), 'wb') as f:
    pickle.dump(sweep_save, f)
print(f"Saved sweep_results.pkl")


# ============================================================
# PHASE 3: Plateau scaling
# ============================================================

print(f"\n{'='*60}")
print("PLATEAU SCALING")
print(f"{'='*60}")

feature_vecs = [features[lang] for lang in langs]
all_results = {}
rng = random.Random(0)

for n in range(1, k + 1):
    all_combos = list(itertools.combinations(range(k), n))
    if len(all_combos) <= args.max_combos:
        combos = all_combos
    else:
        combos = rng.sample(all_combos, args.max_combos)

    print(f"\nn={n}: {len(combos)} combinations (of {len(all_combos)} total)")
    results_n = []

    for ci, combo in enumerate(combos):
        d = sum(feature_vecs[i] for i in combo)
        d_norm = d.norm().item()
        d = F.normalize(d.unsqueeze(0), dim=-1).squeeze(0)

        kl, l2 = sweep_single_direction(
            model, contexts, activations, ref_measure, ref_log_probs,
            d, angles_rad, args.layer, measure_layer, device)

        mean_kl = kl.mean(axis=0)
        ps = measure_plateau_size(mean_kl, angles_deg, 0.5)

        combo_langs = tuple(langs[i] for i in combo)
        results_n.append({
            'combo': combo,
            'combo_langs': combo_langs,
            'kl': kl,
            'l2': l2,
            'plateau_size': ps,
            'sum_norm': d_norm,
        })

        if ci % 10 == 0 or ci == len(combos) - 1:
            print(f"  [{ci+1}/{len(combos)}] {combo_langs}: plateau={ps:.1f}°")

    all_results[n] = results_n
    sizes = [r['plateau_size'] for r in results_n]
    print(f"  n={n}: {np.mean(sizes):.1f}° ± {np.std(sizes):.1f}°")

# Save plateau scaling
save_data = {
    'all_results': all_results,
    'angles_deg': angles_deg,
    'threshold': 0.5,
    'n_values': np.arange(1, k + 1),
    'mean_plateaus': np.array([np.mean([r['plateau_size'] for r in all_results[n]]) for n in range(1, k + 1)]),
    'std_plateaus': np.array([np.std([r['plateau_size'] for r in all_results[n]]) for n in range(1, k + 1)]),
    'layer': args.layer,
    'measure_layer': measure_layer,
    'languages': langs,
}
with open(os.path.join(results_dir, 'plateau_scaling.pkl'), 'wb') as f:
    pickle.dump(save_data, f)
print(f"\nSaved plateau_scaling.pkl")

# Summary
print(f"\n=== Summary ===")
for n in range(1, k + 1):
    sizes = [r['plateau_size'] for r in all_results[n]]
    print(f"  n={n}: {np.mean(sizes):.1f}° ± {np.std(sizes):.1f}°")

print("\nDone!")
