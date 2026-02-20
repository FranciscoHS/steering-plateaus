#!/usr/bin/env python3
"""
Test whether orthogonalized (family-residual) feature directions break the
plateau more sharply than the original features.

For each language family with >1 member, compute:
  - Family mean direction: f_family = mean(f_i for i in family)
  - Pure residual: f_i_pure = f_i - proj(f_i, f_family), then normalize

Then sweep both original and pure features and compare KL/L2 curves.
"""

import sys, os, pickle, argparse, numpy as np
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import (
    load_model, _get_blocks, _get_final_norm, _get_lm_head,
    _is_rope_model, _get_position_embeddings, _run_block,
)
from src.data import load_fineweb_fixed_length

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='qwen3-0.6b')
parser.add_argument('--layer', type=int, default=14)
parser.add_argument('--measure-layer', type=int, default=None)
parser.add_argument('--n_act', type=int, default=50)
parser.add_argument('--seq_len', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--bfloat16', action='store_true')
args = parser.parse_args()

LANG_FAMILIES = {
    'romance': ['es', 'fr', 'it', 'pt'],
    'germanic': ['nl', 'de'],
    'cjk': ['zh', 'ja'],
}
# Singletons (no orthogonalization needed)
SINGLETONS = ['ru', 'ar']


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

results_dir = f'results/allpairs_contrastive/{args.model}'

# Load feature directions
print("Loading feature directions...")
with open(os.path.join(results_dir, 'directions.pkl'), 'rb') as f:
    dir_data = pickle.load(f)
features = dir_data['features']
languages = dir_data['languages']

# Load model
print(f"\nLoading model: {args.model}")
load_dtype = torch.bfloat16 if args.bfloat16 else None
model, tokenizer, device = load_model(args.model, dtype=load_dtype)
if args.bfloat16:
    print("  Loaded in bfloat16")
n_layers = len(_get_blocks(model))
measure_layer = args.measure_layer if args.measure_layer is not None else n_layers - 2
print(f"  {n_layers} layers, measure_layer={measure_layer}")

# Compute orthogonalized features
print("\n=== Computing orthogonalized features ===")
pure_features = {}

for family_name, family_langs in LANG_FAMILIES.items():
    # Family mean (unnormalized — use raw features for proper averaging)
    raw_features = dir_data.get('features_raw', features)
    family_vecs = [raw_features[l] for l in family_langs if l in features]
    if len(family_vecs) < 2:
        continue
    family_mean = sum(family_vecs) / len(family_vecs)
    family_dir = F.normalize(family_mean.unsqueeze(0), dim=-1).squeeze(0)

    print(f"\n  {family_name}: {family_langs}")
    print(f"    ||family_mean|| = {family_mean.norm():.2f}")

    for lang in family_langs:
        if lang not in features:
            continue
        f_lang = features[lang]  # unit vector

        # Orthogonalize against family mean direction
        proj = (f_lang @ family_dir) * family_dir
        residual = f_lang - proj
        residual_norm = residual.norm().item()
        pure = F.normalize(residual.unsqueeze(0), dim=-1).squeeze(0)
        pure_features[lang] = pure

        cos_orig_family = (f_lang @ family_dir).item()
        cos_pure_family = (pure @ family_dir).item()
        print(f"    {lang}: cos(orig, family)={cos_orig_family:.3f}, "
              f"||residual||={residual_norm:.3f}, "
              f"cos(pure, family)={cos_pure_family:.6f}")

# Load FineWeb activations
print(f"\nLoading {args.n_act} FineWeb sequences...")
token_ids_list = load_fineweb_fixed_length(args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
contexts, activations = extract_context_and_activation(
    model, tokenizer, token_ids_list, args.layer, device)
contexts = contexts.to(device)
activations = activations.to(device)

# Reference
with torch.no_grad():
    ref_hidden = torch.cat([contexts, activations.unsqueeze(1)], dim=1)
    ref_measure, ref_logits = forward_with_measure(model, ref_hidden, args.layer, measure_layer)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

angles_deg = np.linspace(3, 60, 40)
angles_rad = np.radians(angles_deg)

# Sweep all directions
print("\n=== Sweeping directions ===")
sweep_results = {}

# Original features (only those in families)
for lang in pure_features:
    name = f'orig_{lang}'
    print(f"  {name}...")
    kl, l2 = sweep_single_direction(
        model, contexts, activations, ref_measure, ref_log_probs,
        features[lang], angles_rad, args.layer, measure_layer, device)
    sweep_results[name] = {'kl': kl, 'l2': l2}

# Pure (orthogonalized) features
for lang in pure_features:
    name = f'pure_{lang}'
    print(f"  {name}...")
    kl, l2 = sweep_single_direction(
        model, contexts, activations, ref_measure, ref_log_probs,
        pure_features[lang], angles_rad, args.layer, measure_layer, device)
    sweep_results[name] = {'kl': kl, 'l2': l2}

# Family mean directions
for family_name, family_langs in LANG_FAMILIES.items():
    raw_features = dir_data.get('features_raw', features)
    family_vecs = [raw_features[l] for l in family_langs if l in features]
    if len(family_vecs) < 2:
        continue
    family_mean = sum(family_vecs) / len(family_vecs)
    family_dir = F.normalize(family_mean.unsqueeze(0), dim=-1).squeeze(0)
    name = f'family_{family_name}'
    print(f"  {name}...")
    kl, l2 = sweep_single_direction(
        model, contexts, activations, ref_measure, ref_log_probs,
        family_dir, angles_rad, args.layer, measure_layer, device)
    sweep_results[name] = {'kl': kl, 'l2': l2}

# Random baseline
print("  random...")
rand_kl, rand_l2 = sweep_random_directions(
    model, contexts, activations, ref_measure, ref_log_probs,
    angles_rad, args.layer, measure_layer, device, n_dirs=10)
sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}

# Save
save_data = {
    'sweep_results': sweep_results,
    'angles_deg': angles_deg,
    'pure_features': {k: v.cpu() for k, v in pure_features.items()},
    'features': {k: v.cpu() for k, v in features.items()},
    'layer': args.layer,
    'measure_layer': measure_layer,
    'model': args.model,
}
out_path = os.path.join(results_dir, 'orthogonal_features.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(save_data, f)
print(f"\nSaved: {out_path}")

# === Plot ===
print("\n=== Generating plots ===")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

family_colors = {'romance': '#e41a1c', 'germanic': '#377eb8', 'cjk': '#4daf4a'}
lang_markers = {'es': 'o', 'fr': 's', 'it': '^', 'pt': 'v',
                'nl': 'o', 'de': 's', 'zh': 'o', 'ja': 's'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

rand_kl_mean = rand_kl.mean(axis=(0, 2))
rand_l2_mean = rand_l2.mean(axis=(0, 2))

for family_name, family_langs in LANG_FAMILIES.items():
    color = family_colors[family_name]
    for lang in family_langs:
        if lang not in pure_features:
            continue
        # Original
        orig_kl = sweep_results[f'orig_{lang}']['kl'].mean(axis=0)
        orig_l2 = sweep_results[f'orig_{lang}']['l2'].mean(axis=0)
        ax1.plot(angles_deg, orig_kl, '-', color=color, alpha=0.4, linewidth=1)
        ax2.plot(angles_deg, orig_l2, '-', color=color, alpha=0.4, linewidth=1)

        # Pure
        pure_kl = sweep_results[f'pure_{lang}']['kl'].mean(axis=0)
        pure_l2 = sweep_results[f'pure_{lang}']['l2'].mean(axis=0)
        ax1.plot(angles_deg, pure_kl, '--', color=color, alpha=0.7, linewidth=1.5)
        ax2.plot(angles_deg, pure_l2, '--', color=color, alpha=0.7, linewidth=1.5)

    # Family mean
    fname = f'family_{family_name}'
    if fname in sweep_results:
        fam_kl = sweep_results[fname]['kl'].mean(axis=0)
        fam_l2 = sweep_results[fname]['l2'].mean(axis=0)
        ax1.plot(angles_deg, fam_kl, '-', color=color, linewidth=2.5,
                 label=f'{family_name} (family mean)')
        ax2.plot(angles_deg, fam_l2, '-', color=color, linewidth=2.5,
                 label=f'{family_name} (family mean)')

# Random
ax1.plot(angles_deg, rand_kl_mean, 'k--', linewidth=2, label='random')
ax2.plot(angles_deg, rand_l2_mean, 'k--', linewidth=2, label='random')

# Add legend entries for line styles
ax1.plot([], [], '-', color='gray', alpha=0.4, linewidth=1, label='original features')
ax1.plot([], [], '--', color='gray', alpha=0.7, linewidth=1.5, label='pure (orthogonalized)')
ax2.plot([], [], '-', color='gray', alpha=0.4, linewidth=1, label='original features')
ax2.plot([], [], '--', color='gray', alpha=0.7, linewidth=1.5, label='pure (orthogonalized)')

ax1.set_xlabel('Perturbation angle (°)')
ax1.set_ylabel('Mean KL divergence')
ax1.set_title('KL divergence')
ax1.legend(fontsize=7, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

ax2.set_xlabel('Perturbation angle (°)')
ax2.set_ylabel('Mean L2 distance')
ax2.set_title(f'L2 distance (layer {measure_layer})')
ax2.legend(fontsize=7, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

fig.suptitle(f'Original vs orthogonalized features — {args.model}, layer {args.layer}',
             fontsize=13)
fig.tight_layout()
out_plot = os.path.join(results_dir, 'orthogonal_features.png')
fig.savefig(out_plot, dpi=150, bbox_inches='tight')
print(f"Saved: {out_plot}")

# Summary: average KL at 30deg for each category
idx30 = np.argmin(np.abs(angles_deg - 30))
print(f"\n=== KL at 30° ===")
for family_name, family_langs in LANG_FAMILIES.items():
    for lang in family_langs:
        if lang not in pure_features:
            continue
        orig_kl = sweep_results[f'orig_{lang}']['kl'].mean(axis=0)[idx30]
        pure_kl = sweep_results[f'pure_{lang}']['kl'].mean(axis=0)[idx30]
        ratio = pure_kl / orig_kl if orig_kl > 0 else float('nan')
        print(f"  {lang} ({family_name}): orig={orig_kl:.3f}, pure={pure_kl:.3f}, ratio={ratio:.2f}")
    fname = f'family_{family_name}'
    if fname in sweep_results:
        fam_kl = sweep_results[fname]['kl'].mean(axis=0)[idx30]
        print(f"  {family_name} mean: {fam_kl:.3f}")
print(f"  random: {rand_kl_mean[idx30]:.3f}")

print("\nDone!")
