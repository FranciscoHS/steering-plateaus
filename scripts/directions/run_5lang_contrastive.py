#!/usr/bin/env python3
"""
Re-estimate contrastive features using 5 languages (one per family),
then run perturbation sweeps. Uses existing activation data from the
10-language allpairs run.

The 5-language estimator is better conditioned because the features
are more orthogonal (no within-family correlations).
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

SUBSET = ['es', 'de', 'zh', 'ru', 'ar']


# --- Helper functions (same as run_orthogonal_features.py) ---

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

# Load 10-language feature directions
print("Loading 10-language feature directions...")
with open(os.path.join(results_dir, 'directions.pkl'), 'rb') as f:
    dir_data = pickle.load(f)
features_10 = dir_data['features']
raw_10 = dir_data.get('features_raw', features_10)

# Re-estimate with 5-language subset
print(f"\nRe-estimating with 5-language subset: {SUBSET}")
raw_subset = {l: raw_10[l].float() for l in SUBSET}
features_5_raw = {}
for lang in SUBSET:
    others = [l for l in SUBSET if l != lang]
    diffs = [raw_subset[lang] - raw_subset[o] for o in others]
    features_5_raw[lang] = sum(diffs) / len(diffs)

features_5 = {l: F.normalize(v.unsqueeze(0), dim=-1).squeeze(0)
              for l, v in features_5_raw.items()}

# Print diagnostics
print("\n  Pairwise cosines (5-lang estimator):")
for i in range(len(SUBSET)):
    for j in range(i + 1, len(SUBSET)):
        cos = (features_5[SUBSET[i]] @ features_5[SUBSET[j]]).item()
        print(f"    {SUBSET[i]} vs {SUBSET[j]}: {cos:+.3f}")

print("\n  Cosine between 10-lang and 5-lang estimates:")
for lang in SUBSET:
    cos10 = features_10[lang].float()
    cos5 = features_5[lang]
    print(f"    {lang}: {(cos10 @ cos5).item():.4f}")

# Load model
print(f"\nLoading model: {args.model}")
load_dtype = torch.bfloat16 if args.bfloat16 else None
model, tokenizer, device = load_model(args.model, dtype=load_dtype)
model_dtype = next(model.parameters()).dtype
if args.bfloat16:
    print("  Loaded in bfloat16")
n_layers = len(_get_blocks(model))
measure_layer = args.measure_layer if args.measure_layer is not None else n_layers - 2
print(f"  {n_layers} layers, measure_layer={measure_layer}")

# Convert features to model dtype
for lang in SUBSET:
    features_5[lang] = features_5[lang].to(dtype=model_dtype)
    features_10[lang] = features_10[lang].to(dtype=model_dtype)

# Load FineWeb activations
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

angles_deg = np.linspace(3, 60, 40)
angles_rad = np.radians(angles_deg)

# Sweep both 10-lang and 5-lang features for the subset languages
print("\n=== Sweeping directions ===")
sweep_results = {}

for lang in SUBSET:
    # 10-lang estimate
    name = f'10lang_{lang}'
    print(f"  {name}...")
    kl, l2 = sweep_single_direction(
        model, contexts, activations, ref_measure, ref_log_probs,
        features_10[lang], angles_rad, args.layer, measure_layer, device)
    sweep_results[name] = {'kl': kl, 'l2': l2}

    # 5-lang estimate
    name = f'5lang_{lang}'
    print(f"  {name}...")
    kl, l2 = sweep_single_direction(
        model, contexts, activations, ref_measure, ref_log_probs,
        features_5[lang], angles_rad, args.layer, measure_layer, device)
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
    'features_5': {k: v.cpu() for k, v in features_5.items()},
    'features_10': {k: features_10[k].cpu() for k in SUBSET},
    'subset': SUBSET,
    'layer': args.layer,
    'measure_layer': measure_layer,
    'model': args.model,
}
out_path = os.path.join(results_dir, '5lang_contrastive.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(save_data, f)
print(f"\nSaved: {out_path}")

# === Plot ===
print("\n=== Generating plots ===")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

lang_colors = {'es': '#e41a1c', 'de': '#377eb8', 'zh': '#4daf4a',
               'ru': '#984ea3', 'ar': '#ff7f00'}
lang_labels = {'es': 'Spanish', 'de': 'German', 'zh': 'Chinese',
               'ru': 'Russian', 'ar': 'Arabic'}

rand_kl_mean = rand_kl.mean(axis=(0, 2))
rand_l2_mean = rand_l2.mean(axis=(0, 2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(angles_deg, rand_kl_mean, 'k--', linewidth=2, label='random')
ax2.plot(angles_deg, rand_l2_mean, 'k--', linewidth=2, label='random')

for lang in SUBSET:
    color = lang_colors[lang]
    label = lang_labels[lang]

    kl_10 = sweep_results[f'10lang_{lang}']['kl'].mean(axis=0)
    kl_5 = sweep_results[f'5lang_{lang}']['kl'].mean(axis=0)
    l2_10 = sweep_results[f'10lang_{lang}']['l2'].mean(axis=0)
    l2_5 = sweep_results[f'5lang_{lang}']['l2'].mean(axis=0)

    ax1.plot(angles_deg, kl_10, '-', color=color, linewidth=1, alpha=0.4)
    ax1.plot(angles_deg, kl_5, '-', color=color, linewidth=2, label=label)
    ax2.plot(angles_deg, l2_10, '-', color=color, linewidth=1, alpha=0.4)
    ax2.plot(angles_deg, l2_5, '-', color=color, linewidth=2, label=label)

# Legend entries for line styles
ax1.plot([], [], '-', color='gray', alpha=0.4, linewidth=1, label='10-lang estimate')
ax1.plot([], [], '-', color='gray', linewidth=2, label='5-lang estimate')
ax2.plot([], [], '-', color='gray', alpha=0.4, linewidth=1, label='10-lang estimate')
ax2.plot([], [], '-', color='gray', linewidth=2, label='5-lang estimate')

ax1.set_xlabel('Perturbation angle (°)')
ax1.set_ylabel('Mean KL divergence')
ax1.set_title('KL divergence')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

ax2.set_xlabel('Perturbation angle (°)')
ax2.set_ylabel('Mean L2 distance')
ax2.set_title(f'L2 distance (layer {measure_layer})')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

fig.suptitle(f'5-lang (thick) vs 10-lang (thin) contrastive features — {args.model}',
             fontsize=13)
fig.tight_layout()
out_plot = os.path.join(results_dir, '5lang_contrastive.png')
fig.savefig(out_plot, dpi=150, bbox_inches='tight')
print(f"Saved: {out_plot}")

# Summary
idx30 = np.argmin(np.abs(angles_deg - 30))
print(f"\n=== KL at 30° ===")
for lang in SUBSET:
    kl_10 = sweep_results[f'10lang_{lang}']['kl'].mean(axis=0)[idx30]
    kl_5 = sweep_results[f'5lang_{lang}']['kl'].mean(axis=0)[idx30]
    ratio = kl_5 / kl_10 if kl_10 > 0 else float('nan')
    print(f"  {lang}: 10-lang={kl_10:.3f}, 5-lang={kl_5:.3f}, ratio={ratio:.2f}")
print(f"  random: {rand_kl_mean[idx30]:.3f}")

print("\nDone!")
