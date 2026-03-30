#!/usr/bin/env python3
"""
Feature purity experiment: do higher-quality contrastive pairs produce
steering directions with shorter plateaus?

Extracts English→Mandarin contrastive directions from three prompt sets
of varying quality (HQ, MQ, LQ), then sweeps steering magnitude and
measures KL divergence to compare plateau shapes.

All three sets use the same English starters (from the HQ set); they
differ only in Mandarin translation quality.
"""

import sys, os, pickle, numpy as np
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import (
    load_model, _get_blocks, _get_final_norm, _get_lm_head,
    _is_rope_model, _get_position_embeddings, _run_block,
)
from src.data import load_fineweb_fixed_length

from data.bilingual_prompts_hq import PROMPT_PAIRS as HQ_PAIRS
from data.bilingual_prompts_mq import PROMPT_PAIRS as MQ_PAIRS
from data.bilingual_prompts_lq import PROMPT_PAIRS as LQ_PAIRS

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='qwen3-8b')
parser.add_argument('--layer', type=int, default=None,
                    help='Layer for direction extraction (default: 6)')
parser.add_argument('--measure-layer', type=int, default=None,
                    help='Layer for L2 measurement (default: n_layers - 2)')
parser.add_argument('--n_act', type=int, default=50)
parser.add_argument('--seq_len', type=int, default=5)
parser.add_argument('--n_prompts', type=int, default=None,
                    help='Number of prompt pairs to use (default: all)')
parser.add_argument('--hq-only', action='store_true',
                    help='Only use HQ and LQ sets (skip MQ)')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--bfloat16', action='store_true',
                    help='Load model in bfloat16 (needed for larger models on 24GB)')
parser.add_argument('--dtype', default=None, help='Model dtype (e.g. bfloat16)')
args = parser.parse_args()
if args.bfloat16:
    args.dtype = 'bfloat16'

MODEL = args.model
N_ACT = args.n_act
SEQ_LEN = args.seq_len
SEED = args.seed

# --- Helper functions (from run_allpairs_contrastive.py) ---

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
    """Perturbation sweep for one fixed direction."""
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
                m_h, p_logits = forward_with_measure(model, hidden, layer, measure_layer)
                p_lp = F.log_softmax(p_logits, dim=-1)
                l2 = (m_h - ref_measure).norm(dim=-1)
                kl = (ref_log_probs.exp() * (ref_log_probs - p_lp)).sum(dim=-1)
                l2_dists[:, ang_idx, dir_idx] = l2.float().cpu().numpy()
                kl_divs[:, ang_idx, dir_idx] = kl.float().cpu().numpy()
    return kl_divs, l2_dists


# =====================================================================
# Main
# =====================================================================

print(f"Loading model: {MODEL}")
dtype = torch.bfloat16 if args.dtype == 'bfloat16' else None
model, tokenizer, device = load_model(MODEL, dtype=dtype)
n_layers = len(_get_blocks(model))

LAYER = args.layer if args.layer is not None else 6
MEASURE_LAYER = args.measure_layer if args.measure_layer is not None else n_layers - 2
print(f"Extraction layer: {LAYER}, Measure layer: {MEASURE_LAYER}, Total layers: {n_layers}")

# --- Phase 1: Extract contrastive directions ---
print("\n=== Phase 1: Extracting contrastive directions ===")

datasets = {'HQ': HQ_PAIRS, 'LQ': LQ_PAIRS}
if not args.hq_only:
    datasets['MQ'] = MQ_PAIRS

# Subsample prompts if requested
N_PROMPTS = args.n_prompts
if N_PROMPTS is not None:
    np.random.seed(SEED)
    for name in list(datasets.keys()):
        pairs = datasets[name]
        if len(pairs) > N_PROMPTS:
            idx = np.random.choice(len(pairs), N_PROMPTS, replace=False)
            datasets[name] = [pairs[i] for i in sorted(idx)]

directions = {}
direction_metadata = {}

for name, pairs in datasets.items():
    en_prompts = [p[0] for p in pairs]
    zh_prompts = [p[1] for p in pairs]

    print(f"\n{name}: {len(pairs)} pairs")
    en_acts = get_acts(model, tokenizer, device, en_prompts, LAYER)
    zh_acts = get_acts(model, tokenizer, device, zh_prompts, LAYER)

    # Contrastive direction: mean(zh - en), normalized (steers toward Mandarin)
    diff = zh_acts - en_acts
    mean_diff = diff.mean(dim=0)
    direction = F.normalize(mean_diff, dim=0)

    directions[name] = direction
    direction_metadata[name] = {
        'mean_diff_norm': mean_diff.norm().item(),
        'n_pairs': len(pairs),
        'en_acts_mean_norm': en_acts.norm(dim=-1).mean().item(),
        'zh_acts_mean_norm': zh_acts.norm(dim=-1).mean().item(),
    }
    print(f"  Mean diff norm: {mean_diff.norm().item():.4f}")

# Cosine similarities between directions
print("\n--- Cosine similarities between directions ---")
dir_keys = sorted(directions.keys())
for n1 in dir_keys:
    for n2 in dir_keys:
        if n1 < n2:
            cos = F.cosine_similarity(directions[n1].unsqueeze(0),
                                       directions[n2].unsqueeze(0)).item()
            print(f"  {n1} vs {n2}: {cos:.4f}")

# --- Phase 2: Get source activations for sweep ---
print("\n=== Phase 2: Loading source activations ===")
np.random.seed(SEED)
torch.manual_seed(SEED)

token_ids_list = load_fineweb_fixed_length(N_ACT, tokenizer, seq_len=SEQ_LEN, seed=SEED)
contexts, activations = extract_context_and_activation(
    model, tokenizer, token_ids_list, LAYER, device
)
contexts = contexts.to(device)
activations = activations.to(device)

# Reference logits and measure
with torch.no_grad():
    ref_hidden = torch.cat([contexts, activations.unsqueeze(1)], dim=1)
    ref_measure, ref_logits = forward_with_measure(model, ref_hidden, LAYER, MEASURE_LAYER)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

print(f"Source activations: {activations.shape}")

# --- Phase 3: Sweep ---
print("\n=== Phase 3: Perturbation sweeps ===")
angles_deg = np.linspace(3, 180, 60)
angles_rad = np.deg2rad(angles_deg)

sweep_results = {}

for name, direction in directions.items():
    print(f"Sweeping {name} direction...")
    kl, l2 = sweep_single_direction(
        model, contexts, activations, ref_measure, ref_log_probs,
        direction, angles_rad, LAYER, MEASURE_LAYER, device
    )
    sweep_results[name] = {'kl': kl, 'l2': l2}
    mean_kl = kl.mean(axis=0)
    print(f"  KL range: [{mean_kl.min():.3f}, {mean_kl.max():.3f}]")

# Random baseline
print("Sweeping random directions...")
rand_kl, rand_l2 = sweep_random_directions(
    model, contexts, activations, ref_measure, ref_log_probs,
    angles_rad, LAYER, MEASURE_LAYER, device, n_dirs=10
)
sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}

# --- Phase 4: Save ---
suffix = ""
if args.hq_only:
    suffix += "_hq_lq"
if N_PROMPTS is not None:
    suffix += f"_{N_PROMPTS}prompts"
if SEED != 42:
    suffix += f"_seed{SEED}"

outdir = f"results/feature_purity/{MODEL}"
os.makedirs(outdir, exist_ok=True)

results = {
    'directions': {k: v.cpu().float().numpy() for k, v in directions.items()},
    'direction_metadata': direction_metadata,
    'sweep_results': sweep_results,
    'angles_deg': angles_deg,
    'model': MODEL,
    'layer': LAYER,
    'measure_layer': MEASURE_LAYER,
    'n_act': N_ACT,
    'seq_len': SEQ_LEN,
    'n_prompts': N_PROMPTS,
    'hq_only': args.hq_only,
    'seed': SEED,
}
pkl_path = os.path.join(outdir, f'feature_purity_results{suffix}.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nResults saved to {pkl_path}")

# --- Phase 5: Plot ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dir_names = [n for n in ['HQ', 'MQ', 'LQ'] if n in sweep_results]
colors = {'HQ': 'green', 'MQ': 'orange', 'LQ': 'red'}
n_prompts_str = f", {N_PROMPTS} prompts" if N_PROMPTS else ""

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metric, label in [(axes[0], 'kl', 'KL divergence'), (axes[1], 'l2', 'L2 distance')]:
    for name in dir_names:
        vals = sweep_results[name][metric]
        mean = vals.mean(axis=0)
        sem = vals.std(axis=0) / np.sqrt(vals.shape[0])
        ax.plot(angles_deg, mean, color=colors[name], label=name, linewidth=2)
        ax.fill_between(angles_deg, mean - sem, mean + sem, color=colors[name], alpha=0.2)

    rand = sweep_results['random'][metric]
    rand_mean = rand.mean(axis=(0, 2)) if rand.ndim == 3 else rand.mean(axis=0)
    ax.plot(angles_deg, rand_mean, 'k--', label='random', linewidth=1, alpha=0.5)

    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel(label)
    ax.set_title(f'Feature purity: {label} ({MODEL}, L{LAYER}{n_prompts_str})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(outdir, f'feature_purity_sweep{suffix}.png')
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
