#!/usr/bin/env python3
"""
Feature purity experiment: English → Portuguese edition.

Extracts English→Portuguese contrastive directions from three prompt sets
of varying quality (HQ, MQ, LQ), then sweeps steering magnitude and
measures KL divergence to compare plateau shapes.

- HQ: Hand-crafted European Portuguese translations
- MQ: Mistral-7B translations, formatting cleaned up, translation errors kept
- LQ: Mistral-7B raw output (stray quotes, meta-text, errors)

All three sets share the same 28 English prompts.
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

from data.bilingual_prompts_pt_hq import PROMPT_PAIRS as HQ_PAIRS
from data.bilingual_prompts_pt_mq import PROMPT_PAIRS as MQ_PAIRS
from data.bilingual_prompts_pt_lq import PROMPT_PAIRS as LQ_PAIRS

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='qwen3-8b')
parser.add_argument('--layer', type=int, default=None,
                    help='Layer for direction extraction (default: 6)')
parser.add_argument('--measure-layer', type=int, default=None,
                    help='Layer for L2 measurement (default: n_layers - 2)')
parser.add_argument('--n_act', type=int, default=50)
parser.add_argument('--seq_len', type=int, default=5)
parser.add_argument('--n_prompts', type=int, default=5,
                    help='Number of prompt pairs per subsample (default: 5)')
parser.add_argument('--n_samples', type=int, default=10,
                    help='Number of random subsamples to draw (default: 10)')
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
N_PROMPTS = args.n_prompts
N_SAMPLES = args.n_samples
SEED = args.seed

# --- Helper functions (from run_feature_purity.py) ---

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

# --- Phase 1: Get all activations for each quality level ---
print("\n=== Phase 1: Extracting activations for all prompts ===")

all_pairs = {'HQ': HQ_PAIRS, 'LQ': LQ_PAIRS}
if not args.hq_only:
    all_pairs['MQ'] = MQ_PAIRS

# Pre-compute activations for all prompts in each set
all_acts = {}
for name, pairs in all_pairs.items():
    en_prompts = [p[0] for p in pairs]
    pt_prompts = [p[1] for p in pairs]
    print(f"{name}: {len(pairs)} pairs")
    en_acts = get_acts(model, tokenizer, device, en_prompts, LAYER)
    pt_acts = get_acts(model, tokenizer, device, pt_prompts, LAYER)
    all_acts[name] = {'en': en_acts, 'pt': pt_acts}

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

# --- Phase 3: Subsample and sweep ---
print(f"\n=== Phase 3: {N_SAMPLES} subsamples of {N_PROMPTS} prompts, perturbation sweeps ===")
angles_deg = np.linspace(3, 60, 20)
angles_rad = np.deg2rad(angles_deg)

rng = np.random.RandomState(SEED)
n_total = len(HQ_PAIRS)

# sweep_results[name] = list of N_SAMPLES dicts, each with 'kl' and 'l2'
sweep_results = {}
direction_data = {}  # store directions and metadata per sample

for sample_idx in range(N_SAMPLES):
    idx = rng.choice(n_total, N_PROMPTS, replace=False)
    idx_sorted = sorted(idx)
    print(f"\n--- Sample {sample_idx+1}/{N_SAMPLES}: prompts {idx_sorted} ---")

    for name in all_acts:
        en_acts = all_acts[name]['en'][idx_sorted]
        pt_acts = all_acts[name]['pt'][idx_sorted]

        diff = pt_acts - en_acts
        mean_diff = diff.mean(dim=0)
        direction = F.normalize(mean_diff, dim=0)

        # Sweep
        kl, l2 = sweep_single_direction(
            model, contexts, activations, ref_measure, ref_log_probs,
            direction, angles_rad, LAYER, MEASURE_LAYER, device
        )
        mean_kl = kl.mean(axis=0)
        print(f"  {name}: mean_diff_norm={mean_diff.norm().item():.4f}, "
              f"KL range=[{mean_kl.min():.3f}, {mean_kl.max():.3f}]")

        if name not in sweep_results:
            sweep_results[name] = []
            direction_data[name] = []
        sweep_results[name].append({'kl': kl, 'l2': l2})
        direction_data[name].append({
            'direction': direction.cpu().float().numpy(),
            'mean_diff_norm': mean_diff.norm().item(),
            'prompt_indices': idx_sorted,
        })

# Random baseline (once)
print("\nSweeping random directions...")
rand_kl, rand_l2 = sweep_random_directions(
    model, contexts, activations, ref_measure, ref_log_probs,
    angles_rad, LAYER, MEASURE_LAYER, device, n_dirs=10
)
sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}

# --- Phase 4: Save ---
suffix = f"_pt_{N_PROMPTS}x{N_SAMPLES}"
if args.hq_only:
    suffix += "_hq_lq"
if SEED != 42:
    suffix += f"_seed{SEED}"

outdir = f"results/feature_purity/{MODEL}"
os.makedirs(outdir, exist_ok=True)

results = {
    'sweep_results': sweep_results,
    'direction_data': direction_data,
    'angles_deg': angles_deg,
    'model': MODEL,
    'layer': LAYER,
    'measure_layer': MEASURE_LAYER,
    'n_act': N_ACT,
    'seq_len': SEQ_LEN,
    'n_prompts': N_PROMPTS,
    'n_samples': N_SAMPLES,
    'hq_only': args.hq_only,
    'seed': SEED,
    'language_pair': 'en-pt',
}
pkl_path = os.path.join(outdir, f'feature_purity_results{suffix}.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nResults saved to {pkl_path}")

# --- Phase 5: Plot ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

quality_names = [n for n in ['HQ', 'MQ', 'LQ'] if n in sweep_results]
colors = {'HQ': 'green', 'MQ': 'orange', 'LQ': 'red'}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metric, label in [(axes[0], 'kl', 'KL divergence'), (axes[1], 'l2', 'L2 distance')]:
    for name in quality_names:
        # Stack all samples: each is (N_ACT, n_angles), average over activations first
        # to get one curve per sample, then plot mean ± sem over samples
        curves = []
        for s in sweep_results[name]:
            curves.append(s[metric].mean(axis=0))  # mean over activations
        curves = np.array(curves)  # (N_SAMPLES, n_angles)
        mean = curves.mean(axis=0)
        sem = curves.std(axis=0) / np.sqrt(curves.shape[0])
        ax.plot(angles_deg, mean, color=colors[name], label=name, linewidth=2)
        ax.fill_between(angles_deg, mean - sem, mean + sem, color=colors[name], alpha=0.2)
        # Also plot individual sample curves faintly
        for c in curves:
            ax.plot(angles_deg, c, color=colors[name], linewidth=0.7, alpha=0.45)

    rand = sweep_results['random'][metric]
    rand_mean = rand.mean(axis=(0, 2)) if rand.ndim == 3 else rand.mean(axis=0)
    ax.plot(angles_deg, rand_mean, 'k--', label='random', linewidth=1, alpha=0.5)

    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel(label)
    ax.set_title(f'Feature purity (EN→PT): {label}\n({MODEL}, L{LAYER}, {N_PROMPTS}×{N_SAMPLES})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(outdir, f'feature_purity_sweep{suffix}.png')
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
