#!/usr/bin/env python3
"""
Plateau scaling with composite feature directions.

Tests whether plateau size scales as √n when perturbing along the normalized
sum of n feature directions. Each feature only gets ~θ/√n perturbation, so
the model can independently correct each one, extending the plateau.

Uses 10 language feature directions from the all-pairwise contrastive experiment.
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='qwen3-0.6b')
parser.add_argument('--layer', type=int, default=14)
parser.add_argument('--measure-layer', type=int, default=None)
parser.add_argument('--n_act', type=int, default=50)
parser.add_argument('--seq_len', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--kl_threshold', type=float, default=0.5)
parser.add_argument('--max_combos', type=int, default=45)
parser.add_argument('--bfloat16', action='store_true')
parser.add_argument('--cross-family', action='store_true',
                    help='Only combine languages from different families')
parser.add_argument('--5lang', dest='five_lang', action='store_true',
                    help='Use 5-language (one per family) re-estimated features')
args = parser.parse_args()

MODEL = args.model
LAYER = args.layer
N_ACT = args.n_act
SEQ_LEN = args.seq_len
SEED = args.seed
KL_THRESHOLD = args.kl_threshold
MAX_COMBOS_PER_N = args.max_combos

# Language family mapping
LANG_FAMILY = {
    'zh': 'cjk', 'ja': 'cjk',
    'nl': 'germanic', 'de': 'germanic',
    'es': 'romance', 'fr': 'romance', 'it': 'romance', 'pt': 'romance',
    'ru': 'slavic',
    'ar': 'semitic',
}


def is_cross_family(combo, languages):
    """Check that no two languages in the combo share a family."""
    families = [LANG_FAMILY[languages[i]] for i in combo]
    return len(families) == len(set(families))


# --- Helper functions (from run_allpairs_contrastive.py) ---

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


def measure_plateau_size(mean_kl, angles_deg, threshold):
    """Find angle where mean KL first crosses threshold, using linear interpolation."""
    for i in range(len(mean_kl) - 1):
        if mean_kl[i] < threshold <= mean_kl[i + 1]:
            # Linear interpolation
            frac = (threshold - mean_kl[i]) / (mean_kl[i + 1] - mean_kl[i])
            return angles_deg[i] + frac * (angles_deg[i + 1] - angles_deg[i])
    # If never crossed, return max angle (right-censored)
    if mean_kl[-1] < threshold:
        return angles_deg[-1]
    # If already above threshold at first angle
    return angles_deg[0]


# ============================================================
# MAIN
# ============================================================

# Load feature directions
results_dir = f'results/allpairs_contrastive/{MODEL}'
if args.five_lang:
    print("Loading 5-language features...")
    with open(os.path.join(results_dir, '5lang_contrastive.pkl'), 'rb') as f:
        lang5_data = pickle.load(f)
    features = lang5_data['features_5']
    languages = lang5_data['subset']
else:
    print("Loading feature directions...")
    with open(os.path.join(results_dir, 'directions.pkl'), 'rb') as f:
        dir_data = pickle.load(f)
    features = dir_data['features']
    languages = dir_data['languages']
print(f"  {len(languages)} languages: {languages}")

# Load model and FineWeb activations
print("\nLoading model...")
load_dtype = torch.bfloat16 if args.bfloat16 else None
model, tokenizer, device = load_model(MODEL, dtype=load_dtype)
if args.bfloat16:
    print("  Loaded in bfloat16")
n_layers = len(_get_blocks(model))
MEASURE_LAYER = args.measure_layer if args.measure_layer is not None else n_layers - 2
print(f"  {n_layers} layers, measure_layer={MEASURE_LAYER}")

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

# === Generate composite directions and sweep ===
print("\n=== Generating composite directions and sweeping ===")
feature_vecs = [features[lang] for lang in languages]
k = len(languages)

all_results = {}  # n → list of {combo, kl, l2, plateau_size}
rng = random.Random(0)

max_n = k
if args.cross_family:
    n_families = len(set(LANG_FAMILY[l] for l in languages))
    max_n = n_families  # can't go beyond number of distinct families
    print(f"  Cross-family mode: {n_families} families, max n={max_n}")

for n in range(1, max_n + 1):
    # Generate combinations
    all_combos = list(itertools.combinations(range(k), n))
    if args.cross_family:
        all_combos = [c for c in all_combos if is_cross_family(c, languages)]
    if len(all_combos) <= MAX_COMBOS_PER_N:
        combos = all_combos
    else:
        combos = rng.sample(all_combos, MAX_COMBOS_PER_N)

    n_combos = len(combos)
    print(f"\nn={n}: {n_combos} combinations (of {len(all_combos)} total)")

    results_n = []
    for ci, combo in enumerate(combos):
        # Sum and normalize
        d = sum(feature_vecs[i] for i in combo)
        d_norm = d.norm().item()
        d = F.normalize(d.unsqueeze(0), dim=-1).squeeze(0)

        # Sweep
        kl, l2 = sweep_single_direction(
            model, contexts, activations, ref_measure, ref_log_probs,
            d, angles_rad, LAYER, MEASURE_LAYER, device)

        # Measure plateau size
        mean_kl = kl.mean(axis=0)
        ps = measure_plateau_size(mean_kl, angles_deg, KL_THRESHOLD)

        combo_langs = tuple(languages[i] for i in combo)
        results_n.append({
            'combo': combo,
            'combo_langs': combo_langs,
            'kl': kl,
            'l2': l2,
            'plateau_size': ps,
            'sum_norm': d_norm,
        })

        if ci % 10 == 0 or ci == n_combos - 1:
            print(f"  [{ci+1}/{n_combos}] {combo_langs}: plateau={ps:.1f}°, ||sum||={d_norm:.2f}")

    all_results[n] = results_n

    # Print summary for this n
    plateau_sizes = [r['plateau_size'] for r in results_n]
    print(f"  n={n} summary: plateau size = {np.mean(plateau_sizes):.1f}° ± {np.std(plateau_sizes):.1f}°")

    # Checkpoint
    ckpt = {
        'all_results': all_results,
        'angles_deg': angles_deg,
        'threshold': KL_THRESHOLD,
        'layer': LAYER,
        'measure_layer': MEASURE_LAYER,
        'languages': languages,
    }
    ckpt_path = os.path.join(results_dir, 'plateau_scaling_checkpoint.pkl')
    with open(ckpt_path, 'wb') as f:
        pickle.dump(ckpt, f)

# === Summary ===
print("\n=== Summary: Plateau size vs n ===")
n_values = []
mean_plateaus = []
std_plateaus = []
for n in range(1, max_n + 1):
    sizes = [r['plateau_size'] for r in all_results[n]]
    m, s = np.mean(sizes), np.std(sizes)
    n_values.append(n)
    mean_plateaus.append(m)
    std_plateaus.append(s)
    print(f"  n={n}: {m:.1f}° ± {s:.1f}° ({len(sizes)} combos)")

n_values = np.array(n_values)
mean_plateaus = np.array(mean_plateaus)
std_plateaus = np.array(std_plateaus)

# === Save final results ===
print("\n=== Saving ===")
save_data = {
    'all_results': all_results,
    'angles_deg': angles_deg,
    'threshold': KL_THRESHOLD,
    'n_values': n_values,
    'mean_plateaus': mean_plateaus,
    'std_plateaus': std_plateaus,
    'layer': LAYER,
    'measure_layer': MEASURE_LAYER,
    'languages': languages,
    'cross_family': args.cross_family,
    'five_lang': args.five_lang,
}
suffix = '_5lang' if args.five_lang else ('_crossfamily' if args.cross_family else '')
with open(os.path.join(results_dir, f'plateau_scaling{suffix}.pkl'), 'wb') as f:
    pickle.dump(save_data, f)
print(f"  Saved: {results_dir}/plateau_scaling{suffix}.pkl")

# === Plot ===
print("\n=== Generating plots ===")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mask = angles_deg <= 60

# --- Plot 1: KL curves by n ---
fig, ax = plt.subplots(figsize=(8, 5))
cmap = plt.cm.viridis
for n in range(1, max_n + 1):
    # Average KL across all combos for this n
    all_kl = np.stack([r['kl'] for r in all_results[n]])  # [n_combos, 50, 60]
    mean_kl = all_kl.mean(axis=(0, 1))  # [60]
    color = cmap((n - 1) / max(max_n - 1, 1))
    ax.plot(angles_deg[mask], mean_kl[mask], color=color, linewidth=1.5, label=f'n={n}')

# Random baseline (load from previous experiment if available)
try:
    rand_file = 'sweep_results.pkl'
    with open(os.path.join(results_dir, rand_file), 'rb') as f:
        prev_data = pickle.load(f)
    rand_kl = prev_data['sweep_results']['random']['kl']
    # Only plot if angles match
    prev_angles = prev_data.get('angles_deg', None)
    if prev_angles is not None and len(prev_angles) == rand_kl.shape[1]:
        prev_mask = prev_angles <= 60
        rand_kl_avg = rand_kl[:, prev_mask, :].mean(axis=2)
        ax.plot(prev_angles[prev_mask], rand_kl_avg.mean(axis=0), '--', color='gray',
                label='random', linewidth=1.5)
except Exception:
    pass  # skip random baseline if file not found

ax.axhline(y=KL_THRESHOLD, color='red', linestyle=':', alpha=0.5, label=f'threshold={KL_THRESHOLD}')
ax.set_xlabel('Perturbation angle (°)')
ax.set_ylabel('Mean KL divergence')
ax.set_title(f'KL curves by number of summed features\n{MODEL}, layer {LAYER}')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(results_dir, f'scaling_kl_curves{suffix}.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {results_dir}/scaling_kl_curves{suffix}.png")

# --- Plot 2: Plateau size vs n (two panels: linear and log-log) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: linear scale
for n in range(1, max_n + 1):
    sizes = [r['plateau_size'] for r in all_results[n]]
    jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(sizes))
    ax1.scatter(np.full(len(sizes), n) + jitter, sizes,
                s=8, alpha=0.3, color='steelblue', zorder=1)

ax1.errorbar(n_values, mean_plateaus, yerr=std_plateaus,
             fmt='o-', color='navy', markersize=6, linewidth=2, capsize=3,
             zorder=2, label='mean ± std')

n_fine = np.linspace(1, max_n, 100)
ax1.plot(n_fine, mean_plateaus[0] * np.sqrt(n_fine), ':', color='green', linewidth=1.5,
         label=f'√n prediction: {mean_plateaus[0]:.1f} × √n')

ax1.set_xlabel('Number of features summed (n)')
ax1.set_ylabel(f'Plateau size (° at KL > {KL_THRESHOLD})')
ax1.set_title('Plateau size vs n')
ax1.set_xticks(range(1, max_n + 1))
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Right: log-log scale
ax2.errorbar(n_values, mean_plateaus, yerr=std_plateaus,
             fmt='o-', color='navy', markersize=6, linewidth=2, capsize=3,
             label='mean ± std')
ax2.plot(n_fine, mean_plateaus[0] * np.sqrt(n_fine), ':', color='green', linewidth=1.5,
         label='slope = 0.5')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Number of features summed (n)')
ax2.set_ylabel(f'Plateau size (° at KL > {KL_THRESHOLD})')
ax2.set_title('Log-log: plateau size vs n')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

label_5lang = ' (5-lang features)' if args.five_lang else ''
fig.suptitle(f'Plateau scaling{label_5lang}\n'
             f'{MODEL}, layer {LAYER}, threshold KL > {KL_THRESHOLD}', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(results_dir, f'plateau_scaling{suffix}.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {results_dir}/plateau_scaling{suffix}.png")

print("\nDone!")
