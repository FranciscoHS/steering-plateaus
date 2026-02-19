#!/usr/bin/env python3
"""
Re-run Qwen3-0.6B plateau-breaking sweeps with L2 measured at layer 26
(2nd-to-last of 28 layers), matching the GPT-2 XL convention.

Produces a two-panel plot:
  Left:  KL divergence (from final logits)
  Right: L2 distance at layer 26
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

MODEL = 'qwen3-0.6b'
LAYER = 14
MEASURE_LAYER = 26  # 2nd-to-last of 28 layers (0-27)

results_dir = 'results/scaling_test/qwen3-0.6b'


# --- Helper functions ---

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
    """Forward pass capturing hidden state at measure_layer AND producing logits.

    Single forward pass through blocks [start_layer+1 .. end], capturing the
    intermediate residual stream at measure_layer for L2 computation.

    Returns:
        measure_hidden: [N, d_model] - last-token hidden after measure_layer block
        logits: [N, vocab_size] - output logits from final layer
    """
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
            l2_dists[:, ang_idx] = l2.cpu().numpy()
            kl_divs[:, ang_idx] = kl.cpu().numpy()
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

    # Pre-compute orthogonal random directions
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


# === Main ===

print("Loading model...")
model, tokenizer, device = load_model(MODEL)

# Load existing directions
print("Loading directions from pickle files...")
with open(os.path.join(results_dir, 'scaling_results.pkl'), 'rb') as f:
    scaling = pickle.load(f)
with open(os.path.join(results_dir, 'validation_results.pkl'), 'rb') as f:
    val = pickle.load(f)
with open('results/multi_contrastive/qwen3-0.6b/optimization_results.pkl', 'rb') as f:
    old_mc = pickle.load(f)
with open(os.path.join(results_dir, 'plateau_sweeps.pkl'), 'rb') as f:
    plateau = pickle.load(f)

ref_mean = scaling['ref_direction']
svd_k10 = val['svd_k10']
svd_k3 = plateau['svd_k3']

neg_dirs = {
    '-k=10 mean': -ref_mean,
    '-k=10 SVD': -svd_k10,
    '-d_EM (old)': -old_mc['directions']['d_EM'],
    '-f_E mean (old k=3)': -old_mc['directions']['f_E'],
    '-f_E SVD (k=3)': -svd_k3,
    '+MELBO_EM': old_mc['directions']['MELBO_EM'],
}

# === Load FineWeb activations (same seed/params as original) ===
print("\nLoading FineWeb activations...")
token_ids_list = load_fineweb_fixed_length(50, tokenizer, seq_len=5, seed=42)
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

# === Run negative direction sweeps ===
print(f"\nRunning negative-direction sweeps (L2 at layer {MEASURE_LAYER})...")
sweep_results = {}
for name, direction in neg_dirs.items():
    print(f"  Sweeping {name}...")
    kl, l2 = sweep_single_direction(
        model, contexts, activations, ref_measure, ref_log_probs,
        direction, angles_rad, LAYER, MEASURE_LAYER, device)
    sweep_results[name] = {'kl': kl, 'l2': l2}
    peak_kl = kl.mean(axis=0).max()
    peak_angle = angles_deg[kl.mean(axis=0).argmax()]
    print(f"    peak KL = {peak_kl:.3f} at {peak_angle:.0f}°")

# === Run random direction sweeps ===
print("\nRunning random direction sweeps (10 dirs per activation)...")
rand_kl, rand_l2 = sweep_random_directions(
    model, contexts, activations, ref_measure, ref_log_probs,
    angles_rad, LAYER, MEASURE_LAYER, device, n_dirs=10)
sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}
print(f"  random KL shape: {rand_kl.shape}")

# === Save ===
save_data = {
    'sweep_results': sweep_results,
    'angles_deg': angles_deg,
    'layer': LAYER,
    'measure_layer': MEASURE_LAYER,
    'svd_k3': svd_k3,
}
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'plateau_sweeps_v2.pkl'), 'wb') as f:
    pickle.dump(save_data, f)
print(f"\nSaved: {results_dir}/plateau_sweeps_v2.pkl")

# === Plot ===
print("\nGenerating plot...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = {
    '-k=10 mean': 'C0',
    '-k=10 SVD': 'C1',
    '-d_EM (old)': 'C2',
    '-f_E mean (old k=3)': 'C3',
    '-f_E SVD (k=3)': 'C4',
    '+MELBO_EM': 'C5',
}
styles = {
    '-f_E mean (old k=3)': '--',
    '-f_E SVD (k=3)': '--',
}

mask = angles_deg <= 60

# Named directions
for name in neg_dirs:
    kl = sweep_results[name]['kl'][:, mask]
    l2 = sweep_results[name]['l2'][:, mask]
    style = styles.get(name, '-')
    c = colors[name]
    ax1.plot(angles_deg[mask], kl.mean(axis=0), style, color=c, label=name, linewidth=1.5)
    ax2.plot(angles_deg[mask], l2.mean(axis=0), style, color=c, label=name, linewidth=1.5)

# Random: average over dirs first, then plot mean ± std over activations
rand_kl_avg = rand_kl[:, mask, :].mean(axis=2)  # [50, n_angles]
rand_l2_avg = rand_l2[:, mask, :].mean(axis=2)
ax1.plot(angles_deg[mask], rand_kl_avg.mean(axis=0), '-', color='gray',
         label='random (10 avg ± std)', linewidth=1.5)
ax1.fill_between(angles_deg[mask],
                 rand_kl_avg.mean(axis=0) - rand_kl_avg.std(axis=0),
                 rand_kl_avg.mean(axis=0) + rand_kl_avg.std(axis=0),
                 color='gray', alpha=0.15)
ax2.plot(angles_deg[mask], rand_l2_avg.mean(axis=0), '-', color='gray',
         label='random (10 avg ± std)', linewidth=1.5)
ax2.fill_between(angles_deg[mask],
                 rand_l2_avg.mean(axis=0) - rand_l2_avg.std(axis=0),
                 rand_l2_avg.mean(axis=0) + rand_l2_avg.std(axis=0),
                 color='gray', alpha=0.15)

ax1.set_xlabel('Perturbation angle (°)')
ax1.set_ylabel('Mean KL divergence')
ax1.set_title('KL divergence (from final logits)')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Perturbation angle (°)')
ax2.set_ylabel(f'Mean L2 distance')
ax2.set_title(f'L2 distance (layer {MEASURE_LAYER})')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

fig.suptitle(f'Plateau-breaking directions: perturbation sweeps\n'
             f'Qwen3-0.6B, steer layer {LAYER}, measure L2 at layer {MEASURE_LAYER}',
             fontsize=12)
fig.tight_layout()

plot_path = os.path.join(results_dir, 'plateau_breaking_sweeps.png')
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved: {plot_path}")
print("Done!")
