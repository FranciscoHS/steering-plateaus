#!/usr/bin/env python3
"""Perturbation sweep: avg-of-per-act direction vs joint-100 direction vs random.

Loads the per-activation optimization results and the v9 checkpoint (for activations),
sweeps across angles measuring KL divergence, L2 distance, and cosine distance.
Produces 3 separate plots (one per metric).

Usage:
    python -u scripts/jacobian/sweep_per_act_vs_joint.py \
        --per_act_results results/delta_opt_v9_per_act/results.pkl \
        --v9_checkpoint results/delta_opt_v9/checkpoint.pkl \
        --output_dir results/delta_opt_v9_per_act
"""

import pickle
import sys
import os
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_to_final,
    _get_final_norm, _get_lm_head,
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--per_act_results', type=str, required=True)
parser.add_argument('--v9_checkpoint', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--n_samples', type=int, default=30,
                    help='Number of activations to sweep (subsample if < 100)')
parser.add_argument('--n_random', type=int, default=10,
                    help='Random perturbation directions per activation')
parser.add_argument('--max_angle', type=float, default=60)
parser.add_argument('--n_angles', type=int, default=30)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Load data
print('Loading per-activation results...', flush=True)
with open(args.per_act_results, 'rb') as f:
    per_act = pickle.load(f)

avg_direction = per_act['avg_direction']
joint_direction = per_act['joint_direction']

print('Loading v9 checkpoint...', flush=True)
with open(args.v9_checkpoint, 'rb') as f:
    v9 = pickle.load(f)

activations = v9['activations']
contexts = v9['contexts']
v9_args = v9['args']
layer = v9_args['layer']
model_name = v9_args['model']
n_total = activations.shape[0]
d_model = activations.shape[1]

print(f'  {n_total} activations, d_model={d_model}')
print(f'  model={model_name}, layer={layer}')

# Subsample if needed
n_samples = min(args.n_samples, n_total)
if n_samples < n_total:
    np.random.seed(42)
    idx = np.random.choice(n_total, n_samples, replace=False)
    idx.sort()
else:
    idx = np.arange(n_total)

print(f'Loading model: {model_name}', flush=True)
model, tokenizer, device = load_model(model_name)
model_dtype = next(model.parameters()).dtype

# Normalize directions
avg_dir = F.normalize(avg_direction.to(device, dtype=model_dtype), dim=-1)
joint_dir = F.normalize(joint_direction.to(device, dtype=model_dtype), dim=-1)

angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
angles_rad = np.deg2rad(angles_deg)
n_angles = len(angles_deg)

print(f'Sweeping {n_angles} angles, {n_samples} activations, {args.n_random} random dirs', flush=True)

# Results: direction_name -> {metric -> [n_angles, n_samples_or_more]}
# For avg and joint: one measurement per activation per angle
# For random: n_random measurements per activation per angle
direction_names = ['Avg of per-act', 'Joint-100', 'Random']
# Store as lists of per-angle arrays, then stack
results = {name: {'kl': [[] for _ in range(n_angles)],
                  'l2': [[] for _ in range(n_angles)],
                  'cos': [[] for _ in range(n_angles)]}
           for name in direction_names}


def perturb_and_measure(a_i, a_hat, R, ctx_i, direction, theta, ref_final, ref_probs, ref_log_probs):
    """Perturb activation along direction at angle theta, return (kl, l2, cos_dist)."""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    proj = (direction * a_hat).sum()
    v = direction - proj * a_hat
    v = v / (v.norm() + 1e-10)
    a_pert = cos_t * a_i + sin_t * R * v
    with torch.no_grad():
        pf = forward_from_layer_to_final(model, ctx_i, a_pert.reshape(1, 1, d_model), layer)
        # L2
        l2 = (pf - ref_final).float().norm().item()
        # Cosine distance
        cos_sim = F.cosine_similarity(pf.float(), ref_final.float(), dim=-1).item()
        cos_dist = 1 - cos_sim
        # KL
        pl = _get_lm_head(model)(_get_final_norm(model)(pf.unsqueeze(1)).squeeze(1))
        plp = F.log_softmax(pl.float(), dim=-1)
        kl = (ref_probs * (ref_log_probs - plp)).sum().item()
    return kl, l2, cos_dist


for si, i in enumerate(idx):
    a_i = activations[i].to(device, dtype=model_dtype)
    ctx_i = contexts[i].to(device, dtype=model_dtype)
    R = a_i.norm()
    a_hat = a_i / R

    # Reference (unperturbed)
    with torch.no_grad():
        ref_final = forward_from_layer_to_final(model, ctx_i, a_i.reshape(1, 1, d_model), layer)
        ref_logits = _get_lm_head(model)(_get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1))
        ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
        ref_probs = ref_log_probs.exp()

    for ang_idx, theta in enumerate(angles_rad):
        # Avg direction
        kl, l2, cos_d = perturb_and_measure(a_i, a_hat, R, ctx_i, avg_dir, theta,
                                             ref_final, ref_probs, ref_log_probs)
        results['Avg of per-act']['kl'][ang_idx].append(kl)
        results['Avg of per-act']['l2'][ang_idx].append(l2)
        results['Avg of per-act']['cos'][ang_idx].append(cos_d)

        # Joint direction
        kl, l2, cos_d = perturb_and_measure(a_i, a_hat, R, ctx_i, joint_dir, theta,
                                             ref_final, ref_probs, ref_log_probs)
        results['Joint-100']['kl'][ang_idx].append(kl)
        results['Joint-100']['l2'][ang_idx].append(l2)
        results['Joint-100']['cos'][ang_idx].append(cos_d)

        # Random directions
        for _ in range(args.n_random):
            rd = F.normalize(torch.randn(d_model, device=device, dtype=model_dtype), dim=-1)
            kl, l2, cos_d = perturb_and_measure(a_i, a_hat, R, ctx_i, rd, theta,
                                                 ref_final, ref_probs, ref_log_probs)
            results['Random']['kl'][ang_idx].append(kl)
            results['Random']['l2'][ang_idx].append(l2)
            results['Random']['cos'][ang_idx].append(cos_d)

    if (si + 1) % 5 == 0:
        print(f'  {si+1}/{n_samples}', flush=True)

# Convert to arrays
for name in direction_names:
    for metric in ['kl', 'l2', 'cos']:
        results[name][metric] = [np.array(vals) for vals in results[name][metric]]

# Save raw data
data_path = os.path.join(args.output_dir, 'sweep_data.pkl')
with open(data_path, 'wb') as f:
    pickle.dump({
        'angles_deg': angles_deg,
        'results': results,
        'idx': idx,
        'args': vars(args),
    }, f)
print(f'Data saved: {data_path}', flush=True)

# Plot
colors = {'Avg of per-act': 'C1', 'Joint-100': 'C0', 'Random': 'gray'}

for metric, ylabel, fname in [
    ('kl', 'KL divergence (nats)', 'sweep_kl.png'),
    ('l2', 'L2 distance (final layer)', 'sweep_l2.png'),
    ('cos', 'Cosine distance (final layer)', 'sweep_cos.png'),
]:
    fig, ax = plt.subplots(figsize=(10, 6))

    for name in direction_names:
        data = results[name][metric]  # list of arrays, one per angle
        medians = [np.median(d) for d in data]
        q25 = [np.percentile(d, 25) for d in data]
        q75 = [np.percentile(d, 75) for d in data]
        ax.plot(angles_deg, medians, '-', color=colors[name],
                label=name, linewidth=2)
        ax.fill_between(angles_deg, q25, q75, color=colors[name], alpha=0.15)

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Perturbation sweep: avg-of-per-act vs joint vs random\n'
                 f'{model_name}, layer {layer}, {n_samples} activations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, args.max_angle)

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, fname)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fig_path}', flush=True)

print('Done.', flush=True)
