#!/usr/bin/env python3
"""Perturbation sweep along v7 optimized direction vs random. Measures both KL and L2."""

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
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--n_samples', type=int, default=20)
parser.add_argument('--n_random', type=int, default=5)
args = parser.parse_args()

with open(args.checkpoint, 'rb') as f:
    cp = pickle.load(f)

activations = cp['activations']
contexts = cp['contexts']
direction = cp['opt_state']['direction']
direction = direction / direction.norm()
run_args = cp['args']
print(f"Loaded checkpoint: step {cp['opt_state']['step']}, angle={run_args['angle']} deg", flush=True)

print('Loading model...', flush=True)
model, tokenizer, device = load_model(run_args['model'])

direction = direction.to(device)
angles = np.arange(0, 91, 5)
d_model = activations.shape[1]

print(f'Sweeping {len(angles)} angles, {args.n_samples} samples, {args.n_random} random dirs', flush=True)

results_opt_kl = {a: [] for a in angles}
results_rand_kl = {a: [] for a in angles}
results_opt_l2 = {a: [] for a in angles}
results_rand_l2 = {a: [] for a in angles}

for i in range(args.n_samples):
    a_i = activations[i].to(device)
    ctx_i = contexts[i].to(device)
    R = a_i.norm()
    a_hat = a_i / R

    with torch.no_grad():
        ref_final = forward_from_layer_to_final(model, ctx_i, a_i.reshape(1, 1, d_model), run_args['layer'])
        ref_logits = _get_lm_head(model)(_get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1))
        ref_lp = F.log_softmax(ref_logits.float(), dim=-1)
        ref_p = ref_lp.exp()

    for angle in angles:
        theta = np.deg2rad(float(angle))
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Optimized direction
        proj = (direction * a_hat).sum()
        v = direction - proj * a_hat
        v = v / (v.norm() + 1e-10)
        a_pert = cos_t * a_i + sin_t * R * v
        with torch.no_grad():
            pf = forward_from_layer_to_final(model, ctx_i, a_pert.reshape(1, 1, d_model), run_args['layer'])
            pl = _get_lm_head(model)(_get_final_norm(model)(pf.unsqueeze(1)).squeeze(1))
            plp = F.log_softmax(pl.float(), dim=-1)
            kl = (ref_p * (ref_lp - plp)).sum().item()
            l2 = (ref_final - pf).norm().item()
        results_opt_kl[angle].append(kl)
        results_opt_l2[angle].append(l2)

        # Random directions
        for _ in range(args.n_random):
            rd = torch.randn(d_model, device=device)
            rd = rd / rd.norm()
            proj_r = (rd * a_hat).sum()
            v_r = rd - proj_r * a_hat
            v_r = v_r / (v_r.norm() + 1e-10)
            a_pert_r = cos_t * a_i + sin_t * R * v_r
            with torch.no_grad():
                pf_r = forward_from_layer_to_final(model, ctx_i, a_pert_r.reshape(1, 1, d_model), run_args['layer'])
                pl_r = _get_lm_head(model)(_get_final_norm(model)(pf_r.unsqueeze(1)).squeeze(1))
                plp_r = F.log_softmax(pl_r.float(), dim=-1)
                kl_r = (ref_p * (ref_lp - plp_r)).sum().item()
                l2_r = (ref_final - pf_r).norm().item()
            results_rand_kl[angle].append(kl_r)
            results_rand_l2[angle].append(l2_r)

    if (i + 1) % 5 == 0:
        print(f'  {i+1}/{args.n_samples}', flush=True)

opt_kl_means = [np.mean(results_opt_kl[a]) for a in angles]
rand_kl_means = [np.mean(results_rand_kl[a]) for a in angles]
rand_kl_stds = [np.std(results_rand_kl[a]) for a in angles]
opt_l2_means = [np.mean(results_opt_l2[a]) for a in angles]
rand_l2_means = [np.mean(results_rand_l2[a]) for a in angles]
rand_l2_stds = [np.std(results_rand_l2[a]) for a in angles]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(angles, opt_kl_means, 'o-', color='C1', label='Optimized direction', linewidth=2)
ax1.plot(angles, rand_kl_means, 's-', color='gray', label='Random directions', linewidth=2)
ax1.fill_between(angles, np.array(rand_kl_means) - np.array(rand_kl_stds),
                np.array(rand_kl_means) + np.array(rand_kl_stds), alpha=0.2, color='gray')
ax1.set_xlabel('Perturbation angle (deg)')
ax1.set_ylabel('KL divergence')
ax1.set_title('KL divergence')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(angles, opt_l2_means, 'o-', color='C1', label='Optimized direction', linewidth=2)
ax2.plot(angles, rand_l2_means, 's-', color='gray', label='Random directions', linewidth=2)
ax2.fill_between(angles, np.array(rand_l2_means) - np.array(rand_l2_stds),
                np.array(rand_l2_means) + np.array(rand_l2_stds), alpha=0.2, color='gray')
ax2.set_xlabel('Perturbation angle (deg)')
ax2.set_ylabel('L2 distance (final layer)')
ax2.set_title('L2 distance')
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle(f'Perturbation sweep: min-Jacobian+L2>{run_args.get("l2_min",250)} direction vs random (layer {run_args["layer"]})', fontsize=12)
plt.tight_layout()
plt.savefig(args.output, dpi=150)
plt.close()

print(flush=True)
for a, ok, rk, ol, rl in zip(angles, opt_kl_means, rand_kl_means, opt_l2_means, rand_l2_means):
    print(f'  {a:3d} deg: opt_KL={ok:.4f}  rand_KL={rk:.4f}  opt_L2={ol:.1f}  rand_L2={rl:.1f}', flush=True)
print(f'Saved: {args.output}', flush=True)

# Save raw data
data_path = args.output.replace('.png', '_data.pkl')
with open(data_path, 'wb') as f:
    pickle.dump({
        'angles': angles,
        'opt_kl': results_opt_kl, 'rand_kl': results_rand_kl,
        'opt_l2': results_opt_l2, 'rand_l2': results_rand_l2,
    }, f)
print(f'Data saved: {data_path}', flush=True)
