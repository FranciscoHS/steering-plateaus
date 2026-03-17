#!/usr/bin/env python3
"""Quick perturbation sweep along the v6 optimized direction vs random."""

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
print(f"Loaded checkpoint: step {cp['opt_state']['step']}, angle={run_args['angle']} deg")

print('Loading model...', flush=True)
model, tokenizer, device = load_model(run_args['model'])

direction = direction.to(device)
angles = np.arange(0, 91, 5)
d_model = activations.shape[1]

print(f'Sweeping {len(angles)} angles, {args.n_samples} samples, {args.n_random} random dirs', flush=True)

kl_opt = {a: [] for a in angles}
kl_rand = {a: [] for a in angles}
l2_opt = {a: [] for a in angles}
l2_rand = {a: [] for a in angles}
cd_opt = {a: [] for a in angles}
cd_rand = {a: [] for a in angles}

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
            cd = 1.0 - F.cosine_similarity(ref_final.float(), pf.float(), dim=-1).item()
        kl_opt[angle].append(kl)
        l2_opt[angle].append(l2)
        cd_opt[angle].append(cd)

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
                cd_r = 1.0 - F.cosine_similarity(ref_final.float(), pf_r.float(), dim=-1).item()
            kl_rand[angle].append(kl_r)
            l2_rand[angle].append(l2_r)
            cd_rand[angle].append(cd_r)

    if (i + 1) % 5 == 0:
        print(f'  {i+1}/{args.n_samples}', flush=True)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# KL subplot
kl_opt_means = [np.mean(kl_opt[a]) for a in angles]
kl_rand_means = [np.mean(kl_rand[a]) for a in angles]
kl_rand_stds = [np.std(kl_rand[a]) for a in angles]
ax1.plot(angles, kl_opt_means, 'o-', color='C1', label='Optimized direction', linewidth=2)
ax1.plot(angles, kl_rand_means, 's-', color='gray', label='Random directions', linewidth=2)
ax1.fill_between(angles, np.array(kl_rand_means) - np.array(kl_rand_stds),
                 np.array(kl_rand_means) + np.array(kl_rand_stds), alpha=0.2, color='gray')
ax1.set_xlabel('Perturbation angle (deg)')
ax1.set_ylabel('KL divergence')
ax1.set_title('KL divergence')
ax1.legend()
ax1.grid(True, alpha=0.3)

# L2 subplot
l2_opt_means = [np.mean(l2_opt[a]) for a in angles]
l2_rand_means = [np.mean(l2_rand[a]) for a in angles]
l2_rand_stds = [np.std(l2_rand[a]) for a in angles]
ax2.plot(angles, l2_opt_means, 'o-', color='C1', label='Optimized direction', linewidth=2)
ax2.plot(angles, l2_rand_means, 's-', color='gray', label='Random directions', linewidth=2)
ax2.fill_between(angles, np.array(l2_rand_means) - np.array(l2_rand_stds),
                 np.array(l2_rand_means) + np.array(l2_rand_stds), alpha=0.2, color='gray')
ax2.set_xlabel('Perturbation angle (deg)')
ax2.set_ylabel('L2 distance (final layer)')
ax2.set_title('L2 distance')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Cosine distance subplot
cd_opt_means = [np.mean(cd_opt[a]) for a in angles]
cd_rand_means = [np.mean(cd_rand[a]) for a in angles]
cd_rand_stds = [np.std(cd_rand[a]) for a in angles]
ax3.plot(angles, cd_opt_means, 'o-', color='C1', label='Optimized direction', linewidth=2)
ax3.plot(angles, cd_rand_means, 's-', color='gray', label='Random directions', linewidth=2)
ax3.fill_between(angles, np.array(cd_rand_means) - np.array(cd_rand_stds),
                 np.array(cd_rand_means) + np.array(cd_rand_stds), alpha=0.2, color='gray')
ax3.set_xlabel('Perturbation angle (deg)')
ax3.set_ylabel('Cosine distance (1=orthogonal)')
ax3.set_title('Cosine distance')
ax3.legend()
ax3.grid(True, alpha=0.3)

fig.suptitle(f'Perturbation sweep: min-Jacobian direction vs random (optimized at {run_args["angle"]} deg)', fontsize=12)
plt.tight_layout()
plt.savefig(args.output, dpi=150)
plt.close()

print(flush=True)
for a, kl_o, kl_r, l2_o, l2_r, cd_o, cd_r in zip(angles, kl_opt_means, kl_rand_means, l2_opt_means, l2_rand_means, cd_opt_means, cd_rand_means):
    print(f'  {a:3d} deg: KL opt={kl_o:.4f} rand={kl_r:.4f}  |  L2 opt={l2_o:.1f} rand={l2_r:.1f}  |  cos_dist opt={cd_o:.4f} rand={cd_r:.4f}', flush=True)
print(f'Saved: {args.output}', flush=True)
