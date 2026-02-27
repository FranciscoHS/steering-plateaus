#!/usr/bin/env python3
"""Compare MELBO direction vs matched probe direction in perturbation sweep."""
import sys, os, pickle
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model
from src.data import generate_chess_games

# Load MELBO results
with open('results/capped_melbo/chess_16L_layer11_target0.5/melbo_results.pkl', 'rb') as f:
    melbo = pickle.load(f)

# Load probe
probe_ckpt = torch.load('results/chess_probes/piece_probe_16L_layer11.pth',
                         map_location='cpu', weights_only=False)
probe = probe_ckpt['linear_probe']
probe_dirs = probe[0].permute(1, 2, 3, 0).reshape(-1, probe.shape[1])

# MELBO_8 and matched probe (N on h7, index 720)
melbo_dir = melbo['directions'][7]
probe_dir = probe_dirs[720]
print(f"Cosine(MELBO_8, N-on-h7): {F.cosine_similarity(melbo_dir.unsqueeze(0), probe_dir.unsqueeze(0)).item():.4f}")

# Load model
model, tokenizer, device = load_model('chess_16L')
layer = 11
blocks = list(model.transformer.h)
ln_f = model.transformer.ln_f
lm_head = model.lm_head

# Generate same data
token_ids_list = generate_chess_games(50, 50, seed=42)

# Extract activations
all_ctx, all_act = [], []
for i in range(0, 50, 64):
    bt = torch.stack(token_ids_list[i:i+64]).to(device)
    with torch.no_grad():
        out = model(bt, output_hidden_states=True)
        h = out.hidden_states[layer + 1]
    all_ctx.append(h[:, :-1, :].float().cpu())
    all_act.append(h[:, -1, :].float().cpu())
contexts = torch.cat(all_ctx, 0)
activations = torch.cat(all_act, 0)

angles_deg = np.linspace(0, 180, 61)[1:]
angles_rad = np.deg2rad(angles_deg)
n_act = activations.shape[0]


def get_ref(ctx, a):
    with torch.no_grad():
        hidden = torch.cat([ctx, a.unsqueeze(0).unsqueeze(0)], dim=1)
        for block in blocks[layer+1:]:
            hidden = block(hidden)[0]
        logits = lm_head(ln_f(hidden[:, -1, :]))
        return F.log_softmax(logits.float(), dim=-1)


def get_kl(ctx, a_pert, ref_lp):
    with torch.no_grad():
        hidden = torch.cat([ctx, a_pert.unsqueeze(0).unsqueeze(0)], dim=1)
        for block in blocks[layer+1:]:
            hidden = block(hidden)[0]
        logits = lm_head(ln_f(hidden[:, -1, :]))
        pert_lp = F.log_softmax(logits.float(), dim=-1)
        return (ref_lp.exp() * (ref_lp - pert_lp)).sum().item()


def sweep_dir(direction):
    d = F.normalize(direction.to(device).float(), dim=-1)
    kl_all = np.zeros((n_act, len(angles_rad)))
    for i in range(n_act):
        ctx = contexts[i:i+1].to(device)
        a = activations[i].to(device)
        a_norm = a.norm()
        a_unit = a / a_norm
        d_orth = d - (d @ a_unit) * a_unit
        d_orth = F.normalize(d_orth, dim=-1)
        ref_lp = get_ref(ctx, a)
        for j, theta in enumerate(angles_rad):
            a_pert = a_norm * (np.cos(theta) * a_unit + np.sin(theta) * d_orth)
            kl_all[i, j] = get_kl(ctx, a_pert, ref_lp)
    return kl_all


def sweep_random(n_dirs=5):
    kl_all = np.zeros((n_act, len(angles_rad), n_dirs))
    for i in range(n_act):
        ctx = contexts[i:i+1].to(device)
        a = activations[i].to(device)
        a_norm = a.norm()
        a_unit = a / a_norm
        ref_lp = get_ref(ctx, a)
        v = torch.randn(n_dirs, 512, device=device)
        v = v - (v @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        v = F.normalize(v, dim=-1)
        for j, theta in enumerate(angles_rad):
            for k in range(n_dirs):
                a_pert = a_norm * (np.cos(theta) * a_unit + np.sin(theta) * v[k])
                kl_all[i, j, k] = get_kl(ctx, a_pert, ref_lp)
    return kl_all


print('Sweeping MELBO_8...')
kl_melbo = sweep_dir(melbo_dir)
print('Sweeping probe (N on h7)...')
kl_probe = sweep_dir(probe_dir)
print('Sweeping random (5 dirs)...')
kl_random = sweep_random(n_dirs=5)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

rand_per_act = np.median(kl_random, axis=2)
ax.fill_between(angles_deg,
                np.percentile(rand_per_act, 25, axis=0),
                np.percentile(rand_per_act, 75, axis=0),
                color='gray', alpha=0.2)
ax.plot(angles_deg, np.median(rand_per_act, axis=0),
        color='gray', lw=2, ls='--', label='Random')

ax.fill_between(angles_deg,
                np.percentile(kl_melbo, 25, axis=0),
                np.percentile(kl_melbo, 75, axis=0),
                color='blue', alpha=0.15)
ax.plot(angles_deg, np.median(kl_melbo, axis=0),
        color='blue', lw=2, label='MELBO_8')

ax.fill_between(angles_deg,
                np.percentile(kl_probe, 25, axis=0),
                np.percentile(kl_probe, 75, axis=0),
                color='green', alpha=0.15)
ax.plot(angles_deg, np.median(kl_probe, axis=0),
        color='green', lw=2, label='Probe: N on h7')

ax.axhline(y=0.5, color='red', ls=':', alpha=0.5, label='target KL=0.5')
ax.set_xlabel('Perturbation angle (degrees)')
ax.set_ylabel('KL divergence')
ax.set_title('chess_16L layer 11: MELBO_8 vs matched probe (cos=-0.177)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 180)
plt.tight_layout()

out_dir = 'results/capped_melbo/chess_16L_layer11_target0.5'
plt.savefig(f'{out_dir}/melbo_vs_probe_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved melbo_vs_probe_sweep.png')

idx_40 = np.argmin(np.abs(angles_deg - 40))
print(f'\nKL at 40 degrees (median over activations):')
print(f'  Random:  {np.median(rand_per_act[:, idx_40]):.4f}')
print(f'  MELBO_8: {np.median(kl_melbo[:, idx_40]):.4f}')
print(f'  Probe:   {np.median(kl_probe[:, idx_40]):.4f}')
