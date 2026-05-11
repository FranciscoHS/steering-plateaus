#!/usr/bin/env python3
"""Combine both bottleneck sweep results into one plot."""

import json
import matplotlib.pyplot as plt

# Load both sweeps
with open('results/trained_autoencoder/bottleneck_sweep/sweep_results.json') as f:
    high = json.load(f)
with open('results/trained_autoencoder/bottleneck_sweep_low/sweep_results.json') as f:
    low = json.load(f)

# Merge (low-k sweep has the 6k-step versions; high-k sweep has 10k-step versions)
# For k=5 which appears in both, prefer the 10k-step version
all_results = {}
for k, v in low.items():
    all_results[int(k)] = v
for k, v in high.items():
    all_results[int(k)] = v  # overwrites k=5 with 10k-step version

ks = sorted(all_results.keys())
cos_vals = [all_results[k]['cos_sim'] for k in ks]
rel_l2_vals = [all_results[k]['rel_l2'] for k in ks]

sae_cos = 0.942

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(ks, cos_vals, 'o-', color='C0', label='Autoencoder (~67M params)')
ax1.axhline(sae_cos, color='C1', linestyle='--', label='SAE 8x k=32 (67M params)')
ax1.set_xlabel('Bottleneck dimension')
ax1.set_ylabel('Cosine similarity')
ax1.set_title('Reconstruction quality vs bottleneck dim')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(ks, rel_l2_vals, 'o-', color='C0', label='Autoencoder (~67M params)')
ax2.set_xlabel('Bottleneck dimension')
ax2.set_ylabel('Relative L2 error')
ax2.set_title('Reconstruction error vs bottleneck dim')
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.tight_layout()
out = 'results/trained_autoencoder/bottleneck_sweep_combined.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved to {out}")
