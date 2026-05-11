"""Plot local ID vs cluster size for layer-10 k-means sweep (match reference style)."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN_PATH = 'results/jacobian/cluster_id_l10_sweep/sweep_results.json'
OUT_PATH = 'results/jacobian/cluster_id_l10_sweep/id_vs_cluster_size.png'

with open(IN_PATH) as f:
    d = json.load(f)

global_id = d['global_id']
all_sizes = np.array(d['all_sizes'])
all_ids = np.array(d['all_ids'])

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(all_sizes, all_ids, alpha=0.4, s=15, color='steelblue')
ax.axhline(global_id, color='red', linestyle='--', linewidth=1.5,
           label=f'Global ID = {global_id:.1f}')

size_bins = np.logspace(np.log10(all_sizes.min()), np.log10(all_sizes.max()), 15)
bin_centers, bin_medians, bin_q25, bin_q75 = [], [], [], []
for i in range(len(size_bins) - 1):
    mask = (all_sizes >= size_bins[i]) & (all_sizes < size_bins[i+1])
    if mask.sum() >= 3:
        bin_centers.append(np.sqrt(size_bins[i] * size_bins[i+1]))
        bin_medians.append(np.median(all_ids[mask]))
        bin_q25.append(np.percentile(all_ids[mask], 25))
        bin_q75.append(np.percentile(all_ids[mask], 75))
bin_centers = np.array(bin_centers)
bin_medians = np.array(bin_medians)
bin_q25 = np.array(bin_q25)
bin_q75 = np.array(bin_q75)
ax.plot(bin_centers, bin_medians, 'o-', color='darkblue', linewidth=2,
        markersize=6, label='Binned median')
ax.fill_between(bin_centers, bin_q25, bin_q75, color='darkblue', alpha=0.15,
                label='IQR')

ax.set_xscale('log')
ax.set_xlabel('Cluster size (number of points)')
ax.set_ylabel('TWO-NN intrinsic dimensionality')
ax.set_title('Local ID vs cluster size (Qwen3-1.7B layer 10, last token, seq_len=10)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"Saved {OUT_PATH}")
