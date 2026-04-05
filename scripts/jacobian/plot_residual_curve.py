#!/usr/bin/env python3
"""Plot variance explained (1 - ||residual||²/||a||²) vs number of directions."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Parsed from the full run log
residual_norms = [
    62.80, 47.04, 39.64, 34.05, 29.77, 25.99, 23.21, 20.54, 18.17, 17.02,
    15.31, 14.02, 13.29, 12.49, 11.17, 10.21, 9.44, 9.02, 8.69, 8.35,
    8.11, 7.84, 7.67, 7.42, 7.26, 6.94, 6.78, 6.70, 6.57, 6.52,
    6.49, 6.40, 6.28, 6.27, 6.25, 6.21, 6.14, 6.12, 6.11, 6.03,
    6.03, 6.02, 6.01, 6.00, 5.93, 5.93, 5.90, 5.89, 5.89, 5.88,
    5.82, 5.81, 5.79, 5.78, 5.77, 5.77, 5.76, 5.75, 5.75, 5.75,
    5.74, 5.73, 5.72, 5.71, 5.70, 5.70, 5.69, 5.69, 5.68, 5.68,
    5.67, 5.66, 5.66, 5.65, 5.65, 5.65, 5.64, 5.64, 5.63, 5.63,
]

a_norm = 62.80
n_dirs = np.arange(len(residual_norms))
var_explained = 1 - (np.array(residual_norms) / a_norm) ** 2

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_dirs, var_explained * 100, 'o-', markersize=3, linewidth=1.5)
ax.set_xlabel('Number of directions')
ax.set_ylabel('Variance explained (%)')
ax.set_title('Response decomposition: cumulative variance explained\n(Qwen3-1.7B, layer 7)')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, len(residual_norms) - 1)
ax.set_ylim(0, 100)

plt.tight_layout()
out = 'results/response_decomp/variance_curve.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved to {out}")
