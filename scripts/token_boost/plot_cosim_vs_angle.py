#!/usr/bin/env python3
"""Plot cosine similarity between optimized directions vs optimization angle."""

import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

angles = [1, 2, 3, 4, 5, 10, 15, 20]
cos_sims = []

for angle in angles:
    if angle == 15:
        path = f'results/token_boost/gpt2_L5_book/results.pkl'
    else:
        path = f'results/token_boost/gpt2_L5_book_angle{angle}/results.pkl'
    with open(path, 'rb') as f:
        results = pickle.load(f)
    cos_sims.append(results['cos_sims'][0, 1])

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(angles, cos_sims, 'o-', color='#2171b5', markersize=7, linewidth=2)
ax.set_xlabel('Optimization angle (degrees)', fontsize=12)
ax.set_ylabel('Cosine similarity (d₀, d₁)', fontsize=12)
ax.set_title('Direction similarity vs optimization angle\n'
             'GPT-2, layer 5, target=" book"', fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
ax.set_xticks(angles)

out = 'results/token_boost/cosim_vs_angle.png'
plt.tight_layout()
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")
