#!/usr/bin/env python3
"""
Compare FP-pruned vs MP-pruned dictionary benchmarks on the same plot.
"""

import pickle
import matplotlib.pyplot as plt

# Load both benchmarks
with open('results/benchmark_dictionary/llama-3.2-1b_layer2_v2/benchmark_results.pkl', 'rb') as f:
    mp = pickle.load(f)
with open('results/benchmark_dictionary_fp/llama-3.2-1b_layer2/benchmark_results.pkl', 'rb') as f:
    fp = pickle.load(f)

n_values = mp['n_values']  # same for both

# Methods: (results_dict, key, label, marker, linestyle, color)
methods = [
    (mp, 'dictionary', 'Dict (MP-pruned)', 'o', '-', 'C0'),
    (fp, 'dictionary', 'Dict (FP-pruned)', 's', '-', 'C1'),
    (mp, 'our_sae',    'SAE 8x (resid)',   '^', '--', 'C2'),
    (mp, 'random',     'Random (MP)',       'x', ':', 'C3'),
]

metrics = [
    ('rel_l2', 'Relative L2 Error', True),
    ('cos_sim', 'Cosine Similarity', False),
    ('kl_div', 'KL Divergence', True),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (key, title, use_log) in zip(axes, metrics):
    for res, method_key, label, marker, ls, color in methods:
        if method_key not in res or not res[method_key]:
            continue
        vals = [res[method_key][n][key] for n in n_values]
        if vals[0] is None:
            continue
        ax.plot(n_values, vals, marker=marker, linestyle=ls,
                label=label, color=color, markersize=5)

    ax.set_xlabel('n (active features)')
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if use_log:
        ax.set_yscale('log')

fig.suptitle('Reconstruction: FP-pruned vs MP-pruned dictionary (20k atoms)', fontsize=11)
fig.tight_layout()
out = 'results/benchmark_dictionary_fp/llama-3.2-1b_layer2/fp_vs_mp_comparison.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved to {out}")
