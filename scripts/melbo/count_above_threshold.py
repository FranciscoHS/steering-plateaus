#!/usr/bin/env python3
"""Count how many MELBO directions exceed multiples of random median response."""

import pickle
import numpy as np

results_path = 'results/melbo_spectrum/qwen3-1.7b_layer2_l2_tgt26_act1/results.pkl'
with open(results_path, 'rb') as f:
    r = pickle.load(f)

sweep_angles_deg = r['sweep_angles_deg']
random_objs = r['random_objs']
melbo_results = r['melbo_results']

multipliers = [2, 5, 10, 20, 50]

print(f"How many MELBO directions exceed Nx random median?")
print(f"{'angle':>6s}", end='')
for m in multipliers:
    print(f"  {m:>3d}x", end='')
print(f"  {'median':>10s}")

for angle in r['opt_angles_deg']:
    objs = np.array(melbo_results[angle]['objs'])
    ang_idx = np.argmin(np.abs(sweep_angles_deg - angle))
    rand_med = np.median(random_objs[ang_idx])

    print(f"{angle:5.1f}°", end='')
    for m in multipliers:
        threshold = m * rand_med
        n_above = np.sum(objs > threshold)
        print(f"  {n_above:>4d}", end='')
    print(f"  {rand_med:>10.2f}")
