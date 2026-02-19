#!/usr/bin/env python3
"""Quick script to print results from melbo_animal experiment."""
import pickle, numpy as np
r = pickle.load(open("results/melbo_animal/qwen3-0.6b/results.pkl", "rb"))

S = r["singular_values"].numpy()
cumvar = (S**2).cumsum() / (S**2).sum()
print(f"Top-5 SV: {S[:5].tolist()}")
print(f"Top-5 cumvar: {cumvar[:5].tolist()}")
print(f"PC1 var explained: {S[0]**2 / (S**2).sum():.3f}")
print(f"cos(PC1, mean_dir): {abs(float(r['pc1'] @ r['mean_dir'])):.3f}")
print()

for i, (kl, d) in enumerate(zip(r["melbo_kls"], r["melbo_directions"])):
    cos_m = abs(float(d @ r["mean_dir"]))
    cos_p = abs(float(d @ r["pc1"]))
    print(f"MELBO_{i+1}: KL={kl:.6f}, cos(mean)={cos_m:.3f}, cos(PC1)={cos_p:.3f}")
    for cat, cd in r["cat_diffs"].items():
        print(f"  cos(MELBO_{i+1}, dogs-{cat}): {abs(float(d @ cd)):.3f}")
    if i > 0:
        for j in range(i):
            cos_prev = abs(float(d @ r["melbo_directions"][j]))
            print(f"  cos(MELBO_{i+1}, MELBO_{j+1}): {cos_prev:.4f}")

angles = r["angles_deg"]
idx15 = np.argmin(np.abs(angles - 15))
sr = r["sweep_results"]
rand_kl = np.median(sr["random"]["kl"][:, idx15, :])
print(f"\nAt 15 deg:")
print(f"  Random median KL: {rand_kl:.6f}")
for name in ["mean_dir", "PC1"]:
    if name in sr:
        val = np.median(sr[name]["kl"][:, idx15])
        print(f"  {name} median KL: {val:.6f}")
for cat_name in ["dogs-cats", "dogs-birds", "dogs-fish", "dogs-snakes"]:
    if cat_name in sr:
        val = np.median(sr[cat_name]["kl"][:, idx15])
        print(f"  {cat_name} median KL: {val:.6f}")
for i in range(len(r["melbo_kls"])):
    print(f"  MELBO_{i+1} mean KL: {r['melbo_kls'][i]:.6f}")
