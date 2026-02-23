#!/usr/bin/env python3
"""
Compute 4-language contrastive features (es, zh, ru, ar) — dropping German.

Tests hypothesis that ES-DE correlation (Assumption 1 violation) contaminates
the Spanish feature. If true, the 4-lang Spanish feature should be cleaner.

Uses existing raw pairwise means from the 10-language allpairs run.
"""

import sys, os, pickle
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F

SUBSET_4 = ['es', 'zh', 'ru', 'ar']

results_dir = 'results/allpairs_contrastive/qwen3-4b'

# Load 10-language feature directions
print("Loading 10-language feature directions...")
with open(os.path.join(results_dir, 'directions.pkl'), 'rb') as f:
    dir_data = pickle.load(f)
features_10 = dir_data['features']
raw_10 = dir_data.get('features_raw', features_10)

# Re-estimate with 4-language subset (no German)
print(f"\nRe-estimating with 4-language subset: {SUBSET_4}")
raw_subset = {l: raw_10[l].float() for l in SUBSET_4}
features_4_raw = {}
for lang in SUBSET_4:
    others = [l for l in SUBSET_4 if l != lang]
    diffs = [raw_subset[lang] - raw_subset[o] for o in others]
    features_4_raw[lang] = sum(diffs) / len(diffs)

features_4 = {l: F.normalize(v.unsqueeze(0), dim=-1).squeeze(0)
              for l, v in features_4_raw.items()}

# Print diagnostics
print("\n  Pairwise cosines (4-lang estimator):")
for i in range(len(SUBSET_4)):
    for j in range(i + 1, len(SUBSET_4)):
        cos = (features_4[SUBSET_4[i]] @ features_4[SUBSET_4[j]]).item()
        print(f"    {SUBSET_4[i]} vs {SUBSET_4[j]}: {cos:+.4f}")

# Compare with 5-lang features
print("\n  Loading 5-lang features for comparison...")
with open(os.path.join(results_dir, '5lang_contrastive.pkl'), 'rb') as f:
    data_5 = pickle.load(f)
features_5 = data_5['features_5']

print("\n  Cosine between 4-lang and 5-lang estimates:")
for lang in SUBSET_4:
    f4 = features_4[lang].float()
    f5 = features_5[lang].float()
    print(f"    {lang}: {(f4 @ f5).item():.4f}")

print("\n  Cosine between 4-lang and 10-lang estimates:")
for lang in SUBSET_4:
    f4 = features_4[lang].float()
    f10 = features_10[lang].float()
    print(f"    {lang}: {(f4 @ f10).item():.4f}")

# Save
save_data = {
    'features_4': {k: v.cpu() for k, v in features_4.items()},
    'features_4_raw': {k: v.cpu() for k, v in features_4_raw.items()},
    'subset': SUBSET_4,
    'layer': dir_data.get('layer', 14),
    'model': 'qwen3-4b',
    'note': 'Dropped German to test Assumption 1 violation hypothesis',
}
out_path = os.path.join(results_dir, '4lang_no_german.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(save_data, f)
print(f"\nSaved: {out_path}")
print("Done!")
