#!/usr/bin/env python3
"""
Quick linear probe: at which layer can a linear classifier distinguish languages?
Extracts final-token activations for 10 languages × 36 prompts at each layer,
trains logistic regression with 5-fold CV, reports accuracy.
"""

import sys, os, numpy as np
sys.path.insert(0, '.')

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from src.model import load_model
from data.multilingual_prompts import PROMPT_SETS, LANGUAGES

MODEL = 'qwen3-0.6b'
LAYERS = list(range(0, 28, 2))  # every 2nd layer: 0, 2, 4, ..., 26

print(f"Model: {MODEL}")
print(f"Languages: {LANGUAGES}")
print(f"Probing layers: {LAYERS}")
print(f"Prompts per language: {len(PROMPT_SETS)}")

print("\nLoading model...")
model, tokenizer, device = load_model(MODEL)

# Extract activations at ALL layers in one pass (more efficient)
print("\nExtracting activations...")
X_by_layer = {l: [] for l in LAYERS}
y = []

for lang_idx, lang in enumerate(LANGUAGES):
    prompts = [ps[lang] for ps in PROMPT_SETS]
    for p in prompts:
        toks = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**toks, output_hidden_states=True)
        for layer in LAYERS:
            act = out.hidden_states[layer + 1][:, -1, :].squeeze(0).cpu().numpy()
            X_by_layer[layer].append(act)
        y.append(lang_idx)

y = np.array(y)
print(f"  Total samples: {len(y)} ({len(LANGUAGES)} languages × {len(PROMPT_SETS)} prompts)")

# Probe each layer
print(f"\n{'Layer':>5}  {'Accuracy':>8}  {'Std':>5}  {'zh vs ja':>8}")
print("-" * 35)

for layer in LAYERS:
    X = np.stack(X_by_layer[layer])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5-fold CV logistic regression (all 10 languages)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')

    # Also check zh (idx 0) vs ja (idx 5) specifically
    zh_ja_mask = np.isin(y, [0, 5])
    X_zj = X_scaled[zh_ja_mask]
    y_zj = y[zh_ja_mask]
    clf_zj = LogisticRegression(max_iter=1000, C=1.0)
    scores_zj = cross_val_score(clf_zj, X_zj, y_zj, cv=5, scoring='accuracy')

    print(f"  {layer:>3}   {scores.mean():>7.1%} ± {scores.std():>4.1%}   {scores_zj.mean():>7.1%}")

print("\nDone!")
