#!/usr/bin/env python3
"""
Identify which prompt pairs drive the PC1-mean anomaly at layers 7, 10-13.

For each anomalous layer, projects difference vectors onto the component of PC1
that's orthogonal to the mean direction. Pairs with largest projections are the
ones pulling PC1 away from the mean.

Also trains linear vs non-linear probes to test whether language is a non-linear
feature.

Usage:
    python scripts/run_svd_outlier_analysis.py --model qwen3-0.6b
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model
from data.bilingual_prompts import ENGLISH_PROMPTS, CHINESE_PROMPTS, PROMPT_PAIRS


def extract_all_layers(model, tokenizer, prompts, device):
    """Extract last-token activations at ALL layers for a list of prompts."""
    all_acts = None
    n_layers = None
    for i, prompt in enumerate(prompts):
        tokens = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if all_acts is None:
            n_layers = len(hidden_states) - 1
            d_model = hidden_states[0].shape[-1]
            all_acts = torch.zeros(n_layers, len(prompts), d_model)
        for layer in range(n_layers):
            all_acts[layer, i, :] = hidden_states[layer + 1][0, -1, :].cpu()
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(prompts)}")
    return all_acts, n_layers


def train_probe(X_train, y_train, X_test, y_test, nonlinear=False, hidden_dim=64,
                lr=1e-3, epochs=500):
    """Train a linear or non-linear probe. Returns test accuracy."""
    d = X_train.shape[1]
    if nonlinear:
        model = torch.nn.Sequential(
            torch.nn.Linear(d, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
    else:
        model = torch.nn.Linear(d, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        logits = model(X_train).squeeze(-1)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_logits = model(X_test).squeeze(-1)
        preds = (test_logits > 0).float()
        acc = (preds == y_test).float().mean().item()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-0.6b')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/svd_steering/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    N = len(ENGLISH_PROMPTS)

    # Load model and extract activations
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    print(f"\nExtracting activations...")
    en_acts, n_layers = extract_all_layers(model, tokenizer, ENGLISH_PROMPTS, device)
    zh_acts, _ = extract_all_layers(model, tokenizer, CHINESE_PROMPTS, device)
    print(f"{n_layers} layers")

    # ==========================================
    # Part 1: Outlier pair identification
    # ==========================================
    anomalous_layers = [7, 10, 11, 12, 13]
    normal_layers = [6, 9, 14, 15]  # for comparison

    print("\n=== Outlier pair analysis ===")

    for layer in anomalous_layers + normal_layers:
        D = en_acts[layer] - zh_acts[layer]  # [N, d]

        # Mean direction
        mean_dir = D.mean(dim=0)
        mean_unit = mean_dir / mean_dir.norm()

        # SVD
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        pc1 = Vh[0]
        cosine = (pc1 @ mean_unit).abs().item()

        # Component of PC1 orthogonal to mean
        pc1_orth = pc1 - (pc1 @ mean_unit) * mean_unit
        pc1_orth_norm = pc1_orth.norm()

        if pc1_orth_norm < 1e-6:
            # PC1 ≈ mean, no anomaly
            if layer in anomalous_layers:
                print(f"\nLayer {layer}: cos(PC1,mean)={cosine:.3f} — PC1 ≈ mean, skipping")
            continue

        pc1_orth_unit = pc1_orth / pc1_orth_norm

        # Project each difference vector onto the orthogonal component
        projections = (D @ pc1_orth_unit).numpy()  # [N]
        proj_magnitudes = np.abs(projections)

        # Top pairs by projection magnitude
        top_idx = np.argsort(proj_magnitudes)[::-1]

        tag = "ANOMALOUS" if layer in anomalous_layers else "normal"
        print(f"\nLayer {layer} ({tag}): cos(PC1,mean)={cosine:.3f}, "
              f"||PC1_orth||={pc1_orth_norm:.3f}")
        print(f"  Top 5 pairs driving anomaly:")
        for rank, idx in enumerate(top_idx[:5]):
            en_prompt, zh_prompt = PROMPT_PAIRS[idx]
            print(f"    {rank+1}. pair {idx}: |proj|={proj_magnitudes[idx]:.2f} "
                  f"(signed={projections[idx]:+.2f})")
            print(f"       EN: \"{en_prompt}\"")
            print(f"       ZH: \"{zh_prompt}\"")

        # What fraction of the anomaly is driven by top 5 pairs?
        total_orth_energy = (proj_magnitudes ** 2).sum()
        top5_energy = (proj_magnitudes[top_idx[:5]] ** 2).sum()
        print(f"  Top 5 pairs explain {top5_energy/total_orth_energy*100:.1f}% "
              f"of orthogonal energy")

    # ==========================================
    # Part 2: Linear vs non-linear probe
    # ==========================================
    print("\n\n=== Linear vs non-linear probe ===")

    # Use leave-one-out-ish: 5-fold cross validation
    n_folds = 5
    fold_size = N // n_folds

    probe_results = {}

    for layer in range(n_layers):
        # Stack English and Chinese activations
        X = torch.cat([en_acts[layer], zh_acts[layer]], dim=0)  # [2N, d]
        y = torch.cat([torch.ones(N), torch.zeros(N)])  # 1=English, 0=Chinese

        linear_accs = []
        mlp_accs = []

        for fold in range(n_folds):
            test_en = list(range(fold * fold_size, (fold + 1) * fold_size))
            test_zh = [i + N for i in test_en]
            test_idx = test_en + test_zh
            train_idx = [i for i in range(2 * N) if i not in test_idx]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            lin_acc = train_probe(X_train, y_train, X_test, y_test, nonlinear=False)
            mlp_acc = train_probe(X_train, y_train, X_test, y_test, nonlinear=True)
            linear_accs.append(lin_acc)
            mlp_accs.append(mlp_acc)

        lin_mean = np.mean(linear_accs)
        mlp_mean = np.mean(mlp_accs)
        probe_results[layer] = {'linear': lin_mean, 'mlp': mlp_mean}
        print(f"  Layer {layer:2d}: linear={lin_mean:.3f}  MLP={mlp_mean:.3f}  "
              f"gap={mlp_mean - lin_mean:+.3f}")

    # Save results
    results = {
        'args': vars(args),
        'probe_results': probe_results,
    }
    with open(os.path.join(args.results_dir, 'outlier_analysis.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Plot: linear vs MLP accuracy across layers
    fig, ax = plt.subplots(figsize=(8, 4))
    layers = list(range(n_layers))
    lin_accs = [probe_results[l]['linear'] for l in layers]
    mlp_accs = [probe_results[l]['mlp'] for l in layers]
    ax.plot(layers, lin_accs, 'o-', markersize=4, label='Linear probe')
    ax.plot(layers, mlp_accs, 's-', markersize=4, label='MLP probe (2-layer)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('5-fold CV accuracy')
    ax.set_title('Language classification: linear vs non-linear probe')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)
    fig.savefig(os.path.join(args.results_dir, 'probe_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved probe_comparison.png")
    print("Done!")


if __name__ == '__main__':
    main()
