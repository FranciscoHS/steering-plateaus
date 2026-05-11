#!/usr/bin/env python3
"""Plot final comparison of AE vs SAE reconstruction quality."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def main():
    results_dir = 'results/final_comparison'

    # Load all eval results
    methods = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith('eval_') and fname.endswith('.json'):
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)
            label = fname.replace('eval_', '').replace('.json', '')
            data['label'] = label
            methods.append(data)

    if not methods:
        print("No eval results found!")
        return

    # Print summary
    print(f"{'Method':<25s} {'KL div':>10s} {'CE increase':>12s} {'Cosine':>10s} {'MSE':>10s}")
    print("-" * 70)
    for m in methods:
        print(f"{m['label']:<25s} {m['kl_div_mean']:>10.6f} {m['ce_increase_mean']:>12.6f} "
              f"{m['cos_sim_mean']:>10.4f} {m['mse_mean']:>10.6f}")

    # Bar chart comparison
    labels = [m['label'] for m in methods]
    metrics = {
        'KL Divergence': [m['kl_div_mean'] for m in methods],
        'CE Increase': [m['ce_increase_mean'] for m in methods],
        'Cosine Similarity': [m['cos_sim_mean'] for m in methods],
        'MSE': [m['mse_mean'] for m in methods],
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63'][:len(labels)]

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(range(len(labels)), values, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_title(metric_name, fontsize=10)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7)

    plt.suptitle('Final Comparison: Deep AE vs TopK SAE\n'
                 '(k=32, 50k steps, held-out CE eval)', fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(results_dir, 'final_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {out_path}")


if __name__ == '__main__':
    main()
