#!/usr/bin/env python3
"""Replot LessWrong figures from saved pkl data.

Generates two variants per config:
  - 3 curves per category (clean individual lines)
  - Median + IQR shading
"""
import os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_curves(pkl_path):
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    angles_deg = d['angles_deg']
    curves = d['curves']
    config = d['config']

    groups = {}
    for prefix, label in [('sae_', 'SAE features'), ('random_', 'Random'), ('contrastive_', 'Contrastive EN↔ZH')]:
        keys = sorted([k for k in curves if k.startswith(prefix)],
                       key=lambda k: int(k.split('_')[-1]))
        groups[prefix] = {
            'label': label,
            'keys': keys,
            'data': np.array([curves[k] for k in keys]),  # [n_curves, n_angles]
        }
    return angles_deg, groups, config


COLORS = {'random_': 'C0', 'contrastive_': 'C2', 'sae_': 'C3'}


def plot_3curves(angles_deg, groups, config, save_path):
    """3 individual curves per category."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for prefix in ['random_', 'contrastive_', 'sae_']:
        g = groups[prefix]
        color = COLORS[prefix]
        # Pick 3 curves spread across the range (low/mid/high at midpoint angle)
        data = g['data']
        mid_idx = len(angles_deg) // 2
        mid_vals = data[:, mid_idx]
        order = np.argsort(mid_vals)
        # Pick 25th, 50th, 75th percentile curves
        picks = [order[len(order)//4], order[len(order)//2], order[3*len(order)//4]]
        for i, idx in enumerate(picks):
            label = g['label'] if i == 0 else None
            ax.plot(angles_deg, data[idx], color=color, lw=1.5, alpha=0.7, label=label)

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel(config['ylabel'], fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, angles_deg[-1])
    ax.set_title(f'{config["title_suffix"]}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_median_shading(angles_deg, groups, config, save_path):
    """Median + IQR shading."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for prefix in ['random_', 'contrastive_', 'sae_']:
        g = groups[prefix]
        color = COLORS[prefix]
        data = g['data']
        med = np.median(data, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.15)
        ax.plot(angles_deg, med, color=color, lw=2.5, label=g['label'])

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ax.set_ylabel(config['ylabel'], fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, angles_deg[-1])
    ax.set_title(f'{config["title_suffix"]}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    results_dir = 'results/lesswrong_plot/gemma-2-2b'
    for cfg_name in ['main', 'cosine', 'layer8']:
        pkl_path = os.path.join(results_dir, f'curves_{cfg_name}.pkl')
        if not os.path.exists(pkl_path):
            print(f"Skipping {cfg_name}: {pkl_path} not found")
            continue
        angles_deg, groups, config = load_curves(pkl_path)
        plot_3curves(angles_deg, groups, config,
                     os.path.join(results_dir, f'plot_{cfg_name}_3curves.png'))
        plot_median_shading(angles_deg, groups, config,
                            os.path.join(results_dir, f'plot_{cfg_name}_median.png'))


if __name__ == '__main__':
    main()
