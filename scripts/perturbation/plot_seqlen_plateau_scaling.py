#!/usr/bin/env python3
"""Plot plateau size vs sequence length.

Loads perturbation sweep results for different seq_len values,
measures the plateau size (angle at which median crosses a threshold),
and plots plateau size vs seq_len for three metrics.

Usage:
    python scripts/perturbation/plot_seqlen_plateau_scaling.py
    python scripts/perturbation/plot_seqlen_plateau_scaling.py --model gemma-2-9b --seq_lens 1 2 3 4 5 --layers 2
"""
import argparse, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(model, seq_len, layer=2):
    """Try to load results for a given model, seq_len and layer."""
    candidates = [
        f'results/seqlen_scaling/{model}_layer{layer}_seq{seq_len}/results.pkl',
        f'results/seqlen{seq_len}_sweep/{model}_layer{layer}/results.pkl',
        f'results/seqlen1_sweep/{model}_layer{layer}/results.pkl' if seq_len == 1 else None,
    ]
    for path in candidates:
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    return None


def measure_plateau(angles_deg, median_curve, threshold):
    """Find the angle where the median first crosses threshold.

    Returns the angle in degrees, or NaN if it never crosses.
    """
    crossings = np.where(median_curve > threshold)[0]
    if len(crossings) > 0:
        return angles_deg[crossings[0]]
    return float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--seq_lens', type=int, nargs='+',
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30])
    parser.add_argument('--layers', type=int, nargs='+', default=[2, 7, 15])
    args = parser.parse_args()

    thresholds = {
        'cos': 0.1,
        'kl': 2.0,
        'l2': 500.0,
    }
    metric_labels = {
        'cos': 'Cosine distance',
        'kl': 'KL divergence',
        'l2': 'L2 distance',
    }
    layer_colors = {l: f'C{i}' for i, l in enumerate(args.layers)}

    # Load all results per layer
    all_results = {}  # (layer, seq_len) -> results
    for layer in args.layers:
        for sl in args.seq_lens:
            r = load_results(args.model, sl, layer=layer)
            if r is not None:
                all_results[(layer, sl)] = r
                print(f"layer={layer}, seq_len={sl}: loaded ({r['angles_deg'][-1]:.0f}° max)")

    if not all_results:
        print("No results found!")
        sys.exit(1)

    # Measure plateau sizes
    plateau_sizes = {}  # (layer, metric) -> {seq_len: angle}
    for (layer, sl), r in sorted(all_results.items()):
        angles_deg = r['angles_deg']
        real_data = r['real']
        for metric, thresh in thresholds.items():
            key = (layer, metric)
            if key not in plateau_sizes:
                plateau_sizes[key] = {}
            per_act = np.median(real_data[metric], axis=2)  # [n_act, n_angles]
            median_curve = np.median(per_act, axis=0)        # [n_angles]
            angle = measure_plateau(angles_deg, median_curve, thresh)
            plateau_sizes[key][sl] = angle
            print(f"  layer={layer}, seq_len={sl}, {metric}: plateau = {angle:.1f}°")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (metric, thresh) in zip(axes, thresholds.items()):
        for layer in args.layers:
            key = (layer, metric)
            if key not in plateau_sizes:
                continue
            sls = sorted(plateau_sizes[key].keys())
            angles = [plateau_sizes[key][sl] for sl in sls]
            ax.plot(sls, angles, 'o-', color=layer_colors[layer], markersize=6,
                    linewidth=2, label=f'Layer {layer}')

        ax.set_xlabel('Sequence length')
        ax.set_ylabel('Plateau size (degrees)')
        ax.set_title(f'{metric_labels[metric]} > {thresh}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{args.model}: plateau size vs sequence length', fontsize=14)
    plt.tight_layout()

    out_dir = 'results/seqlen_scaling'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'plateau_vs_seqlen_{args.model}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
