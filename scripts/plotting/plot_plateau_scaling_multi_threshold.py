#!/usr/bin/env python3
"""
Analyze plateau scaling with multiple thresholds in both KL and L2 space.
Works from existing plateau_scaling.pkl data (which contains both KL and L2).
"""

import pickle, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize


def measure_threshold_crossing(mean_curve, angles_deg, threshold):
    """Find angle where curve first crosses threshold, using linear interpolation."""
    for i in range(len(mean_curve) - 1):
        if mean_curve[i] < threshold <= mean_curve[i + 1]:
            frac = (threshold - mean_curve[i]) / (mean_curve[i + 1] - mean_curve[i])
            return angles_deg[i] + frac * (angles_deg[i + 1] - angles_deg[i])
    if mean_curve[-1] < threshold:
        return angles_deg[-1]  # right-censored
    return angles_deg[0]


def power_law(n, a, alpha):
    return a * n ** alpha


models = ['qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b']
kl_thresholds = [0.1, 0.3, 0.5, 1.0]
l2_fractions = [0.1, 0.2, 0.3, 0.5]  # fraction of random L2 at 60deg

for model in models:
    pkl_path = f'results/allpairs_contrastive/{model}/plateau_scaling.pkl'
    if not os.path.exists(pkl_path):
        print(f"Skipping {model} — no data yet")
        continue

    print(f"\n{'='*60}")
    print(f"  {model}")
    print(f"{'='*60}")

    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)

    angles_deg = d['angles_deg']
    languages = d['languages']
    k = len(languages)

    # Also load random baseline from sweep_results
    sweep_path = f'results/allpairs_contrastive/{model}/sweep_results.pkl'
    with open(sweep_path, 'rb') as f:
        sweep = pickle.load(f)
    rand_kl = sweep['sweep_results']['random']['kl']  # [N, 60, 10]
    rand_l2 = sweep['sweep_results']['random']['l2']

    rand_kl_mean = rand_kl.mean(axis=(0, 2))  # [60]
    rand_l2_mean = rand_l2.mean(axis=(0, 2))

    # Compute L2 thresholds as fractions of random L2 at 60deg
    idx60 = np.argmin(np.abs(angles_deg - 60))
    rand_l2_at_60 = rand_l2_mean[idx60]
    l2_thresholds = [frac * rand_l2_at_60 for frac in l2_fractions]

    # --- KL threshold analysis ---
    print("\nKL threshold analysis:")
    fig_kl, axes_kl = plt.subplots(2, 2, figsize=(12, 10))
    axes_kl = axes_kl.flatten()

    for ti, thresh in enumerate(kl_thresholds):
        ax = axes_kl[ti]
        n_values = []
        mean_plateaus = []
        std_plateaus = []
        all_sizes = []

        for n in range(1, k + 1):
            sizes = []
            for r in d['all_results'][n]:
                mean_kl = r['kl'].mean(axis=0)
                ps = measure_threshold_crossing(mean_kl, angles_deg, thresh)
                sizes.append(ps)
            n_values.append(n)
            mean_plateaus.append(np.mean(sizes))
            std_plateaus.append(np.std(sizes))
            all_sizes.append(sizes)

        n_values = np.array(n_values)
        mean_plateaus = np.array(mean_plateaus)
        std_plateaus = np.array(std_plateaus)

        # Random baseline threshold crossing
        rand_ps = measure_threshold_crossing(rand_kl_mean, angles_deg, thresh)

        # Fit power law
        try:
            popt, _ = optimize.curve_fit(power_law, n_values, mean_plateaus,
                                          p0=[mean_plateaus[0], 0.5], maxfev=5000)
            a_fit, alpha_fit = popt
        except:
            a_fit, alpha_fit = mean_plateaus[0], 0.0

        print(f"  KL>{thresh}: exponent={alpha_fit:.3f}, "
              f"n=1: {mean_plateaus[0]:.1f}°, n=10: {mean_plateaus[-1]:.1f}°, "
              f"random: {rand_ps:.1f}°")

        # Plot
        for n_idx, n in enumerate(range(1, k + 1)):
            jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(all_sizes[n_idx]))
            ax.scatter(np.full(len(all_sizes[n_idx]), n) + jitter, all_sizes[n_idx],
                       s=6, alpha=0.25, color='steelblue', zorder=1)

        ax.errorbar(n_values, mean_plateaus, yerr=std_plateaus,
                     fmt='o-', color='navy', markersize=5, linewidth=2, capsize=3, zorder=2)

        n_fine = np.linspace(1, 10, 100)
        ax.plot(n_fine, power_law(n_fine, a_fit, alpha_fit), '--', color='red',
                linewidth=1.5, label=f'fit: n^{{{alpha_fit:.2f}}}')
        ax.plot(n_fine, mean_plateaus[0] * np.sqrt(n_fine), ':', color='green',
                linewidth=1.5, alpha=0.5, label='√n prediction')
        ax.axhline(y=rand_ps, color='gray', linestyle='--', alpha=0.5, label=f'random={rand_ps:.0f}°')

        ax.set_xlabel('Number of features (k)')
        ax.set_ylabel('Threshold angle (°)')
        ax.set_title(f'KL > {thresh}')
        ax.set_xticks(range(1, k + 1))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig_kl.suptitle(f'KL plateau scaling — {model}, layer {d["layer"]}', fontsize=13)
    fig_kl.tight_layout()
    out_kl = f'results/allpairs_contrastive/{model}/plateau_scaling_kl_multi.png'
    fig_kl.savefig(out_kl, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_kl}")

    # --- L2 threshold analysis ---
    print("\nL2 threshold analysis:")
    fig_l2, axes_l2 = plt.subplots(2, 2, figsize=(12, 10))
    axes_l2 = axes_l2.flatten()

    for ti, (thresh, frac) in enumerate(zip(l2_thresholds, l2_fractions)):
        ax = axes_l2[ti]
        n_values = []
        mean_plateaus = []
        std_plateaus = []
        all_sizes = []

        for n in range(1, k + 1):
            sizes = []
            for r in d['all_results'][n]:
                mean_l2 = r['l2'].mean(axis=0)
                ps = measure_threshold_crossing(mean_l2, angles_deg, thresh)
                sizes.append(ps)
            n_values.append(n)
            mean_plateaus.append(np.mean(sizes))
            std_plateaus.append(np.std(sizes))
            all_sizes.append(sizes)

        n_values = np.array(n_values)
        mean_plateaus = np.array(mean_plateaus)
        std_plateaus = np.array(std_plateaus)

        # Random baseline
        rand_ps = measure_threshold_crossing(rand_l2_mean, angles_deg, thresh)

        # Fit
        try:
            popt, _ = optimize.curve_fit(power_law, n_values, mean_plateaus,
                                          p0=[mean_plateaus[0], 0.5], maxfev=5000)
            a_fit, alpha_fit = popt
        except:
            a_fit, alpha_fit = mean_plateaus[0], 0.0

        print(f"  L2>{thresh:.0f} ({frac:.0%} of rand@60°): exponent={alpha_fit:.3f}, "
              f"n=1: {mean_plateaus[0]:.1f}°, n=10: {mean_plateaus[-1]:.1f}°, "
              f"random: {rand_ps:.1f}°")

        # Plot
        for n_idx, n in enumerate(range(1, k + 1)):
            jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(all_sizes[n_idx]))
            ax.scatter(np.full(len(all_sizes[n_idx]), n) + jitter, all_sizes[n_idx],
                       s=6, alpha=0.25, color='steelblue', zorder=1)

        ax.errorbar(n_values, mean_plateaus, yerr=std_plateaus,
                     fmt='o-', color='navy', markersize=5, linewidth=2, capsize=3, zorder=2)

        n_fine = np.linspace(1, 10, 100)
        ax.plot(n_fine, power_law(n_fine, a_fit, alpha_fit), '--', color='red',
                linewidth=1.5, label=f'fit: n^{{{alpha_fit:.2f}}}')
        ax.plot(n_fine, mean_plateaus[0] * np.sqrt(n_fine), ':', color='green',
                linewidth=1.5, alpha=0.5, label='√n prediction')
        ax.axhline(y=rand_ps, color='gray', linestyle='--', alpha=0.5, label=f'random={rand_ps:.0f}°')

        ax.set_xlabel('Number of features (k)')
        ax.set_ylabel('Threshold angle (°)')
        ax.set_title(f'L2 > {thresh:.0f} ({frac:.0%} of rand@60°)')
        ax.set_xticks(range(1, k + 1))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig_l2.suptitle(f'L2 plateau scaling — {model}, layer {d["layer"]}, measure layer {d["measure_layer"]}',
                    fontsize=13)
    fig_l2.tight_layout()
    out_l2 = f'results/allpairs_contrastive/{model}/plateau_scaling_l2_multi.png'
    fig_l2.savefig(out_l2, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_l2}")

    # --- KL and L2 curves by k (zoomed) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    mask = angles_deg <= 30
    cmap = plt.cm.viridis

    for n in [1, 2, 3, 5, 10]:
        all_kl = np.stack([r['kl'] for r in d['all_results'][n]])
        all_l2 = np.stack([r['l2'] for r in d['all_results'][n]])
        mean_kl = all_kl.mean(axis=(0, 1))
        mean_l2 = all_l2.mean(axis=(0, 1))
        color = cmap((n - 1) / 9)
        ax1.plot(angles_deg[mask], mean_kl[mask], color=color, linewidth=2, label=f'k={n}')
        ax2.plot(angles_deg[mask], mean_l2[mask], color=color, linewidth=2, label=f'k={n}')

    # Random
    ax1.plot(angles_deg[mask], rand_kl_mean[mask], '--', color='gray', linewidth=2, label='random')
    ax1.fill_between(angles_deg[mask],
                     rand_kl_mean[mask] - rand_kl.mean(axis=0).std(axis=-1)[mask],
                     rand_kl_mean[mask] + rand_kl.mean(axis=0).std(axis=-1)[mask],
                     color='gray', alpha=0.15)
    ax2.plot(angles_deg[mask], rand_l2_mean[mask], '--', color='gray', linewidth=2, label='random')
    ax2.fill_between(angles_deg[mask],
                     rand_l2_mean[mask] - rand_l2.mean(axis=0).std(axis=-1)[mask],
                     rand_l2_mean[mask] + rand_l2.mean(axis=0).std(axis=-1)[mask],
                     color='gray', alpha=0.15)

    ax1.set_xlabel('Perturbation angle (°)')
    ax1.set_ylabel('Mean KL divergence')
    ax1.set_title('KL divergence')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.set_xlabel('Perturbation angle (°)')
    ax2.set_ylabel('Mean L2 distance')
    ax2.set_title(f'L2 distance (layer {d["measure_layer"]})')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    fig.suptitle(f'Response curves by k — {model}, layer {d["layer"]}', fontsize=13)
    fig.tight_layout()
    out_curves = f'results/allpairs_contrastive/{model}/plateau_scaling_curves.png'
    fig.savefig(out_curves, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_curves}")

plt.close('all')
print("\nDone!")
