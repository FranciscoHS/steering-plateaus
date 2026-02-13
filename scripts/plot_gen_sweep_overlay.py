#!/usr/bin/env python3
"""
Overlay generation Chinese fraction with perturbation sweep KL divergence.

Creates a 2x2 grid:
  - Rows: direction (diff-of-means, MELBO_1)
  - Columns: method (spherical, additive)
  - Each panel: Chinese char fraction (left y-axis) + KL divergence (right y-axis)
  - X-axis: angle in degrees (additive magnitudes converted via arctan(m/R))
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def chinese_char_fraction(text):
    """Fraction of characters in text that are CJK unified ideographs."""
    if not text:
        return 0.0
    n_chinese = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    n_total = sum(1 for ch in text if not ch.isspace() and ch not in '.,;:!?()[]{}"\'-/')
    if n_total == 0:
        return 0.0
    return n_chinese / n_total


def load_gen_results(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_fractions(gen_data):
    """Extract per-prompt Chinese fractions at each steering strength.

    Returns: x_values (angles in degrees), fractions_per_prompt dict {prompt: [frac_per_x]}
    """
    results = gen_data['results']
    mean_norm = gen_data['mean_norm']

    # Determine if additive or spherical
    if 'magnitudes' in gen_data:
        magnitudes = gen_data['magnitudes']
        # Convert to equivalent angles
        angles = [np.degrees(np.arctan(m / mean_norm)) if m > 0 else 0.0 for m in magnitudes]
        x_key = 'magnitude'
        x_values_raw = magnitudes
    else:
        angles = gen_data['angles']
        x_key = 'angle_deg'
        x_values_raw = angles

    # Group by prompt
    prompts = []
    seen = set()
    for r in results:
        if r['prompt'] not in seen:
            prompts.append(r['prompt'])
            seen.add(r['prompt'])

    fractions = {p: [] for p in prompts}
    for r in results:
        frac = chinese_char_fraction(r['generated_text'])
        fractions[r['prompt']].append(frac)

    return angles, fractions, prompts


def main():
    # Load perturbation sweep data
    sweep_path = 'results/melbo_constrained/qwen3-0.6b/results.pkl'
    with open(sweep_path, 'rb') as f:
        sweep_data = pickle.load(f)

    sweep_angles = sweep_data['angles_deg']  # (60,)
    sweep_kl_mean = sweep_data['sweep_results']['mean']['kl']  # (50, 60)
    sweep_kl_melbo1 = sweep_data['sweep_results']['MELBO_1']['kl']  # (50, 60)

    # Generation results
    configs = [
        ('Spherical + Diff-of-means', 'results/gen50/spherical_mean/results.pkl', sweep_kl_mean),
        ('Additive + Diff-of-means', 'results/gen50/additive_mean/results.pkl', sweep_kl_mean),
        ('Spherical + MELBO_1', 'results/gen50/spherical_melbo1/results.pkl', sweep_kl_melbo1),
        ('Additive + MELBO_1', 'results/gen50/additive_melbo1/results.pkl', sweep_kl_melbo1),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Layout: [spherical_mean, additive_mean; spherical_melbo, additive_melbo]
    panel_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (title, gen_path, sweep_kl), (row, col) in zip(configs, panel_positions):
        gen_data = load_gen_results(gen_path)
        angles, fractions, prompts = extract_fractions(gen_data)

        ax1 = axes[row, col]
        ax2 = ax1.twinx()

        # Plot KL divergence (perturbation sweep) - background, right axis
        kl_median = np.median(sweep_kl, axis=0)
        kl_q25 = np.percentile(sweep_kl, 25, axis=0)
        kl_q75 = np.percentile(sweep_kl, 75, axis=0)

        ax2.fill_between(sweep_angles, kl_q25, kl_q75, alpha=0.15, color='gray')
        ax2.plot(sweep_angles, kl_median, color='gray', alpha=0.6, linewidth=1.5,
                 linestyle='--', label='KL div (perturbation sweep)')
        ax2.set_ylabel('KL divergence (perturbation sweep)', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        # Plot Chinese fraction - foreground, left axis
        # Per-prompt lines (thin)
        for i, prompt in enumerate(prompts):
            ax1.plot(angles, fractions[prompt], alpha=0.3, linewidth=0.8, color=f'C{i}')

        # Median across prompts
        frac_array = np.array([fractions[p] for p in prompts])  # (n_prompts, n_angles)
        frac_median = np.median(frac_array, axis=0)
        frac_q25 = np.percentile(frac_array, 25, axis=0)
        frac_q75 = np.percentile(frac_array, 75, axis=0)

        ax1.fill_between(angles, frac_q25, frac_q75, alpha=0.2, color='C0')
        ax1.plot(angles, frac_median, color='C0', linewidth=2.5, label='Chinese fraction (median)')

        ax1.set_xlabel('Angle (degrees)')
        ax1.set_ylabel('Chinese character fraction', color='C0')
        ax1.tick_params(axis='y', labelcolor='C0')
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlim(0, 35)
        ax1.set_title(title)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    fig.suptitle('Qwen3-0.6B Layer 14: Generation vs Perturbation Sweep\n'
                 '(50 tokens, rolling window, 5 prompts)',
                 fontsize=13)
    plt.tight_layout()

    save_path = 'results/gen50/generation_vs_sweep_overlay.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    main()
