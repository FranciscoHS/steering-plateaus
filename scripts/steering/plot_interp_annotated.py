#!/usr/bin/env python3
"""Annotated interpolation plot: overlay completion regimes on d(alpha) curve."""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import sys

# Default paths, overridable via CLI args
DATA_PATH = 'results/latent_interpolation/big_to_in_logits_200/results.pkl'
OUTPUT_PATH = 'results/latent_interpolation/big_to_in_logits_200/interpolation_annotated.png'

if len(sys.argv) >= 3:
    DATA_PATH = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]


def first_n_tokens(text, n=5):
    return ' '.join(text.strip().split()[:n])


def find_regimes(alphas, texts, n_tokens=5, min_length=5):
    """Find contiguous regimes, merging short flickers into neighbours."""
    # First pass: raw regime detection
    raw_regimes = []
    prev = None
    start = 0
    for i, (a, t) in enumerate(zip(alphas, texts)):
        cur = first_n_tokens(t, n_tokens)
        if cur != prev:
            if prev is not None:
                raw_regimes.append((start, i - 1, prev))
            prev = cur
            start = i
    raw_regimes.append((start, len(alphas) - 1, prev))

    # Second pass: merge short regimes into previous
    merged = [raw_regimes[0]]
    for s, e, txt in raw_regimes[1:]:
        if (e - s + 1) < min_length:
            # Extend previous regime to cover this flicker
            prev_s, prev_e, prev_txt = merged[-1]
            merged[-1] = (prev_s, e, prev_txt)
        else:
            # Check if same text as previous (interrupted by flicker)
            prev_s, prev_e, prev_txt = merged[-1]
            if txt == prev_txt:
                merged[-1] = (prev_s, e, prev_txt)
            else:
                merged.append((s, e, txt))

    return [(s, e, alphas[s], alphas[e], txt) for s, e, txt in merged]


def pick_label(text, max_words=6):
    """Pick a short, meaningful label from the completion."""
    words = text.strip().split()[:max_words]
    label = ' '.join(words)
    if len(label) > 35:
        label = label[:35] + '...'
    return label


def main():
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    r = data['results'][0]
    alphas = np.array(r['alphas'])
    raw_d = np.array(r['raw']['rel_dist'])
    lat_d = np.array(r['latent']['rel_dist'])

    # Detect latent method from saved args
    args = data.get('args', {})
    if isinstance(args, dict) and args.get('sae_weights'):
        latent_label = 'Latent (SAE k=32, d=16384)'
    else:
        latent_label = 'Latent (AE 30-dim)'

    raw_regimes = find_regimes(r['alphas'], r['raw']['texts'], n_tokens=4)
    lat_regimes = find_regimes(r['alphas'], r['latent']['texts'], n_tokens=4)

    # Filter out regimes shorter than 3 steps (flickering)
    raw_regimes = [reg for reg in raw_regimes if reg[1] - reg[0] >= 3]
    lat_regimes = [reg for reg in lat_regimes if reg[1] - reg[0] >= 3]

    print("Raw regimes:")
    for s, e, a0, a1, txt in raw_regimes:
        print(f"  [{a0:.3f}, {a1:.3f}] ({e-s+1} steps): {txt}")
    print("\nLatent regimes:")
    for s, e, a0, a1, txt in lat_regimes:
        print(f"  [{a0:.3f}, {a1:.3f}] ({e-s+1} steps): {txt}")

    # ── Plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.12})

    regime_colors = ['#BBDEFB', '#FFE0B2', '#C8E6C9', '#F3E5F5', '#FBE9E7']

    for ax_idx, (ax, d_vals, regimes, method, color) in enumerate([
        (axes[0], raw_d, raw_regimes, 'Raw (activation space)', '#2196F3'),
        (axes[1], lat_d, lat_regimes, latent_label, '#E91E63'),
    ]):
        # Shade regimes
        for ri, (s, e, a0, a1, txt) in enumerate(regimes):
            rc = regime_colors[ri % len(regime_colors)]
            ax.axvspan(a0, a1, color=rc, alpha=0.4, zorder=0)

        # Plot curve on top
        ax.plot(alphas, d_vals, '-', color=color, linewidth=2, alpha=0.95, zorder=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.15, linewidth=1, zorder=1)

        # Labels — place alternating above/below curve
        for ri, (s, e, a0, a1, txt) in enumerate(regimes):
            mid_alpha = (a0 + a1) / 2
            mid_idx = (s + e) // 2
            mid_d = d_vals[mid_idx]
            label = pick_label(txt, max_words=5)

            # Alternate above/below
            if ri % 2 == 0:
                y_pos = min(mid_d + 0.15, 0.92)
            else:
                y_pos = max(mid_d - 0.15, 0.08)

            ax.annotate(
                f'"{label}"',
                xy=(mid_alpha, mid_d), xytext=(mid_alpha, y_pos),
                fontsize=9, ha='center', va='center',
                fontstyle='italic', color='#333',
                arrowprops=dict(arrowstyle='->', color='#888', lw=0.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#999', alpha=0.95, linewidth=0.8),
                zorder=3)

        ax.set_ylabel('Relative distance d', fontsize=13)
        ax.set_title(method, fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=11)

    prompt_a = r['prompt_a']
    prompt_b = r['prompt_b']
    axes[1].set_xlabel(f'Interpolation α  (0 = "{prompt_a}", 1 = "{prompt_b}")', fontsize=13)

    fig.suptitle(f'SLERP interpolation: "{prompt_a}" → "{prompt_b}"\n'
                 f'Qwen3-1.7B layer 2, logit-space L2 relative distance, {len(alphas)} steps',
                 fontsize=14, fontweight='bold', y=0.99)

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
