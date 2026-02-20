#!/usr/bin/env python3
"""Plot orthogonal features results — one row per family, columns for KL and L2."""

import pickle, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

models = ['qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b']
families = {
    'Romance': ['es', 'fr', 'it', 'pt'],
    'Germanic': ['nl', 'de'],
    'CJK': ['zh', 'ja'],
}
lang_labels = {
    'es': 'Spanish', 'fr': 'French', 'it': 'Italian', 'pt': 'Portuguese',
    'nl': 'Dutch', 'de': 'German', 'zh': 'Chinese', 'ja': 'Japanese',
}
lang_colors = {
    'es': '#e41a1c', 'fr': '#ff7f00', 'it': '#984ea3', 'pt': '#a65628',
    'nl': '#377eb8', 'de': '#4daf4a',
    'zh': '#377eb8', 'ja': '#e41a1c',
}

for model in models:
    path = f'results/allpairs_contrastive/{model}/orthogonal_features.pkl'
    if not os.path.exists(path):
        print(f"Skipping {model} — no results")
        continue

    with open(path, 'rb') as f:
        data = pickle.load(f)

    sr = data['sweep_results']
    angles = data['angles_deg']
    rand_kl = sr['random']['kl'].mean(axis=(0, 2))
    rand_l2 = sr['random']['l2'].mean(axis=(0, 2))

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))

    for row, (family_name, langs) in enumerate(families.items()):
        ax_kl = axes[row, 0]
        ax_l2 = axes[row, 1]

        # Random baseline
        ax_kl.plot(angles, rand_kl, 'k--', linewidth=1.5, alpha=0.5, label='random')
        ax_l2.plot(angles, rand_l2, 'k--', linewidth=1.5, alpha=0.5, label='random')

        # Family mean
        fkey = f'family_{family_name.lower()}'
        if fkey in sr:
            fam_kl = sr[fkey]['kl'].mean(axis=0)
            fam_l2 = sr[fkey]['l2'].mean(axis=0)
            ax_kl.plot(angles, fam_kl, 'k-', linewidth=2, alpha=0.7,
                       label=f'{family_name} mean')
            ax_l2.plot(angles, fam_l2, 'k-', linewidth=2, alpha=0.7,
                       label=f'{family_name} mean')

        # Per-language: original (solid) and pure (dashed)
        for lang in langs:
            if f'orig_{lang}' not in sr:
                continue
            color = lang_colors[lang]
            label = lang_labels[lang]

            orig_kl = sr[f'orig_{lang}']['kl'].mean(axis=0)
            orig_l2 = sr[f'orig_{lang}']['l2'].mean(axis=0)
            pure_kl = sr[f'pure_{lang}']['kl'].mean(axis=0)
            pure_l2 = sr[f'pure_{lang}']['l2'].mean(axis=0)

            ax_kl.plot(angles, orig_kl, '-', color=color, linewidth=1.5,
                       label=f'{label}')
            ax_kl.plot(angles, pure_kl, '--', color=color, linewidth=1.5,
                       label=f'{label} (pure)')

            ax_l2.plot(angles, orig_l2, '-', color=color, linewidth=1.5,
                       label=f'{label}')
            ax_l2.plot(angles, pure_l2, '--', color=color, linewidth=1.5,
                       label=f'{label} (pure)')

        ax_kl.set_ylabel('Mean KL divergence')
        ax_l2.set_ylabel('Mean L2 distance')
        ax_kl.set_title(f'{family_name} — KL')
        ax_l2.set_title(f'{family_name} — L2')
        ax_kl.legend(fontsize=7, loc='upper left')
        ax_l2.legend(fontsize=7, loc='upper left')
        ax_kl.grid(True, alpha=0.3)
        ax_l2.grid(True, alpha=0.3)
        ax_kl.set_ylim(bottom=0)
        ax_l2.set_ylim(bottom=0)

    axes[2, 0].set_xlabel('Perturbation angle (°)')
    axes[2, 1].set_xlabel('Perturbation angle (°)')

    fig.suptitle(f'Original (solid) vs orthogonalized (dashed) features — {model}',
                 fontsize=13)
    fig.tight_layout()
    out = f'results/allpairs_contrastive/{model}/orthogonal_features_by_family.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)

    # --- Zoomed version: 0-20 degrees ---
    fig2, axes2 = plt.subplots(3, 2, figsize=(12, 11))
    zoom_mask = angles <= 20

    for row, (family_name, langs) in enumerate(families.items()):
        ax_kl = axes2[row, 0]
        ax_l2 = axes2[row, 1]

        ax_kl.plot(angles[zoom_mask], rand_kl[zoom_mask], 'k--', linewidth=1.5,
                   alpha=0.5, label='random')
        ax_l2.plot(angles[zoom_mask], rand_l2[zoom_mask], 'k--', linewidth=1.5,
                   alpha=0.5, label='random')

        fkey = f'family_{family_name.lower()}'
        if fkey in sr:
            fam_kl = sr[fkey]['kl'].mean(axis=0)
            fam_l2 = sr[fkey]['l2'].mean(axis=0)
            ax_kl.plot(angles[zoom_mask], fam_kl[zoom_mask], 'k-', linewidth=2,
                       alpha=0.7, label=f'{family_name} mean')
            ax_l2.plot(angles[zoom_mask], fam_l2[zoom_mask], 'k-', linewidth=2,
                       alpha=0.7, label=f'{family_name} mean')

        for lang in langs:
            if f'orig_{lang}' not in sr:
                continue
            color = lang_colors[lang]
            label = lang_labels[lang]

            orig_kl = sr[f'orig_{lang}']['kl'].mean(axis=0)
            orig_l2 = sr[f'orig_{lang}']['l2'].mean(axis=0)
            pure_kl = sr[f'pure_{lang}']['kl'].mean(axis=0)
            pure_l2 = sr[f'pure_{lang}']['l2'].mean(axis=0)

            ax_kl.plot(angles[zoom_mask], orig_kl[zoom_mask], '-', color=color,
                       linewidth=1.5, label=f'{label}')
            ax_kl.plot(angles[zoom_mask], pure_kl[zoom_mask], '--', color=color,
                       linewidth=1.5, label=f'{label} (pure)')
            ax_l2.plot(angles[zoom_mask], orig_l2[zoom_mask], '-', color=color,
                       linewidth=1.5, label=f'{label}')
            ax_l2.plot(angles[zoom_mask], pure_l2[zoom_mask], '--', color=color,
                       linewidth=1.5, label=f'{label} (pure)')

        ax_kl.set_ylabel('Mean KL divergence')
        ax_l2.set_ylabel('Mean L2 distance')
        ax_kl.set_title(f'{family_name} — KL')
        ax_l2.set_title(f'{family_name} — L2')
        ax_kl.legend(fontsize=7, loc='upper left')
        ax_l2.legend(fontsize=7, loc='upper left')
        ax_kl.grid(True, alpha=0.3)
        ax_l2.grid(True, alpha=0.3)
        ax_kl.set_ylim(bottom=0)
        ax_l2.set_ylim(bottom=0)

    axes2[2, 0].set_xlabel('Perturbation angle (°)')
    axes2[2, 1].set_xlabel('Perturbation angle (°)')

    fig2.suptitle(f'Zoomed 0–20° — Original (solid) vs orthogonalized (dashed) — {model}',
                  fontsize=13)
    fig2.tight_layout()
    out2 = f'results/allpairs_contrastive/{model}/orthogonal_features_by_family_zoom.png'
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")
    plt.close(fig2)

print("Done!")
