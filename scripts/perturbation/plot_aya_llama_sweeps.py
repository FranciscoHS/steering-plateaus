#!/usr/bin/env python3
"""1D perturbation sweeps (L² vs angle) for non-language contrastive directions
on aya and llama, comparing to a random baseline.

For each model, we plot the L² response when perturbing along ONE direction
(d2 angle = 0). The 1D sweep is grid[:, 0] from a 2D pair PKL — the perturbed
direction is `direction_labels[0]` (which in our 30-pair set is always a SEM
concept: Gender / Sentiment / Formal).

Output: writeup/figures/sweeps_aya_llama.png/.pdf
"""
import os, sys, glob, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = 'results/fig4_robustness'
OUT_PNG = 'writeup/figures/sweeps_aya_llama.png'
OUT_PDF = 'writeup/figures/sweeps_aya_llama.pdf'

# Per-model: pick one pair per SEM direction (the d2 doesn't affect grid[:,0]
# since axis 1 is the sweep axis with axis 2 angle = 0)
SEM_DIRS = ['Gender', 'Sentiment', 'Formal']
SEM_COLORS = {'Gender': '#d62728', 'Sentiment': '#2ca02c', 'Formal': '#9467bd'}
PARTNER = 'Portuguese'   # any language; only used to find the pair PKL


def load_grid(model, sem):
    pkl = os.path.join(DIR, f'v2_H_{model}_{sem}_{PARTNER}.pkl')
    if not os.path.exists(pkl):
        # gemma's are stored under A_, others under H_
        pkl = os.path.join(DIR, f'v2_A_{model}_{sem}_{PARTNER}.pkl')
    if not os.path.exists(pkl): return None, None, None
    with open(pkl, 'rb') as f: d = pickle.load(f)
    angles = d['angles_deg']
    # grid shape: (n_seeds, n_angles, n_angles); marginal sweep along d1 = grid[:, :, 0]
    sweep = d['l2'][:, :, 0]   # (n_seeds, n_angles)
    return angles, sweep, d.get('direction_labels')


def load_random(model):
    """Concatenate all v2_B_<model>_*.pkl 1D marginals as random baselines."""
    pkls = sorted(glob.glob(os.path.join(DIR, f'v2_B_{model}_*.pkl')))
    if not pkls: return None, None
    sweeps = []
    angles = None
    for pkl in pkls:
        with open(pkl, 'rb') as f: d = pickle.load(f)
        angles = d['angles_deg']
        sweeps.append(d['l2'][:, :, 0])   # axis-1 marginal
    return angles, np.concatenate(sweeps, axis=0)


def panel(ax, model, model_label):
    rnd_angles, rnd_mat = load_random(model)
    # IQR band for random
    if rnd_mat is not None:
        rnd_med = np.median(rnd_mat, axis=0)
        rnd_lo  = np.percentile(rnd_mat, 25, axis=0)
        rnd_hi  = np.percentile(rnd_mat, 75, axis=0)
        ax.fill_between(rnd_angles, rnd_lo, rnd_hi, color='gray', alpha=0.22)
        ax.plot(rnd_angles, rnd_med, '-', color='gray', lw=2.0, label='random (median)')

    ymax = 0.0
    for sem in SEM_DIRS:
        angles, mat, labels = load_grid(model, sem)
        if mat is None:
            print(f'  no data for {model} {sem}×{PARTNER}')
            continue
        med = np.median(mat, axis=0)
        lo  = np.percentile(mat, 25, axis=0)
        hi  = np.percentile(mat, 75, axis=0)
        c = SEM_COLORS[sem]
        ax.fill_between(angles, lo, hi, color=c, alpha=0.15)
        ax.plot(angles, med, '-', color=c, lw=2.2, label=sem)
        ymax = max(ymax, float(hi.max()))

    ax.set_xlim(0, 30)
    if ymax > 0:
        # add headroom for random IQR + label
        if rnd_mat is not None:
            ymax = max(ymax, float(np.percentile(rnd_mat, 75, axis=0).max()))
        ax.set_ylim(0, ymax * 1.10)
    ax.set_xlabel(r'Perturbation angle $\theta$ (deg)', fontsize=11)
    ax.set_ylabel(r'$L^2$ distance at penultimate layer', fontsize=11)
    ax.set_title(model_label, fontsize=12)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, fontsize=9, loc='upper left')


def main():
    plt.rcParams['font.family'] = 'Inter'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'cm'

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    panel(axes[0], 'aya',   'Aya-Expanse-8B  ·  perturb L4 → measure L30')
    panel(axes[1], 'llama', 'Llama-3.1-8B  ·  perturb L4 → measure L30')

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    print(f'Saved {OUT_PNG}')


if __name__ == '__main__':
    main()
