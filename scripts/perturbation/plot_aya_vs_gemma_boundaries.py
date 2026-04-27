#!/usr/bin/env python3
"""Compare aya vs gemma boundary plots for a few representative pairs.

Aya wins on closest-to-p=2 across mean/median/std. This generates side-by-side
boundary plots so we can visually check whether aya's contours are cleaner /
more Euclidean-looking than gemma's.

Reads from results/fig4_robustness/v2_A_gemma_*.pkl + v2_H_aya_*.pkl.
Output: writeup/figures/aya_vs_gemma_boundaries.png/.pdf
"""
import os, sys, glob, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse, superellipse_curve)

DIR = 'results/fig4_robustness'
OUT_PNG = 'writeup/figures/aya_vs_gemma_boundaries.png'
OUT_PDF = 'writeup/figures/aya_vs_gemma_boundaries.pdf'

PAIRS = [('Formal', 'Portuguese'), ('Gender', 'Portuguese'),
         ('Sentiment', 'Portuguese'), ('Formal', 'Spanish')]


def find_theta(curve, ang, thr):
    idx = np.searchsorted(curve, thr)
    if idx == 0 or idx >= len(ang): return ang[min(idx, len(ang) - 1)]
    x0, x1 = ang[idx-1], ang[idx]; y0, y1 = curve[idx-1], curve[idx]
    if y1 == y0: return x0
    return x0 + (thr - y0) / (y1 - y0) * (x1 - x0)


def fit_one(pkl_path, frac=0.6):
    with open(pkl_path, 'rb') as f: d = pickle.load(f)
    grid = np.median(d['l2'], axis=0)
    angles = d['angles_deg']
    c1max = grid[:,0].max(); c2max = grid[0,:].max()
    thr = frac * min(c1max, c2max)
    t1 = find_theta(grid[:,0], angles, thr)
    t2 = find_theta(grid[0,:], angles, thr)
    pts = extract_contour_points(angles, grid, thr)
    if len(pts) == 0: return None
    cn = pts.astype(float).copy(); cn[:,0]/=t1; cn[:,1]/=t2
    p, _, _ = fit_superellipse(cn, (t1+t2)/2)
    return {'cn': cn, 'p': p, 't1': t1, 't2': t2, 'thr': thr,
            'labels': d['direction_labels']}


def panel(ax, fit, title):
    if fit is None:
        ax.text(0.5, 0.5, 'no fit', ha='center', va='center')
        ax.set_title(title); return
    # Reference: p=2 (Euclidean, solid blue) and p=20 (per-feature-ish, dashed orange)
    xi, yi = superellipse_curve(20)
    ax.plot(xi, yi, ls=(0, (2.5, 3.0)), color='#D9822B', lw=1.6, alpha=0.85,
            label=r'$p\to\infty$')
    xc, yc = superellipse_curve(2)
    ax.plot(xc, yc, '-', color='#1F77B4', lw=2.2, alpha=0.85,
            label=r'$p=2$')
    # Fitted superellipse (green)
    xf, yf = superellipse_curve(fit['p'])
    ax.plot(xf, yf, '-', color='#2ca02c', lw=1.4, alpha=0.85,
            label=f"fit $p$={fit['p']:.2f}")
    # Data points
    cn = fit['cn']
    cont_angle = np.arctan2(cn[:,1], cn[:,0]); order = np.argsort(cont_angle)
    ax.plot(cn[order,0], cn[order,1], 'o', color='#3a3a3a',
            markersize=4.5, alpha=0.9, markeredgecolor='none')
    ax.set_xlim(-0.05, 1.20); ax.set_ylim(-0.05, 1.20)
    ax.set_aspect('equal')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel(fit['labels'][0], fontsize=10)
    ax.set_ylabel(fit['labels'][1], fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_title(title, fontsize=11, pad=8)


def main():
    plt.rcParams['font.family'] = 'Inter'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'cm'

    fig, axes = plt.subplots(2, len(PAIRS), figsize=(3.4 * len(PAIRS), 7.0))

    for col, (a, b) in enumerate(PAIRS):
        gemma_pkl = os.path.join(DIR, f'v2_A_gemma_{a}_{b}.pkl')
        aya_pkl   = os.path.join(DIR, f'v2_H_aya_{a}_{b}.pkl')
        gemma_fit = fit_one(gemma_pkl) if os.path.exists(gemma_pkl) else None
        aya_fit   = fit_one(aya_pkl)   if os.path.exists(aya_pkl)   else None
        panel(axes[0, col], gemma_fit, f'Gemma-2-9B  —  {a}×{b}')
        panel(axes[1, col], aya_fit,   f'Aya-Expanse-8B  —  {a}×{b}')

    # one shared legend
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    print(f'Saved {OUT_PNG}')


if __name__ == '__main__':
    main()
