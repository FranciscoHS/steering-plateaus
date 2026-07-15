#!/usr/bin/env python3
"""Plots for the J-Lens superellipse experiment (qwen3-8b L20).

1. Beeswarm of fitted p per category (EN, ZH) across L2 thresholds.
2. Per-pair normalized contour + superellipse fit panels, one figure per
   threshold, for visual inspection of fit quality.

Pure post-processing of saved sweep pkls — no model, runs locally.
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# CJK-capable fallback font for Chinese token labels
plt.rcParams['font.family'] = ['DejaVu Sans', 'AR PL UMing CN']

from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse, superellipse_curve)
from scripts.jlens.sweep_isoplateau_jlens import (
    L2_THRESHOLDS, SWEEP_DIR, SUMMARY_PKL, RESULTS_DIR, find_threshold,
    iter_pairs, DIRECTIONS_PKL)

COLORS = {'EN': '#377eb8', 'ZH': '#e41a1c'}


def beeswarm():
    with open(SUMMARY_PKL, 'rb') as f:
        rows = pickle.load(f)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    rng = np.random.default_rng(0)
    for ti, thresh in enumerate(L2_THRESHOLDS):
        for ci, cat in enumerate(['EN', 'ZH']):
            ps = [r['p'] for r in rows
                  if r['threshold'] == thresh and r['category'] == cat]
            x = ti + (-0.17 if cat == 'EN' else 0.17)
            xs = x + rng.uniform(-0.08, 0.08, len(ps))
            ax.scatter(xs, ps, s=28, alpha=0.75, color=COLORS[cat],
                       label=cat if ti == 0 else None, zorder=3)
            if ps:
                ax.hlines(np.median(ps), x - 0.13, x + 0.13,
                          color=COLORS[cat], lw=2, zorder=4)
    ax.axhline(1.0, color='#999999', ls='--', lw=1, label='p=1 (diamond)')
    ax.axhline(2.0, color='#555555', ls='--', lw=1, label='p=2 (circle)')
    ax.set_xticks(range(len(L2_THRESHOLDS)))
    ax.set_xticklabels([f'L2={t:g}' for t in L2_THRESHOLDS])
    ax.set_ylabel('superellipse p')
    ax.set_title('J-Lens direction pairs — superellipse p by category\n'
                 'qwen3-8b, perturb L20, measure L34, median over 30 acts')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'jlens_p_beeswarm_L20.png')
    plt.savefig(out, dpi=200, facecolor='white')
    plt.close()
    print(f"Saved: {out}")


def shape_panels():
    with open(DIRECTIONS_PKL, 'rb') as f:
        dirs = pickle.load(f)
    pairs = list(iter_pairs(dirs))

    for thresh in L2_THRESHOLDS:
        n = len(pairs)
        ncol = 5
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.1 * nrow))
        axes = np.atleast_2d(axes)
        for ax in axes.flat[n:]:
            ax.axis('off')

        for k, (cat, i, pair) in enumerate(pairs):
            ax = axes.flat[k]
            pkl = os.path.join(SWEEP_DIR, f'sweep_{cat}_{i:02d}.pkl')
            with open(pkl, 'rb') as f:
                res = pickle.load(f)
            angles = res['angles_deg']
            med = np.median(res['l2'], axis=0)
            t1 = find_threshold(med[:, 0], angles, thresh)
            t2 = find_threshold(med[0, :], angles, thresh)
            title = f"{cat}{i}: {res['tokens'][0]}×{res['tokens'][1]}"
            ok = not (t1 <= angles[1] or t2 <= angles[1]
                      or t1 >= angles[-1] or t2 >= angles[-1])
            contour = extract_contour_points(angles, med, thresh) if ok else []
            if ok and len(contour):
                cn = contour.copy()
                cn[:, 0] /= t1
                cn[:, 1] /= t2
                p_fit, ci, info = fit_superellipse(cn, (t1 + t2) / 2)
                for pref, c in [(1, '#cccccc'), (2, '#999999')]:
                    xc, yc = superellipse_curve(pref)
                    ax.plot(xc, yc, '--', color=c, lw=1)
                if p_fit is not None:
                    xf, yf = superellipse_curve(p_fit)
                    ax.plot(xf, yf, '-', color=COLORS[cat], lw=2)
                    title += f"\np={p_fit:.2f} [{ci[0]:.2f},{ci[1]:.2f}]"
                order = np.argsort(np.arctan2(cn[:, 1], cn[:, 0]))
                ax.plot(cn[order, 0], cn[order, 1], 'o', color='#333333',
                        ms=2.5, alpha=0.7)
            else:
                title += "\n(no valid contour)"
            ax.set_title(title, fontsize=8)
            ax.set_xlim(-0.05, 1.45)
            ax.set_ylim(-0.05, 1.45)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=6)

        fig.suptitle(f'J-Lens pair iso-plateau shapes, L2={thresh:g} '
                     f'(normalized by per-axis θ*) — qwen3-8b L20',
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = os.path.join(RESULTS_DIR, f'jlens_shapes_L2_{int(thresh)}.png')
        plt.savefig(out, dpi=170, facecolor='white')
        plt.close()
        print(f"Saved: {out}")


if __name__ == '__main__':
    beeswarm()
    shape_panels()
