#!/usr/bin/env python3
"""Single-pair iso-plateau shape plots for J-Lens sweeps, in the original
plot_shape() style from sweep_isoplateau_finegrid.py (full-size figure,
p=1/2/inf references, radial-error box, legend).

Usage: python scripts/jlens/plot_jlens_shape_single.py [PAIR ...] [--thresh T]
       PAIR like EN_00 or ZH_03; defaults: EN_00 ZH_00 at L2=120.
"""
import os, sys, pickle, argparse
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['DejaVu Sans', 'AR PL UMing CN']

from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse, superellipse_curve)
from scripts.jlens.sweep_isoplateau_jlens import (
    SWEEP_DIR, RESULTS_DIR, find_threshold)


def plot_one(tag, thresh):
    pkl = os.path.join(SWEEP_DIR, f'sweep_{tag}.pkl')
    with open(pkl, 'rb') as f:
        res = pickle.load(f)
    angles = res['angles_deg']
    med = np.median(res['l2'], axis=0)
    t1, t2 = res['tokens']

    theta_1 = find_threshold(med[:, 0], angles, thresh)
    theta_2 = find_threshold(med[0, :], angles, thresh)
    contour = extract_contour_points(angles, med, thresh)
    contour_norm = contour.copy()
    contour_norm[:, 0] /= theta_1
    contour_norm[:, 1] /= theta_2
    p_fit, ci, info = fit_superellipse(contour_norm, (theta_1 + theta_2) / 2)

    fig, ax = plt.subplots(figsize=(6, 6))
    x1 = 1 - np.linspace(0, 1, 200)
    y1 = np.linspace(0, 1, 200)
    ax.plot(x1, y1, '--', color='#aaaaaa', lw=1.5, label='p=1 (diamond)')
    xc, yc = superellipse_curve(2)
    ax.plot(xc, yc, '--', color='#888888', lw=1.5, label='p=2 (circle)')
    xs, ys = superellipse_curve(20)
    ax.plot(xs, ys, '--', color='#bbbbbb', lw=1.5, label='p→∞ (square)')

    if p_fit is not None:
        xf, yf = superellipse_curve(p_fit)
        ax.plot(xf, yf, '-', color='#e41a1c', lw=2.5,
                label=f'fit: p={p_fit:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]')
        ax.text(0.03, 0.97,
                f'radial error: {info["mean_radial_deg"]:.2f}° mean, '
                f'{info["max_radial_deg"]:.2f}° max\n'
                f'({info["n_pts"]} contour pts)',
                transform=ax.transAxes, fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    order = np.argsort(np.arctan2(contour_norm[:, 1], contour_norm[:, 0]))
    ax.plot(contour_norm[order, 0], contour_norm[order, 1], 'o',
            color='#377eb8', markersize=3, alpha=0.7, label='data')

    ax.set_xlabel(f'α₁ / θ₁*  ({t1}, normalized)', fontsize=12)
    ax.set_ylabel(f'α₂ / θ₂*  ({t2}, normalized)', fontsize=12)
    ax.set_title(f'Iso-Plateau Shape (L2={thresh:g}) — J-Lens {tag}\n'
                 f'θ₁*={theta_1:.2f}°, θ₂*={theta_2:.2f}° · qwen3-8b L20',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-0.05, 1.5)
    ax.set_ylim(-0.05, 1.5)
    ax.set_aspect('equal')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.plot(1, 0, 'kv', markersize=8, zorder=5)
    ax.plot(0, 1, 'k>', markersize=8, zorder=5)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR,
                       f'jlens_shape_single_{tag}_L2_{int(thresh)}.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs', nargs='*', default=['EN_00', 'ZH_00'])
    ap.add_argument('--thresh', type=float, default=120.0)
    args = ap.parse_args()
    for tag in args.pairs:
        plot_one(tag, args.thresh)
