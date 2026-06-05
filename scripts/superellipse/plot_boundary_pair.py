#!/usr/bin/env python3
"""
Fig 3-style iso-plateau boundary plots for the learned (d1, d2) pair, for both
the optimized (training) and fresh (held-out) anchor sets.

Reproduces fsec-anonymous/scripts/plotting/plot_fig3_boundary.py but reads the
median L2 grids already computed by fit_superellipse_pair_v2.py (stored in
fit_canonical.pkl), so no model forward is needed.

Each panel plots the iso-L2 contour in raw (alpha_1, alpha_2) degrees with three
references: the p=2 ellipse ("circle"), the p->inf rectangle (per-feature
threshold), and the fitted superellipse. Axis limits are widened to show the
full contour, so over-suppression that blows past the p->inf box is visible.

Usage:
    python scripts/superellipse/plot_boundary_pair.py \
        --fit_pkl results/superellipse_dict/qwen3-1.7b_L2_a30/fit_canonical/fit_canonical.pkl
"""
import os, sys, argparse, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.superellipse.canonical.superellipse import (
    extract_contour, axis_intercept, fit_superellipse, superellipse_curve,
)


def boundary_for_set(angles, median_grid, exact):
    """Return everything needed to draw one panel."""
    base = min(median_grid[:, 0].max(), median_grid[0, :].max())
    thresh = 0.5 * base
    theta1 = axis_intercept(angles, median_grid[:, 0], thresh)
    theta2 = axis_intercept(angles, median_grid[0, :], thresh)
    contour = extract_contour(angles, median_grid, thresh)  # raw (a1,a2) deg
    cn_raw = contour.astype(float)
    # normalized coords (for fit + masking), exact-geodesic or small-angle
    cn_norm = cn_raw.copy()
    if exact:
        r_deg = np.hypot(cn_norm[:, 0], cn_norm[:, 1])
        r_safe = np.where(r_deg > 1e-12, r_deg, 1.0)
        r_rad = np.deg2rad(r_deg)
        cn_norm[:, 0] = (np.sin(r_rad) / np.sin(np.deg2rad(theta1))) * (cn_norm[:, 0] / r_safe)
        cn_norm[:, 1] = (np.sin(r_rad) / np.sin(np.deg2rad(theta2))) * (cn_norm[:, 1] / r_safe)
    else:
        cn_norm[:, 0] /= theta1; cn_norm[:, 1] /= theta2
    keep = (cn_norm[:, 0] > 0.05) & (cn_norm[:, 1] > 0.05) \
        & (cn_norm[:, 0] < 1.5) & (cn_norm[:, 1] < 1.5)
    fit = fit_superellipse(cn_norm[keep, 0], cn_norm[keep, 1])
    return {'thresh': thresh, 'theta1': theta1, 'theta2': theta2,
            'cn_raw': cn_raw, 'keep': keep, 'fit': fit,
            'axis_max': (median_grid[:, 0].max(), median_grid[0, :].max())}


def draw_panel(ax, b, title, d1lab='d1', d2lab='d2'):
    t1, t2 = b['theta1'], b['theta2']
    p_fit = b['fit']['p']
    cn_raw, keep = b['cn_raw'], b['keep']
    # p->inf rectangle (semi-axes t1, t2)
    xi, yi = superellipse_curve(20)
    ax.plot(t1 * xi, t2 * yi, ls=(0, (1.8, 2.5)), color='#D9822B', lw=1.6,
            label=r'Per-feature threshold ($p\to\infty$)')
    # p=2 ellipse ("circle")
    xc, yc = superellipse_curve(2)
    ax.plot(t1 * xc, t2 * yc, '-', color='#1F77B4', lw=2.6,
            label=r'Euclidean ($p=2$)')
    # fitted superellipse
    if np.isfinite(p_fit):
        xf, yf = superellipse_curve(p_fit)
        ax.plot(t1 * xf, t2 * yf, '-', color='#3a7d3a', lw=2.0, zorder=4,
                label=rf'Fit ($p={p_fit:.2f}$, n={b["fit"]["n_pts"]})')
    # data: contour points (kept = used in fit; dropped = out-of-box)
    cont_angle = np.arctan2(cn_raw[:, 1], cn_raw[:, 0]); order = np.argsort(cont_angle)
    ax.plot(cn_raw[keep][np.argsort(np.arctan2(cn_raw[keep, 1], cn_raw[keep, 0]))][:, 0],
            cn_raw[keep][np.argsort(np.arctan2(cn_raw[keep, 1], cn_raw[keep, 0]))][:, 1],
            'o', color='#3a3a3a', ms=4, alpha=0.85, mec='none', label='Data (in fit)', zorder=5)
    dropped = ~keep
    if dropped.any():
        ax.plot(cn_raw[dropped, 0], cn_raw[dropped, 1], 'x', color='#c0392b',
                ms=5, alpha=0.7, label='Data (out of box, dropped)', zorder=5)
    # widen limits to show the full contour
    xmax = max(1.15 * t1, cn_raw[:, 0].max() * 1.05)
    ymax = max(1.15 * t2, cn_raw[:, 1].max() * 1.05)
    ax.set_xlim(-0.03 * t1, xmax); ax.set_ylim(-0.03 * t2, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel(rf'$\alpha_1$ ({d1lab}, deg)')
    ax.set_ylabel(rf'$\alpha_2$ ({d2lab}, deg)')
    ax.set_title(title, fontsize=11)
    ax.grid(True, color='#bbbbbb', lw=0.5, alpha=0.45)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit_pkl', type=str,
                    default='results/superellipse_dict/qwen3-1.7b_L2_a30/fit_canonical/fit_canonical.pkl')
    ap.add_argument('--exact', action='store_true', default=True,
                    help='exact-geodesic normalization (default True)')
    ap.add_argument('--small_angle', dest='exact', action='store_false')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    with open(args.fit_pkl, 'rb') as f:
        d = pickle.load(f)
    angles = d['angles_deg']
    out = args.out or os.path.join(os.path.dirname(args.fit_pkl), 'boundary_pair.png')

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, set_name in zip(axes, ['optimized', 'fresh']):
        median_grid = np.median(d['sets'][set_name]['l2_grid'], axis=0)
        b = boundary_for_set(angles, median_grid, args.exact)
        print(f'{set_name}: thresh={b["thresh"]:.0f} theta1={b["theta1"]:.1f} '
              f'theta2={b["theta2"]:.1f} p_fit={b["fit"]["p"]:.2f} '
              f'n_pts={b["fit"]["n_pts"]} dropped={int((~b["keep"]).sum())}')
        n_anch = d['sets'][set_name]['l2_grid'].shape[0]
        draw_panel(ax, b, f'{set_name} ({n_anch} anchors): p={b["fit"]["p"]:.2f}')
    norm = 'exact-geodesic' if args.exact else 'small-angle'
    fig.suptitle(f'Iso-plateau boundary, learned pair (norm: {norm})', fontsize=13)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'saved {out}')
    print(f'saved {out.replace(".png", ".pdf")}')


if __name__ == '__main__':
    main()
