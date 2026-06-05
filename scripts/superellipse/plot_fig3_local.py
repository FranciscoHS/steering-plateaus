#!/usr/bin/env python3
"""Local variant of fsec-anonymous's plot_fig3_boundary.py with a wider
plot window so contour points past 1.15 * theta are visible. Reads a
sweep2d pkl path directly.

Usage:
    python scripts/superellipse/plot_fig3_local.py \\
        --pkl results/fdc_pfit/sweep2d_gemma_L15_Chinese__df_fineweb_60deg_dirfdc.pkl \\
        --d1 Chinese --d2 df --out results/fdc_pfit/fig3_Chinese_df_wide.png
"""
import argparse, os, pickle, sys
sys.path.insert(0, os.path.expanduser('~/Research/fsec-anonymous'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.lib.superellipse import (
    extract_contour, axis_intercept, fit_superellipse, superellipse_curve,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--d1', required=True)
    ap.add_argument('--d2', required=True)
    ap.add_argument('--thresh', type=float, default=None)
    ap.add_argument('--axes_pad', type=float, default=None,
                    help='multiplier for plot window in units of theta. '
                         'Default: auto-fit to max(1.15, max(|contour|)/theta + 0.05)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    d = pickle.load(open(args.pkl, 'rb'))
    angles = d['angles_deg']; grid = d['l2']
    mg = np.median(grid, axis=0)
    if args.thresh is None:
        thresh = 0.5 * min(mg[:, 0].max(), mg[0, :].max())
    else:
        thresh = args.thresh
    t1 = axis_intercept(angles, mg[:, 0], thresh)
    t2 = axis_intercept(angles, mg[0, :], thresh)
    print(f'thresh={thresh:.2f}  theta_{args.d1}={t1:.2f}  theta_{args.d2}={t2:.2f}')

    contour = extract_contour(angles, mg, thresh)
    raw = contour.astype(float).copy()
    norm = raw.copy(); norm[:, 0] /= t1; norm[:, 1] /= t2
    keep = (norm[:, 0] > 0.05) & (norm[:, 1] > 0.05) \
        & (norm[:, 0] < 1.5) & (norm[:, 1] < 1.5)
    norm = norm[keep]; raw = raw[keep]
    print(f'contour pts kept: {keep.sum()}/{len(keep)}')

    fit = fit_superellipse(norm[:, 0], norm[:, 1])
    p_fit = fit['p']
    print(f'p_fit={p_fit:.3f}  mean_radial_frac={fit["mean_radial_frac"]:.3f}  '
          f'n_pts={fit["n_pts"]}')

    # Auto-determine axis pad
    if args.axes_pad is None:
        pad1 = max(1.15, raw[:, 0].max() / t1 + 0.05)
        pad2 = max(1.15, raw[:, 1].max() / t2 + 0.05)
    else:
        pad1 = pad2 = args.axes_pad

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    # p=inf rectangle
    xi, yi = superellipse_curve(20)
    ax.plot(t1 * xi, t2 * yi, ls=(0, (1.8, 2.5)),
            color='#D9822B', lw=1.6, alpha=0.95,
            label=r'Per-feature threshold ($p\to\infty$)')
    # p=2 ellipse
    xc, yc = superellipse_curve(2)
    ax.plot(t1 * xc, t2 * yc, '-', color='#1F77B4', lw=2.6, alpha=0.95,
            label=r'Euclidean ($p=2$)')
    # Fitted superellipse
    xf, yf = superellipse_curve(p_fit)
    ax.plot(t1 * xf, t2 * yf, '-', color='#3a7d3a', lw=2.0, alpha=0.95,
            label=rf'Fit ($p={p_fit:.2f}$, resid={100*fit["mean_radial_frac"]:.1f}%)')
    # Data points (sorted by angle)
    phi = np.arctan2(raw[:, 1], raw[:, 0])
    order = np.argsort(phi)
    ax.plot(raw[order, 0], raw[order, 1], 'o',
            color='#3a3a3a', ms=4, alpha=0.85, mec='none', label='Data')

    ax.set_xlim(-0.03 * t1, pad1 * t1)
    ax.set_ylim(-0.03 * t2, pad2 * t2)
    ax.set_aspect('equal')
    ax.set_xlabel(rf'$\alpha\cos\varphi$  ({args.d1}, deg)', fontsize=12)
    ax.set_ylabel(rf'$\alpha\sin\varphi$  ({args.d2}, deg)', fontsize=12)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(True, color='#bbbbbb', lw=0.5, alpha=0.45)
    ax.legend(fontsize=9, loc='upper right', frameon=True, framealpha=0.95)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=220, bbox_inches='tight', facecolor='white')
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
