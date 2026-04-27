#!/usr/bin/env python3
"""Clean iso-plateau shape plot: data + p=2 circle + p=∞ square, no fit line.

Reads an existing sweep_isoplateau_<tag>.pkl, extracts the contour at a given
threshold, normalizes by axis intercepts, and plots against reference shapes.
Defaults to the mandarin_haskell pair on Gemma-2-9b.
"""
import os, sys, pickle, argparse
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_isoplateau import extract_contour_points, superellipse_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='results/mandarin_haskell/gemma-2-9b/sweep_isoplateau_mandarin_haskell.pkl')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--tag', default='mandarin_haskell')
    ap.add_argument('--labels', default='Mandarin,Haskell',
                    help='Comma-separated axis labels')
    ap.add_argument('--metrics', default='l2,cosine',
                    help='Comma-separated list: l2, cosine')
    ap.add_argument('--l2_thresh', type=float, default=50.0)
    ap.add_argument('--cos_thresh', type=float, default=0.01)
    ap.add_argument('--include_diamond', action='store_true',
                    help='Include p=1 diamond reference as well (default: no)')
    ap.add_argument('--phi_sample', type=int, default=0,
                    help='If >0, sample the contour radially at this many mixing '
                         'angles phi in [0, pi/2], color by phi. Otherwise plot raw '
                         'matplotlib-extracted contour points.')
    ap.add_argument('--unnormalized', action='store_true',
                    help='Plot in degrees (axes = theta*cos(phi), theta*sin(phi)); '
                         'reference shapes scaled by (theta_1, theta_2).')
    args = ap.parse_args()

    with open(args.pkl, 'rb') as f:
        results = pickle.load(f)
    angles = results['angles_deg']
    out_dir = args.out_dir or os.path.dirname(args.pkl)
    labels = args.labels.split(',')

    metrics_info = []
    if 'l2' in args.metrics.split(','):
        metrics_info.append(('l2', args.l2_thresh, 'L2 distance'))
    if 'cosine' in args.metrics.split(','):
        metrics_info.append(('cosine', args.cos_thresh, 'Cosine distance'))

    for metric, threshold, label in metrics_info:
        data = results[metric]  # [N_ACT, N, N]
        median_grid = np.median(data, axis=0)

        pt_curve = median_grid[:, 0]   # α₂=0
        gen_curve = median_grid[0, :]  # α₁=0
        def find_threshold(curve):
            idx = np.searchsorted(curve, threshold)
            if idx == 0 or idx >= len(angles):
                return angles[idx] if idx < len(angles) else angles[-1]
            x0, x1 = angles[idx-1], angles[idx]
            y0, y1 = curve[idx-1], curve[idx]
            return x0 + (threshold - y0) / (y1 - y0) * (x1 - x0) if y1 != y0 else x0
        theta1 = find_threshold(pt_curve)
        theta2 = find_threshold(gen_curve)

        if args.phi_sample > 0:
            # Sample phi in [0, pi/2], for each phi find the radial breaking magnitude
            from scipy.interpolate import RegularGridInterpolator
            phis = np.linspace(0.02, np.pi/2 - 0.02, args.phi_sample)
            # Interpolator on the (α1, α2) grid
            interp = RegularGridInterpolator(
                (angles, angles), median_grid,
                bounds_error=False, fill_value=None)
            cn_list = []
            phis_used = []
            # March outward along each radial until we cross threshold
            r_grid = np.linspace(0.2, min(angles[-1], max(angles))*1.4, 400)
            for phi in phis:
                pts = np.stack([r_grid * np.cos(phi), r_grid * np.sin(phi)], axis=1)
                vals = interp(pts)
                # Find first crossing
                above = np.where(vals >= threshold)[0]
                if len(above) == 0:
                    continue
                k = above[0]
                if k == 0:
                    continue
                # Linear interpolate
                v0, v1 = vals[k-1], vals[k]
                r0, r1 = r_grid[k-1], r_grid[k]
                if v1 == v0: continue
                t = (threshold - v0) / (v1 - v0)
                r_star = r0 + t * (r1 - r0)
                a1 = r_star * np.cos(phi); a2 = r_star * np.sin(phi)
                cn_list.append([a1 / theta1, a2 / theta2])
                phis_used.append(phi)
            cn = np.array(cn_list)
            phis_used = np.array(phis_used)
            if args.unnormalized:
                cn[:, 0] *= theta1; cn[:, 1] *= theta2
        else:
            contour_pts = extract_contour_points(angles, median_grid, threshold)
            if len(contour_pts) == 0:
                print(f'no contour for {metric}'); continue
            cn = contour_pts.astype(float).copy()
            if not args.unnormalized:
                cn[:, 0] /= theta1; cn[:, 1] /= theta2
            phis_used = None

        fig, ax = plt.subplots(figsize=(6, 6))
        sx = theta1 if args.unnormalized else 1.0
        sy = theta2 if args.unnormalized else 1.0
        # per-feature threshold prediction — square (normalized) / rectangle (unnormalized)
        xs, ys = superellipse_curve(20)
        ax.plot(xs * sx, ys * sy, '--', color='#d62728', lw=2.0,
                label='per-feature threshold')
        # Euclidean combination — circle (normalized) / ellipse (unnormalized)
        xc, yc = superellipse_curve(2)
        ax.plot(xc * sx, yc * sy, '--', color='#1f77b4', lw=2.0,
                label='Euclidean combination')
        if args.include_diamond:
            x1 = 1 - np.linspace(0, 1, 200); y1 = np.linspace(0, 1, 200)
            ax.plot(x1 * sx, y1 * sy, ':', color='#7f7f7f', lw=1.5, label='diamond')
        # data (plain dots, no color scale)
        if phis_used is not None and len(phis_used) > 0:
            ax.plot(cn[:, 0], cn[:, 1], 'o', color='#2ca02c',
                    markersize=5, alpha=0.9, label='data')
        else:
            cont_angle = np.arctan2(cn[:, 1], cn[:, 0])
            order = np.argsort(cont_angle)
            ax.plot(cn[order, 0], cn[order, 1], 'o', color='#2ca02c',
                    markersize=4, alpha=0.8, label='data')
        ax.plot(1 * sx, 0, 'kv', markersize=8, zorder=5)
        ax.plot(0, 1 * sy, 'k>', markersize=8, zorder=5)

        if args.unnormalized:
            xmax = max(theta1, theta2) * 1.15
            ax.set_xlim(-0.03 * xmax, xmax); ax.set_ylim(-0.03 * xmax, xmax)
            ax.set_xlabel(rf'$\theta(\varphi)\cos\varphi$  [deg, along {labels[0]}]',
                          fontsize=12)
            ax.set_ylabel(rf'$\theta(\varphi)\sin\varphi$  [deg, along {labels[1]}]',
                          fontsize=12)
        else:
            ax.set_xlim(-0.03, 1.15); ax.set_ylim(-0.03, 1.15)
            ax.set_xlabel(rf'$\theta(\varphi)\cos\varphi / \theta_1$  ({labels[0]})',
                          fontsize=12)
            ax.set_ylabel(rf'$\theta(\varphi)\sin\varphi / \theta_2$  ({labels[1]})',
                          fontsize=12)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=10, loc='lower left', frameon=True, framealpha=0.95)

        suffix = '_unnorm' if args.unnormalized else ''
        out = os.path.join(out_dir, f'isoplateau_clean_{args.tag}_{metric}{suffix}.png')
        plt.tight_layout()
        plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'Saved {out}')


if __name__ == '__main__':
    main()
