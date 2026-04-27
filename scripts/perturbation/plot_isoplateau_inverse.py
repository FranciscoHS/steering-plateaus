#!/usr/bin/env python3
"""Inverse of the iso-plateau (circle-boundary) plot.

Original (forward): fix a response threshold T, sweep mixing angle phi,
find the input magnitude r(phi) needed to reach T → contour in INPUT space.

Inverse (this script): fix input magnitude r, sweep mixing angle phi,
read off the response magnitude → curve in RESPONSE space. One curve per r.
"""
import os, pickle, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


def load_grid(pkl_path):
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    angles = np.asarray(d['angles_deg'], dtype=float)
    l2 = np.median(d['l2'], axis=0).astype(float)        # [n_a, n_a]
    return angles, l2, d.get('direction_labels', ('A', 'B'))


def sample_radial(angles, grid, r_values, phis):
    """For each (r, phi) return grid value at (alpha1, alpha2) = (r cos phi, r sin phi)."""
    interp = RegularGridInterpolator((angles, angles), grid,
                                     bounds_error=False, fill_value=np.nan)
    out = np.full((len(r_values), len(phis)), np.nan)
    for i, r in enumerate(r_values):
        a1 = r * np.cos(phis)
        a2 = r * np.sin(phis)
        in_box = (a1 <= angles[-1]) & (a2 <= angles[-1])
        pts = np.stack([a1[in_box], a2[in_box]], axis=1)
        out[i, in_box] = interp(pts)
    return out  # [n_r, n_phi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl',
                    default='results/paper_L2_gemma2_9b/sweep_isoplateau_L2_xwide.pkl')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--tag', default='L2_xwide')
    ap.add_argument('--r_values', default='5,10,20,40,70',
                    help='Comma-separated input magnitudes r (deg)')
    ap.add_argument('--n_phi', type=int, default=180)
    args = ap.parse_args()

    angles, l2, labels = load_grid(args.pkl)
    out_dir = args.out_dir or os.path.dirname(args.pkl)
    r_values = np.array([float(x) for x in args.r_values.split(',')])
    phis = np.linspace(0.0, np.pi / 2, args.n_phi)
    phis_deg = np.degrees(phis)

    l2_curves = sample_radial(angles, l2, r_values, phis)

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(1, len(r_values) - 1)) for i in range(len(r_values))]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, r in enumerate(r_values):
        ax.plot(phis_deg, l2_curves[i], color=colors[i], lw=2,
                label=fr'$r={r:g}^\circ$')
    ax.set_xlabel(fr'mixing angle $\varphi$ (deg) — '
                  fr'$0^\circ$={labels[0]}, $90^\circ$={labels[1]}')
    ax.set_ylabel(r'$\|\Delta h\|_2$  (response magnitude)')
    ax.set_xlim(0, 90)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, frameon=False, ncol=2,
              title='input magnitude', title_fontsize=9)
    ax.set_title('Response magnitude vs perturbation direction\n'
                 'at fixed input magnitudes', fontsize=11)
    fig.tight_layout()
    out = os.path.join(out_dir, f'isoplateau_inverse_{args.tag}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
