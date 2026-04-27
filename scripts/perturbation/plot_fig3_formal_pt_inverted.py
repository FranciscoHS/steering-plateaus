#!/usr/bin/env python3
"""Inverted view of the Formal x Portuguese iso-plateau plot.

Axes are MARGINAL responses (the response you would get if a single component
of the perturbation were applied along its direction alone):
  x = response(0, r * sin(phi))   "response when fully along Portuguese"
  y = response(r * cos(phi), 0)   "response when fully along Formal"

Each curve corresponds to a fixed rotation magnitude r. As phi rotates from
0 (pure Formal) to 90 deg (pure Portuguese), the curve sweeps from
(0, response(r, 0)) to (response(0, r), 0).
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator

PKL = 'results/paper_L2_gemma2_9b/sweep_isoplateau_L2_wide.pkl'
OUT_PNG = 'writeup/figures/fig3_pt_gender_inverted.png'
OUT_PDF = 'writeup/figures/fig3_pt_gender_inverted.pdf'

R_VALUES = np.arange(2.0, 60.0 + 1e-9, 2.0)


def main():
    with open(PKL, 'rb') as f:
        res = pickle.load(f)
    angles = np.asarray(res['angles_deg'], dtype=float)  # 0..30, 41 pts
    median_grid = np.median(res['l2'], axis=0)           # (41, 41)
    labels = res['direction_labels']                     # ('Formal','Portuguese')

    interp = RegularGridInterpolator(
        (angles, angles), median_grid,
        bounds_error=False, fill_value=np.nan)

    # Marginal response curves along each axis (perturbation only along F or P).
    # marg_F(t) = response(t, 0), marg_P(t) = response(0, t).
    def marg_F(t):
        return interp(np.stack([t, np.zeros_like(t)], axis=-1))

    def marg_P(t):
        return interp(np.stack([np.zeros_like(t), t], axis=-1))

    # No normalization: plot raw marginal responses on each axis.

    phi = np.linspace(0.0, 90.0, 361)
    phi_rad = np.deg2rad(phi)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    cmap = plt.get_cmap('viridis')
    norm_color = Normalize(vmin=float(R_VALUES.min()),
                           vmax=float(R_VALUES.max()))

    r_max_grid = angles[-1]
    x_all_max, y_all_max = 0.0, 0.0
    for r in R_VALUES:
        tF = r * np.cos(phi_rad)
        tP = r * np.sin(phi_rad)
        mask = (tF <= r_max_grid + 1e-9) & (tP <= r_max_grid + 1e-9)
        x = marg_P(tP[mask])
        y = marg_F(tF[mask])
        color = cmap(norm_color(r))
        ax.plot(x, y, '-', color=color, lw=1.2, alpha=0.85, zorder=2)
        # Mark phi=0 (top, pure Formal) and phi=90 (right, pure Portuguese).
        ax.plot(x[0], y[0], 'o', color=color, markersize=3.5,
                markeredgecolor='white', markeredgewidth=0.5, zorder=3)
        ax.plot(x[-1], y[-1], 'o', color=color, markersize=3.5,
                markeredgecolor='white', markeredgewidth=0.5, zorder=3)
        x_all_max = max(x_all_max, np.nanmax(x))
        y_all_max = max(y_all_max, np.nanmax(y))

    lim_x = x_all_max * 1.05
    lim_y = y_all_max * 1.05
    ax.set_xlim(0, lim_x)
    ax.set_ylim(0, lim_y)
    ax.set_xlabel(f'response when fully along {labels[1]} '
                  '(L2)', fontsize=11)
    ax.set_ylabel(f'response when fully along {labels[0]} '
                  '(L2)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(f'Marginal response curves '
                 f'({labels[0]} x {labels[1]})',
                 fontsize=12)

    sm = ScalarMappable(norm=norm_color, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'rotation magnitude $r$ (deg)', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=220, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    print(f'Saved {OUT_PNG}')
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
