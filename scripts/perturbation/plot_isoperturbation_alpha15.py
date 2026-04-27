#!/usr/bin/env python3
"""TMS-style cartoon (5 features in 2D) vs zoomed-in 2-feature data.

LHS (orange): synthetic 5-feature superposition cartoon. Iso-perturbation
boundary at fixed rotation alpha approximates the p->infty limit:
R(phi) = r_min + (r_max - r_min) * (max_k |cos(phi - alpha_k)|)^q
for q large -> the response is large near each of the 5 feature spokes and
flatter (but never zero) in between.

RHS (blue): real Gemma 9B (Formal x Portuguese) data, alpha=15deg, L2 metric.
This corresponds to zooming in on a single wedge of the LHS between two
consecutive feature spokes -- so we frame it as an inset/zoom of the cartoon.
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Wedge
from scipy.interpolate import RegularGridInterpolator

# Avenir-ish font: try Avenir first, fall back through Lato / Inter / Open Sans.
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [
    'Avenir', 'Avenir Next', 'Avenir Next LT Pro',
    'Lato', 'Inter', 'Open Sans',
    'Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'cm'


PKL = 'results/fig4_robustness/v2_A_gemma_Formal_Portuguese.pkl'
OUT_DIR = 'results/fig4_robustness'

ALPHA_DEG = 15.0
N_PHI = 60

ORANGE = '#F5A623'
BLUE   = '#3B7DD8'
SPOKE_GREY = '#222222'


# -------- LHS cartoon -----------------------------------------------------
N_FEATS = 5
def tms_polygon(r_outer, r_inner, n_per_edge=24):
    """Star polygon: outer vertices at the N feature spokes (radius r_outer),
    inner vertices at the angular midpoints between consecutive spokes
    (radius r_inner << r_outer). Straight-line edges convey 'feature
    directions much more responsive than between'."""
    spoke_angles = np.linspace(0, 2 * np.pi, N_FEATS, endpoint=False) + np.pi / 2
    inner_angles = spoke_angles + np.pi / N_FEATS
    outer_v = np.stack([r_outer * np.cos(spoke_angles),
                        r_outer * np.sin(spoke_angles)], axis=1)
    inner_v = np.stack([r_inner * np.cos(inner_angles),
                        r_inner * np.sin(inner_angles)], axis=1)
    # Interleave outer and inner vertices: o0, i0, o1, i1, ..., o_{N-1}, i_{N-1}.
    verts = np.empty((2 * N_FEATS, 2))
    verts[0::2] = outer_v
    verts[1::2] = inner_v
    xs, ys = [], []
    for k in range(2 * N_FEATS):
        x0, y0 = verts[k]
        x1, y1 = verts[(k + 1) % (2 * N_FEATS)]
        ts = np.linspace(0, 1, n_per_edge, endpoint=False)
        xs.append(x0 + ts * (x1 - x0))
        ys.append(y0 + ts * (y1 - y0))
    return np.concatenate(xs), np.concatenate(ys), spoke_angles


# -------- RHS data --------------------------------------------------------
def data_curve():
    with open(PKL, 'rb') as f:
        d = pickle.load(f)
    angles = d['angles_deg']
    grids = d['l2']
    labels = d['direction_labels']
    n_act = grids.shape[0]
    phi = np.linspace(0.0, np.pi / 2, N_PHI)
    sin_alpha = np.sin(np.deg2rad(ALPHA_DEG))
    s1 = sin_alpha * np.cos(phi); s2 = sin_alpha * np.sin(phi)
    t1 = np.rad2deg(np.arcsin(np.clip(s1, -1, 1)))
    t2 = np.rad2deg(np.arcsin(np.clip(s2, -1, 1)))
    pts = np.stack([t1, t2], axis=-1)
    per_act = np.zeros((n_act, N_PHI))
    for a in range(n_act):
        f_interp = RegularGridInterpolator(
            (angles, angles), grids[a],
            bounds_error=False, fill_value=None)
        per_act[a] = f_interp(pts)
    med = np.median(per_act, axis=0)
    return phi, med, labels


# -------- panels ----------------------------------------------------------
def draw_lhs(ax, xs, ys, spoke_angles, label_idx_d1, label_idx_d2, rmax):
    arrow_len = rmax * 0.95
    for k, ang in enumerate(spoke_angles):
        x, y = arrow_len * np.cos(ang), arrow_len * np.sin(ang)
        ax.annotate('', xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', lw=1.6,
                                    color=SPOKE_GREY))
        if k == label_idx_d1:
            ax.text(x * 1.10, y * 1.10, '$f_1$',
                    fontsize=18, ha='left', va='center')
        elif k == label_idx_d2:
            ax.text(x * 1.10, y * 1.10, '$f_2$',
                    fontsize=18, ha='center', va='bottom')
    # close the loop and draw as a line
    xs_closed = np.append(xs, xs[0]); ys_closed = np.append(ys, ys[0])
    ax.plot(xs_closed, ys_closed, '-', color=ORANGE, lw=2.0)
    ax.set_aspect('equal')
    ax.set_xlim(-rmax, rmax); ax.set_ylim(-rmax, rmax)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(False)


def draw_rhs(ax, phi, R, rmax_sq):
    arrow_len = rmax_sq * 0.95
    for dx, dy in [(arrow_len, 0), (0, arrow_len)]:
        ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', lw=1.6,
                                    color=SPOKE_GREY))
    pad = rmax_sq * 0.04
    ax.text(arrow_len + pad, 0, '$f_1$',
            fontsize=18, ha='left', va='center')
    ax.text(0, arrow_len + pad, '$f_2$',
            fontsize=18, ha='center', va='bottom')
    x = R * np.cos(phi); y = R * np.sin(phi)
    ax.plot(x, y, '-', color=BLUE, lw=2.0)
    ax.set_aspect('equal')
    ax.set_xlim(-rmax_sq * 0.12, rmax_sq); ax.set_ylim(-rmax_sq * 0.12, rmax_sq)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(False)


def main():
    phi_data, R_data, labels = data_curve()

    xs_lhs, ys_lhs, spoke_angles = tms_polygon(
        r_outer=1.0, r_inner=0.1, n_per_edge=24)

    # spoke ordering: spokes are at pi/2 + 2*pi*k/5 (k=0..4). Index 0 is +y
    # (top), index 4 is at pi/2 - 2*pi/5 = 18 deg (top-right). Label these
    # two as d_2 and d_1 respectively to mirror the RHS axes.
    label_idx_d2 = 0
    label_idx_d1 = 4

    fig = plt.figure(figsize=(13, 8.0))
    ax_l = fig.add_axes([0.04, 0.04, 0.45, 0.70])
    ax_r = fig.add_axes([0.55, 0.04, 0.42, 0.70])
    fig.suptitle('Feature directions are not special', fontsize=28, y=0.96)
    ax_l.set_title('Predicted under Superposition', fontsize=19, pad=60)
    ax_r.set_title('Measured', fontsize=19, pad=60)

    rmax_lhs = 1.18
    draw_lhs(ax_l, xs_lhs, ys_lhs, spoke_angles,
             label_idx_d1, label_idx_d2, rmax=rmax_lhs)

    rmax_rhs = max(R_data.max(), 1.0) * 1.18
    draw_rhs(ax_r, phi_data, R_data, rmax_sq=rmax_rhs)

    # "Model Response" annotation: along the f_1 arrow on LHS (rotated, on the
    # upper side of the arrow so it sits inside the wedge), and below the
    # f_1 arrow on RHS (horizontal).
    arrow_len_l = rmax_lhs * 0.95
    f1_lhs_angle = spoke_angles[label_idx_d1]   # ~18 deg
    f1_lhs_deg = np.rad2deg(f1_lhs_angle)
    mid_x_l = 0.55 * arrow_len_l * np.cos(f1_lhs_angle)
    mid_y_l = 0.55 * arrow_len_l * np.sin(f1_lhs_angle)
    perp_off = rmax_lhs * 0.05
    ax_l.text(mid_x_l - perp_off * np.sin(f1_lhs_angle),
              mid_y_l + perp_off * np.cos(f1_lhs_angle),
              'Model Response',
              fontsize=12, ha='center', va='bottom',
              rotation=f1_lhs_deg, color='#444444')
    arrow_len_r = rmax_rhs * 0.95
    pad_box_r = rmax_rhs * 0.03
    ax_r.text(arrow_len_r * 0.5, -(pad_box_r + rmax_rhs * 0.07),
              'Model Response',
              fontsize=12, ha='center', va='top', color='#444444')


    # Full box on each panel + dashed connectors as a zoom indicator.
    box_alpha = 0.3
    arrow_len_l = rmax_lhs * 0.95
    pad_l = rmax_lhs * 0.03
    box_l_lo = -pad_l
    box_l_hi = arrow_len_l + pad_l
    box_l = plt.Rectangle((box_l_lo, box_l_lo),
                          box_l_hi - box_l_lo, box_l_hi - box_l_lo,
                          fill=False, edgecolor='black', lw=1.4,
                          alpha=box_alpha, zorder=10,
                          transform=ax_l.transData)
    ax_l.add_patch(box_l)

    arrow_len_r = rmax_rhs * 0.95
    pad_r = rmax_rhs * 0.03
    box_r_lo = -pad_r
    box_r_hi = arrow_len_r + pad_r
    box_r = plt.Rectangle((box_r_lo, box_r_lo),
                          box_r_hi - box_r_lo, box_r_hi - box_r_lo,
                          fill=False, edgecolor='black', lw=1.4,
                          alpha=box_alpha, zorder=10,
                          transform=ax_r.transData)
    ax_r.add_patch(box_r)

    # Dashed connecting lines: right side of LHS box -> left side of RHS box.
    for p_l, p_r in [((box_l_hi, box_l_hi), (box_r_lo, box_r_hi)),
                     ((box_l_hi, box_l_lo), (box_r_lo, box_r_lo))]:
        con = ConnectionPatch(
            xyA=p_l, coordsA=ax_l.transData,
            xyB=p_r, coordsB=ax_r.transData,
            color='black', lw=1.0, ls='--', alpha=box_alpha, zorder=10)
        fig.add_artist(con)

    out = os.path.join(OUT_DIR,
                       f'isoperturbation_alpha{int(ALPHA_DEG)}_tms_vs_data.png')
    fig.savefig(out, dpi=150)
    print(f'  saved {out}')

    # Also save to writeup/figures/ and the ICML paper figures dir as PDF.
    extra_outputs = [
        'writeup/figures/fig1_feature_directions.pdf',
        'writeup/figures/fig1_feature_directions.png',
        os.path.expanduser(
            '~/Research/error-correction-paper/icml2026/figures/'
            'fig1_feature_directions.pdf'),
    ]
    for path in extra_outputs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150 if path.endswith('.png') else None)
        print(f'  saved {path}')


if __name__ == '__main__':
    main()
