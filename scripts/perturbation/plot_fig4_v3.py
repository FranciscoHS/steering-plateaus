#!/usr/bin/env python3
"""Fig 4 v3 beeswarm: $p$ across many axes of variation, denser.

Each column = one axis of variation. Reads from results/fig4_robustness/v2_*.

Columns (left → right):
  Direction family (4 sub-cols):
    1. Contrastive DoM      (Phase A: v2_A_gemma_*)
    2. SAE latent           (Phase I: v2_I_sae_gemma_*)
    3. MELBO                (Phase I: v2_I_melbo_gemma_*)
    4. Random×Random        (Phase B: v2_B_gemma_*)
  Other axes:
    5. Metric L²/cos/KL     (re-fit v2_A_gemma_* with each metric)
    6. Perturb layer        (v2_E_gemma_*)
    7. Measure layer        (v2_F_gemma_*)
    8. Perturb method add.  (v2_G_gemma_*)
    9. Model {Gemma..Yi}    (v2_A_gemma_* + v2_H_*)
    10. Threshold sweep     (re-fit v2_A_gemma_Formal_Portuguese_*)

Per column: scatter all dots (blue), mean line + ±2σ shaded band (blue),
median tick (orange). Reference dashed line at p=2.
"""
import os, sys, glob, pickle, json
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse)

DIR = 'results/fig4_robustness'
OUT_PNG = 'writeup/figures/fig4_n_robust_beeswarm_v3.png'
OUT_PDF = 'writeup/figures/fig4_n_robust_beeswarm_v3.pdf'

DEFAULT_L2 = 50.0
DEFAULT_COS = 0.01
DEFAULT_KL = 0.05
THRESHOLDS_L2 = [20, 30, 40, 50, 60, 75, 100, 125, 150, 200]
FLAGSHIP = ('Formal', 'Portuguese')

C_DATA   = '#1F77B4'
C_MEAN   = '#1F77B4'
C_BAND   = '#1F77B4'
C_MEDIAN = '#D9822B'
C_REF    = '#D9822B'


def find_theta(curve, ang, thr):
    idx = np.searchsorted(curve, thr)
    if idx == 0 or idx >= len(ang):
        return ang[min(idx, len(ang) - 1)]
    x0, x1 = ang[idx - 1], ang[idx]
    y0, y1 = curve[idx - 1], curve[idx]
    if y1 == y0: return x0
    return x0 + (thr - y0) / (y1 - y0) * (x1 - x0)


def fit_p(grid, angles, thr):
    c1 = grid[:, 0]; c2 = grid[0, :]
    noise = grid[0, 0]
    if thr < noise * 1.5: return None
    t1 = find_theta(c1, angles, thr)
    t2 = find_theta(c2, angles, thr)
    if t1 >= angles[-1] * 0.97 or t2 >= angles[-1] * 0.97: return None
    if t1 <= angles[1] or t2 <= angles[1]: return None
    pts = extract_contour_points(angles, grid, thr)
    if len(pts) == 0: return None
    cn = pts.astype(float).copy()
    cn[:, 0] /= t1; cn[:, 1] /= t2
    p, _, _ = fit_superellipse(cn, (t1 + t2) / 2)
    return p


def fit_additive(grid, mags, thr):
    c1 = grid[:, 0]; c2 = grid[0, :]
    noise = grid[0, 0]
    if thr < noise * 1.5: return None
    t1 = find_theta(c1, mags, thr)
    t2 = find_theta(c2, mags, thr)
    if t1 >= mags[-1] * 0.97 or t2 >= mags[-1] * 0.97: return None
    if t1 <= mags[1] or t2 <= mags[1]: return None
    pts = extract_contour_points(mags, grid, thr)
    if len(pts) == 0: return None
    cn = pts.astype(float).copy()
    cn[:, 0] /= t1; cn[:, 1] /= t2
    p, _, _ = fit_superellipse(cn, (t1 + t2) / 2)
    return p


def load_pkl(path):
    with open(path, 'rb') as f: return pickle.load(f)


# -----------------------------------------------------------------------------
# Per-column collectors
# -----------------------------------------------------------------------------

def col_dom_gemma():
    """Direction family: contrastive DoM (Phase A on Gemma)."""
    out = []
    for pkl in sorted(glob.glob(os.path.join(DIR, 'v2_A_gemma_*.pkl'))):
        d = load_pkl(pkl)
        grid = np.median(d['l2'], axis=0)
        p = fit_p(grid, d['angles_deg'], DEFAULT_L2)
        if p is not None: out.append(p)
    return out


def col_sae_gemma():
    out = []
    for pkl in sorted(glob.glob(os.path.join(DIR, 'v2_I_sae_gemma_*.pkl'))):
        d = load_pkl(pkl)
        grid = np.median(d['l2'], axis=0)
        p = fit_p(grid, d['angles_deg'], DEFAULT_L2)
        if p is not None: out.append(p)
    return out


def col_melbo_gemma():
    out = []
    for pkl in sorted(glob.glob(os.path.join(DIR, 'v2_I_melbo_gemma_*.pkl'))):
        d = load_pkl(pkl)
        grid = np.median(d['l2'], axis=0)
        p = fit_p(grid, d['angles_deg'], DEFAULT_L2)
        if p is not None: out.append(p)
    return out


def col_random_gemma():
    out = []
    for pkl in sorted(glob.glob(os.path.join(DIR, 'v2_B_gemma_*.pkl'))):
        d = load_pkl(pkl)
        grid = np.median(d['l2'], axis=0)
        p = fit_p(grid, d['angles_deg'], DEFAULT_L2)
        if p is not None: out.append(p)
    return out


def col_metric():
    """L², cos, KL on the same 30 v2_A_gemma_* sweeps."""
    out = []
    for pkl in sorted(glob.glob(os.path.join(DIR, 'v2_A_gemma_*.pkl'))):
        d = load_pkl(pkl)
        for metric, thr in [('l2', DEFAULT_L2), ('cosine', DEFAULT_COS),
                             ('kl', DEFAULT_KL)]:
            if metric not in d: continue
            grid = np.median(d[metric], axis=0)
            p = fit_p(grid, d['angles_deg'], thr)
            if p is not None: out.append(p)
    return out


def col_perturb_layer():
    out = []
    for pkl in sorted(glob.glob(os.path.join(DIR, 'v2_E_gemma_L*_*.pkl'))):
        d = load_pkl(pkl)
        grid = np.median(d['l2'], axis=0)
        p = fit_p(grid, d['angles_deg'], DEFAULT_L2)
        if p is not None: out.append(p)
    return out


def col_measure_layer():
    out = []
    for pkl in sorted(glob.glob(os.path.join(DIR, 'v2_F_gemma_meas*.pkl'))):
        d = load_pkl(pkl)
        grid = np.median(d['l2'], axis=0)
        p = fit_p(grid, d['angles_deg'], DEFAULT_L2)
        if p is not None: out.append(p)
    return out


def col_additive():
    out = []
    for pkl in sorted(glob.glob(os.path.join(DIR, 'v2_G_gemma_additive_*.pkl'))):
        d = load_pkl(pkl)
        grid = np.median(d['l2'], axis=0)
        p = fit_additive(grid, d['mags'], DEFAULT_L2)
        if p is not None: out.append(p)
    return out


def col_model():
    """Gemma A v2 + each non-Gemma model's Phase H (all 30 pairs each)."""
    out = []
    # Gemma contribution = Phase A v2
    for pkl in sorted(glob.glob(os.path.join(DIR, 'v2_A_gemma_*.pkl'))):
        d = load_pkl(pkl)
        grid = np.median(d['l2'], axis=0)
        p = fit_p(grid, d['angles_deg'], DEFAULT_L2)
        if p is not None: out.append(p)
    # Other models: Phase H
    for tgt in ['qwen', 'llama', 'mistral', 'aya', 'yi']:
        for pkl in sorted(glob.glob(os.path.join(DIR, f'v2_H_{tgt}_*.pkl'))):
            d = load_pkl(pkl)
            grid = np.median(d['l2'], axis=0)
            p = fit_p(grid, d['angles_deg'], DEFAULT_L2)
            if p is not None: out.append(p)
    return out


def col_threshold():
    """All 30 Gemma A pairs × 10 thresholds."""
    out = []
    for pkl in sorted(glob.glob(os.path.join(DIR, 'v2_A_gemma_*.pkl'))):
        d = load_pkl(pkl)
        grid = np.median(d['l2'], axis=0)
        for thr in THRESHOLDS_L2:
            p = fit_p(grid, d['angles_deg'], float(thr))
            if p is not None: out.append(p)
    return out


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

def beeswarm_x(values, x_center, jitter_scale=0.20, rng=None):
    if rng is None: rng = np.random.RandomState(0)
    return x_center + rng.uniform(-jitter_scale, jitter_scale, size=len(values))


def main():
    cols = [
        ('Contrastive\ndifference of means',   col_dom_gemma()),
        ('SAE latent\nGemma Scope L2',         col_sae_gemma()),
        ('MELBO\noptimization, $\\Delta$KL=1', col_melbo_gemma()),
        ('Random\n$\\mathcal{N}(0, I)$',       col_random_gemma()),
        ('Metric\n$L^2$, cos, KL',             col_metric()),
        ('Perturb layer\n$L \\in \\{2,5,10,20\\}$', col_perturb_layer()),
        ('Measure layer\npenult $\\pm$ offsets', col_measure_layer()),
        ('Perturb method\nadditive',           col_additive()),
        ('Model\n6 across families',           col_model()),
        ('Threshold\n$L^2 \\in [20,200]$',     col_threshold()),
    ]
    print('Column counts:')
    for name, vs in cols:
        clean = name.replace('\n', ' / ')
        if not vs:
            print(f'  {clean:50s}  (empty)')
        else:
            v = np.asarray(vs)
            print(f'  {clean:50s}  n={len(vs):3d}  '
                  f'mean={v.mean():.2f}  med={np.median(v):.2f}  '
                  f'±2σ=[{v.mean()-2*v.std():.2f}, {v.mean()+2*v.std():.2f}]')

    # Drop empty columns
    cols = [(n, v) for n, v in cols if v]
    if not cols:
        print('No data — nothing to plot')
        return

    plt.rcParams['font.family'] = 'Inter'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'cm'

    fig, ax = plt.subplots(figsize=(14.5, 5.2))
    rng = np.random.RandomState(7)
    box_w = 0.32

    # Vertical separators between groups: after col 4 (direction-family group)
    if len(cols) >= 5:
        ax.axvline(3.5, color='#bbbbbb', lw=0.6, ls=':', alpha=0.7, zorder=1)

    for i, (name, values) in enumerate(cols):
        vals = np.asarray(values, dtype=float)
        mean = vals.mean()
        std = vals.std(ddof=1) if len(vals) > 1 else 0.0
        lo, hi = mean - 2 * std, mean + 2 * std
        med = np.median(vals)

        # ±2σ translucent band
        ax.fill_between([i - box_w, i + box_w], [lo, lo], [hi, hi],
                        color=C_BAND, alpha=0.13, linewidth=0, zorder=2)
        # mean line
        ax.plot([i - box_w, i + box_w], [mean, mean],
                color=C_MEAN, lw=1.6, alpha=0.85, zorder=4)

        xs = beeswarm_x(vals, i, jitter_scale=0.22, rng=rng)
        ax.scatter(xs, vals, s=14, color=C_DATA, alpha=0.55,
                   edgecolor='none', zorder=3)

        # median tick (orange) — slightly thicker, on top of everything
        ax.plot([i - box_w, i + box_w], [med, med],
                color=C_MEDIAN, lw=2.4, zorder=5)

    ax.axhline(2.0, color=C_REF, lw=1.4, ls='--', alpha=0.85,
               label=r'$p = 2$ (Euclidean)')

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([''] * len(cols))
    for i, (label, _) in enumerate(cols):
        top, bot = label.split('\n', 1)
        ax.annotate(top, xy=(i, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -8), textcoords='offset points',
                    fontsize=10, ha='center', va='top')
        ax.annotate(bot, xy=(i, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -23), textcoords='offset points',
                    fontsize=8.5, color='#666', ha='center', va='top')

    ax.set_ylabel(r'fitted superellipse exponent $p$', fontsize=12)
    all_v = np.concatenate([np.asarray(v) for _, v in cols])
    pad = (all_v.max() - all_v.min()) * 0.08 + 0.05
    ax.set_ylim(all_v.min() - pad, all_v.max() + pad)
    ax.grid(True, axis='y', which='major', color='#cccccc', lw=0.4, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Custom legend: median (orange), mean (blue), p=2 ref
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0], [0], color=C_MEDIAN, lw=2.4, label='median'),
        Line2D([0], [0], color=C_MEAN, lw=1.6, label='mean'),
        Patch(facecolor=C_BAND, alpha=0.13, label=r'mean $\pm 2\sigma$'),
        Line2D([0], [0], color=C_REF, lw=1.4, ls='--',
               label=r'$p = 2$ (Euclidean)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=True,
              fontsize=9, ncol=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    print(f'\nSaved {OUT_PNG}')


if __name__ == '__main__':
    main()
