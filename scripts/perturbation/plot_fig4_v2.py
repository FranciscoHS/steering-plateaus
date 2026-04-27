#!/usr/bin/env python3
"""Fig 4 v2 beeswarm: $p$ across many axes of variation.

Each column = one axis of variation around a base case. Reads from
results/fig4_robustness/. The base case is Gemma-2-9B at L2, contrastive ×
contrastive, $L^2$ metric, default threshold = 50.

Axes (columns):
  1. Different contrastive pairs (the base — many pairs)
  2. Random baselines              (vary direction type)
  3. Different metrics             (vary L2 → cos / KL on rep pairs)
  4. Different perturbation layer  (vary L=2/5/10/20 on rep pairs)
  5. Different measurement layer   (vary penult ± offsets on rep pairs)
  6. Additive perturbation         (vary spherical → additive on rep pairs)
  7. Different model               (vary Gemma → Qwen3 / Llama)
  8. Different threshold           (vary thr on flagship)

Aesthetics: data points blue, median tick + p=2 reference orange. No title.
Diagonal x labels.
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
OUT_PNG = 'writeup/figures/fig4_n_robust_beeswarm.png'
OUT_PDF = 'writeup/figures/fig4_n_robust_beeswarm.pdf'

DEFAULT_L2 = 50.0
DEFAULT_COS = 0.01
DEFAULT_KL = 0.05
THRESHOLDS_L2 = [20, 30, 40, 50, 60, 75, 100, 125, 150, 200]
FLAGSHIP = ('Formal', 'Portuguese')

C_DATA   = '#1F77B4'
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
    """Returns p_fit or None if invalid."""
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
    """Same as fit_p but on additive (mag) grid. Coords are magnitudes, not angles."""
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


# ----------------------------------------------------------------------------
# Aggregate per-column
# ----------------------------------------------------------------------------

def col_pair_identity():
    """Phase A — many ctr×ctr pairs, base config."""
    out = []
    for pkl in sorted(glob.glob(os.path.join(DIR, 'A_gemma_*.pkl'))):
        d = load_pkl(pkl)
        grid = np.median(d['l2'], axis=0)
        p = fit_p(grid, d['angles_deg'], DEFAULT_L2)
        if p is not None: out.append(p)
    return out


def col_random_baselines():
    out = []
    for pkl in sorted(glob.glob(os.path.join(DIR, 'B_*_gemma_*.pkl'))):
        d = load_pkl(pkl)
        grid = np.median(d['l2'], axis=0)
        p = fit_p(grid, d['angles_deg'], DEFAULT_L2)
        if p is not None: out.append(p)
    return out


def col_metric():
    """Phase D — same rep pairs at L², cos, KL."""
    out = []
    for pkl in sorted(glob.glob(os.path.join(DIR, 'D_gemma_*_withkl.pkl'))):
        d = load_pkl(pkl)
        for metric, thr in [('l2', DEFAULT_L2), ('cosine', DEFAULT_COS),
                             ('kl', DEFAULT_KL)]:
            grid = np.median(d[metric], axis=0)
            p = fit_p(grid, d['angles_deg'], thr)
            if p is not None: out.append(p)
    return out


def col_perturb_layer():
    """Phase E — rep pairs at varying perturb layer."""
    out = []
    for pkl in sorted(glob.glob(os.path.join(DIR, 'E_gemma_L*_*.pkl'))):
        d = load_pkl(pkl)
        grid = np.median(d['l2'], axis=0)
        p = fit_p(grid, d['angles_deg'], DEFAULT_L2)
        if p is not None: out.append(p)
    return out


def col_measure_layer():
    """Phase F — rep pairs at varying measure layer."""
    out = []
    for pkl in sorted(glob.glob(os.path.join(DIR, 'F_gemma_meas*.pkl'))):
        d = load_pkl(pkl)
        grid = np.median(d['l2'], axis=0)
        p = fit_p(grid, d['angles_deg'], DEFAULT_L2)
        if p is not None: out.append(p)
    return out


def col_additive():
    """Phase G — additive perturbation."""
    out = []
    for pkl in sorted(glob.glob(os.path.join(DIR, 'G_gemma_additive_*.pkl'))):
        d = load_pkl(pkl)
        grid = np.median(d['l2'], axis=0)
        # Calibrate threshold on this scale: use 50 too, additive uses ||a|| units
        # Mag→L² magnitude isn't 1:1 with degrees, but L²-at-penultimate scale is
        # comparable (we measure same downstream metric).
        p = fit_additive(grid, d['mags'], DEFAULT_L2)
        if p is not None: out.append(p)
    return out


def col_model():
    """Phase H — rep pairs on Qwen3 + Llama."""
    out = []
    for tgt in ['qwen', 'llama']:
        for pkl in sorted(glob.glob(os.path.join(DIR, f'H_{tgt}_*.pkl'))):
            d = load_pkl(pkl)
            grid = np.median(d['l2'], axis=0)
            p = fit_p(grid, d['angles_deg'], DEFAULT_L2)
            if p is not None: out.append(p)
    return out


def col_threshold():
    """Vary threshold on flagship pair (Formal × Portuguese)."""
    out = []
    pkl = os.path.join(DIR, 'A_gemma_Formal_Portuguese.pkl')
    if not os.path.exists(pkl): return out
    d = load_pkl(pkl)
    grid = np.median(d['l2'], axis=0)
    for thr in THRESHOLDS_L2:
        p = fit_p(grid, d['angles_deg'], float(thr))
        if p is not None: out.append(p)
    return out


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------

def beeswarm_x(values, x_center, jitter_scale=0.18, rng=None):
    if rng is None: rng = np.random.RandomState(0)
    return x_center + rng.uniform(-jitter_scale, jitter_scale, size=len(values))


def main():
    cols = [
        ('Concept pair\nlanguage, gender, register, code',  col_pair_identity()),
        ('Direction type\n+ random',            col_random_baselines()),
        ('Metric\n$L^2$, cos, KL',              col_metric()),
        ('Perturb layer\n$L=2, 5, 10, 20$',     col_perturb_layer()),
        ('Measure layer\npenult ± offsets',     col_measure_layer()),
        ('Perturb method\nadditive',            col_additive()),
        ('Model\nQwen3, Llama',                 col_model()),
        ('Threshold\n$L^2 \\in [20, 200]$',     col_threshold()),
    ]
    print('Column counts:')
    for name, vs in cols:
        if not vs:
            print(f'  {name:36s}  (empty)')
        else:
            print(f'  {name:36s}  n={len(vs):3d}  med={np.median(vs):.2f}  '
                  f'IQR=[{np.percentile(vs,25):.2f},{np.percentile(vs,75):.2f}]')

    # Drop empty columns from the plot
    cols = [(n, v) for n, v in cols if v]

    plt.rcParams['font.family'] = 'Inter'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'cm'

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    rng = np.random.RandomState(7)
    box_w = 0.36
    for i, (name, values) in enumerate(cols):
        vals = np.asarray(values)
        q25, q75 = np.percentile(vals, [25, 75])
        # IQR box (faint blue rectangle)
        ax.fill_between([i - box_w, i + box_w], [q25, q25], [q75, q75],
                        color=C_DATA, alpha=0.13, linewidth=0, zorder=2)
        xs = beeswarm_x(vals, i, jitter_scale=0.20, rng=rng)
        ax.scatter(xs, vals, s=22, color=C_DATA, alpha=0.70,
                   edgecolor='none', zorder=3)
        med = np.median(vals)
        ax.plot([i - box_w, i + box_w], [med, med],
                color=C_MEDIAN, lw=2.2, zorder=4)

    ax.axhline(2.0, color=C_REF, lw=1.4, ls='--', alpha=0.85,
               label=r'$p = 2$ (Euclidean)')

    # Two-line labels: top line at base size, bottom line smaller + lighter
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([''] * len(cols))   # blank — we draw the labels manually
    ymin, _ = ax.get_ylim()
    for i, (label, _) in enumerate(cols):
        top, bot = label.split('\n', 1)
        # compute label y in axes coords for stable placement
        ax.annotate(top, xy=(i, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -8), textcoords='offset points',
                    fontsize=11, ha='center', va='top')
        ax.annotate(bot, xy=(i, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -24), textcoords='offset points',
                    fontsize=9, color='#666', ha='center', va='top')
    ax.set_ylabel(r'fitted superellipse exponent $p$', fontsize=12)
    # Y range: fit to actual data with small padding, no forced ceiling
    all_v = np.concatenate([v for _, v in cols])
    lo = all_v.min() - 0.10
    hi = all_v.max() + 0.10
    ax.set_ylim(lo, hi)
    ax.grid(True, axis='y', which='major', color='#cccccc', lw=0.4, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=True, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    print(f'\nSaved {OUT_PNG}')


if __name__ == '__main__':
    main()
