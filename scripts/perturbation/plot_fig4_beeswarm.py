#!/usr/bin/env python3
"""Fig 4: $n$-robustness beeswarm.

Per-pair fitted superellipse exponent p, across:
  (1) many contrastive pairs (pair identity)
  (2) threshold (sweep on the flagship pair)
  (3) metric (L2 vs cosine, on the flagship pair)
  (4) [optional later] feature-type categories

Headline reference line at p = 2.

Inputs are the existing 200 sweep_2d_*.pkl files in
results/pair_exploration_L2_v2/.
"""
import os, sys, glob, pickle, json
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse)

DIR = 'results/pair_exploration_L2_v2'
OUT_PNG = 'writeup/figures/fig4_n_robust_beeswarm.png'
OUT_PDF = 'writeup/figures/fig4_n_robust_beeswarm.pdf'

L2_THRESHOLDS = np.array([20, 30, 40, 50, 60, 75, 100, 125, 150])
COS_THRESHOLDS = np.array(
    [0.0005, 0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03])
DEFAULT_L2 = 50.0
DEFAULT_COS = 0.01
FLAGSHIP = ('Formal', 'Portuguese')

FEATURE_TYPE = {}
LANG = {'Portuguese', 'Chinese', 'Spanish', 'French', 'German', 'Dutch',
        'Italian', 'Japanese', 'Russian', 'Arabic'}
CODE = {'Python', 'JavaScript', 'Java', 'Rust', 'Haskell', 'Cpp', 'Go', 'Lisp',
        'TypeScript'}
SEMANTIC = {'Gender', 'Sentiment', 'Formal', 'Literary', 'Children', 'News'}
for n in LANG: FEATURE_TYPE[n] = 'lang'
for n in CODE: FEATURE_TYPE[n] = 'code'
for n in SEMANTIC: FEATURE_TYPE[n] = 'semantic'


def find_theta(curve, ang, thr):
    idx = np.searchsorted(curve, thr)
    if idx == 0 or idx >= len(ang):
        return ang[min(idx, len(ang) - 1)]
    x0, x1 = ang[idx - 1], ang[idx]
    y0, y1 = curve[idx - 1], curve[idx]
    if y1 == y0: return x0
    return x0 + (thr - y0) / (y1 - y0) * (x1 - x0)


def fit_one(grid, angles, thr):
    """Returns p_fit (or None if fit fails / contour invalid)."""
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


def collect_data():
    """Returns:
       ps_pair_l2 : list of (pair_label, type_pair, p) at default L2 thr
       ps_pair_cos: same at default cosine thr
       ps_thr_l2  : list of (thr, p) for flagship pair, sweeping L2 thresholds
       ps_thr_cos : same for cosine
    """
    pkls = sorted(glob.glob(os.path.join(DIR, 'sweep_2d_*.pkl')))
    ps_pair_l2, ps_pair_cos = [], []
    ps_thr_l2, ps_thr_cos = [], []

    for pkl in pkls:
        with open(pkl, 'rb') as f:
            res = pickle.load(f)
        a, b = res['direction_labels']
        type_pair = tuple(sorted([FEATURE_TYPE.get(a, '?'),
                                  FEATURE_TYPE.get(b, '?')]))
        angles = res['angles_deg']
        grid_l2 = np.median(res['l2'], axis=0)
        grid_cos = np.median(res['cosine'], axis=0)

        p_l2 = fit_one(grid_l2, angles, DEFAULT_L2)
        if p_l2 is not None:
            ps_pair_l2.append((f'{a}×{b}', type_pair, p_l2))

        p_cos = fit_one(grid_cos, angles, DEFAULT_COS)
        if p_cos is not None:
            ps_pair_cos.append((f'{a}×{b}', type_pair, p_cos))

        if (a, b) == FLAGSHIP or (b, a) == FLAGSHIP:
            for thr in L2_THRESHOLDS:
                p = fit_one(grid_l2, angles, float(thr))
                if p is not None: ps_thr_l2.append((float(thr), p))
            for thr in COS_THRESHOLDS:
                p = fit_one(grid_cos, angles, float(thr))
                if p is not None: ps_thr_cos.append((float(thr), p))

    return ps_pair_l2, ps_pair_cos, ps_thr_l2, ps_thr_cos


def beeswarm_x(values, x_center, jitter_scale=0.07, rng=None):
    """Compute x-positions for points so they spread sideways when overlapping."""
    if rng is None: rng = np.random.RandomState(0)
    return x_center + rng.uniform(-jitter_scale, jitter_scale, size=len(values))


def main():
    ps_pair_l2, ps_pair_cos, ps_thr_l2, ps_thr_cos = collect_data()
    print(f'pair fits (L2): {len(ps_pair_l2)}')
    print(f'pair fits (cos): {len(ps_pair_cos)}')
    print(f'threshold fits L2 (flagship): {len(ps_thr_l2)}')
    print(f'threshold fits cos (flagship): {len(ps_thr_cos)}')

    ps_l2 = np.array([p for _, _, p in ps_pair_l2])
    ps_cos = np.array([p for _, _, p in ps_pair_cos])
    ps_thr_l2_v = np.array([p for _, p in ps_thr_l2])
    ps_thr_cos_v = np.array([p for _, p in ps_thr_cos])

    print(f'\\nL2 metric across pairs:    n={len(ps_l2)}, '
          f'median={np.median(ps_l2):.2f}, '
          f'IQR=[{np.percentile(ps_l2, 25):.2f}, '
          f'{np.percentile(ps_l2, 75):.2f}]')
    print(f'cosine metric across pairs: n={len(ps_cos)}, '
          f'median={np.median(ps_cos):.2f}, '
          f'IQR=[{np.percentile(ps_cos, 25):.2f}, '
          f'{np.percentile(ps_cos, 75):.2f}]')
    print(f'L2 threshold sweep (flagship): n={len(ps_thr_l2_v)}, '
          f'values={ps_thr_l2_v.round(2)}')
    print(f'cos threshold sweep (flagship): n={len(ps_thr_cos_v)}, '
          f'values={ps_thr_cos_v.round(2)}')

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    rng = np.random.RandomState(7)

    # Categories on x-axis
    cats = [
        ('pairs ($L^2$, thr=50)', ps_l2, '#3a3a3a'),
        ('pairs (cosine, thr=0.01)', ps_cos, '#3a3a3a'),
        ('thresholds (Formal × PT, $L^2$)', ps_thr_l2_v, '#1F77B4'),
        ('thresholds (Formal × PT, cos)', ps_thr_cos_v, '#1F77B4'),
    ]
    for i, (name, values, col) in enumerate(cats):
        if len(values) == 0: continue
        # Slight horizontal jitter so overlapping points spread
        xs = beeswarm_x(values, i, jitter_scale=0.18, rng=rng)
        ax.scatter(xs, values, s=22, color=col, alpha=0.55,
                   edgecolor='none', zorder=3)
        # Median tick
        med = np.median(values)
        ax.plot([i - 0.32, i + 0.32], [med, med], color='k', lw=1.6, zorder=4)

    ax.axhline(2.0, color='#D9822B', lw=1.6, ls='--', alpha=0.85,
               label=r'$p = 2$ (Euclidean)')
    ax.axhline(20.0, color='#a05050', lw=1.0, ls=':', alpha=0.5)
    ax.text(len(cats) - 0.5, 19.5, r'$p \to \infty$ (per-feature)',
            fontsize=8.5, color='#a05050', ha='right', va='top')

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c[0] for c in cats], fontsize=10)
    ax.set_ylabel(r'fitted superellipse exponent $p$', fontsize=12)
    ax.set_yscale('log')
    ax.set_yticks([1, 2, 3, 5, 10, 20])
    ax.set_yticklabels(['1', '2', '3', '5', '10', '20'])
    ax.set_ylim(0.9, 25)
    ax.grid(True, axis='y', which='major', color='#bbbbbb', lw=0.4, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Fitted $p$ across pairs, metrics, thresholds (Gemma-2-9B L2)',
                 fontsize=12)
    ax.legend(loc='upper right', frameon=True, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    print(f'\\nSaved {OUT_PNG}')


if __name__ == '__main__':
    main()
