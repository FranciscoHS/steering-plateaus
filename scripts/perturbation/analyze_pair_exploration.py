#!/usr/bin/env python3
"""Analysis of pair_exploration_L2 pkls: rank pairs by threshold-stability of p.

Inputs (local):
  results/pair_exploration_L2/sweep_2d_<A>_<B>.pkl   (L2 + cosine grids)
  results/pair_exploration_L2/pair_list.json
Outputs:
  writeup/figures/pair_explore_ranking.png  — summary plot per pair
  writeup/figures/pair_explore_thr_panel.png — p vs threshold per pair
  results/pair_exploration_L2/summary.json  — ranked metrics
"""
import os, sys, json, glob, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse)

DIR = 'results/pair_exploration_L2'
OUT_DIR = 'writeup/figures'
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLDS = np.array([10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400])


def find_theta_at(curve, angles, thr):
    idx = np.searchsorted(curve, thr)
    if idx == 0 or idx >= len(angles):
        return angles[min(idx, len(angles) - 1)]
    x0, x1 = angles[idx - 1], angles[idx]
    y0, y1 = curve[idx - 1], curve[idx]
    if y1 == y0: return x0
    return x0 + (thr - y0) / (y1 - y0) * (x1 - x0)


def analyze_pair(pkl_path):
    with open(pkl_path, 'rb') as f:
        res = pickle.load(f)
    angles = res['angles_deg']
    grid = np.median(res['l2'], axis=0)
    c1 = grid[:, 0]
    c2 = grid[0, :]
    noise = grid[0, 0]
    a_lbl, b_lbl = res['direction_labels']

    rows = []
    for thr in THRESHOLDS:
        if thr < noise * 1.5:
            continue
        t1 = find_theta_at(c1, angles, thr)
        t2 = find_theta_at(c2, angles, thr)
        if t1 >= angles[-1] * 0.97 or t2 >= angles[-1] * 0.97:
            continue
        if t1 <= angles[1] or t2 <= angles[1]:
            continue
        pts = extract_contour_points(angles, grid, thr)
        if len(pts) == 0: continue
        cn = pts.astype(float).copy()
        cn[:, 0] /= t1; cn[:, 1] /= t2
        avg = (t1 + t2) / 2
        p_fit, ci, info = fit_superellipse(cn, avg)
        if p_fit is None: continue
        rows.append({
            'thr': float(thr), 't1': float(t1), 't2': float(t2),
            'p': float(p_fit),
            'ci': [float(ci[0]), float(ci[1])],
            'err': float(info['mean_radial_frac']),
            'err_deg': float(info['mean_radial_deg']),
        })
    return {'pair': [a_lbl, b_lbl], 'rows': rows}


def summarise(fit_rows):
    ps = np.array([r['p'] for r in fit_rows])
    thrs = np.array([r['thr'] for r in fit_rows])
    errs = np.array([r['err'] for r in fit_rows])
    # Use only "stable-window" thresholds: drop lowest if it's <1.5*noise-ish
    # (already pre-filtered) and highest if it hits grid ceiling (already done).
    return {
        'p_mean': float(ps.mean()),
        'p_median': float(np.median(ps)),
        'p_std': float(ps.std()),
        'p_min': float(ps.min()),
        'p_max': float(ps.max()),
        'p_range': float(ps.max() - ps.min()),
        'n_thresholds': int(len(ps)),
        'err_mean': float(errs.mean()),
        'thr_min': float(thrs.min()),
        'thr_max': float(thrs.max()),
    }


def plot_ranking(results):
    names = [f"{r['pair'][0][:3]}×{r['pair'][1][:3]}" for r in results]
    p_mid = [r['summary']['p_median'] for r in results]
    p_lo = [r['summary']['p_min'] for r in results]
    p_hi = [r['summary']['p_max'] for r in results]
    p_std = [r['summary']['p_std'] for r in results]
    n = len(results)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.35)))
    for i, r in enumerate(results):
        ax.plot([p_lo[i], p_hi[i]], [i, i], '-', color='gray', lw=1.2)
        ax.plot(p_mid[i], i, 'o', color='#1f77b4', markersize=7)
    ax.axvline(2.0, ls='--', color='#d62728', alpha=0.7, label=r'$p=2$ Euclidean')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{r["pair"][0]} × {r["pair"][1]}' for r in results])
    ax.invert_yaxis()
    ax.set_xlabel(r'fitted exponent $p$  (median $\bullet$ with min–max range)')
    ax.set_title('Pair exploration @ Gemma-2-9B L2 — p across threshold sweep')
    ax.grid(alpha=0.25, axis='x')
    ax.legend(loc='upper right')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'pair_explore_ranking.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved {out}')


def plot_threshold_panel(results):
    n = len(results)
    cols = 3
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(14, 3.2 * rows_n),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()
    for i, r in enumerate(results):
        ax = axes[i]
        thrs = np.array([x['thr'] for x in r['rows']])
        ps = np.array([x['p'] for x in r['rows']])
        lo = np.array([x['ci'][0] for x in r['rows']])
        hi = np.array([x['ci'][1] for x in r['rows']])
        ax.fill_between(thrs, lo, hi, color='#1f77b4', alpha=0.2)
        ax.plot(thrs, ps, 'o-', color='#1f77b4', lw=1.5, markersize=5)
        ax.axhline(2.0, ls='--', color='#d62728', alpha=0.6)
        ax.set_xscale('log')
        ax.set_title(f'{r["pair"][0]} × {r["pair"][1]}  '
                     f'(std={r["summary"]["p_std"]:.2f})', fontsize=10)
        ax.grid(alpha=0.25)
    for j in range(n, len(axes)):
        axes[j].axis('off')
    for ax in axes[:n]:
        ax.set_ylim(1.5, 3.5)
    axes[0].set_ylabel('fitted $p$')
    fig.supxlabel(r'$L^2$ threshold')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'pair_explore_thr_panel.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved {out}')


def main():
    pkls = sorted(glob.glob(os.path.join(DIR, 'sweep_2d_*.pkl')))
    print(f'Found {len(pkls)} 2D pair pkls')

    results = []
    for pkl in pkls:
        print(f'[{os.path.basename(pkl)}]')
        ana = analyze_pair(pkl)
        if not ana['rows']:
            print('  no valid fits'); continue
        ana['summary'] = summarise(ana['rows'])
        s = ana['summary']
        print(f"  n_thr={s['n_thresholds']}  "
              f"p_med={s['p_median']:.2f}  std={s['p_std']:.2f}  "
              f"range=[{s['p_min']:.2f}, {s['p_max']:.2f}]")
        results.append(ana)

    # Rank by lowest std of p
    results.sort(key=lambda r: r['summary']['p_std'])

    print('\n=== Ranking (best = lowest std of p across thresholds) ===')
    print(f'{"pair":30s}  {"std":>6s}  {"median":>7s}  {"range":>12s}  {"mean-err":>9s}')
    for r in results:
        s = r['summary']
        name = f'{r["pair"][0]} × {r["pair"][1]}'
        print(f'{name:30s}  {s["p_std"]:6.3f}  {s["p_median"]:7.3f}  '
              f'[{s["p_min"]:4.2f},{s["p_max"]:4.2f}]  {s["err_mean"]:9.4f}')

    with open(os.path.join(DIR, 'summary.json'), 'w') as f:
        json.dump({'ranked': results}, f, indent=2)
    print(f'\nSaved {DIR}/summary.json')

    plot_ranking(results)
    plot_threshold_panel(results)


if __name__ == '__main__':
    main()
