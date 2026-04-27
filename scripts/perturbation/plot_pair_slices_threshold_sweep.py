#!/usr/bin/env python3
"""Sweep thresholds and plot median fitted p vs threshold for the pair-slice data.

Reuses the saved 2D grids in pair_slices.pkl. No re-sweep needed.
"""
import os, sys, pickle
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.sweep_pair_slices_10contrastive import (
    extract_contour, find_axis_intercept, fit_superellipse_p, superellipse_curve,
)


import argparse as _argparse
_ap = _argparse.ArgumentParser()
_ap.add_argument('--pkl', default='results/pair_slices_10contrastive/qwen3-1.7b_L14/pair_slices.pkl')
_ap.add_argument('--outdir', default=None)
_args, _ = _ap.parse_known_args()
PKL = _args.pkl
OUTDIR = _args.outdir or os.path.dirname(PKL)


def fit_all_pairs(grids, angles_2d, thr):
    """grids: [N_PAIRS, N_ACT, N, N]. Return list of fitted p's (None for bad)."""
    ps = []; contours = []
    for pi in range(grids.shape[0]):
        median_grid = np.median(grids[pi], axis=0)
        row0 = median_grid[:, 0]; col0 = median_grid[0, :]
        th1 = find_axis_intercept(row0, angles_2d, thr)
        th2 = find_axis_intercept(col0, angles_2d, thr)
        if not (np.isfinite(th1) and np.isfinite(th2) and th1 > 0 and th2 > 0):
            ps.append(None); contours.append(None); continue
        cp = extract_contour(angles_2d, median_grid, thr)
        if len(cp) == 0:
            ps.append(None); contours.append(None); continue
        cn = cp.copy().astype(float)
        cn[:, 0] /= th1; cn[:, 1] /= th2
        ps.append(fit_superellipse_p(cn))
        contours.append(cn)
    return ps, contours


def main():
    with open(PKL, 'rb') as f:
        r = pickle.load(f)
    angles_2d = r['angles_2d_deg']
    grids = r['grids']

    thr_sets = {
        'cos': [0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.02, 0.03, 0.05],
        'l2':  [50, 100, 150, 250, 352.2, 500, 700, 900, 1100],
        'kl':  [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.4],
    }

    # Per-metric: fit p across thresholds, collect summary stats.
    summary = {}
    for m, thr_list in thr_sets.items():
        entry = {'thresholds': thr_list, 'medians': [], 'lo': [], 'hi': [],
                 'n_valid': [], 'all_ps': []}
        for thr in thr_list:
            ps, _ = fit_all_pairs(grids[m], angles_2d, thr)
            vals = [p for p in ps if p is not None]
            entry['n_valid'].append(len(vals))
            if vals:
                entry['medians'].append(np.median(vals))
                lo, hi = np.percentile(vals, [5, 95])
                entry['lo'].append(lo); entry['hi'].append(hi)
            else:
                entry['medians'].append(np.nan)
                entry['lo'].append(np.nan); entry['hi'].append(np.nan)
            entry['all_ps'].append(vals)
        summary[m] = entry

    # Print a table
    print()
    for m in ('cos', 'l2', 'kl'):
        s = summary[m]
        print(f'=== {m} ===')
        print(f'  {"threshold":>12}  {"n_valid":>8}  {"median p":>10}  {"5–95%":>12}')
        for thr, nv, med, lo, hi in zip(s['thresholds'], s['n_valid'],
                                          s['medians'], s['lo'], s['hi']):
            if np.isfinite(med):
                print(f'  {thr:>12.4g}  {nv:>8d}  {med:>10.2f}  [{lo:.2f}, {hi:.2f}]')
            else:
                print(f'  {thr:>12.4g}  {nv:>8d}  {"—":>10}  {"—":>12}')

    # Plot: median p vs threshold, shaded 5-95% band
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m, label in zip(axes, ('cos', 'l2', 'kl'),
                             ('Cosine', 'L2', 'KL')):
        s = summary[m]
        thrs = np.array(s['thresholds'], dtype=float)
        med = np.array(s['medians']); lo = np.array(s['lo']); hi = np.array(s['hi'])
        ok = np.isfinite(med)
        ax.fill_between(thrs[ok], lo[ok], hi[ok], color='#377eb8', alpha=0.25, label='5–95%')
        ax.plot(thrs[ok], med[ok], 'o-', color='#1f77b4', lw=1.5, label='median p')
        ax.axhline(2, color='k', ls='--', lw=1, alpha=0.6, label='p=2')
        ax.set_xscale('log')
        ax.set_xlabel(f'{label} threshold')
        ax.set_ylabel('fitted p')
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim(0.5, 6)
    plt.suptitle('Fitted p vs threshold, 45 pair slices (median, 5–95%)', fontsize=12)
    plt.tight_layout()
    out1 = os.path.join(OUTDIR, 'p_vs_threshold.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight'); plt.close()
    print(f'\nSaved {out1}')

    # For each metric, also re-do the overlay at the "best" threshold
    # (the threshold where fitted p is closest to 2, as a sensibility check)
    best_thr = {}
    for m in ('cos', 'l2', 'kl'):
        s = summary[m]
        diffs = [abs(med - 2) if np.isfinite(med) else np.inf for med in s['medians']]
        idx = int(np.argmin(diffs))
        best_thr[m] = s['thresholds'][idx]
        print(f'  best threshold ({m}) = {best_thr[m]:.4g}  (median p={s["medians"][idx]:.2f})')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, m, label in zip(axes, ('cos', 'l2', 'kl'),
                             ('Cosine distance', 'L2 distance', 'KL divergence')):
        thr = best_thr[m]
        ps, contours = fit_all_pairs(grids[m], angles_2d, thr)
        valid_ps = [p for p in ps if p is not None]
        xc, yc = superellipse_curve(2); ax.plot(xc, yc, '--', color='#888', lw=1.5, label='p=2 circle')
        x1 = 1 - np.linspace(0, 1, 200); y1 = np.linspace(0, 1, 200)
        ax.plot(x1, y1, ':', color='#ccc', lw=1.5, label='p=1 diamond')
        xs, ys = superellipse_curve(20); ax.plot(xs, ys, ':', color='#bbb', lw=1.5, label='p=∞ square')
        for cn in contours:
            if cn is None: continue
            ang = np.arctan2(cn[:, 1], cn[:, 0])
            order = np.argsort(ang)
            ax.plot(cn[order, 0], cn[order, 1], '-', color='#377eb8', lw=0.9, alpha=0.35)
        ax.plot(1, 0, 'kv', markersize=7); ax.plot(0, 1, 'k>', markersize=7)
        ax.set_xlim(-0.05, 1.5); ax.set_ylim(-0.05, 1.5)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        if valid_ps:
            med = np.median(valid_ps); lo, hi = np.percentile(valid_ps, [5, 95])
            ax.set_title(f'{label}\nbest thr={thr:.4g}, {len(valid_ps)}/45 pairs, '
                         f'p median={med:.2f} [5–95: {lo:.2f}, {hi:.2f}]')
        else:
            ax.set_title(f'{label}\n(none valid)')
        ax.set_xlabel(r'$\alpha_i / \theta_i^*$')
        ax.set_ylabel(r'$\alpha_j / \theta_j^*$')
        ax.legend(fontsize=9, loc='upper right')
    plt.suptitle('Pair-slice overlay at threshold chosen to match p=2 (median)',
                 fontsize=12)
    plt.tight_layout()
    out2 = os.path.join(OUTDIR, 'pair_slices_overlay_besttthr.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {out2}')


if __name__ == '__main__':
    main()
