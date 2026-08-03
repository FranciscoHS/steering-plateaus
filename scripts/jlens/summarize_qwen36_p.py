#!/usr/bin/env python3
"""Summarize superellipse p across both legs of the Qwen3.6-27B run:
  - Contrastive: median_p_for_pair over the 6 robustness cells of
    fits_qwen36_L36.pkl (same statistic the canonical beeswarm plots).
  - J-Lens ZH: per-pair p is the fit at each L2 threshold in
    jlens_p_summary_L36.pkl; we report the per-pair median over thresholds.

Prints and writes results/jlens/qwen3.6-27b/p_summary_combined.txt.
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np

from scripts.exp_map.plotting.plot_robustness_beeswarm import median_p_for_pair

FITS = 'results/exp_map/data/fits/fits_qwen36_L36.pkl'
JLENS = 'results/jlens/qwen3.6-27b/jlens_p_summary_L36.pkl'
OUT = 'results/jlens/qwen3.6-27b/p_summary_combined.txt'


def summ(ps, label, lines):
    ps = np.asarray([p for p in ps if np.isfinite(p)])
    lines.append(f"{label}: n={len(ps)}  median={np.median(ps):.3f}  "
                 f"mean={ps.mean():.3f}  IQR=[{np.percentile(ps, 25):.3f}, "
                 f"{np.percentile(ps, 75):.3f}]  "
                 f"frac>2={ (ps > 2).mean():.2f}")
    return ps


def main():
    lines = ["Qwen3.6-27B superellipse p summary — L36 (measure L62)"]

    # Contrastive
    with open(FITS, 'rb') as f:
        full = pickle.load(f)
    con_ps, con_rows = [], []
    for key, fit in full.items():
        p = median_p_for_pair(fit)
        if np.isfinite(p):
            con_ps.append(p)
            con_rows.append((p, key))
    lines.append("\n== Contrastive (median_p over 6 robustness cells) ==")
    summ(con_ps, "Contrastive", lines)
    for p, key in sorted(con_rows):
        lines.append(f"    {str(key):40s} p={p:.3f}")

    # J-Lens ZH: median over thresholds per pair
    with open(JLENS, 'rb') as f:
        rows = pickle.load(f)
    by_pair = {}
    for r in rows:
        by_pair.setdefault((r['category'], r['pair_idx']), []).append(r['p'])
    zh_ps = []
    lines.append("\n== J-Lens ZH (median p over L2 thresholds per pair) ==")
    zh_rows = []
    for (cat, idx), ps in sorted(by_pair.items()):
        if cat != 'ZH':
            continue
        med = float(np.median(ps))
        zh_ps.append(med)
        zh_rows.append((med, cat, idx))
    summ(zh_ps, "J-Lens ZH", lines)
    for med, cat, idx in sorted(zh_rows):
        lines.append(f"    {cat} {idx:02d}  p={med:.3f}")

    txt = '\n'.join(lines)
    with open(OUT, 'w') as f:
        f.write(txt)
    print(txt)
    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
