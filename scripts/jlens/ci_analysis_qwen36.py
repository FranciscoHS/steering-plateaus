#!/usr/bin/env python3
"""How reliably is the J-Lens ZH median p separated from 2?

Reports, per family: bootstrap CIs on the median at 95/99/99.9%, the bootstrap
mass at or above p=2, and two distribution-free checks against the null
"median = 2" (sign test and Wilcoxon signed-rank). Pure post-processing.
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
from scipy import stats

from scripts.exp_map.plotting.plot_robustness_beeswarm import median_p_for_pair
from scripts.jlens.plot_qwen36_beeswarm import jlens_medians, FITS, JDIR

N_BOOT = 200_000
OUT = os.path.join(JDIR, 'ci_analysis_L36.txt')


def boot_ci(ps, rng, n_boot=N_BOOT):
    ps = np.asarray(ps, dtype=float)
    draws = rng.choice(ps, size=(n_boot, len(ps)), replace=True)
    meds = np.median(draws, axis=1)
    return meds


def main():
    with open(FITS, 'rb') as f:
        full = pickle.load(f)
    con = [p for p in (median_p_for_pair(v) for v in full.values())
           if np.isfinite(p)]
    zh = jlens_medians(os.path.join(JDIR, 'jlens_p_summary_L36_zh100.pkl'), 'ZH')
    en = jlens_medians(os.path.join(JDIR, 'jlens_p_summary_L36_en_concrete.pkl'),
                       'EN')

    lines = ["Bootstrap CI on the median p — qwen3.6-27b L36→L62",
             f"percentile bootstrap, {N_BOOT:,} resamples with replacement, "
             f"seed 0", ""]
    rng = np.random.default_rng(0)
    for label, ps in [('Contrastive', con), ('J-Lens ZH', zh),
                      ('J-Lens EN (concrete)', en)]:
        ps = np.asarray(ps, dtype=float)
        meds = boot_ci(ps, rng)
        med = float(np.median(ps))
        lines.append(f"{label}  (n={len(ps)}, median={med:.3f})")
        for lvl, (a, b) in [(95, (2.5, 97.5)), (99, (0.5, 99.5)),
                            (99.9, (0.05, 99.95))]:
            lo, hi = np.percentile(meds, [a, b])
            lines.append(f"   {lvl:>5}% CI: [{lo:.3f}, {hi:.3f}]   "
                         f"width={hi - lo:.3f}")
        frac_ge2 = float((meds >= 2.0).mean())
        lines.append(f"   bootstrap mass at median>=2: {frac_ge2:.5f} "
                     f"({int(frac_ge2 * N_BOOT)}/{N_BOOT})")
        # distribution-free checks vs the null "median = 2"
        n_above = int((ps > 2).sum()); n_below = int((ps < 2).sum())
        sign_p = stats.binomtest(n_above, n_above + n_below, 0.5).pvalue
        try:
            w_p = stats.wilcoxon(ps - 2.0).pvalue
        except Exception:
            w_p = float('nan')
        lines.append(f"   pairs above/below 2: {n_above}/{n_below}   "
                     f"sign-test p={sign_p:.3g}   wilcoxon p={w_p:.3g}")
        lines.append("")

    txt = '\n'.join(lines)
    with open(OUT, 'w') as f:
        f.write(txt)
    print(txt)
    print(f"Saved: {OUT}")


if __name__ == '__main__':
    main()
