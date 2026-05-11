"""Report mean, std, N, naive SEM, and direction-bootstrap 95% CI for
each (column, sub-group) in the robustness beeswarm. Mirrors the data
loaders in plot_robustness_beeswarm.

The naive SEM = std/sqrt(N) treats every pair as independent, which is
wrong here: each direction appears in N-1 pairs. The direction
bootstrap resamples directions (the true unit of variation) with
replacement within each sub-group, rebuilds pairs, and recomputes the
mean — giving an honest CI."""
from __future__ import annotations
import sys
sys.path.insert(0, ".")
import numpy as np

from scripts.exp_map.plotting.plot_robustness_beeswarm import (
    col_model, col_perturb_layer, col_measure_layer, col_metric,
    col_threshold, col_method, col_direction_type,
    _bootstrap_pooled_mean_ci,
)


def report(label, subs):
    print(f"\n{label}")
    for sub, pair_dict in subs.items():
        a = np.array(list(pair_dict.values()), dtype=float)
        a = a[np.isfinite(a)]
        n = len(a)
        if n == 0:
            print(f"  {sub:>16s}  n=0")
            continue
        mean = a.mean()
        std = a.std(ddof=1) if n > 1 else float("nan")
        sem = std / np.sqrt(n) if n > 1 else float("nan")
        # Direction bootstrap on this single sub-group only.
        bm, lo, hi = _bootstrap_pooled_mean_ci({sub: pair_dict}, n_iter=2000)
        print(f"  {sub:>16s}  n={n:4d}  mean={mean:.4f}  std={std:.4f}  "
              f"naive SEM={sem:.4f}  dir-boot 95% CI=[{lo:.4f}, {hi:.4f}]"
              f"  (half={0.5*(hi-lo):.4f})")


def main():
    target, layer = "gemma", 2
    report("Model",               col_model(layer))
    report("Perturbation layer",  col_perturb_layer(target))
    report("Measurement layer",   col_measure_layer(target, layer))
    report("Fit metric",          col_metric(target, layer))
    report("Response threshold",  col_threshold(target, layer))
    report("Perturbation method", col_method(target, layer))
    report("Direction family",    col_direction_type(target, layer))


if __name__ == "__main__":
    main()
