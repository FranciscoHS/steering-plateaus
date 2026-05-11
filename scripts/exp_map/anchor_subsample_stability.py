"""Subsample N in {10, 20, 30} anchors from existing 30-anchor sweep
grids and look at the standard deviation of the fitted superellipse
exponent n across B random subsamples per pair.

Reads sweep2d pkls for a given anchor source (default: wiki_en) and
writes:
  - per-pair: mean and std of n across B subsamples for each N.
  - across-pairs aggregate: median std (and quartiles) for each N.

Usage:
  python scripts/exp_map/anchor_subsample_stability.py \
    --anchor_source wiki_en --B 20
"""
from __future__ import annotations
import os, sys, glob, pickle, argparse
sys.path.insert(0, ".")
import numpy as np

from scripts.exp_map.lib.superellipse import (
    extract_contour, axis_intercept, fit_superellipse,
)


def fit_n_from_grid(angles, grid, threshold_frac=0.5,
                     max_alpha_deg=60.0) -> float:
    """Same fit as plot_robustness_beeswarm.fit_pair_at_threshold but
    takes the (n, n) median grid directly."""
    base = min(grid[:, 0].max(), grid[0, :].max())
    thresh = threshold_frac * base
    contour = extract_contour(angles, grid, thresh)
    if contour.size == 0: return np.nan
    edge_eps = 1.0
    grid_max = float(angles.max())
    contour = contour[(contour[:, 0] < grid_max - edge_eps)
                       & (contour[:, 1] < grid_max - edge_eps)]
    contour = contour[(contour[:, 0] <= max_alpha_deg)
                       & (contour[:, 1] <= max_alpha_deg)]
    if len(contour) < 3: return np.nan
    t1 = axis_intercept(angles, grid[:, 0], thresh)
    t2 = axis_intercept(angles, grid[0, :], thresh)
    if not (np.isfinite(t1) and np.isfinite(t2) and t1 > 0 and t2 > 0):
        return np.nan
    return fit_superellipse(contour[:, 0] / t1, contour[:, 1] / t2)["p"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor_source", default="wiki_en")
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--B", type=int, default=20,
                    help="bootstrap resamples per pair per N")
    ap.add_argument("--Ns", default="10,15,20,25,30")
    ap.add_argument("--replace", action="store_true", default=True,
                    help="bootstrap with replacement so N=30 is non-trivial")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    Ns = [int(x) for x in args.Ns.split(",")]

    src_suffix = f"_src{args.anchor_source}" if args.anchor_source != "fineweb" else ""
    pat = (f"results/exp_map/data/sweeps_2d/sweep2d_{args.target}_L"
           f"{args.layer}_*deg{src_suffix}.pkl")
    files = sorted(glob.glob(pat))
    print(f"found {len(files)} pkls for {pat}")
    if not files: return

    rng = np.random.default_rng(args.seed)
    per_pair_std = {N: [] for N in Ns}

    for fp in files:
        with open(fp, "rb") as f: d = pickle.load(f)
        angles = d["angles_deg"]
        per_anchor_grid = d["l2"]   # (n_anchors, n, n)
        n_anchors = per_anchor_grid.shape[0]
        for N in Ns:
            ns = []
            for _ in range(args.B):
                idx = rng.choice(n_anchors, size=N, replace=True)
                grid = np.median(per_anchor_grid[idx], axis=0)
                v = fit_n_from_grid(angles, grid)
                if np.isfinite(v): ns.append(v)
            if len(ns) >= 2:
                per_pair_std[N].append(float(np.std(ns)))

    print(f"\n{args.anchor_source} — n stability across B={args.B} "
          f"anchor subsamples, per-pair std then aggregated:")
    print(f"{'N':>4s}  {'pairs':>6s}  {'std_mean':>8s}  "
          f"{'std_med':>8s}  {'std_q25':>8s}  {'std_q75':>8s}")
    for N in Ns:
        arr = np.array(per_pair_std[N])
        if arr.size == 0:
            print(f"{N:>4d}  no data"); continue
        print(f"{N:>4d}  {arr.size:>6d}  {arr.mean():>8.4f}  "
              f"{np.median(arr):>8.4f}  {np.percentile(arr,25):>8.4f}  "
              f"{np.percentile(arr,75):>8.4f}")

    # 1/sqrt(N) prediction: std(N) = std(N0) * sqrt(N0/N).
    # Fit this against the measured medians and extrapolate to N=50.
    arrs = {N: np.array(per_pair_std[N]) for N in Ns}
    meds = {N: float(np.median(arrs[N])) if arrs[N].size else np.nan
            for N in Ns}
    print("\n1/sqrt(N) prediction check (medians):")
    if 30 in meds and np.isfinite(meds[30]):
        s30 = meds[30]
        for N in Ns + [50]:
            pred = s30 * np.sqrt(30.0 / N)
            obs = meds.get(N, np.nan)
            obs_s = f"{obs:.4f}" if np.isfinite(obs) else "  (extrap) "
            print(f"  N={N:>3d}  predicted={pred:.4f}  observed={obs_s}")
    out = {"anchor_source": args.anchor_source, "Ns": Ns,
           "B": args.B, "per_pair_std": per_pair_std}
    out_pkl = (f"results/exp_map/data/anchor_subsample_stability_"
               f"{args.anchor_source}.pkl")
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, "wb") as f: pickle.dump(out, f)
    print(f"\nsaved {out_pkl}")


if __name__ == "__main__":
    main()
