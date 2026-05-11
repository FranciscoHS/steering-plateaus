"""Per-anchor decomposition of the per-pair n estimate.

For each pair, the canonical fit medians the L^2 grid across 30 anchors
before fitting. Here we instead fit n on each anchor's own grid (using
the same threshold derived from the median grid, for comparability),
giving 30 n values per pair.

Then ask: how much of the per-pair variation in the beeswarm is
"between-pair" (real per-pair geometry) vs "within-pair" (anchor
sampling noise)?

Usage:
  python scripts/exp_map/per_anchor_decomposition.py \\
      --target gemma --layer 2
"""
from __future__ import annotations
import os, sys, glob, pickle, argparse
sys.path.insert(0, ".")
import numpy as np

from scripts.exp_map.lib.superellipse import (
    fit_p_one_pass, threshold_levels,
)

SWEEPS_DIR = "results/exp_map/data/sweeps_2d"
OUT_DIR = "results/exp_map/data/fits"


def per_anchor_fits(d: dict,
                     thresh_label: str = "50",
                     max_alpha: float = 60.0) -> dict:
    """Return {anchor_idx: p, ...} plus the canonical median-grid fit
    using the same threshold."""
    angles = d["angles_deg"]
    grid = d["l2"]                          # (n_anchors, n, n)
    median_grid = np.median(grid, axis=0)
    levels = threshold_levels(median_grid)
    thresh = levels[thresh_label]
    canonical = fit_p_one_pass(angles, median_grid, thresh, max_alpha)
    per_anchor = {}
    for i in range(grid.shape[0]):
        f = fit_p_one_pass(angles, grid[i], thresh, max_alpha)
        per_anchor[i] = {
            "p": f["p"],
            "n_pts": f["n_pts"],
            "mrf": f["mean_radial_frac"],
        }
    return {
        "thresh_label": thresh_label,
        "thresh_value": float(thresh),
        "max_alpha_deg": float(max_alpha),
        "canonical_p": canonical["p"],
        "canonical_n_pts": canonical["n_pts"],
        "canonical_mrf": canonical["mean_radial_frac"],
        "per_anchor": per_anchor,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--variant_suffix", default="",
                    help="match sweep filename suffix; default '' = canonical")
    ap.add_argument("--thresh", default="50", choices=["50", "75"])
    ap.add_argument("--max_alpha", type=float, default=60.0)
    ap.add_argument("--out_suffix", default="_peranchor")
    args = ap.parse_args()

    pat = (f"{SWEEPS_DIR}/sweep2d_{args.target}_L{args.layer}"
           f"_*deg{args.variant_suffix}.pkl")
    files = sorted(glob.glob(pat))
    if not files:
        print(f"no files match {pat}")
        return
    print(f"found {len(files)} pair sweeps", flush=True)

    all_fits = {}
    for fp in files:
        with open(fp, "rb") as f:
            d = pickle.load(f)
        a, b = d["direction_labels"]
        fit = per_anchor_fits(d, thresh_label=args.thresh,
                                max_alpha=args.max_alpha)
        all_fits[(a, b)] = fit

    # Aggregate stats
    rows = []
    for pair, fit in all_fits.items():
        ps = np.array([v["p"] for v in fit["per_anchor"].values()],
                      dtype=float)
        ps_finite = ps[np.isfinite(ps)]
        rows.append({
            "pair": pair,
            "canonical_p": fit["canonical_p"],
            "n_anchors_finite": int(len(ps_finite)),
            "mean_p": float(np.mean(ps_finite)) if len(ps_finite) else np.nan,
            "median_p": float(np.median(ps_finite)) if len(ps_finite) else np.nan,
            "std_p": float(np.std(ps_finite)) if len(ps_finite) > 1 else np.nan,
            "p25_p": float(np.percentile(ps_finite, 25)) if len(ps_finite) else np.nan,
            "p75_p": float(np.percentile(ps_finite, 75)) if len(ps_finite) else np.nan,
        })

    # Within-pair vs between-pair
    canonical_ps = np.array([r["canonical_p"] for r in rows
                              if np.isfinite(r["canonical_p"])])
    mean_ps_per_pair = np.array([r["mean_p"] for r in rows
                                  if np.isfinite(r["mean_p"])])
    std_ps_per_pair = np.array([r["std_p"] for r in rows
                                 if np.isfinite(r["std_p"])])

    print(f"\n=== variance decomposition  ({args.target} L{args.layer}, "
          f"thresh={args.thresh}%, max_alpha={args.max_alpha:g}°) ===")
    print(f"Canonical (median-grid) per-pair n:")
    print(f"  n_pairs with finite p = {len(canonical_ps)}")
    print(f"  mean = {np.mean(canonical_ps):.3f}   "
          f"median = {np.median(canonical_ps):.3f}   "
          f"std = {np.std(canonical_ps):.3f}")
    print(f"  p25={np.percentile(canonical_ps,25):.3f}  "
          f"p75={np.percentile(canonical_ps,75):.3f}  "
          f"min={canonical_ps.min():.3f}  max={canonical_ps.max():.3f}")
    print()
    print(f"Per-anchor refits (one n per (pair, anchor)):")
    print(f"  Between-pair spread = std(mean_p_per_pair) = "
          f"{np.std(mean_ps_per_pair):.3f}")
    print(f"  Within-pair spread  = mean(std_p_per_pair)  = "
          f"{np.mean(std_ps_per_pair):.3f}")
    print(f"  Ratio (within / between) = "
          f"{np.mean(std_ps_per_pair) / max(np.std(mean_ps_per_pair), 1e-9):.3f}")
    print()
    print("If within ≪ between → the per-pair beeswarm spread is real geometry.")
    print("If within ~ between → it's anchor noise; aggregate harder or report "
          "uncertainty.")

    # Top/bottom by canonical_p, with their per-anchor IQR
    rows_sorted = sorted([r for r in rows if np.isfinite(r["canonical_p"])],
                          key=lambda r: r["canonical_p"])
    print(f"\nLowest canonical n (10):")
    for r in rows_sorted[:10]:
        print(f"  {r['pair'][0]:>10s} x {r['pair'][1]:<10s}  "
              f"canon={r['canonical_p']:.2f}  "
              f"per-anchor mean={r['mean_p']:.2f} "
              f"std={r['std_p']:.2f}  "
              f"IQR=[{r['p25_p']:.2f},{r['p75_p']:.2f}]  "
              f"n_ok={r['n_anchors_finite']}/30")
    print(f"\nHighest canonical n (10):")
    for r in rows_sorted[-10:]:
        print(f"  {r['pair'][0]:>10s} x {r['pair'][1]:<10s}  "
              f"canon={r['canonical_p']:.2f}  "
              f"per-anchor mean={r['mean_p']:.2f} "
              f"std={r['std_p']:.2f}  "
              f"IQR=[{r['p25_p']:.2f},{r['p75_p']:.2f}]  "
              f"n_ok={r['n_anchors_finite']}/30")

    out_path = os.path.join(
        OUT_DIR,
        f"fits_{args.target}_L{args.layer}{args.variant_suffix}{args.out_suffix}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"fits": all_fits, "rows": rows,
                     "args": vars(args)}, f)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
