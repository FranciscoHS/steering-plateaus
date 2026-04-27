"""Run the robustness protocol (threshold x range x bootstrap) on every
sweep2d_<target>_L<layer>_*.pkl in results/exp_map/data/sweeps_2d/.

Output:
  results/exp_map/data/fits/fits_<target>_L<layer>.pkl  -- full per-pair dicts
  also prints a compact table to stdout.
"""
from __future__ import annotations
import os, sys, pickle, glob, argparse
sys.path.insert(0, ".")
import numpy as np
from scripts.exp_map.lib.superellipse import robust_p_fit

OUT_DIR = "results/exp_map/data/fits"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--layer", type=int, default=2)
    args = ap.parse_args()

    pat = f"results/exp_map/data/sweeps_2d/sweep2d_{args.target}_L{args.layer}_*.pkl"
    files = sorted(glob.glob(pat))
    if not files:
        print(f"no files match {pat}"); return

    rows = []
    full = {}
    for fp in files:
        with open(fp, "rb") as f: d = pickle.load(f)
        ang = d["angles_deg"]; grid = d["l2"]   # (n_anchors, n, n)
        a, b = d["direction_labels"]
        fit = robust_p_fit(ang, grid)
        full[(a, b)] = fit
        # collapse cells into a row: low/high across all (threshold x range)
        ps = []
        for k, v in fit.items():
            if k == "__levels__": continue
            if np.isfinite(v["p_lo"]) and np.isfinite(v["p_hi"]):
                ps.append(v["p_lo"]); ps.append(v["p_hi"])
        if ps:
            row = {
                "pair": f"{a} x {b}",
                "p_lo_overall": float(min(ps)),
                "p_hi_overall": float(max(ps)),
                "n_cells_ok": int(sum(1 for k, v in fit.items()
                                       if k != "__levels__" and
                                       np.isfinite(v["p_median"]))),
            }
        else:
            row = {"pair": f"{a} x {b}", "p_lo_overall": np.nan,
                   "p_hi_overall": np.nan, "n_cells_ok": 0}
        rows.append(row)

    # sort by mid-p
    def mid(r):
        if np.isnan(r["p_lo_overall"]): return 99
        return 0.5 * (r["p_lo_overall"] + r["p_hi_overall"])
    rows = sorted(rows, key=mid)

    print(f"\n=== {args.target} L{args.layer}  {len(rows)} pairs ===")
    print(f"{'pair':40s}  p_range (across thresh x range x bootstrap)  cells_ok")
    for r in rows:
        if np.isnan(r["p_lo_overall"]):
            rng = "no fit"
        else:
            rng = f"[{r['p_lo_overall']:.2f}, {r['p_hi_overall']:.2f}]"
        print(f"{r['pair']:40s}  {rng:20s}  {r['n_cells_ok']}/6")

    out_pkl = os.path.join(OUT_DIR, f"fits_{args.target}_L{args.layer}.pkl")
    with open(out_pkl, "wb") as f: pickle.dump(full, f)
    print(f"\nsaved {out_pkl}")


if __name__ == "__main__":
    main()
