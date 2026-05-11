"""Beeswarm with one column per threshold-percentage of the per-pair
max axis L^2. Reuses render() from plot_robustness_beeswarm.

For each filtered pair (Gemma2 L=2, overlap < 0.10, excl 3, no MWE),
loads the canonical sweep2d pkl and refits the super-ellipse at
thresh = pct * min(max(L2[:,0]), max(L2[0,:])) for each requested pct.
Per-pair value is the median of fits across max_alpha ∈ {30, 45, 60}.
"""
from __future__ import annotations
import os, sys, pickle, argparse
sys.path.insert(0, ".")
import numpy as np

from scripts.exp_map.plotting.plot_robustness_beeswarm import (
    render, _load_fit_pair_dict, _filter_by_overlap,
    _filter_exclude, _filter_exclude_mwe, FITS_DIR, OUT_DIR,
)
from scripts.exp_map.lib.superellipse import fit_p_one_pass

SWEEP_DIR = "results/exp_map/data/sweeps_2d"


def _sweep_path(target, layer, a, b, suffix=""):
    sfx = f"_{suffix}" if suffix else ""
    for u, v in [(a, b), (b, a)]:
        p = os.path.join(SWEEP_DIR,
                         f"sweep2d_{target}_L{layer}_{u}__{v}_fineweb_60deg{sfx}.pkl")
        if os.path.exists(p): return p
    return None


def _per_pair_p_at_pct(pkl_path, pct, max_alphas=(30.0, 45.0, 60.0)):
    with open(pkl_path, "rb") as f: dd = pickle.load(f)
    ang = dd["angles_deg"]
    mean_grid = dd["l2"].mean(axis=0)
    axis_max = min(mean_grid[:, 0].max(), mean_grid[0, :].max())
    thresh = pct * axis_max
    ps = []
    for ma in max_alphas:
        fit = fit_p_one_pass(ang, mean_grid, thresh, ma)
        if np.isfinite(fit["p"]): ps.append(fit["p"])
    return float(np.median(ps)) if ps else np.nan


def _per_pair_p_at_fixed(pkl_path, thresh, max_alphas=(30.0, 45.0, 60.0)):
    with open(pkl_path, "rb") as f: dd = pickle.load(f)
    ang = dd["angles_deg"]
    mean_grid = dd["l2"].mean(axis=0)
    ps = []
    for ma in max_alphas:
        fit = fit_p_one_pass(ang, mean_grid, thresh, ma)
        if np.isfinite(fit["p"]): ps.append(fit["p"])
    return float(np.median(ps)) if ps else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--max_overlap", type=float, default=0.10)
    ap.add_argument("--exclude_dirs",
                    default="HonestyShort,TensePresent,Formal")
    ap.add_argument("--exclude_mwe", action="store_true", default=True)
    ap.add_argument("--pcts", default="5,15,25,35,45,55,65,75",
                    help="comma-sep percentages of axis max")
    ap.add_argument("--fixed", default="",
                    help="if set, use these fixed L^2 thresholds (comma-"
                         "sep) instead of pcts. e.g. '10,51,93,134,176,"
                         "217,259,300'.")
    ap.add_argument("--family", default="dom",
                    choices=["dom", "sae", "melbo", "random"],
                    help="dom: contrastive DoMs (overlap-filtered). "
                         "sae|melbo|random: corresponding sweep set, "
                         "no overlap filter (family-specific).")
    args = ap.parse_args()

    excl = {n.strip() for n in args.exclude_dirs.split(",") if n.strip()}
    use_fixed = bool(args.fixed)
    if use_fixed:
        thresh_vals = [float(x) for x in args.fixed.split(",")]
    else:
        pcts = [float(x) for x in args.pcts.split(",")]

    if args.family == "dom":
        fits_suffix = ""; sweep_suffix = ""
    elif args.family == "sae":
        fits_suffix = "_dirsae_random"; sweep_suffix = "dirsae_random"
    elif args.family == "melbo":
        fits_suffix = "_dirmelbo"; sweep_suffix = "dirmelbo"
    elif args.family == "random":
        fits_suffix = "_dirrandom"; sweep_suffix = "dirrandom"

    fp = os.path.join(FITS_DIR,
                      f"fits_{args.target}_L{args.layer}{fits_suffix}.pkl")
    d = _load_fit_pair_dict(fp)
    if args.family == "dom":
        d = _filter_by_overlap(d, args.target, args.layer, args.max_overlap)
        d = _filter_exclude(d, excl)
        d = _filter_exclude_mwe(d, args.exclude_mwe)
    print(f"filtered set ({args.family}): n={len(d)} pairs")

    columns = []
    iter_vals = thresh_vals if use_fixed else pcts
    for v in iter_vals:
        col = {}
        n_skip = 0
        for pair in d:
            a, b = sorted(pair)
            spath = _sweep_path(args.target, args.layer, a, b, sweep_suffix)
            if spath is None: n_skip += 1; continue
            if use_fixed:
                p = _per_pair_p_at_fixed(spath, v)
            else:
                p = _per_pair_p_at_pct(spath, v / 100.0)
            if np.isfinite(p):
                col[pair] = p
        label = f"{int(v)}" if use_fixed else f"{int(v)}%"
        med = (np.median(list(col.values())) if col else float('nan'))
        print(f"  {label:>5s}  n={len(col)}  median={med:.3f}  skip={n_skip}")
        if col:
            columns.append((label, {label: col}))

    fam_tag = "" if args.family == "dom" else f"_{args.family}"
    mode_tag = "_threshold_fixed" if use_fixed else "_threshold_pct"
    out_png = os.path.join(
        OUT_DIR,
        f"robustness_beeswarm_{args.target}_L{args.layer}{fam_tag}{mode_tag}.png")
    out_pdf = out_png.replace(".png", ".pdf")
    render(columns, out_png, out_pdf, show_sub_legends=False)


if __name__ == "__main__":
    main()
