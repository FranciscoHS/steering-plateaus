"""Per-pair 2D iso-L^2 contour plot with normalized coords + super-ellipse fits.

Plots, per pair PKL:
  - mean L^2 grid as a heatmap
  - 50% and 75% threshold contours overlaid in normalized (alpha/intercept) coords
  - reference shapes p=1 (diamond), p=2 (circle)
"""
from __future__ import annotations
import os, sys, pickle, argparse, glob
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.exp_map.lib.superellipse import (
    extract_contour, axis_intercept, threshold_levels,
    fit_p_one_pass, superellipse_curve,
)

FIG_DIR = "results/exp_map/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_one(fp: str):
    with open(fp, "rb") as f: d = pickle.load(f)
    ang = d["angles_deg"]
    grid = d["l2"].mean(axis=0)
    lvl = threshold_levels(grid)
    a, b = d["direction_labels"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # heatmap with both contours
    im = axes[0].imshow(grid.T, origin="lower",
                        extent=[ang[0], ang[-1], ang[0], ang[-1]],
                        aspect="auto", cmap="viridis")
    axes[0].contour(ang, ang, grid.T, levels=[lvl["50"], lvl["75"]],
                    colors=["white", "yellow"], linewidths=1.2)
    axes[0].set_xlabel(f"alpha_1 ({a}, deg)")
    axes[0].set_ylabel(f"alpha_2 ({b}, deg)")
    axes[0].set_title(f"L^2 grid; thresholds={lvl['50']:.2f}, {lvl['75']:.2f}")
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    # normalized contour + fit
    for tl, color in [("50", "C0"), ("75", "C3")]:
        thresh = lvl[tl]
        fit = fit_p_one_pass(ang, grid, thresh, max_alpha_deg=60.0)
        if not np.isfinite(fit["p"]):
            continue
        # extract the normalized contour for plotting
        raw = extract_contour(ang, grid, thresh)
        if raw.size == 0: continue
        t1, t2 = fit["intercepts"]
        xn = raw[:, 0] / t1
        yn = raw[:, 1] / t2
        m = (xn > 0.05) & (yn > 0.05) & (xn < 1.5) & (yn < 1.5)
        axes[1].plot(xn[m], yn[m], "o", color=color, ms=3, alpha=0.6,
                     label=f"th={tl}%, p={fit['p']:.2f} "
                           f"(N={fit['n_pts']})")
        xs, ys = superellipse_curve(fit["p"])
        axes[1].plot(xs, ys, color=color, lw=1.5, ls="--")
    xc, yc = superellipse_curve(2.0)
    axes[1].plot(xc, yc, "k:", lw=0.8, label="p=2 ref")
    xd, yd = superellipse_curve(1.0)
    axes[1].plot(xd, yd, "k-.", lw=0.8, label="p=1 ref")
    axes[1].set_xlabel(f"alpha_1 / theta_1*"); axes[1].set_ylabel(f"alpha_2 / theta_2*")
    axes[1].set_xlim(0, 1.4); axes[1].set_ylim(0, 1.4)
    axes[1].set_aspect("equal")
    axes[1].set_title(f"normalized iso-contours")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"{d['model']} L{d['perturb_layer']}  {a} x {b}")
    fig.tight_layout()
    name = os.path.basename(fp).replace(".pkl", ".png")
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--pair", default=None,
                    help="optional 'A,B' to plot one pair")
    args = ap.parse_args()
    if args.pair:
        a, b = args.pair.split(",")
        files = [f"results/exp_map/data/sweeps_2d/"
                 f"sweep2d_{args.target}_L{args.layer}_{a}__{b}_fineweb_60deg.pkl"]
    else:
        pat = f"results/exp_map/data/sweeps_2d/sweep2d_{args.target}_L{args.layer}_*.pkl"
        files = sorted(glob.glob(pat))
    for fp in files:
        if not os.path.exists(fp):
            print(f"missing {fp}"); continue
        plot_one(fp)


if __name__ == "__main__":
    main()
