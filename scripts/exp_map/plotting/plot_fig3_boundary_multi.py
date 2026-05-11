"""Multi-threshold version of plot_fig3_boundary.

Same polar (theta cos phi / theta_1, theta sin phi / theta_2) layout as
plot_fig3_boundary.py but loops over many absolute L^2 thresholds, fits
a superellipse per contour, and plots each in a distinct color.
No reference shapes (p=2 / p=inf) — kept clean for many overlays.

Usage:
  python scripts/exp_map/plotting/plot_fig3_boundary_multi.py \\
      --target gemma --d1 Gender --d2 Tense --layer 2 \\
      --threshs 5 10 25 50 100 200 300
"""
from __future__ import annotations
import os, sys, argparse, pickle
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.exp_map.lib.superellipse import (
    extract_contour, axis_intercept, fit_superellipse,
)

OUT_DIR = "results/exp_map/figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--d1", required=True)
    ap.add_argument("--d2", required=True)
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--threshs", type=float, nargs="+", required=True,
                    help="absolute L^2 thresholds to overlay")
    ap.add_argument("--max_alpha", type=float, default=60.0)
    ap.add_argument("--sbr", action="store_true",
                    help="use the _sbr variant (single-batch-ref, no "
                         "alpha=(0,0) floor)")
    ap.add_argument("--variant_suffix", default="",
                    help="extra filename suffix appended after _sbr (e.g. "
                         "'_d3-0.1-2' for the dense+coarse sweep)")
    ap.add_argument("--uniform_interp_step", type=float, default=0.1,
                    help="bilinearly interpolate the median L^2 grid onto a "
                         "uniform angle grid at this step (deg) before "
                         "contour extraction. Fixes contour distortion on "
                         "non-uniform sweeps (e.g. dense-near-origin grids "
                         "where matplotlib's contour() draws straight chords "
                         "across the dense/coarse seam). Set to 0 to disable.")
    ap.add_argument("--max_pts_per_contour", type=int, default=30,
                    help="subsample each contour evenly to at most this "
                         "many points before plotting, so dense contours "
                         "render as discrete dots not solid lines. Set 0 "
                         "to disable subsampling.")
    ap.add_argument("--marker_size", type=float, default=5.0,
                    help="matplotlib markersize for the dots")
    ap.add_argument("--no_normalize", action="store_true",
                    help="plot contours in raw (alpha1, alpha2) degrees "
                         "instead of normalized (theta cos / theta_1, "
                         "theta sin / theta_2). Filename gets a _raw "
                         "suffix. p_fit is still computed (on the "
                         "normalized version) and shown in the legend.")
    ap.add_argument("--coherence_json", default=None,
                    help="path to a coherence_boundary.py output JSON; "
                         "if set, overlay the per-ray boundary points "
                         "(only meaningful with --no_normalize since the "
                         "coherence boundary lives in raw alpha space).")
    ap.add_argument("--out_tag", default="")
    args = ap.parse_args()

    sbr_suffix = "_sbr" if args.sbr else ""
    pkl_path = (f"results/exp_map/data/sweeps_2d/sweep2d_{args.target}_L"
                f"{args.layer}_{args.d1}__{args.d2}_fineweb_"
                f"{int(args.max_alpha)}deg{sbr_suffix}{args.variant_suffix}.pkl")
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    angles = d["angles_deg"]
    grid = d["l2"]                    # (n_anchors, n, n)
    median_grid = np.median(grid, axis=0)

    # Optionally re-grid onto uniform angles. matplotlib's contour() draws
    # linear chords between adjacent grid cells, so on the dense+coarse
    # sweep (0.1deg up to 3deg, then 2deg) contours straddling the seam are
    # distorted (verified to bias p_fit by up to ~0.12 in T~10..100).
    if args.uniform_interp_step and args.uniform_interp_step > 0:
        from scipy.interpolate import RegularGridInterpolator
        uangles = np.arange(0.0, angles[-1] + 1e-9,
                            args.uniform_interp_step)
        if len(uangles) != len(angles) or not np.allclose(uangles, angles):
            interp = RegularGridInterpolator(
                (angles, angles), median_grid,
                method="linear", bounds_error=False, fill_value=np.nan)
            Aa, Bb = np.meshgrid(uangles, uangles, indexing="ij")
            median_grid = interp(np.stack([Aa.ravel(), Bb.ravel()],
                                          axis=-1)
                                 ).reshape(len(uangles), len(uangles))
            angles = uangles
            print(f"interpolated onto uniform {args.uniform_interp_step}deg "
                  f"grid: n={len(angles)}")

    ax_max1 = median_grid[:, 0].max()
    ax_max2 = median_grid[0, :].max()
    print(f"axis maxes: {ax_max1:.2f} / {ax_max2:.2f}")

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    if args.no_normalize:
        ax.grid(True, which="major", color="#bbbbbb", lw=0.5, alpha=0.45)
    else:
        ax.set_xticks(np.arange(0, 1.61, 0.2))
        ax.set_yticks(np.arange(0, 1.61, 0.2))
        ax.grid(True, which="major", color="#bbbbbb", lw=0.5, alpha=0.45)

    threshs = sorted(set(float(t) for t in args.threshs))
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(threshs), vmax=max(threshs)) \
        if len(threshs) > 1 else plt.Normalize(vmin=0, vmax=1)

    # In normalized mode the data is O(1) so 1.15 is a sensible lower
    # bound. In raw (degree) mode we don't pre-pad — use the actual
    # data range (which can be sub-degree for tiny thresholds).
    xmax_seen = 0.0 if args.no_normalize else 1.15
    ymax_seen = 0.0 if args.no_normalize else 1.15
    for thresh in threshs:
        color = cmap(norm(thresh)) if len(threshs) > 1 else cmap(0.5)
        theta1 = axis_intercept(angles, median_grid[:, 0], thresh)
        theta2 = axis_intercept(angles, median_grid[0, :], thresh)
        if not (np.isfinite(theta1) and np.isfinite(theta2)
                and theta1 > 0 and theta2 > 0):
            print(f"thresh={thresh:.2f}: axis intercepts out of range "
                  f"(theta1={theta1}, theta2={theta2}) — skipping")
            continue
        contour = extract_contour(angles, median_grid, thresh)
        if contour.size == 0:
            print(f"thresh={thresh:.2f}: no contour points — skipping")
            continue
        contour_raw = contour.astype(float).copy()  # keep alpha-deg coords
        cn = contour.astype(float).copy()
        cn[:, 0] /= theta1; cn[:, 1] /= theta2
        keep = (cn[:, 0] > 0.05) & (cn[:, 1] > 0.05) \
            & (cn[:, 0] < 1.5) & (cn[:, 1] < 1.5)
        cn = cn[keep]
        contour_raw = contour_raw[keep]
        if cn.shape[0] < 4:
            print(f"thresh={thresh:.2f}: only {cn.shape[0]} points after "
                  f"filtering — skipping fit")
            continue
        fit = fit_superellipse(cn[:, 0], cn[:, 1])
        p_fit = fit["p"]
        print(f"thresh={thresh:.2f}: theta1={theta1:.2f}, theta2={theta2:.2f}, "
              f"p_fit={p_fit:.3f}, n_pts={fit['n_pts']}, "
              f"mrf={fit['mean_radial_frac']:.3f}")

        cont_angle = np.arctan2(cn[:, 1], cn[:, 0])
        order = np.argsort(cont_angle)
        plot_pts = contour_raw if args.no_normalize else cn
        plot_pts_sorted = plot_pts[order]
        if args.max_pts_per_contour and \
                plot_pts_sorted.shape[0] > args.max_pts_per_contour:
            idx = np.linspace(0, plot_pts_sorted.shape[0] - 1,
                              args.max_pts_per_contour).astype(int)
            plot_pts_sorted = plot_pts_sorted[idx]
        mrf = fit["mean_radial_frac"]
        ax.plot(plot_pts_sorted[:, 0], plot_pts_sorted[:, 1], "o",
                color=color, markersize=args.marker_size, alpha=0.9,
                markeredgecolor="none",
                label=(rf"$L^2={thresh:g}$  ($n={p_fit:.2f}$, "
                       rf"err={mrf*100:.1f}%)"))
        xmax_seen = max(xmax_seen, plot_pts[:, 0].max())
        ymax_seen = max(ymax_seen, plot_pts[:, 1].max())

    # Optionally overlay the coherence regions as shaded polygons (only
    # meaningful in raw alpha space — coordinates are degrees).
    if args.coherence_json and args.no_normalize:
        import json as _json
        from matplotlib.patches import Polygon as _Polygon
        with open(args.coherence_json) as f:
            coh = _json.load(f)
        phis = np.array([r["phi_deg"] for r in coh["rays"]])
        cb_x = np.array([r["alpha1_deg"] for r in coh["rays"]])
        cb_y = np.array([r["alpha2_deg"] for r in coh["rays"]])
        order_phi = np.argsort(phis)
        cb_x = cb_x[order_phi]
        cb_y = cb_y[order_phi]
        # Track for axis limits
        xmax_seen = max(xmax_seen, float(cb_x.max()))
        ymax_seen = max(ymax_seen, float(cb_y.max()))

    if args.no_normalize:
        # Pad proportional to the data range so this looks right whether
        # the contours live at 0.05° or 50°.
        pad = max(0.02, 0.05 * max(xmax_seen, ymax_seen))
        ax.set_xlim(-pad, xmax_seen + pad)
        ax.set_ylim(-pad, ymax_seen + pad)
    else:
        ax.set_xlim(-0.03, xmax_seen + 0.05)
        ax.set_ylim(-0.03, ymax_seen + 0.05)
    ax.set_aspect("equal")

    # If we have a coherence overlay, draw shaded coherent / incoherent
    # regions now that the axes are sized. The coherent polygon is the
    # quadrant-bounded region inside the boundary curve (origin -> sorted
    # boundary points -> back to origin).
    if args.coherence_json and args.no_normalize:
        from matplotlib.patches import Polygon as _Polygon
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        # Incoherent: full visible rectangle (drawn first, behind contours)
        incoh = _Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                         closed=True, facecolor="#d62728", alpha=0.10,
                         edgecolor="none", zorder=0)
        ax.add_patch(incoh)
        # Coherent: origin + boundary curve
        coh_poly_pts = [(0.0, 0.0)]
        coh_poly_pts += list(zip(cb_x.tolist(), cb_y.tolist()))
        coh_poly_pts += [(0.0, 0.0)]
        coh = _Polygon(coh_poly_pts, closed=True,
                       facecolor="#2ca02c", alpha=0.18,
                       edgecolor="#2ca02c", linewidth=1.2, zorder=0.5)
        ax.add_patch(coh)
        # Region labels — horizontally aligned, each centered in its
        # half (coherent under boundary, incoherent past the L^2=500
        # contour at the same height).
        # placed so each sits between two adjacent contour lines at the
        # chosen y, with the bbox not crossing any contour
        label_x = 18.0
        ax.text(label_x, 35.0, "coherent",
                color="#1c6b1c", fontsize=11,
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="#2ca02c",
                          alpha=0.95))
        ax.text(label_x, 65.0, "incoherent",
                color="#a02020", fontsize=11,
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="#d62728",
                          alpha=0.95))
    if args.no_normalize:
        ax.set_xlabel(rf"$\alpha_1$ (deg)  [{args.d1}]", fontsize=12)
        ax.set_ylabel(rf"$\alpha_2$ (deg)  [{args.d2}]", fontsize=12)
    else:
        ax.set_xlabel(rf"$\theta(\varphi)\cos\varphi / \theta_1$  ({args.d1})",
                      fontsize=12)
        ax.set_ylabel(rf"$\theta(\varphi)\sin\varphi / \theta_2$  ({args.d2})",
                      fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.95,
              ncol=1)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    raw_suffix = "_raw" if args.no_normalize else ""
    out_png = os.path.join(OUT_DIR,
        f"fig3_boundary_multi_{args.d1}_{args.d2}_{args.target}_"
        f"L{args.layer}{sbr_suffix}{args.variant_suffix}{raw_suffix}{tag}.png")
    out_pdf = out_png.replace(".png", ".pdf")
    plt.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")


if __name__ == "__main__":
    main()
