"""Inspect the three high-p outlier pairs from the robustness beeswarm.

Shows, for each pair, the (alpha_1, alpha_2) response grid as a heatmap,
the 50% iso-contour overlaid in white, and the best-fit superellipse
|x|^p + |y|^p = 1 (after rescaling by axis-intercepts) overlaid in
orange. Title shows the fit p.
"""
from __future__ import annotations
import os, sys, pickle
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.exp_map.lib.superellipse import (
    extract_contour, axis_intercept, fit_superellipse,
)

OUT = "results/exp_map/figures/outlier_iso_fits.png"


CASES = [
    ("results/exp_map/data/sweeps_2d/sweep2d_llama_L2_Gender__Tense_fineweb_60deg.pkl",
     "l2",
     "llama  •  Gender × Tense  •  L² (model column outlier)"),
    ("results/exp_map/data/sweeps_2d/sweep2d_gemma_L2_HonestyShort__Russian_fineweb_60deg_cos+kl+l2.pkl",
     "kl",
     "gemma  •  HonestyShort × Russian  •  KL (metric column outlier)"),
    ("results/exp_map/data/sweeps_2d/sweep2d_gemma_L2_Honesty__Russian_fineweb_60deg_cos+kl+l2.pkl",
     "kl",
     "gemma  •  Honesty × Russian  •  KL (metric column outlier)"),
]


def fit_at_threshold_frac(angles, mean_grid, frac=0.5, max_alpha_deg=60.0):
    base = min(mean_grid[:, 0].max(), mean_grid[0, :].max())
    thresh = frac * base
    contour = extract_contour(angles, mean_grid, thresh)
    grid_max = float(angles.max())
    if contour.size:
        contour = contour[(contour[:, 0] < grid_max - 1.0)
                           & (contour[:, 1] < grid_max - 1.0)]
        contour = contour[(contour[:, 0] <= max_alpha_deg)
                           & (contour[:, 1] <= max_alpha_deg)]
    t1 = axis_intercept(angles, mean_grid[:, 0], thresh)
    t2 = axis_intercept(angles, mean_grid[0, :], thresh)
    fit = fit_superellipse(contour[:, 0] / t1, contour[:, 1] / t2) \
        if (np.isfinite(t1) and np.isfinite(t2) and t1 > 0 and t2 > 0
            and len(contour) >= 3) else {"p": np.nan}
    return contour, t1, t2, thresh, fit


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (path, metric, title) in zip(axes, CASES):
        d = pickle.load(open(path, "rb"))
        angles = d["angles_deg"]
        grid = np.median(d[metric], axis=0)
        contour, t1, t2, thresh, fit = fit_at_threshold_frac(angles, grid, 0.5)
        # Plot in NORMALIZED axis coords (x = alpha_1 / t1, y = alpha_2 / t2)
        # — the same coords used by the superellipse fit, so the unit fit
        # |x|^p + |y|^p = 1 sits directly on top of the rescaled contour.
        if not (np.isfinite(t1) and np.isfinite(t2) and t1 > 0 and t2 > 0):
            ax.set_title(title + "  (no axis intercepts)", fontsize=9)
            continue
        # Plot the 50% iso-contour samples in normalized coords.
        if contour.size:
            ax.scatter(contour[:, 0] / t1, contour[:, 1] / t2, s=10,
                        c="#3a6ea5", alpha=0.85, label="contour samples")
        # Reference: unit circle (p=2) and unit square (p=∞).
        theta = np.linspace(0, np.pi / 2, 200)
        ax.plot(np.cos(theta), np.sin(theta), ls=":", color="#888",
                 lw=1.0, label=r"unit circle ($p=2$)")
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], ls=":", color="#bbb",
                 lw=1.0, label=r"unit square ($p\to\infty$)")
        # Best-fit unit superellipse.
        if np.isfinite(fit["p"]):
            p = fit["p"]
            ux = np.cos(theta) ** (2 / p)
            uy = np.sin(theta) ** (2 / p)
            ax.plot(ux, uy, color="#D9822B", lw=2.2,
                     label=f"fit:  $p = {p:.2f}$")
        ax.set_xlabel(r"$\alpha_1 / t_1$")
        ax.set_ylabel(r"$\alpha_2 / t_2$")
        ax.set_title(f"{title}\n($t_1={t1:.1f}^\\circ$, "
                     f"$t_2={t2:.1f}^\\circ$, threshold = "
                     f"{metric.upper()} of {thresh:.2f})",
                     fontsize=9)
        ax.set_xlim(0, 1.6); ax.set_ylim(0, 1.6)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
