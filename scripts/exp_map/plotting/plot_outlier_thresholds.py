"""Re-fit the llama Gender x Tense pair at multiple thresholds and
plot the rescaled iso-contour + unit superellipse fit at each."""
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

PATH = "results/exp_map/data/sweeps_2d/sweep2d_llama_L2_Gender__Tense_fineweb_60deg.pkl"
OUT = "results/exp_map/figures/outlier_llama_gender_tense_thresholds.png"
FRACS = [0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
MAX_ALPHA = 60.0


def fit_at(angles, grid, frac):
    base = min(grid[:, 0].max(), grid[0, :].max())
    thresh = frac * base
    contour = extract_contour(angles, grid, thresh)
    grid_max = float(angles.max())
    if contour.size:
        contour = contour[(contour[:, 0] < grid_max - 1.0)
                           & (contour[:, 1] < grid_max - 1.0)]
        contour = contour[(contour[:, 0] <= MAX_ALPHA)
                           & (contour[:, 1] <= MAX_ALPHA)]
    t1 = axis_intercept(angles, grid[:, 0], thresh)
    t2 = axis_intercept(angles, grid[0, :], thresh)
    if not (np.isfinite(t1) and np.isfinite(t2) and t1 > 0 and t2 > 0
            and len(contour) >= 3):
        return contour, t1, t2, thresh, {"p": np.nan}
    fit = fit_superellipse(contour[:, 0] / t1, contour[:, 1] / t2)
    return contour, t1, t2, thresh, fit


def main():
    d = pickle.load(open(PATH, "rb"))
    angles = d["angles_deg"]
    grid = np.median(d["l2"], axis=0)
    n = len(FRACS)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.0))
    axes = axes.flatten()
    theta = np.linspace(0, np.pi / 2, 200)
    for ax, frac in zip(axes, FRACS):
        contour, t1, t2, thresh, fit = fit_at(angles, grid, frac)
        if not (np.isfinite(t1) and np.isfinite(t2) and t1 > 0 and t2 > 0):
            ax.set_title(f"frac = {int(frac*100)}%  (no fit)")
            continue
        ax.scatter(contour[:, 0] / t1, contour[:, 1] / t2, s=10,
                    c="#3a6ea5", alpha=0.85, label="contour samples")
        ax.plot(np.cos(theta), np.sin(theta), ls=":", color="#888",
                 lw=1.0, label=r"$p=2$")
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], ls=":", color="#bbb",
                 lw=1.0, label=r"$p\to\infty$")
        if np.isfinite(fit["p"]):
            p = fit["p"]
            ux = np.cos(theta) ** (2 / p)
            uy = np.sin(theta) ** (2 / p)
            ax.plot(ux, uy, color="#D9822B", lw=2.2,
                     label=f"fit:  $p = {p:.2f}$")
        ax.set_xlim(0, 1.6); ax.set_ylim(0, 1.6)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\alpha_1 / t_1$")
        ax.set_ylabel(r"$\alpha_2 / t_2$")
        ax.set_title(f"threshold = {int(frac*100)}%  •  "
                     f"$t_1={t1:.1f}^\\circ$, $t_2={t2:.1f}^\\circ$  •  "
                     f"$L^2={thresh:.1f}$",
                     fontsize=9)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
        ax.grid(alpha=0.25)
    fig.suptitle(
        "llama-3.1-8b  •  Gender × Tense  •  threshold sweep "
        "(default beeswarm uses 50%)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
