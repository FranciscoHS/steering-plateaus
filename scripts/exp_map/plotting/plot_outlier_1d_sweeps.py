"""1D L^2 sweeps along each of the two llama directions in the
Gender x Tense outlier pair. Each panel = one axis of the 2D grid
(alpha_2=0 or alpha_1=0). Horizontal lines = threshold levels used
by the superellipse fit, computed as fraction *  min(max axis_1,
max axis_2) per the threshold_levels() rule.
"""
from __future__ import annotations
import os, sys, pickle
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATH = "results/exp_map/data/sweeps_2d/sweep2d_llama_L2_Gender__Tense_fineweb_60deg.pkl"
OUT = "results/exp_map/figures/outlier_llama_gender_tense_1d.png"
FRACS = [0.25, 0.50, 0.75]
COLORS = {0.25: "#3a6ea5", 0.50: "#D9822B", 0.75: "#7B3294"}


def main():
    d = pickle.load(open(PATH, "rb"))
    angles = d["angles_deg"]
    # Mean over 30 anchors → 31x31 grid.
    grid = np.median(d["l2"], axis=0)
    # 1D sweeps along each axis (the other one held at alpha=0).
    axis_d1 = grid[:, 0]   # vary alpha_1 (Gender), alpha_2 = 0
    axis_d2 = grid[0, :]   # vary alpha_2 (Tense),  alpha_1 = 0
    # Per-anchor 1D sweeps, for the spread band.
    per_anchor_d1 = d["l2"][:, :, 0]
    per_anchor_d2 = d["l2"][:, 0, :]
    # Threshold rule used by the fit:
    #   thresh = frac * min(max(axis_d1), max(axis_d2))
    base = min(axis_d1.max(), axis_d2.max())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for ax, mean_curve, per_anchor, name in [
        (axes[0], axis_d1, per_anchor_d1, "Gender ($d_1$)"),
        (axes[1], axis_d2, per_anchor_d2, "Tense ($d_2$)"),
    ]:
        # Per-anchor cloud (faded).
        for i in range(per_anchor.shape[0]):
            ax.plot(angles, per_anchor[i], color="#888", lw=0.5, alpha=0.25)
        ax.plot(angles, mean_curve, color="black", lw=2.0,
                 label="anchor mean")
        # Threshold lines + the angle at which each is reached.
        for f in FRACS:
            t = f * base
            ax.axhline(t, color=COLORS[f], lw=1.4, ls="--",
                        label=f"thresh = {int(f*100)}%  ($L^2={t:.1f}$)")
            # First crossing on the mean curve.
            cross = np.where(mean_curve >= t)[0]
            if len(cross):
                # linear-interpolate between cross-1 and cross
                i = cross[0]
                if i > 0:
                    a0, a1 = angles[i-1], angles[i]
                    v0, v1 = mean_curve[i-1], mean_curve[i]
                    if v1 != v0:
                        a = a0 + (t - v0) * (a1 - a0) / (v1 - v0)
                    else:
                        a = a1
                else:
                    a = angles[i]
                ax.plot([a, a], [0, t], color=COLORS[f], lw=1.0, ls=":",
                         alpha=0.8)
                ax.scatter([a], [t], color=COLORS[f], s=30, zorder=5)
                ax.annotate(f"{a:.1f}°", xy=(a, t), xytext=(a + 1.5, t),
                            color=COLORS[f], fontsize=8, va="center")
        ax.set_xlabel(r"$\alpha$ (deg)")
        ax.set_ylabel(r"$L^2$ at penult")
        ax.set_title(f"1D sweep along {name}  •  llama L=2  •  "
                     f"max = {mean_curve.max():.1f}",
                     fontsize=10)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, angles[-1])
        ax.set_ylim(0, max(per_anchor.max(), mean_curve.max()) * 1.05)

    fig.suptitle(
        "llama-3.1-8b · Gender × Tense · 1D sweeps  "
        "(threshold rule: frac × min(max $d_1$, max $d_2$) = "
        f"frac × {base:.1f})",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}")
    print(f"axis_d1 max = {axis_d1.max():.2f}  axis_d2 max = {axis_d2.max():.2f}")
    print(f"base (= min) = {base:.2f}; thresholds:")
    for f in FRACS:
        print(f"  {int(f*100)}%  ->  L^2 = {f*base:.2f}")


if __name__ == "__main__":
    main()
