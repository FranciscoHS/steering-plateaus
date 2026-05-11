"""Beeswarm with one column per direction family (DoM contrastive,
MELBO, SAE decoder, random). Each column shows all C(33,2)=528 pair
fits for that family, with the pooled mean, direction-bootstrap 95% CI,
and median. Same visual style as plot_robustness_beeswarm.

Usage:
  python scripts/exp_map/plotting/plot_directions_beeswarm.py
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.exp_map.plotting.plot_robustness_beeswarm import (
    col_direction_type, _bootstrap_pooled_mean_ci,
)

OUT_DIR = "results/exp_map/figures"
COLORS = ["#3a6ea5", "#D9822B", "#3aa54a", "#a53a3a"]
ORDER = ["DoM contrastive", "MELBO", "SAE decoder", "random"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    args = ap.parse_args()

    families_all = col_direction_type(args.target, args.layer)
    families = {k: families_all[k] for k in ORDER if k in families_all}
    if not families:
        print("no direction-family data found")
        return

    n_cols = len(families)
    fig, ax = plt.subplots(figsize=(2.6 * n_cols + 2.5, 8.0))
    rng = np.random.RandomState(0)
    x_pos = 1.0
    label_y_main = -0.03

    print("Family stats:")
    for s_idx, (family, pair_dict) in enumerate(families.items()):
        color = COLORS[s_idx % len(COLORS)]
        data = np.array(list(pair_dict.values()), dtype=float)
        data = data[np.isfinite(data)]

        xs = x_pos + rng.uniform(-0.30, 0.30, size=len(data))
        ax.scatter(xs, data, color=color, s=12, alpha=0.55,
                   edgecolor="none", zorder=3)

        mean, lo, hi = _bootstrap_pooled_mean_ci({family: pair_dict})
        ax.errorbar(x_pos, mean, yerr=[[mean - lo], [hi - mean]],
                    fmt="D", color="#222222", ecolor="#222222",
                    markersize=7, markerfacecolor="white",
                    markeredgewidth=1.5, elinewidth=1.5,
                    capsize=6, capthick=1.5, zorder=5)
        ax.hlines(np.median(data), x_pos - 0.36, x_pos + 0.36,
                  colors="black", lw=2.5, zorder=6)
        ax.text(x_pos, label_y_main, family, ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=11,
                fontweight="bold", color=color)

        print(f"  {family:>16s}  n={len(data):4d}  mean={mean:.4f}  "
              f"95% CI=[{lo:.4f}, {hi:.4f}]  median={np.median(data):.4f}")
        x_pos += 1.0

    p2 = ax.axhline(2.0, ls="--", color="#D9822B", lw=1.2, alpha=0.85,
                    label=r"$n=2$ (Euclidean)")
    mean_handle = plt.Line2D([0], [0], marker="D", color="#222222",
                              linestyle="", markerfacecolor="white",
                              markeredgewidth=1.5, markersize=7,
                              label="mean (bar = 95% CI, dir bootstrap)")
    median_handle = plt.Line2D([0], [0], color="black", lw=2.5,
                               label="median")
    ax.set_ylabel(r"fitted superellipse exponent $n$", fontsize=12)
    ax.set_xlim(0.2, x_pos - 1.0 + 0.8)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=[p2, mean_handle, median_handle],
              loc="upper right", fontsize=9,
              frameon=True, framealpha=0.9)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.97, bottom=0.10)

    out_png = os.path.join(
        OUT_DIR, f"directions_beeswarm_{args.target}_L{args.layer}.png")
    out_pdf = out_png.replace(".png", ".pdf")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_png, dpi=220, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)
    print(f"\nsaved {out_png}")
    print(f"saved {out_pdf}")


if __name__ == "__main__":
    main()
