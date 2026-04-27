"""Quick L^2 vs angle plot for a 1D exp-map sweep."""
from __future__ import annotations
import os, sys, pickle, argparse, glob
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "results/exp_map/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--layer", type=int, default=2)
    args = ap.parse_args()

    pat = f"results/exp_map/data/sweeps_1d/sweep1d_{args.target}_L{args.layer}_*.pkl"
    files = sorted(glob.glob(pat))
    if not files:
        print(f"no files match {pat}"); return

    fig, axes = plt.subplots(4, 4, figsize=(20, 14), sharex=True, sharey=False)
    axes = axes.flatten()
    for i, fp in enumerate(files):
        if i >= len(axes): break
        with open(fp, "rb") as f: d = pickle.load(f)
        ang = d["angles_deg"]
        l2_med = np.median(d["l2"], axis=0)
        l2_lo = np.percentile(d["l2"], 25, axis=0)
        l2_hi = np.percentile(d["l2"], 75, axis=0)
        rand_med = np.median(d["l2_random"].reshape(-1, len(ang)), axis=0)
        ax = axes[i]
        ax.fill_between(ang, l2_lo, l2_hi, alpha=0.25, color="C0")
        ax.plot(ang, l2_med, color="C0", lw=2,
                label=f"{d['direction_name']} ({d['source_name']})")
        ax.plot(ang, rand_med, color="grey", lw=1.2, ls="--",
                label="random (median)")
        ax.set_title(d["direction_name"], fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25)
    for i in range(len(files), len(axes)): axes[i].axis("off")
    fig.suptitle(f"E1 1D exp-map sweeps  -  {args.target}  L{args.layer}",
                 y=0.995)
    fig.supxlabel("alpha (deg)"); fig.supylabel("L2 at penultimate layer")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"e1_1d_{args.target}_L{args.layer}.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
