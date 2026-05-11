"""Per-anchor-set 1D sweeps overlaying the original MWE direction with
Gender and French at the same anchors, plus random baseline."""
from __future__ import annotations
import os, sys, pickle, glob
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "results/exp_map/data/sweeps_1d"
FIG_DIR = "results/exp_map/figures"
os.makedirs(FIG_DIR, exist_ok=True)

mwe_files = sorted(glob.glob(
    os.path.join(DATA_DIR, "sweep1d_gemma_L2_*_mwe_*_match_75deg.pkl")))
# Keep only the original-MWE files: those where direction_name == anchor concept.
mwe_self = []
for fp in mwe_files:
    with open(fp, "rb") as f: d = pickle.load(f)
    if d["source_name"] == f"mwe_{d['direction_name']}_match":
        mwe_self.append(fp)
print(f"matched {len(mwe_self)} MWE-self files")

ncols = min(len(mwe_self), 3)
nrows = int(np.ceil(len(mwe_self) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.6*nrows),
                         sharex=True, squeeze=False)
axes = axes.flatten()

def quartiles(arr_2d):
    return (np.median(arr_2d, axis=0),
            np.percentile(arr_2d, 25, axis=0),
            np.percentile(arr_2d, 75, axis=0))

for i, fp in enumerate(mwe_self):
    with open(fp, "rb") as f: dm = pickle.load(f)
    name = dm["direction_name"]
    src = dm["source_name"]
    ang = dm["angles_deg"]
    med, lo, hi = quartiles(dm["l2"])
    rand = dm["l2_random"].reshape(-1, len(ang))
    rmed, rlo, rhi = quartiles(rand)

    ax = axes[i]
    ax.fill_between(ang, lo, hi, alpha=0.22, color="C0")
    ax.plot(ang, med, color="C0", lw=2, label=f"DoM: {name}")
    ax.fill_between(ang, rlo, rhi, alpha=0.15, color="grey")
    ax.plot(ang, rmed, color="grey", lw=1.3, ls="--",
            label=f"random (n={rand.shape[0]//dm['n_anchors']}, median)")

    for color, dname in [("C1", "Gender"), ("C2", "French")]:
        gp = os.path.join(
            DATA_DIR, f"sweep1d_gemma_L2_{dname}_{src}_75deg.pkl")
        if not os.path.exists(gp): continue
        with open(gp, "rb") as f: dg = pickle.load(f)
        gmed, glo, ghi = quartiles(dg["l2"])
        ax.fill_between(ang, glo, ghi, alpha=0.18, color=color)
        ax.plot(ang, gmed, color=color, lw=1.8, label=f"DoM: {dname}")

    ax.set_title(name, fontsize=8.5)
    ax.set_xlabel("alpha (deg)"); ax.set_ylabel("L2 at penult. layer")
    ax.legend(fontsize=6.5, loc="upper left"); ax.grid(alpha=0.25)

for j in range(len(mwe_self), len(axes)): axes[j].axis("off")
fig.suptitle("E1 1D sweeps - 5 MWE anchor sets, gemma L2 - "
             "DoM(MWE) vs Gender vs French (same anchors)",
             y=1.00, fontsize=10)
fig.tight_layout()
out = os.path.join(FIG_DIR, "e1_1d_gemma_L2_mwe5_with_gender_french.png")
fig.savefig(out, dpi=140); plt.close(fig)
print(f"saved {out}")
