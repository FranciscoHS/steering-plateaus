"""Plot 1D holdout sweeps for selected concepts on a single axis.

Loads per-concept pickles produced by `sweep_1d.py` from
`results/exp_map/data/sweeps_1d/`, plots median L^2 vs perturbation angle
(IQR shading) for each requested concept, and overlays a single random
baseline curve pooled from the requested concepts' `l2_random` arrays
(so the random sweeps are taken from the same mix of holdout activations).

Usage:
  python scripts/exp_map/plotting/plot_holdout_sweeps.py \
      --target gemma --layer 2 \
      --concepts Gender_male Tense_past
"""
from __future__ import annotations
import os, sys, pickle, argparse
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "results/exp_map/data/sweeps_1d"
FIG_DIR = "results/exp_map/figures"

LABEL = {
    "gemma":   "Gemma-2-9B",
    "qwen":    "Qwen3-1.7B",
    "llama":   "Llama-3.1-8B",
    "mistral": "Mistral-7B-v0.3",
    "aya":     "Aya-Expanse-8B",
    "yi":      "Yi-1.5-9B",
}

COLORS = {
    "Gender":       "#ff7f00",
    "Tense":        "#377eb8",
    "Sentiment":    "#4daf4a",
    "Age":          "#984ea3",
    "Era":          "#ff7f00",
    "Formal":       "#a65628",
    "Honesty":      "#f781bf",
    "HonestyShort": "#999999",
    "Literary":     "#66c2a5",
    "Number":       "#fc8d62",
    "Person":       "#8da0cb",
    "Refusal":      "#e78ac3",
    "Status":       "#a6d854",
    "TensePresent": "#ffd92f",
}


def load_concept(target: str, layer: int, concept: str):
    """concept is '<DirectionName>_<promptset>' e.g. 'Gender_male'."""
    fp = os.path.join(
        DATA_DIR,
        f"sweep1d_{target}_L{layer}_{concept}_prompts_holdout_75deg.pkl",
    )
    with open(fp, "rb") as f:
        return pickle.load(f), fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(LABEL))
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument(
        "--concepts", nargs="+", required=True,
        help="Per-concept stems, e.g. Gender_male Tense_past",
    )
    ap.add_argument("--out_name", default=None,
                    help="Override output filename stem.")
    ap.add_argument("--n_random_per_concept", type=int, default=15,
                    help="Random sweeps subsampled per concept for the "
                         "pooled baseline (matches contrastive n_anchors "
                         "across both concepts when set to n_anchors/2).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    plt.rcParams["font.family"] = "Inter"
    plt.rcParams["font.sans-serif"] = ["Inter", "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "cm"

    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.6))

    angles = None
    measure_layer = None
    perturb_layer = None
    pooled_random = []

    for concept in args.concepts:
        d, fp = load_concept(args.target, args.layer, concept)
        ang = d["angles_deg"]
        if angles is None:
            angles = ang
            measure_layer = d["measure_layer"]
            perturb_layer = d["perturb_layer"]
        else:
            assert np.allclose(ang, angles), f"angle grid mismatch in {fp}"

        name = d["direction_name"]
        source = d["source_name"].replace("_prompts_holdout", " holdout")
        c = COLORS.get(name, "k")
        l2 = d["l2"]                                  # (n_anchors, n_ang)
        med = np.median(l2, axis=0)
        lo = np.percentile(l2, 25, axis=0)
        hi = np.percentile(l2, 75, axis=0)
        ax.fill_between(ang, lo, hi, color=c, alpha=0.18)
        ax.plot(ang, med, "-", color=c, lw=2.4, label=name)

        # subsample this concept's random sweeps for the pooled baseline
        rflat = d["l2_random"].reshape(-1, d["l2_random"].shape[-1])
        rng = np.random.default_rng(args.seed + hash(concept) % (2**32))
        n_pick = min(args.n_random_per_concept, rflat.shape[0])
        idx = rng.choice(rflat.shape[0], size=n_pick, replace=False)
        pooled_random.append(rflat[idx])

    # one pooled random baseline curve (median + IQR) over all collected
    # (concept × random_dir × anchor) sweeps
    rand = np.concatenate(pooled_random, axis=0)
    rmed = np.median(rand, axis=0)
    rlo = np.percentile(rand, 25, axis=0)
    rhi = np.percentile(rand, 75, axis=0)
    ax.fill_between(angles, rlo, rhi, color="grey", alpha=0.18)
    ax.plot(angles, rmed, "--", color="grey", lw=1.6, label="Random")

    ax.axhline(50, color="black", ls=":", lw=1.2,
               label=r"Threshold ($L^2=50$)")

    ax.set_xlim(0, float(angles.max()))
    ax.set_xlabel(r"Perturbation angle $\theta$ (deg)", fontsize=12)
    ax.set_ylabel(r"$L^2$ distance", fontsize=12)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, fontsize=9, loc="upper left")

    plt.tight_layout()
    stem = args.out_name or (
        f"sweeps_holdout_{args.target}_L{args.layer}_"
        + "_".join(args.concepts)
    )
    out_png = os.path.join(FIG_DIR, f"{stem}.png")
    out_pdf = os.path.join(FIG_DIR, f"{stem}.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")


if __name__ == "__main__":
    main()
