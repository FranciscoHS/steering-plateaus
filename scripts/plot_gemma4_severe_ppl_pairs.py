"""Paired per-record PPL plot: no-AE vs +AE for the two attack settings.

Goal: distinguish whether the +AE PPL shift on leetrules is uniform across
records (continuous amplification) or outlier-driven (a small subset is
hard-flipped to gibberish, dragging the mean/median).

Reads gemma4_advbench_severe_ppl_qwen3.json (Qwen3-1.7B unconditional PPL
on Gemma-4 continuations) and writes a 2-panel scatter PNG.
"""
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

IN_JSON = "/home/francisco/Research/steering-plateaus/results/qwen3_advbench/gemma4_advbench_severe_ppl_qwen3.json"
OUT_PNG = "/home/francisco/Research/steering-plateaus/results/qwen3_advbench/gemma4_severe_ppl_pairs.png"


def get_records(d, key):
    return {r["idx"]: r["ppl"] for r in d["results"][key]["per_record"] if math.isfinite(r["ppl"])}


def paired(noAE, AE):
    xs, ys, idxs = [], [], []
    for k in sorted(set(noAE) & set(AE)):
        xs.append(noAE[k]); ys.append(AE[k]); idxs.append(k)
    return np.array(xs), np.array(ys), idxs


def main():
    with open(IN_JSON) as f:
        d = json.load(f)

    plain_noAE  = get_records(d, "plain_noAE")
    plain_AE    = get_records(d, "plain_AEfirst10")
    leet_noAE   = get_records(d, "leetrules_noAE")
    leet_AE     = get_records(d, "leetrules_AEfirst10")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, (lab, (no, ae)) in zip(
        axes,
        [("plain (no attack)", (plain_noAE, plain_AE)),
         ("leetrules jailbreak", (leet_noAE, leet_AE))],
    ):
        x, y, idxs = paired(no, ae)
        ratio = y / x
        ax.scatter(x, y, s=42, alpha=0.7, edgecolor="black", linewidth=0.5)
        for i, k in enumerate(idxs):
            if ratio[i] > 3 or ratio[i] < 1/3:  # annotate outliers
                ax.annotate(str(k), (x[i], y[i]), fontsize=7, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")
        lo, hi = 5, max(x.max(), y.max()) * 1.2
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, lw=0.8, label="y = x")
        ax.plot([lo, hi], [2 * lo, 2 * hi], ":", color="gray", alpha=0.6, lw=0.8, label="y = 2x")
        ax.plot([lo, hi], [0.5 * lo, 0.5 * hi], ":", color="gray", alpha=0.6, lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("PPL no AE")
        ax.set_ylabel("PPL with AE (first10)")
        n_up = int((ratio > 1).sum()); n_dn = int((ratio < 1).sum())
        med_ratio = float(np.median(ratio))
        geo_ratio = float(np.exp(np.mean(np.log(ratio))))
        ax.set_title(
            f"{lab}\nn={len(x)}  median(y/x)={med_ratio:.2f}  geomean(y/x)={geo_ratio:.2f}  "
            f"({n_up} up, {n_dn} down)"
        )
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, which="both", alpha=0.2)

    fig.suptitle("Qwen3-1.7B PPL of Gemma-4 continuations: no-AE vs AE-first10 (paired by record)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")

    print("\nPer-record ratios (PPL_AE / PPL_noAE):")
    for lab, (no, ae) in [("plain", (plain_noAE, plain_AE)), ("leetrules", (leet_noAE, leet_AE))]:
        x, y, idxs = paired(no, ae)
        r = y / x
        order = np.argsort(r)[::-1]
        print(f"\n  {lab}: median ratio = {np.median(r):.2f}, geomean = {np.exp(np.mean(np.log(r))):.2f}")
        print(f"  top-5 up:   " + "  ".join(f"idx={idxs[i]:2d} {x[i]:6.1f}->{y[i]:7.1f} x{r[i]:5.2f}" for i in order[:5]))
        print(f"  top-5 down: " + "  ".join(f"idx={idxs[i]:2d} {x[i]:6.1f}->{y[i]:7.1f} x{r[i]:5.2f}" for i in order[-5:][::-1]))


if __name__ == "__main__":
    main()
