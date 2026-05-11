"""Cross-check: does raw DoM overlap predict the per-pair fitted n?

Hypothesis: language-language pairs have very non-orthogonal DoMs;
orthonormalization renormalizes the perpendicular residual, and the
resulting d2p represents a small "language-A minus language-B" nudge
that the model treats as correlated with d1p — producing n<2.

Computes |<d1, d2>| for each pair using the raw DoMs in
results/exp_map/data/directions/dirs_gemma_L2.pkl, joins to canonical n
from fits_gemma_L2.pkl, prints summary stats and a scatter, and writes a
PNG.
"""
from __future__ import annotations
import os, sys, pickle
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR_PKL  = "results/exp_map/data/directions/dirs_gemma_L2.pkl"
FITS_PKL = "results/exp_map/data/fits/fits_gemma_L2.pkl"
OUT_PNG  = "results/exp_map/figures/overlap_vs_n_gemma_L2.png"

# Group classification — based on registry.TARGETS conventions
LANG_NAMES = {"Arabic", "Chinese", "Dutch", "French", "German", "Italian",
              "Japanese", "Korean", "Portuguese", "Russian", "Spanish"}
CODE_NAMES = {"Cpp", "Go", "Haskell", "Java", "JavaScript", "Lisp",
              "Python", "Rust", "TypeScript"}


def kind(name: str) -> str:
    if name in LANG_NAMES: return "lang"
    if name in CODE_NAMES: return "code"
    return "sem"


def pair_kind(a: str, b: str) -> str:
    ka, kb = kind(a), kind(b)
    pair = tuple(sorted([ka, kb]))
    return "x".join(pair)


def median_p_for_pair(robust_fit_dict) -> float:
    ps = []
    for k, cell in robust_fit_dict.items():
        if k == "__levels__": continue
        if np.isfinite(cell.get("p_median", np.nan)):
            ps.append(cell["p_median"])
    return float(np.median(ps)) if ps else np.nan


def main():
    with open(DIR_PKL, "rb") as f:
        dblob = pickle.load(f)
    dirs = {k: v.cpu().numpy().astype(np.float64)
            for k, v in dblob["directions"].items()}
    # confirm unit norm
    norms = np.array([np.linalg.norm(v) for v in dirs.values()])
    print(f"DoM norms: min={norms.min():.4f} max={norms.max():.4f} "
          f"(should all be 1.0)")

    with open(FITS_PKL, "rb") as f:
        fits = pickle.load(f)

    rows = []
    for pair_key, fit in fits.items():
        a, b = pair_key
        if a not in dirs or b not in dirs:
            continue
        n = median_p_for_pair(fit)
        if not np.isfinite(n):
            continue
        ov = float(abs(np.dot(dirs[a], dirs[b])))
        rows.append({"a": a, "b": b, "kind": pair_kind(a, b),
                      "overlap": ov, "n": n})
    print(f"joined {len(rows)} pairs (fits ∩ dirs)")

    # Group stats
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    print("\nGroup stats (mean ± std over pairs):")
    print(f"{'group':12s}  {'n_pairs':>7s}  {'<|<d1,d2>|>':>14s}  "
          f"{'<n>':>10s}  {'std n':>8s}")
    for k, group in sorted(by_kind.items()):
        ovs = np.array([r["overlap"] for r in group])
        ns  = np.array([r["n"] for r in group])
        print(f"{k:12s}  {len(group):7d}  "
              f"{ovs.mean():.3f}±{ovs.std():.3f}  "
              f"{ns.mean():6.3f}    {ns.std():.3f}")

    # Pearson + Spearman correlation
    ovs = np.array([r["overlap"] for r in rows])
    ns  = np.array([r["n"] for r in rows])
    pearson = float(np.corrcoef(ovs, ns)[0, 1])
    rk_o = ovs.argsort().argsort()
    rk_n = ns.argsort().argsort()
    spearman = float(np.corrcoef(rk_o, rk_n)[0, 1])
    print(f"\nPearson r(|<d1,d2>|, n)  = {pearson:+.3f}  (all pairs)")
    print(f"Spearman ρ                = {spearman:+.3f}")

    # Within lang×lang
    ll = by_kind.get("langxlang", [])
    if ll:
        ovs_l = np.array([r["overlap"] for r in ll])
        ns_l  = np.array([r["n"] for r in ll])
        pl = float(np.corrcoef(ovs_l, ns_l)[0, 1])
        print(f"Pearson r within lang×lang only = {pl:+.3f}  "
              f"(n={len(ll)})")

    # Scatter
    KIND_COLORS = {
        "langxlang": "#3a6ea5",  "langxsem": "#7B3294", "langxcode": "#1f9d9d",
        "semxsem":   "#D9822B",  "codexsem": "#3aa54a", "codexcode": "#a53a3a",
    }
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for k in sorted(by_kind.keys()):
        group = by_kind[k]
        ax.scatter([r["overlap"] for r in group],
                   [r["n"] for r in group],
                   color=KIND_COLORS.get(k, "#888"),
                   s=18, alpha=0.7, edgecolor="none",
                   label=f"{k} (n={len(group)})")
    ax.axhline(2.0, ls="--", color="#444", lw=0.8, alpha=0.7)
    ax.set_xlabel(r"raw DoM overlap  $|\langle d_1, d_2 \rangle|$",
                  fontsize=11)
    ax.set_ylabel(r"fitted exponent $n$", fontsize=11)
    ax.set_title(rf"DoM overlap vs fitted $n$  (gemma L=2;  "
                 rf"Pearson r={pearson:+.2f}, Spearman ρ={spearman:+.2f})",
                 fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, facecolor="white")
    fig.savefig(OUT_PNG.replace(".png", ".pdf"), facecolor="white")
    print(f"\nsaved {OUT_PNG}")


if __name__ == "__main__":
    main()
