#!/usr/bin/env python3
"""Plot KL-div and MSE vs dataset size for AE vs SAE on Qwen3-1.7B layer 10."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/l10_scaling_comparison")
OUT_DIR = Path("results/l10_scaling_comparison")

AE_RUNS = [
    (1_000_000,  "l10_1M_drop_ce_eval.json"),
    (2_000_000,  "l10_2M_drop_ce_eval.json"),
    (4_000_000,  "l10_4M_drop_ce_eval.json"),
    (6_000_000,  "l10_6M_drop_ce_eval.json"),
    (8_000_000,  "l10_8M_drop_ce_eval.json"),
    (10_000_000, "l10_10M_drop_ce_eval.json"),
]

SAE_RUNS = [
    (2_000_000,  "l10_doubled_k32_2M_ce_eval.json"),
    (10_000_000, "l10_doubled_k32_10M_ce_eval.json"),
]


def load(runs):
    xs, kls, mses = [], [], []
    for n, fname in runs:
        with open(RESULTS_DIR / fname) as f:
            d = json.load(f)
        xs.append(n)
        kls.append(d["kl_div_mean"])
        mses.append(d["mse_mean"])
    return xs, kls, mses


ae_x, ae_kl, ae_mse = load(AE_RUNS)
sae_x, sae_kl, sae_mse = load(SAE_RUNS)

for metric, ae_y, sae_y, ylabel, fname in [
    ("kl",  ae_kl,  sae_kl,  "KL divergence", "kl_vs_data.png"),
    ("mse", ae_mse, sae_mse, "MSE",           "mse_vs_data.png"),
]:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ae_x, ae_y, marker="o", label="Deep AE (67M params, k=32)")
    ax.plot(sae_x, sae_y, marker="s", label="TopK SAE (134M params, k=32, d_sae=32768)")
    ax.set_xlabel("Training activations")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"{ylabel} vs training data size (Qwen3-1.7B layer 10)")
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
