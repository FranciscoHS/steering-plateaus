#!/bin/bash
# Phase 1 pilot for Plans/exp_map_mwe_concepts.md.
# Extract DoM for caa_sycophancy, run 33 cross-pair sweeps with the existing
# semantic+lang+code directions, fit, and dump overlap diagnostics.
set -e
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
cd /workspace/steering-plateaus
echo "[$(date)] installing deps"
pip install -q transformers accelerate
echo "[$(date)] HF login"
python -c "from huggingface_hub import login; login(token='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')"

echo "[$(date)] extracting caa_sycophancy DoM (gemma L=2)"
python -u - <<'PY'
import sys, pickle
sys.path.insert(0, ".")
from src.model import load_model
from scripts.exp_map.lib import directions as dirlib, registry
import torch
import numpy as np

model, tok, dev = load_model("gemma-2-9b", dtype=torch.bfloat16)
blob = dirlib.extract_all(model, tok, dev, "gemma", 2,
                           names=["caa_sycophancy"])
d = blob["directions"]["caa_sycophancy"]
print(f"  ||d_sycophancy|| = {d.norm().item():.4f}")
existing = [n for n in blob["directions"] if n != "caa_sycophancy"]
overlaps = sorted(((abs(float(torch.dot(d, blob["directions"][n]))), n)
                   for n in existing), reverse=True)
print("\n  top-5 overlaps with existing DoMs:")
for o, n in overlaps[:5]: print(f"    {n:24s}  |<.,.>| = {o:.4f}")
print("\n  bottom-5 overlaps:")
for o, n in overlaps[-5:]: print(f"    {n:24s}  |<.,.>| = {o:.4f}")
flagged = [(o, n) for (o, n) in overlaps if o > 0.3]
print(f"\n  flagged (overlap > 0.3): {len(flagged)}")
for o, n in flagged: print(f"    {n}  {o:.4f}")
PY

echo "[$(date)] running sweep_2d for 33 caa_sycophancy x <existing> pairs"
PAIRS=$(python -c "
import sys; sys.path.insert(0, '.')
from scripts.exp_map.lib import registry
existing = list(registry.DIRECTIONS) + list(registry.LANG_KEYS) + list(registry.CODE_KEYS)
print(';'.join(f'caa_sycophancy,{n}' for n in existing))
")
python -u scripts/exp_map/sweep_2d.py \
  --target gemma --layer 2 \
  --pairs "$PAIRS" \
  --skip_existing

echo "[$(date)] fitting just the 33 new pilot pairs"
python -u - <<'PY'
import sys, os, glob, pickle
sys.path.insert(0, ".")
import numpy as np
from scripts.exp_map.lib.superellipse import robust_p_fit

pat = ("results/exp_map/data/sweeps_2d/"
       "sweep2d_gemma_L2_*caa_sycophancy*_fineweb_60deg.pkl")
files = sorted(glob.glob(pat))
print(f"  {len(files)} pilot sweep pkls")

pilot_fits = {}
rows = []
for fp in files:
    with open(fp, "rb") as f: d = pickle.load(f)
    a, b = d["direction_labels"]
    fit = robust_p_fit(d["angles_deg"], d["l2"], n_bootstrap=200)
    pilot_fits[(a, b)] = fit
    ps = []
    for k, v in fit.items():
        if k == "__levels__": continue
        if np.isfinite(v["p_lo"]) and np.isfinite(v["p_hi"]):
            ps += [v["p_lo"], v["p_hi"]]
    if ps:
        rows.append({"pair": f"{a} x {b}",
                     "p_lo": float(min(ps)),
                     "p_hi": float(max(ps)),
                     "p_med": float(np.median([
                         v["p_median"] for k, v in fit.items()
                         if k != "__levels__" and
                         np.isfinite(v.get("p_median", np.nan))]))})
    else:
        rows.append({"pair": f"{a} x {b}", "p_lo": np.nan,
                     "p_hi": np.nan, "p_med": np.nan})

# Also pull the raw DoM overlap into the table.
with open("results/exp_map/data/directions/dirs_gemma_L2.pkl", "rb") as f:
    blob = pickle.load(f)
import torch
d_syc = blob["directions"]["caa_sycophancy"]
ovs = {n: abs(float(torch.dot(d_syc, blob["directions"][n])))
       for n in blob["directions"] if n != "caa_sycophancy"}

rows = sorted(rows, key=lambda r: r["p_med"] if np.isfinite(r["p_med"]) else 99)
print(f"\n  {'pair':40s} {'overlap':>8s} {'p_med':>7s} {'p_range':>14s}")
for r in rows:
    other = r["pair"].split(" x ")[1]
    ov = ovs.get(other, np.nan)
    if np.isfinite(r["p_lo"]):
        rng = f"[{r['p_lo']:.2f},{r['p_hi']:.2f}]"
    else:
        rng = "no fit"
    pm = f"{r['p_med']:.3f}" if np.isfinite(r['p_med']) else "  nan"
    print(f"  {r['pair']:40s}  {ov:6.4f}  {pm:>7s}  {rng:>14s}")

os.makedirs("results/exp_map/data/fits", exist_ok=True)
with open("results/exp_map/data/fits/fits_gemma_L2_pilot_caa_sycophancy.pkl", "wb") as f:
    pickle.dump(pilot_fits, f)
print(f"\n  saved fits_gemma_L2_pilot_caa_sycophancy.pkl ({len(pilot_fits)} pairs)")
PY

echo "[$(date)] done"
