#!/bin/bash
# Phase 2 + 3 of Plans/exp_map_mwe_concepts.md.
# (1) Extract DoMs for all 177 MWE concepts (~1.7 GPU-hr).
# (2) Select bottom-K lowest-overlap MWE-touching pairs (overlap < 0.15).
# (3) Run sweep_2d.py on those (default ~1000 pairs => ~6 GPU-hr).
#
# Args (positional):
#   $1 = K (default 1000)        target pair count for the sweep
#   $2 = TAU (default 0.15)      overlap threshold
#   $3 = SHARD (optional, e.g. "0/4") for parallel pods
set -e
K="${1:-1000}"
TAU="${2:-0.15}"
SHARD="${3:-}"
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
cd /workspace/steering-plateaus
echo "[$(date)] installing deps"
pip install -q transformers accelerate
echo "[$(date)] HF login"
python -c "from huggingface_hub import login; login(token='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')"

echo "[$(date)] Phase 2 — extracting DoMs for all 177 MWE concepts"
python -u - <<'PY'
import sys
sys.path.insert(0, ".")
from src.model import load_model
from scripts.exp_map.lib import directions as dirlib, registry
import torch
model, tok, dev = load_model("gemma-2-9b", dtype=torch.bfloat16)
all_names = (list(registry.DIRECTIONS) + list(registry.LANG_KEYS)
             + list(registry.CODE_KEYS) + list(registry.MWE_KEYS))
print(f"[ext] requesting {len(all_names)} directions ("
      f"{len(registry.MWE_KEYS)} MWE)")
blob = dirlib.extract_all(model, tok, dev, "gemma", 2, names=all_names)
print(f"[ext] done; {len(blob['directions'])} dirs in cache")
PY

echo "[$(date)] Phase 3a — selecting lowest-overlap pair list"
python -u scripts/exp_map/select_mwe_pairs.py \
  --target gemma --layer 2 --tau "$TAU" --k "$K"

PAIRS_FILE="results/exp_map/data/pair_lists/mwe_pairs_gemma_L2_ov${TAU}_top${K}.csv"
echo "[$(date)] Phase 3b — running sweep_2d on $PAIRS_FILE (shard='$SHARD')"

if [ -n "$SHARD" ]; then
  python -u scripts/exp_map/sweep_2d.py \
    --target gemma --layer 2 \
    --pairs_file "$PAIRS_FILE" \
    --shard "$SHARD" \
    --skip_existing
else
  python -u scripts/exp_map/sweep_2d.py \
    --target gemma --layer 2 \
    --pairs_file "$PAIRS_FILE" \
    --skip_existing
fi

echo "[$(date)] done"
