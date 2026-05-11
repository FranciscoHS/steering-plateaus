#!/bin/bash
# Anchor-source ablation sweep on gemma L=2 (Plans/exp_map_anchor_source.md).
# Re-runs all 528 pairs with anchors drawn from wiki_en, wiki_zh, code.
# Each phase uses --skip_existing so re-launching after a pod death resumes.
set -e
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
cd /network/steering-plateaus
echo "[$(date)] installing deps"
pip install -q transformers accelerate
echo "[$(date)] HF login"
python -c "from huggingface_hub import login; login(token='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')"

for SRC in wiki_en wiki_zh code; do
  echo "[$(date)] === sweep_2d --anchor_source $SRC ==="
  python -u scripts/exp_map/sweep_2d.py \
    --target gemma --layer 2 --anchor_source "$SRC" --skip_existing
  echo "[$(date)] === fit_pairs --anchor_source $SRC ==="
  python -u scripts/exp_map/fit_pairs.py \
    --target gemma --layer 2 --anchor_source "$SRC"
done

echo "[$(date)] done"
