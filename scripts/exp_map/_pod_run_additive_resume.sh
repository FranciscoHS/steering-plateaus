#!/bin/bash
# Resume the additive variant sweep on gemma L=2.
set -e
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
cd /network/steering-plateaus
echo "[$(date)] installing deps"
pip install -q transformers accelerate
echo "[$(date)] HF login"
python -c "from huggingface_hub import login; login(token='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')"
echo "[$(date)] launching sweep_2d --mode additive (skip_existing)"
python -u scripts/exp_map/sweep_2d.py \
  --target gemma --layer 2 --mode additive --skip_existing
echo "[$(date)] done"
