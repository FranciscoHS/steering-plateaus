#!/bin/bash
# Run on the Random pod. Pipes everything to /workspace/random_run.log.
set -e
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
cd /workspace/steering-plateaus
echo "[$(date)] installing deps"
pip install -q transformers accelerate
echo "[$(date)] HF login"
python -c "from huggingface_hub import login; login(token='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')"
# Random direction cache already shipped with the synced directions/ dir;
# regenerate locally for reproducibility (idempotent).
echo "[$(date)] (re)building random direction cache"
python -u scripts/exp_map/random_directions.py
echo "[$(date)] launching sweep_2d --directions_pkl"
python -u scripts/exp_map/sweep_2d.py \
  --target gemma --layer 2 \
  --directions_pkl results/exp_map/data/directions/dirs_gemma_L2_random.pkl \
  --skip_existing
echo "[$(date)] done"
